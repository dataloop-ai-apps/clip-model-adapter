import os
import clip
import json
import logging
import time
import dtlpy as dl
import numpy as np
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger('[openai-clip]')


# clip available models: ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

class ImageTextDataset(Dataset):
    def __init__(self, list_image_path, list_txt, preprocess):
        self.image_path = list_image_path
        # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.title = clip.tokenize(list_txt, context_length=77, truncate=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_path[idx]))  # Image from PIL module
        title = self.title[idx]
        return image, title


class ClipAdapter(dl.BaseModelAdapter):
    """
    Model Adapter for CLIP text and image embedding model from OpenAI
    """

    def load(self, local_path, **kwargs):
        self.arch_name = self.configuration.get("model_name", "ViT-B/32")
        self.weights_filename = self.configuration.get('weights_filename', 'best.pt')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.arch_name not in clip.available_models():
            raise ValueError(f"Model {self.arch_name} is not an available architecture for CLIP.")
        model_filepath = os.path.join(local_path, self.weights_filename) if Path(
            self.weights_filename).stem not in clip.available_models() \
            else self.weights_filename
        self.model, self.preprocess = clip.load(name=self.arch_name, device=self.device, jit=False)
        if os.path.isfile(model_filepath) is True:  # and self.model.status != 'pre-trained':
            checkpoint = torch.load(model_filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.info("No previously saved model found, loading from default pre-trained weights.")
        self.model.eval()
        logger.info(f"Loaded model CLIP {self.arch_name} successfully")

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally
            the function is called in save_to_model which first save locally and then uploads to model entity

        :param local_path: `str` directory path in local FileSystem
        """
        model_path = os.path.join(local_path, self.weights_filename)
        torch.save({'model_state_dict': self.model.state_dict()}, model_path)
        logger.info(f"Model saved to {model_path}")

    def prepare_item_func(self, item: dl.Item):
        return item

    def embed(self, batch, **kwargs):
        embeddings = []
        for i, item in enumerate(batch):
            if 'image/' in item.mimetype:
                item_img = Image.fromarray(item.download(save_locally=False, to_array=True))
                image = self.preprocess(item_img).unsqueeze(0).to(self.device)
                features = self.model.encode_image(image)
                embedding = features[0].cpu().detach().numpy().tolist()
            elif 'text/' in item.mimetype:
                text = item.download(save_locally=False).read().decode()
                tokens = clip.tokenize([text], context_length=77, truncate=True).to(self.device)
                features = self.model.encode_text(tokens)
                embedding = features[0].cpu().detach().numpy().tolist()
            else:
                logger.info(
                    f'Unsupported mimetype for CLIP: {type(item)}. '
                    f'Features not extracted for item {item.name} ID {item.id}, skipping.')
                embedding = None
            embeddings.append(embedding)
        return embeddings

    def train(self, data_path, output_path, **kwargs):
        self.model.to(device=self.device)
        self.model.train()

        batch_size = self.configuration.get('batch_size', 32)
        num_epochs = self.configuration.get('num_epochs', 20)
        learning_rate = self.configuration.get('learning_rate', 5e-5)
        betas = self.configuration.get('betas', (0.9, 0.98))
        episilon = self.configuration.get('episilon', 1e-6)
        weight_decay = self.configuration.get('weight_decay', 0.2)

        # early stopping params
        best_loss = np.inf
        not_improving_epochs = 0
        early_stop = self.configuration.get('early_stopping', True)
        early_stopping_epochs = self.configuration.get('early_stopping_epochs', 5)
        end_training = False

        logger.info("Model set to train mode.")

        ################
        # prepare data #
        ################
        # use downloaded prompt items to get image and text pairs
        train_items, train_captions = self.get_img_txt_pairs(os.path.join(data_path, 'train'))
        val_items, val_captions = self.get_img_txt_pairs(os.path.join(data_path, 'validation'))
        train_dataset = ImageTextDataset(train_items, train_captions, self.preprocess)
        val_dataset = ImageTextDataset(val_items, val_captions, self.preprocess)

        dataloaders = {'train': DataLoader(train_dataset,
                                           batch_size=batch_size),
                       'val': DataLoader(val_dataset,
                                         batch_size=batch_size)}
        logger.info(
            f"Dataloaders created. Train dataset: {len(train_dataset)} items, validation dataset: "
            f"{len(val_dataset)} items.")

        #################
        # prepare model #
        #################
        if self.device == "cpu":
            self.model.float()

        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=learning_rate,
                                     betas=betas,
                                     eps=episilon,
                                     weight_decay=weight_decay)

        for epoch in range(num_epochs):
            if end_training:
                break
            logger.info('Epoch {}/{} Start...'.format(epoch, num_epochs))
            tepoch_time = time.time()
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                total_imgs = 0

                with tqdm(dataloaders[phase], unit='batch', desc=f"Epoch {epoch}/{num_epochs} - {phase} phase: ") as tepoch:
                    for idx, batch in enumerate(tepoch):
                        optimizer.zero_grad()

                        images, texts = batch
                        images = images.to(self.device)
                        texts = texts.to(self.device)
                        num_pairs = len(images)
                        if num_pairs == 1:
                            logger.warning("Must have batch size > 1. Skipping item.")
                            continue

                        # forward pass for model predictions
                        logits_per_image, logits_per_text = self.model(images, texts)

                        # calc ground truth + loss
                        ground_truth = torch.arange(num_pairs, dtype=torch.long, device=self.device)
                        total_loss = (loss_img(logits_per_image, ground_truth) +
                                      loss_txt(logits_per_text, ground_truth)) / 2
                        # backprop
                        if phase == 'train':
                            total_loss.backward()

                            if self.device == "cpu":
                                optimizer.step()
                            else:
                                ClipAdapter._convert_models_to_fp32(self.model)
                                optimizer.step()
                                clip.model.convert_weights(self.model)
                            tepoch.set_postfix(Training_loss=f"{total_loss.item():.4f}")

                        # statistics
                        total_imgs += num_pairs
                        running_loss += (total_loss.item() * num_pairs)
                        epoch_loss = running_loss / total_imgs

                        if phase == "val":
                            val_loss = epoch_loss

                    logger.info(
                        f'Epoch {epoch}/{num_epochs} - {phase} '
                        f'Loss: {total_loss.item():.4f},'
                        f'Duration {(time.time() - tepoch_time):.2f}')

                    self.model_entity.metrics.create(samples=dl.PlotSample(figure='loss',
                                                                           legend=phase,
                                                                           x=epoch,
                                                                           y=epoch_loss),
                                                     dataset_id=self.model_entity.dataset_id)

            if val_loss < best_loss:
                not_improving_epochs = 0
                best_loss = val_loss
                logger.info(
                    f'Best validation loss decreased ({best_loss:.4f} --> {val_loss:.4f}). Saving model ...')
                torch.save({'model_state_dict': self.model.state_dict()},
                           os.path.join(output_path, self.weights_filename))
            else:
                not_improving_epochs += 1
            if not_improving_epochs > early_stopping_epochs and early_stop is True:
                logger.info("Early stop achieved at epoch ", epoch + 1)
                end_training = True
        return

    def convert_from_dtlpy(self, data_path, **kwargs):
        # Subsets validation
        subsets = self.model_entity.metadata.get("system", {}).get("subsets", None)
        if 'train' not in subsets:
            raise ValueError('Could not find train set. CLIP requires train and validation set for training. '
                             'Add a train set DQL filter in the dl.Model metadata')
        if 'validation' not in subsets:
            raise ValueError('Could not find validation set. CLIP requires train and validation set for training. '
                             'Add a validation set DQL filter in the dl.Model metadata')

        for subset, filters_dict in subsets.items():
            filters = dl.Filters(custom_filter=filters_dict)
            pages = self.model_entity.dataset.items.list(filters=filters)
            if pages.items_count == 0:
                raise ValueError(f'Could not find items with free-text annotations in subset {subset}. '
                                 f'Make sure there are items with annotations in the data subsets.')

    @staticmethod
    def get_img_txt_pairs(data_path, overwrite=False):
        logger.debug(f"Data path: {data_path}")
        path = Path(data_path)
        # list all downloaded prompt item jsons and download images from link
        item_jsons = (path / "items").rglob("*.json")
        with ThreadPoolExecutor() as executor:
            image_paths = list(
                executor.map(lambda item_file: ClipAdapter._download_stream(item_file, overwrite), item_jsons))
        # image_paths = []  # DEBUG
        # for item_file in item_jsons:
        #     image_paths.append(ClipAdapter._download_stream(item_file, overwrite))

        item_captions = []
        annots_files = (path / 'json').rglob("*.json")
        for src_file in annots_files:
            with open(src_file, 'r') as f:
                data = json.load(f)
            if len(data['annotations']) > 0:
                annot = data['annotations'][0]
                if annot['label'] == 'free-text':
                    item_captions.append(annot.get('coordinates', ''))
                else:
                    raise TypeError(
                        f"No free-text annotation found in json file {src_file}. Please check annotation type.")
            else:
                raise ValueError(f"No annotations found in json file {src_file} to use as image caption.")

        for root, dirs, files in os.walk(data_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
        return image_paths, item_captions

    @staticmethod
    def _download_stream(item_file, overwrite=False):
        with open(item_file) as json_data:
            d = json.load(json_data)
        img_prompt = next(iter(d['prompts'].values()))

        item_url = None
        for dictionary in img_prompt:
            if dictionary.get("mimetype") == "image/*":
                item_url = dictionary.get("value")
                break
        if item_url is None:
            raise ValueError(f"Image URL not found in prompt item {Path(item_file).name}.")
        item_id = item_url.split('/')[-2]
        item = dl.items.get(item_id=item_id)
        download_path = item.download(local_path=Path(item_file).parents[0])
        image_name = Path(item_file).stem + Path(download_path).suffix
        new_path = Path(item_file).parents[0] / image_name
        try:
            os.rename(Path(download_path), new_path)
        except FileExistsError:
            if overwrite is True:
                logger.debug(f"Overwriting file {new_path}.")
                os.remove(new_path)
                os.rename(Path(download_path), new_path)
            else:
                logger.debug(f"File {new_path} already exists. Skipping.")
        return new_path

    @staticmethod
    def _convert_models_to_fp32(model):
        for p in model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()
