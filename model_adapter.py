import os
import clip
import json
import time
import datetime
import logging
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
    def __init__(self, data_path, preprocess):
        json_path = os.path.join(data_path, "json")
        image_path = os.path.join(data_path, "images")
        self.pairs = []
        json_files = Path(json_path).rglob("*.json")
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            for item in data:
                image_filepath = os.path.join(image_path, item['filename'][1:])
                if not os.path.exists(image_filepath):
                    continue
                # check if its a prompt item
                if Path(image_filepath).suffix == '.json':
                    image_filepath = ClipAdapter._download_stream(image_filepath)

                    if item["annotationsCount"] > 0:
                        annot = item["annotations"][0]
                        if annot["label"] == "free-text":
                            caption = annot.get("coordinates", "")
                        else:
                            raise TypeError(
                                f"No free-text annotation found in json file {image_filepath}. Please check item and annotation."
                            )
                    else:
                        raise ValueError(f"No annotations found in json file {image_filepath} to use as image caption.")
                else:
                    caption = item.get("metadata", {}).get("system", {}).get("description", "")

                self.pairs.append({"image_filepath": image_filepath, "caption": caption})
        # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.preprocess = preprocess

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Image
        item = self.pairs[idx]
        image = self.preprocess(Image.open(item['image_filepath']).convert('RGB'))  # Image from PIL module
        # Text
        title = clip.tokenize(item['caption'], context_length=77, truncate=True)
        return image, title


class ClipAdapter(dl.BaseModelAdapter):
    """
    Model Adapter for CLIP text and image embedding model from OpenAI
    """

    def load(self, local_path, **kwargs):
        self.arch_name = self.configuration.get("model_name", "ViT-B/32")
        self.configuration['embeddings_size'] = 512
        self.weights_filename = self.configuration.get('weights_filename', 'best.pt')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.arch_name not in clip.available_models():
            raise ValueError(f"Model {self.arch_name} is not an available architecture for CLIP.")
        model_filepath = (
            os.path.join(local_path, self.weights_filename)
            if Path(self.weights_filename).stem not in clip.available_models()
            else self.weights_filename
        )
        self.model, self.preprocess = clip.load(name=self.arch_name, device=self.device, jit=False)
        if os.path.isfile(model_filepath) is True:  # and self.model.status != 'pre-trained':
            checkpoint = torch.load(model_filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.info("No previously saved model found, loading from default pre-trained weights.")
        self.model.eval()
        logger.info(f"Loaded model CLIP {self.arch_name} successfully")

    def save(self, local_path, **kwargs):
        """
        Saves configuration and weights locally

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
                    f"Unsupported mimetype for CLIP: {type(item)}. "
                    f"Features not extracted for item {item.name} ID {item.id}, skipping."
                )
                embedding = None
            embeddings.append(embedding)
        return embeddings

    def prepare_data(
        self, dataset: dl.Dataset, root_path=None, data_path=None, output_path=None, overwrite=False, **kwargs
    ):
        # define paths
        dataloop_path = dl.service_defaults.DATALOOP_PATH
        root_path = self.adapter_defaults.resolve("root_path", root_path)
        data_path = self.adapter_defaults.resolve("data_path", data_path)
        output_path = self.adapter_defaults.resolve("output_path", output_path)

        if root_path is None:
            now = datetime.datetime.now()
            root_path = os.path.join(
                dataloop_path,
                "model_data",
                "{s_id}_{s_n}".format(s_id=self.model_entity.id, s_n=self.model_entity.name),
                now.strftime("%Y-%m-%d-%H%M%S"),
            )
        if data_path is None:
            data_path = os.path.join(root_path, 'datasets', self.model_entity.dataset.id)
            os.makedirs(data_path, exist_ok=True)
        if output_path is None:
            output_path = os.path.join(root_path, 'output')
            os.makedirs(output_path, exist_ok=True)

        if len(os.listdir(data_path)) > 0:
            self.logger.warning(f"Data path directory ({data_path}) is not empty..")

        # Download the subset items
        subsets = self.model_entity.metadata.get("system", dict()).get("subsets", None)
        if subsets is None:
            raise ValueError("Model (id: {}) must have subsets in metadata.system.subsets".format(self.model_entity.id))
        for subset, filters_dict in subsets.items():
            filters = dl.Filters(custom_filter=filters_dict)
            data_subset_base_path = os.path.join(data_path, subset)
            if os.path.isdir(data_subset_base_path) and not overwrite:
                # existing and dont overwrite
                self.logger.debug(f"Subset {subset!r} Existing (and overwrite=False). Skipping.")
                continue
            else:
                self.logger.debug(f"Downloading subset {subset!r} of {self.model_entity.dataset.name}")

            jsons_path = os.path.join(data_subset_base_path, "json")
            os.makedirs(jsons_path, exist_ok=True)
            jsons = dataset.export(filters=filters, local_path=jsons_path, include_annotations=True)
            images_path = os.path.join(data_subset_base_path, "images")
            os.makedirs(images_path, exist_ok=True)
            images = dataset.items.download(filters=filters, local_path=images_path, to_items_folder=False)

        return root_path, data_path, output_path

    def train(self, data_path, output_path, **kwargs):
        self.model.to(device=self.device)
        self.model.train()

        batch_size = self.configuration.get('batch_size', 128)
        num_epochs = self.configuration.get('num_epochs', 100)
        betas = self.configuration.get('betas', (0.9, 0.98))
        episilon = self.configuration.get('episilon', 1e-6)
        learning_rate = self.configuration.get('learning_rate', 5e-8)
        weight_decay = self.configuration.get('weight_decay', 0.001)

        # early stopping params
        best_loss = np.inf
        not_improving_epochs = 0
        early_stop = self.configuration.get('early_stopping', True)
        early_stopping_epochs = self.configuration.get('early_stopping_epochs', 5)
        end_training = False

        # Callback for updating progress bar
        faas_callback = kwargs.get('on_epoch_end_callback')

        logger.info("Model set to train mode.")

        ################
        # prepare data #
        ################
        # Use downloaded items to get image and text pairs
        train_dataset = ImageTextDataset(data_path=os.path.join(data_path, 'train'), preprocess=self.preprocess)
        val_dataset = ImageTextDataset(data_path=os.path.join(data_path, 'validation'), preprocess=self.preprocess)

        # DataLoaders with optimizations
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        }
        logger.info(
            f"Dataloaders created. Train dataset: {len(train_dataset)} items, validation dataset: "
            f"{len(val_dataset)} items. Batch size: {batch_size}, Num workers: 4, Pin memory: {self.device.type == 'cuda'}"
        )
        logger.info(
            f"Training for {num_epochs} epochs. Learning rate: {learning_rate}, Weight decay: {weight_decay}, Early stopping: {early_stop}"
        )

        #################
        # prepare model #
        #################
        if self.device == "cpu":
            self.model.float()

        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=betas, eps=episilon, weight_decay=weight_decay
        )

        for epoch in range(num_epochs):
            if end_training:
                break
            logger.info(f"Epoch {epoch+1}/{num_epochs} Start...")
            tepoch_time = time.time()
            val_loss = None  # Initialize val_loss for each epoch
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                total_imgs = 0

                with tqdm(
                    dataloaders[phase], unit='batch', desc=f"Epoch {epoch+1}/{num_epochs} - {phase} phase: "
                ) as tepoch:
                    for idx, batch in enumerate(tepoch):
                        images, texts = batch
                        images = images.to(self.device)  # [B, 3, 224, 224]
                        texts = texts.to(self.device).squeeze(1)  # [B, 77]
                        num_pairs = len(images)
                        if num_pairs == 1:
                            logger.warning("Must have batch size > 1. Skipping item.")
                            continue

                        ground_truth = torch.arange(num_pairs, dtype=torch.long, device=self.device)
                        if phase == 'train':
                            optimizer.zero_grad()
                            logits_per_image, logits_per_text = self.model(images, texts)
                            total_loss = (
                                loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)
                            ) / 2
                            total_loss.backward()

                            if self.device == "cpu":
                                optimizer.step()
                            else:
                                ClipAdapter._convert_models_to_fp32(self.model)
                                optimizer.step()
                                clip.model.convert_weights(self.model)
                            tepoch.set_postfix(Training_loss=f"{total_loss.item():.4f}")
                        else:
                            with torch.no_grad():
                                logits_per_image, logits_per_text = self.model(images, texts)
                                total_loss = (
                                    loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)
                                ) / 2

                        # statistics
                        total_imgs += num_pairs
                        running_loss += total_loss.item() * num_pairs
                        epoch_loss = running_loss / total_imgs

                        if phase == "val":
                            val_loss = epoch_loss

                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - {phase} "
                    f"Loss: {total_loss.item():.4f}, "
                    f"Duration {(time.time() - tepoch_time):.2f}"
                )

                self.model_entity.metrics.create(
                    samples=dl.PlotSample(figure='loss', legend=phase, x=epoch + 1, y=epoch_loss),
                    dataset_id=self.model_entity.dataset_id,
                )


            if val_loss is not None and val_loss < best_loss:
                not_improving_epochs = 0
                best_loss = val_loss
                logger.info(
                    f'Best validation loss decreased (prev: {best_loss:.4f} --> new: {val_loss:.4f}). Saving model ...'
                )
                torch.save(
                    {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'best_loss': best_loss,
                    },
                    os.path.join(output_path, self.weights_filename),
                )
            else:
                not_improving_epochs += 1
                logger.info(f"Not improving epochs: {not_improving_epochs}")
            if not_improving_epochs > early_stopping_epochs and early_stop is True:
                logger.info(f"Early stop achieved at epoch {epoch + 1}")
                end_training = True
            if faas_callback is not None:
                faas_callback(epoch, num_epochs)

        return

    def convert_from_dtlpy(self, data_path, **kwargs):
        # Subsets validation
        subsets = self.model_entity.metadata.get("system", {}).get("subsets", None)
        if "train" not in subsets:
            raise ValueError(
                "Could not find train set in the model metadata. CLIP requires train and validation set for training. "
                "Add a train set DQL filter in the dl.Model metadata"
            )
        if "validation" not in subsets:
            raise ValueError(
                "Could not find validation set in the model metadata. CLIP requires train and validation set for training. "
                "Add a validation set DQL filter in the dl.Model metadata"
            )

        for subset, filters_dict in subsets.items():
            filters = dl.Filters(custom_filter=filters_dict)
            pages = self.model_entity.dataset.items.list(filters=filters)
            if pages.items_count == 0:
                raise ValueError(
                    f"Could not find matching items in subset {subset}. "
                    f"Make sure there are items in the data subsets."
                )

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
