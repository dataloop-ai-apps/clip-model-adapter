import os
import clip
import json
import time
import datetime
import logging
import dtlpy as dl
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger('[openai-clip]')
# Suppress PIL debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

# clip available models: ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']


class ImageTextDataset(Dataset):
    def __init__(self, data_path, preprocess, model_labels=None):
        json_path = os.path.join(data_path, "json")
        image_path = os.path.join(data_path, "images")

        self.classification_mode = None
        self.label_to_id = {}

        self.pairs = []
        json_files = Path(json_path).rglob("*.json")
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            for item in data:
                if self.classification_mode is None:
                    self.classification_mode = ImageTextDataset._check_classification_mode(item)
                    if model_labels is not None:
                        self.label_to_id = {label: i for i, label in enumerate(model_labels)}

                # get the image filepath. if it didn't download, skip
                image_filepath = os.path.join(image_path, item['filename'][1:])
                if not os.path.exists(image_filepath):
                    continue
                
                # get caption from label annotation
                if self.classification_mode is True:
                    if item["annotationsCount"] == 0:
                        logger.warning(f"No annotations found in item {item['id']} to use as image classification label. Skipping item.")
                        continue
                    # get the class annotation
                    class_annot = next((a for a in item["annotations"] if a["type"] == "class"), None)
                    if class_annot is None:
                        logger.warning(f"No class annotation found in item {item['id']}. Skipping item.")
                        continue
                    
                    label = class_annot.get("label")
                    try:
                        caption = self.label_to_id[label]
                    except:
                        self.label_to_id[label] = len(self.label_to_id)
                        caption = self.label_to_id[label]
                        logger.warning(f"Label {label} not found in model labels. Adding to model labels.")
                # check if its a prompt item
                elif Path(image_filepath).suffix == '.json':
                    image_filepath = ClipAdapter._download_stream(image_filepath)

                    if not item["annotationsCount"]:
                        logger.warning(f"No annotations found in json file {image_filepath} to use as image caption. Skipping item.")
                        continue
                    annot = item["annotations"][0]
                    if annot["label"] != "free-text":
                        logger.warning(f"No free-text annotation found in json file {image_filepath}. Skipping item.")
                        continue
                    caption = annot.get("coordinates", "")
                else:
                    # get caption from item description
                    caption = item["description"]
                    if caption == "":
                        logger.warning(f"No caption found in json file {image_filepath} to use as image caption. Skipping item.")
                        continue

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
        if hasattr(self, 'classification_mode') and self.classification_mode:
            # For classification mode, return the caption string directly
            title = [item['caption']]
        else:
            # For regular mode, tokenize the caption
            title = clip.tokenize(item['caption'], context_length=77, truncate=True)
        return image, title
    
    @staticmethod
    def _check_classification_mode(item):
        """Helper function to check if any of the annotations are of type 'class'"""
        try:
            for annot in item["annotations"]:
                if annot["type"] == "class":
                    mode = True
                    break
            else:
                mode = False
        except:
            mode = False
        return mode


class ClipAdapter(dl.BaseModelAdapter):
    """
    Model Adapter for CLIP text and image embedding model from OpenAI
    """

    def load(self, local_path, **kwargs):
        self.arch_name = self.configuration.get("model_name", "ViT-B/32")
        self.configuration['embeddings_size'] = 512
        self.weights_filename = self.configuration.get('weights_filename', 'best.pt')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model on device: {self.device}")

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

        self.as_classifier = self.configuration.get('as_classifier', False)
        # create label to id mapping for classifier
        if self.as_classifier is True:
            self.label_to_id = {}
            for i, label in enumerate(self.model_entity.labels):
                label_token = clip.tokenize(label, context_length=77, truncate=True)
                self.label_to_id[label_token] = i

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
        """
        Prepare data paths for dataset download for training
        """
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
            raise ValueError(
                f"Model (id: {self.model_entity.id}) must have subsets in metadata.system.subsets"
                "Add a subset DQL filters to the model metadata"
            )
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

            # Check that jsons and images directories are not empty
            if not os.listdir(jsons_path) or not os.listdir(images_path):
                raise ValueError(
                    f"No items were downloaded for subset {subset} with filter {filters_dict}."
                    "Please check items have been assigned to the subset."
                )

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

        # Progress bars
        progress = kwargs.get('progress', None)
        faas_callback = kwargs.get('on_epoch_end_callback')

        logger.info("Model set to train mode.")

        ################
        # prepare data #
        ################
        # Use downloaded items to get image and text pairs
        train_dataset = ImageTextDataset(
            data_path=os.path.join(data_path, 'train'),
            preprocess=self.preprocess,
            model_labels=self.model_entity.labels,
        )
        val_dataset = ImageTextDataset(
            data_path=os.path.join(data_path, 'validation'),
            preprocess=self.preprocess,
            model_labels=self.model_entity.labels,
        )

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

                        num_pairs = len(images)
                        if num_pairs == 1:
                            logger.warning("Must have batch size > 1. Skipping item.")
                            continue

                        if self.as_classifier is False: # for regular mode
                            texts = texts.to(self.device).squeeze(1)  # [B, 77]  #TODO See what this is for a normal dataset
                            ground_truth = torch.arange(num_pairs, dtype=torch.long, device=self.device)
                        else:
                            # example of texts when using classification mode:
                            # texts = [tensor([2, 3, 1, 1, 1, 0, 3, 3])]
                            # TODO FIX THIS
                            # some earlier auto generated code...
                            # # convert labels to label encoding
                            # texts = torch.tensor(
                            #     [self.label_to_id[label] for label in texts.squeeze(1)], dtype=torch.long, device=self.device
                            # )
                            ground_truth = torch.tensor(
                                [self.label_to_id[label] for label in texts.squeeze(1)], dtype=torch.long, device=self.device
                            )

                        if phase == 'train':
                            optimizer.zero_grad()
                            logits_per_image, logits_per_text = self.model(images, texts)
                            if self.as_classifier is False:
                                total_loss = (
                                    loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)
                                ) / 2
                            else:
                                total_loss = loss_img(logits_per_image, ground_truth)
                                logger.info(f"Logits per image: {logits_per_image.item()}") # DEBUG
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
                                if self.as_classifier is False:
                                    total_loss = (
                                        loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)
                                    ) / 2
                                else:
                                    total_loss = loss_img(logits_per_image, ground_truth)

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
                logger.info(
                    f'Best validation loss decreased (prev: {best_loss:.4f} --> new: {val_loss:.4f}). Saving model ...'
                )
                best_loss = val_loss
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

            if end_training is True:
                if progress is not None:
                    progress.update(
                        progress=100, message=f'Not improving after {not_improving_epochs} epochs, stopping training'
                    )
            elif faas_callback is not None:
                faas_callback(epoch, num_epochs)
        return

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
