# CLIP Model Adapter

## Introduction

This repo is a model integration between [OpenAI's CLIP](https://github.com/openai/CLIP)
model and [Dataloop](https://dataloop.ai/)

CLIP is a neural network trained on image and text pairs that can query the most relevant images for a given text
snippet. It works without output specific task optimizing, similar to the GPT models' zero-shot capabilities. CLIP
matches performance of ResNet50 on ImageNet without using its labeled examples, overcoming major challenges in computer
vision.

The model can generate embeddings for the text queries and the images to the same feature vector space such that
search results are returned based on similarity.

Additionally the model can be fine-tuned to improve search performance on specific datasets with caption annotations.

## Model Details

The CLIP model was developed by researchers at OpenAI to learn about what contributes to robustness in computer vision
tasks. The model was also developed to test the ability of models to generalize to arbitrary image classification tasks
in a zero-shot manner. It was not developed for general model deployment - to deploy models like CLIP, researchers will
first need to carefully study their capabilities in relation to the specific context they’re being deployed within.

## Requirements

* `dtlpy`
* `ftfy`
* `matplotlib`
* `pandas`
* `regex`
* `scikit-learn`
* `git+https://github.com/openai/CLIP.git`
* An account in the [Dataloop platform](https://console.dataloop.ai/)

## Installation

To install the package and create the CLIP model adapter, you will need
a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and
a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) of images
in the Dataloop platform.

### Model Fine-tuning

For training, _**_items must be converted into prompt item objects**_, with the image as the prompt
and the corresponding caption as the response (i.e. a text annotation).

First, upload all images into a dataset in the Dataloop platform. This will also serve as the dataset to be searched 
after CLIP training is complete.

Then, create a prompt items dataset with the images captions that points to the images in the first dataset. You can 
find some example code and functions in the utils script [here](./utils/prepare_dataset.py).

Make sure the dataset has training and validation subsets are defined in the prompt items dataset (see docs 
[here](https://developers.dataloop.ai/tutorials/model_management/marketplace/chapter/#define-dataset-subsets) for 
further SDK information, or use ML Data Split in the dataset browser of the Dataloop platform).


### Editing the configuration

To edit configurations via the platform, find the CLIP model page in the Model Management and edit the json
file displayed there.

To edit via the SDK, change the model configuration.
Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more
information.

The basic configurations included are:
* ```model_name```: architecture of the model (default: 'ViT-B/32', available options: 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px', however use of some other models may require more powerful podtypes)
* ```batch_size```: batch size to be used during the training (default: 128)
* ```num_epochs```: number of epochs to train the model (default: 100)
* ```early_stop```: boolean for early stopping (default: ```True```)
* ```early_stopping_epochs```: number of epochs to wait before stopping training if no improvement (default: 5)
* ```betas```: betas for the optimizer (default: (0.9, 0.98))
* ```epsilon```: epsilon for the optimizer (default: 1e-6)
* ```learning_rate```: learning rate for the optimizer (default: 5e-8)
* ```weight_decay```: weight decay for the optimizer (default: 0.001)

Note: CLIP finetuning is very sensitive to hyperparameter values. It is recommended to use as large a batch size as 
possible, and to use smaller learning rate and weight decay if the model is not converging.

## Deployment

After installing the pretrained model or fine-tuning it on your data, it is necessary to deploy the model in order for
it to be available for prediction.

## Attribution

This application uses OpenAI's CLIP, which is licensed under the MIT License. CLIP is a powerful open-source model for image and text understanding developed by OpenAI.

## Acknowledgments

The CLIP model and code are the intellectual property of OpenAI.
Thank you to the contributors of the CLIP project for their work in advancing multimodal AI research.

## Sources and Further Reading

* [OpenAI documentation](https://openai.com/index/clip/)
