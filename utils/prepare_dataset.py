import dtlpy as dl
import random, string
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor


class ClipPrepare(dl.BaseServiceRunner):
    @staticmethod
    def convert_dataset(dataset):
        dataset_to = ClipPrepare.convert_to_prompt_dataset(dataset_from=dataset)
        return dataset_to

    @staticmethod
    def convert_to_prompt_dataset(dataset_from: dl.Dataset):
        items = dataset_from.items.list()
        try:
            dataset_to = dataset_from.project.datasets.create(dataset_name=f"{dataset_from.name} prompt items")
        except Exception as e:
            print("Prompt item dataset already exists. Creating new prompt item dataset.")
            suffix = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(5))
            dataset_to = dataset_from.project.datasets.create(dataset_name=f"{dataset_from.name} prompt items-{suffix}")

        # use thread multiprocessing to get items and convert them to prompt items
        all_items = items.all()
        with ThreadPoolExecutor() as executor:
            _ = executor.map(lambda item: ClipPrepare._convert_item(item_id=item.id, dataset=dataset_to), all_items)
        # for item in items.all():
        #     item = dataset_from.items.get(item_id=item.id)
        #     _ = _convert_item(item, dataset_to)

        new_recipe = dataset_from.get_recipe_ids()[0]
        dataset_to.switch_recipe(new_recipe)
        return dataset_to

    # add captions for the item either from description or from directory name
    @staticmethod
    def _convert_item(item_id, dataset: dl.Dataset, existing_subsets=False):
        item = dl.items.get(item_id=item_id)
        if item.description is not None:
            caption = item.description
        else:
            print(f"Item {item.id} has no description. Trying directory name.")
            item_dir = item.dir.split('/')[-1]
            if item_dir != "":
                print(f"Using directory name: {item_dir}")
                caption = "this is a photo of a " + item_dir
            else:
                print(f"Item {item.id} has no directory name. Using empty string.")
                caption = ""
        new_name = Path(item.name).stem + '.json'

        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.IMAGE, "value": item.stream}]}  # role default is user
        )
        new_metadata = item.metadata
        if existing_subsets is True:
            new_metadata["system"] = new_metadata.get("system", {})
            new_metadata["system"]["subsets"] = item.metadata.get("system", {}).get("subsets", {})
        new_item = dataset.items.upload(
            prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True, item_metadata=new_metadata
        )
        prompt_item._item = new_item
        prompt_item.add(message={"role": "assistant", "content": [{"mimetype": dl.PromptType.TEXT, "value": caption}]})

        return new_item


if __name__ == "__main__":
    # dl.login()
    PROJECT_NAME = "test mars surface yy" # "<your project name>"
    DATASET_NAME = "Mars Surface Images with Captions" # "<your dataset name>"
    project = dl.projects.get(PROJECT_NAME)
    dataset = project.datasets.get(DATASET_NAME)
    prompt_dataset = ClipPrepare.convert_dataset(dataset)
