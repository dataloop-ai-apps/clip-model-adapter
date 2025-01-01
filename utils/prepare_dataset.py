import dtlpy as dl
import random, string
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor


class ClipPrepare:
    @staticmethod
    def convert_dataset(dataset):
        dataset_to = ClipPrepare.convert_to_prompt_dataset(dataset_from=dataset)
        return dataset_to

    @staticmethod
    def convert_to_prompt_dataset(dataset_from: dl.Dataset):
        items = dataset_from.items.list()
        try:
            dataset_to = dataset_from.project.datasets.get(dataset_name=f"{dataset_from.name} prompt items")
            if dataset_to.items_count > 0:
                suffix = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(5))
                dataset_to = dataset_from.project.datasets.create(
                    dataset_name=f"{dataset_from.name} prompt items-{suffix}")
        except dl.exceptions.BadRequest:
            suffix = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(5))
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
            if item_dir != '':
                print(f"Using directory name: {item_dir}")
                caption = "this is a photo of a " + item_dir
            else:
                print(f"Item {item.id} has no directory name. Using empty string.")
                caption = ''
        new_name = Path(item.name).stem + '.json'

        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(message={"content": [{"mimetype": dl.PromptType.IMAGE,  # role default is user
                                              "value": item.stream}]})
        new_item = dataset.items.upload(prompt_item,
                                        remote_name=new_name,
                                        remote_path=item.dir,
                                        overwrite=True)
        prompt_item._item = new_item
        prompt_item.add(message={"role": "assistant",
                                 "content": [{"mimetype": dl.PromptType.TEXT,
                                              "value": caption}]})

        if existing_subsets is True:
            try:
                new_item.metadata['system']['subsets'] = item.metadata['system']['subsets']
            except KeyError:
                new_item.metadata['system'] = {}
                new_item.metadata['system']['subsets'] = item.metadata['system']['subsets']
            new_item.update()
        return new_item

    @staticmethod
    def create_new_dataset(dataset_name, pairs_csv, img_path_col='filepath', img_desc_col='img_description'):
        try:
            dataset = project.datasets.get(dataset_name=dataset_name)
        except dl.exceptions.NotFound:
            dataset = project.datasets.create(dataset_name=dataset_name)

        for index, row in pairs_csv.iterrows():
            file_path = row[img_path_col]
            annots_path = file_path.replace('items', 'json')
            file_name = Path(file_path).name
            item = dataset.items.upload(local_path=file_path,
                                        local_annotations_path=annots_path,
                                        item_metadata=dl.ExportMetadata.FROM_JSON,
                                        overwrite=True)

            item.set_description(text=row[img_desc_col])
            item.update()
            print(f"Uploaded {file_name} with description: '{row[img_desc_col]}'")

        return dataset


if __name__ == '__main__':
    ENV = 'rc'
    # PROJECT_NAME = 'test yaya'
    # DATASET_NAME = 'Data Management Demo Dataset'
    # CSV_PATH = r"C:\Users\Yaya Tang\Documents\DATASETS\TACO 100\taco_100_INPUTS.csv"
    PROJECT_NAME = "Model mgmt demo"
    DATASET_NAME = "ceyda_fashion"

    dl.setenv(ENV)
    project = dl.projects.get(project_name=PROJECT_NAME)
    dataset = project.datasets.get(dataset_name=DATASET_NAME)
    prepared_dataset = ClipPrepare.convert_dataset(project=project, dataset=dataset)
    print(f"Prepared dataset: {prepared_dataset.name}")