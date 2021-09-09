import openml
from openml.tasks import TaskType
from pathlib import Path
import os

import pickle
import json

root_path = Path('datasets')
openml.config.cache_directory = os.path.expanduser('cache')

def save_task_info(task_name: str, tasks):
    "Save information of the tasks"
    task_path = root_path / task_name

    os.makedirs(task_path, exist_ok=True)
    tasks.to_json(open(task_path / 'tasks_info.json', 'w+'))
    return task_path


def save_dataset(dataset, dataset_path):
    "Save the information and data of a dataset in a specific path"
    if os.path.exists(dataset_path):
        return

    os.makedirs(dataset_path)
    pickle.dump(dataset, open(dataset_path / 'dataset_info.pkl', 'wb'))

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe")

    if y is None or X is None:
        return

    X.to_json(dataset_path / 'X.json')
    y.to_json(dataset_path / 'y.json')
    try:
        json.dump(categorical_indicator, open(dataset_path / 'categorical_indicator.json', 'w+'))
        json.dump(attribute_names, open(dataset_path / 'attribute_names.json', 'w+'))
    except:
        pass

def download_datasets(tasks, task_path, amount=10000):
    "Download and save the information of a list of task in a specific path"
    print(len(list(tasks['did'])[:amount]))
    for did in list(tasks['did'])[:amount]:
        try:
            dataset = openml.datasets.get_dataset(did)
            dataset_path = task_path / f'{did}'
            save_dataset(dataset, dataset_path)
        except Exception as e:
            print(f'error downloading dataset {did}', e)


def download_classification_datasets():
    tasks = openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_CLASSIFICATION,
                                    output_format="dataframe")
    print(tasks.head())

    # Filtering the tasks
    filtered_tasks = tasks.query("NumberOfInstances > 500 and NumberOfInstances < 500000 and NumberOfFeatures < 600")
    print('Index of the tasks:', list(filtered_tasks.index))
    print('Number of tasks:', len(filtered_tasks))

    # Save the information of the filtered tasks
    task_path = save_task_info('classification', filtered_tasks)

    # Get the datasets
    download_datasets(filtered_tasks, task_path)


def download_regression_datasets():
    tasks_df = openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_REGRESSION,
                                       output_format="dataframe")
    print(tasks_df.head())

    # We filter the tasks to only contain datasets with more than 300 instances and less than  500 000 samples
    filtered_tasks = tasks_df.query("NumberOfInstances > 500 and NumberOfInstances < 500000 and NumberOfFeatures < 600")
    print('Index of the tasks:', list(filtered_tasks.index))
    print('Number of tasks:', len(filtered_tasks))

    # Save the information of the filtered tasks
    task_path = save_task_info('regression', filtered_tasks)
    download_datasets(filtered_tasks, task_path)


if __name__ == '__main__':
    download_classification_datasets()
