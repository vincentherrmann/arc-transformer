import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import random
import os
import json


class ArcDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.tasks_list = sorted(os.listdir(dataset_path))

    def __getitem__(self, item):
        with open(self.dataset_path + "/" + self.tasks_list[item], 'r') as f:
            task = json.load(f)

        data = task_map(task, lambda t: torch.LongTensor(t))

        # TODO random transform (rotate, reflect, scale(?))
        # TODO random color permutation

        #priors =

        return data

    def __len__(self):
        return len(self.tasks_list)


def task_map(task_data, f):
    out_data = {}
    for key, value in task_data.items():
        l = []
        for element in value:
            l.append({k: f(v) for k, v in element.items()})
        out_data[key] = l
    return out_data


def task_reduce(task_data, f, start_value):
    result = start_value
    for key in ["train", "test"]:
        value = task_data[key]
        for element in value:
            for k in ["input", "output"]:
                result = f(result, element[k])
    return result


def task_flatten(task_data):
    components = task_reduce(task_data, lambda r, t: r + [t], [])
    out_data = torch.stack(components, dim=0)
    return out_data
    