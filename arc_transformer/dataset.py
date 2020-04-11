import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import random
import os
import json

from arc_transformer.preprocessing import Preprocessing


class ArcDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, max_size=(30, 30), augment=True):
        self.dataset_path = dataset_path
        self.tasks_list = sorted(os.listdir(dataset_path))
        self.index_lookup = list(range(len(self.tasks_list)))
        self.preprocessing = Preprocessing(max_height=max_size[0], max_width=max_size[1])
        new_lookup = []
        self.augment = augment

        for i in self.index_lookup:
            with open(self.dataset_path + "/" + self.tasks_list[i], 'r') as f:
                task = json.load(f)

            data = task_map(task, lambda t: torch.LongTensor(t))
            size = task_reduce(data,
                               f=lambda r, t: [max(r[0], t.shape[0]), max(r[1], t.shape[1])],
                               start_value=[0, 0])
            if size[0] <= max_size[0] and size[1] <= max_size[1]:
                new_lookup.append(i)
        self.index_lookup = new_lookup

    def __getitem__(self, item):
        with open(self.dataset_path + "/" + self.tasks_list[item], 'r') as f:
            task = json.load(f)

        data = task_map(task, lambda t: torch.LongTensor(t))

        # only allow one test example
        if len(data["test"]) > 1:
            data["test"] = data["test"][0:1]

        if self.augment:
            # reflect
            transpose = random.randint(0, 1)
            if transpose == 1:
                data = task_map(data, lambda t: t.t())

            # rotate
            rotate = random.randint(0, 3)
            f = lambda t: t
            if rotate == 1:
                f = lambda t: torch.flip(t, [0])
            elif rotate == 2:
                f = lambda t: torch.flip(t, [0, 1])
            elif rotate == 3:
                f = lambda t: torch.flip(t, [1])
            data = task_map(data, f)

            # maybe scale?

            # permute color (except color 0)
            colors = list(range(1, 10))
            random.shuffle(colors)
            colors = torch.LongTensor([0] + colors)
            data = task_map(data, lambda t: colors[t])

        priors = task_map(data, self.preprocessing.get_relative_priors)
        data = task_map(data, self.preprocessing)

        priors = task_flatten(priors)
        data = task_flatten(data)

        return data, priors

    def __len__(self):
        return len(self.index_lookup)


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
    