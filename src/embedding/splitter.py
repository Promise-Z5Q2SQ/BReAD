from torch.utils.data import random_split, Subset
from dataset import EEGImageNetDataset
import numpy as np
from typing import Tuple
import json


class Splitter:
    def __init__(self, train_ratio=0.9, splitter_path=None):
        self.train_ratio = train_ratio
        self.train_indices = []
        self.test_indices = []
        self.splitter_path = splitter_path

    def split(self, dataset) -> Tuple[EEGImageNetDataset, EEGImageNetDataset]:
        if self.splitter_path is not None:
            self.load_splitter(self.splitter_path)
        else:
            split_num = int(50 * self.train_ratio)
            self.train_indices = np.array([i for i in range(len(dataset)) if i % 50 < split_num])
            self.test_indices = np.array([i for i in range(len(dataset)) if i % 50 >= split_num])

        train_dataset = Subset(dataset, self.train_indices)
        test_dataset = Subset(dataset, self.test_indices)

        return train_dataset, test_dataset

    def save(self, path):
        save = {"train_indices": self.train_indices.tolist(), "test_indices": self.test_indices.tolist()}
        with open(path, "w") as f:
            json.dump(save, f, indent=4)

    def load_splitter(self, path):
        with open(path, "r") as f:
            load = json.load(f)
            self.train_indices = load["train_indices"]
            self.test_indices = load["test_indices"]
