import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import mne
import pickle


class EEGImageNetDataset(Dataset):
    def __init__(self, dataset_dir, subject, granularity, transform=None):
        self.dataset_dir = dataset_dir
        dataset_loaded = torch.load(os.path.join(dataset_dir, "EEG-ImageNet.pth"))
        self.labels = dataset_loaded["labels"]
        self.images = dataset_loaded["images"]
        self.transform = transform
        if subject != -1:
            dataset_chosen = [
                dataset_loaded["dataset"][i]
                for i in range(len(dataset_loaded["dataset"]))
                if dataset_loaded["dataset"][i]["subject"] == subject
            ]
        else:
            dataset_chosen = dataset_loaded["dataset"]
        if granularity == "coarse":
            self.data = [i for i in dataset_chosen if i["granularity"] == "coarse"]
        elif granularity == "all":
            self.data = dataset_chosen
        else:
            fine_num = int(granularity[-1])
            fine_category_range = np.arange(8 * fine_num, 8 * fine_num + 8)
            self.data = [
                i
                for i in dataset_chosen
                if i["granularity"] == "fine" and self.labels.index(i["label"]) in fine_category_range
            ]
        self.use_frequency_feat = False
        self.frequency_feat = None
        self.use_image_label = False
        self.clip_label = None
        self.use_clip_label = False

    def __getitem__(self, index):
        if self.use_frequency_feat:
            feat = self.frequency_feat[index]
        else:
            eeg_data = self.data[index]["eeg_data"].float()
            feat = eeg_data[:, 40:440]

        if self.use_image_label:
            path = self.data[index]["image"]
            # use self.transform to control using filename or Image
            if self.transform:
                label = Image.open(os.path.join(self.dataset_dir, "imageNet_images", path.split("_")[0], path))
                label = self.transform(label)
            else:
                label = path
        elif self.use_clip_label:
            label = self.clip_label[self.data[index]["image"]]
        else:
            label = self.labels.index(self.data[index]["label"])
        return feat, label

    def __len__(self):
        return len(self.data)

    def add_frequency_feat(self, feat):
        if len(feat) == len(self.data):
            self.frequency_feat = torch.from_numpy(feat).float()
        else:
            raise ValueError("Frequency features must have the same length.")

    def add_clip_label(self, label):
        if len(label) == len(self.images):
            self.clip_label = label
        else:
            raise ValueError("Clip features must contain all images.")


class ThingsEEGDataset(Dataset):
    def __init__(self, dataset_dir="../data/THING-EEG", split="train"):
        self.split = split
        if split == "train":
            self.preprocessed_eeg = torch.from_numpy(
                np.load(os.path.join(dataset_dir, "processed_data", "preprocessed_eeg_training.npy"),
                        allow_pickle=True)['preprocessed_eeg_data'])
            self.images_features = torch.load(
                os.path.join(dataset_dir, "things_eeg_images", "train_images_features.pth"))
            self.data_repetition = self.preprocessed_eeg.shape[1]
        elif split == "test":
            self.preprocessed_eeg = torch.from_numpy(
                np.load(os.path.join(dataset_dir, "processed_data", "preprocessed_eeg_test.npy"),
                        allow_pickle=True).item()['preprocessed_eeg_data'])
            self.images_features = torch.load(
                os.path.join(dataset_dir, "things_eeg_images", "test_images_features.pth"))
            self.data_repetition = self.preprocessed_eeg.shape[1]
        else:
            raise ValueError("split should be either 'train' or 'test'")
        self.image_metadata = np.load(os.path.join(dataset_dir, "things_eeg_images", "image_metadata.npy"),
                                      allow_pickle=True).item()
        self.use_frequency_feat = False
        self.frequency_feat = None
        self.use_image_label = False
        self.clip_label = None
        self.use_clip_label = False

    def __getitem__(self, idx):
        if not self.use_frequency_feat:
            if self.split == "train":
                feat = self.preprocessed_eeg[idx // self.data_repetition, idx % self.data_repetition, :62, :]
            elif self.split == "test":
                feat = torch.mean(self.preprocessed_eeg[idx], dim=0)[:62, :]
        else:
            if self.split == "train":
                feat = self.frequency_feat[idx]
            elif self.split == "test":
                feat = torch.mean(
                    self.frequency_feat[idx * self.data_repetition:idx * self.data_repetition + self.data_repetition],
                    dim=0)

        if self.split == "train":
            label = \
                self.images_features[self.image_metadata[f'{self.split}_img_concepts'][idx // self.data_repetition]][
                    self.image_metadata[f'{self.split}_img_files'][idx // self.data_repetition]]
        elif self.split == "test":
            label = self.images_features[self.image_metadata[f'{self.split}_img_concepts'][idx]][
                self.image_metadata[f'{self.split}_img_files'][idx]]

        return feat, label

    def __len__(self):
        if self.split == "train":
            return self.preprocessed_eeg.shape[0] * self.data_repetition
        elif self.split == "test":
            return self.preprocessed_eeg.shape[0]

    def add_frequency_feat(self, feat):
        self.frequency_feat = torch.from_numpy(feat).float()


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir="../data/THINGS-MEG"):
        file_path = os.path.join(data_dir, "preprocessed", "preprocessed_P1-epo.fif")
        image_path = os.path.join(data_dir, "images", "clip_features.pth")

        if os.path.exists(os.path.join(data_dir, "metadata.pkl")):
            with open(os.path.join(data_dir, "metadata.pkl"), "rb") as f:
                self.metadata = pickle.load(f)
        else:
            epochs = mne.read_epochs(file_path)
            self.metadata = epochs.metadata
            with open(os.path.join(data_dir, "metadata.pkl"), "wb") as f:
                pickle.dump(self.metadata, f)

        if os.path.exists(os.path.join(data_dir, "eeg_data.pkl")):
            with open(os.path.join(data_dir, "eeg_data.pkl"), "rb") as f:
                self.eeg_data = pickle.load(f)
        else:
            self.eeg_data = epochs.get_data()
            with open(os.path.join(data_dir, "eeg_data.pkl"), "wb") as f:
                pickle.dump(self.eeg_data, f)

        self.clip_label = torch.load(image_path)
        self.use_clip_label = True

        self.use_frequency_feat = False
        self.frequency_feat = None
        self.use_image_label = False

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        image_file_name = self.metadata["image_path"][index].split("/")[-1]
        label = self.clip_label[image_file_name]
        eeg_index = self.metadata["index"][index]
        feature = torch.tensor(self.eeg_data[eeg_index])
        return feature.flatten().float(), label

    def add_frequency_feat(self, feat):
        self.frequency_feat = torch.from_numpy(feat).float()
