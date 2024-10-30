from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import lightning as l

class PatachedDataModule(l.LightningDataModule):
    """
    parameters
    ---
    dataname and configname: str
        local config file or repository name on the huggingface hub
    batch: int (default: 32, optional)
        batch size
    n_train: int or float (default: None, optional)
        if specified, use only n_train samples in the train/val process
        otherwise, use the default split
    """

    def __init__(
        self,
        batch: int = 2,
        path_train: str = None,
        path_val: str = None,
        path_test: str = None,
        **kwargs,
    ):
        super().__init__()
        # input parameters
        self.batch = batch
        # datasets
        self.dataset = {
            "train": None,
            "val": None,
            "test": None,
        }
        self.path = {
            "train": path_train,
            "val": path_val,
            "test": path_test,
        }   
        self.kwargs = kwargs

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.dataset["train"] = self.get_dataset("train")
            self.dataset["val"] = self.get_dataset("val")
        elif stage == "test":
            self.dataset["test"] =  self.get_dataset("test")

    def get_dataset(self, split):
        return PatchedImages(root=self.path[split])

    def set_dataset(self, split):
        """
        split: str
            train, val, or test
        """
        self.dataset[split] = self.get_dataset(split)

    # loaders
    def get_dataloader(self, split):
        if split == "train":
            return self.train_dataloader()
        elif split == "val":
            return self.val_dataloader()
        elif split == "test":
            return self.test_dataloader()

    def train_dataloader(self):
        if self.dataset["train"] is None:
            self.set_dataset("train")
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch,
            shuffle=True,
            pin_memory=True,
            # num_workers=4,
            # persistent_workers=True,
            # collate_fn=self._collate_fn_train,
        )

    def val_dataloader(self):
        if self.dataset["val"] is None:
            self.set_dataset("val")
        return DataLoader(
            self.dataset["val"],
            batch_size=self.batch,
            shuffle=False,
            pin_memory=True,
            # num_workers=1,
            # collate_fn=self._collate_fn_val,
        )

    def test_dataloader(self):
        if self.dataset["test"] is None:
            self.set_dataset("test")
        return DataLoader(
            self.dataset["test"],
            batch_size=self.batch,
            shuffle=False,
            pin_memory=True,
            # num_workers=1,
            # collate_fn=self._collate_fn_test,
        )

class PatchedImages(Dataset):
    """
    Image shape is (720, 1280, 3) --> (768, 1280, 3) --> 6x10 128x128 patches
    """
    def __init__(self, root: str, transform=None):
        self.files = sorted(Path(root).iterdir())
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray, str]:
        path = str(self.files[index % len(self.files)])
        img = Image.open(path)
        img, patches = from_img_to_patches(img)
        return img, patches, path

    def __len__(self):
        return len(self.files)
    
def from_path_to_patches(path):
    img = Image.open(path)
    return from_img_to_patches(img)    

def from_img_to_patches(img):
    """
    img: PIL.Image
    """
    img = img.resize((1280, 720))
    img = np.array(img)

    # expanding the first dimension from 720 to 768
    pad = ((24, 24), (0, 0), (0, 0))

    # img = np.pad(img, pad, 'constant', constant_values=0) / 255
    img = np.pad(img, pad, mode="edge") / 255.0

    # from (768, 1280, 3) to (3, 768, 1280)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float()

    # from (3, 768, 1280) to (3, 6, 128, 10, 128)
    patches = np.reshape(img, (3, 6, 128, 10, 128))
    # from (3, 6, 128, 10, 128) to (3, 6, 10, 128, 128)
    patches = np.transpose(patches, (0, 1, 3, 2, 4))

    return img, patches