"""Pascal VOC dataset."""

import pickle as pkl
from pathlib import Path

import cv2
import numpy as np

from core.data.base_dataset import iSegBaseDataset
from core.data.data_sample import DSample


class PascalVocDataset(iSegBaseDataset):
    def __init__(self, dataset_path: str, split: str = "train", **kwargs) -> None:
        super().__init__(**kwargs)
        assert split in {"train", "val", "trainval", "test"}

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / "JPEGImages"
        self._insts_path = self.dataset_path / "SegmentationObject"
        self.dataset_split = split

        if split == "test":
            with open(
                self.dataset_path / f"ImageSets/Segmentation/test.pickle", "rb"
            ) as f:
                self.dataset_samples, self.instance_ids = pkl.load(f)
        else:
            with open(
                self.dataset_path / f"ImageSets/Segmentation/{split}.txt", "r"
            ) as f:
                self.dataset_samples = [name.strip() for name in f.readlines()]

    def get_sample(self, index: int) -> DSample:
        sample_id = self.dataset_samples[index]
        image_path = str(self._images_path / f"{sample_id}.jpg")
        mask_path = str(self._insts_path / f"{sample_id}.png")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(
            np.int32
        )
        if self.dataset_split == "test":
            instance_id = self.instance_ids[index]
            mask = np.zeros_like(instances_mask)
            mask[instances_mask == 220] = 220  # ignored area
            mask[instances_mask == instance_id] = 1
            objects_ids = [1]
            instances_mask = mask
        else:
            objects_ids = np.unique(instances_mask)
            objects_ids = [x for x in objects_ids if x != 0 and x != 220]

        return DSample(
            image,
            instances_mask,
            objects_ids=objects_ids,
            ignore_ids=[220],
            sample_id=index,
        )
