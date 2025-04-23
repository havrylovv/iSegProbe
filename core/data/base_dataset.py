"""Base class for interactive segmentation datasets."""

import pickle
import random
from typing import Dict

import numpy as np
from albumentations import Compose
from torch.utils.data import Dataset
from torchvision import transforms

from .data_sample import DSample
from .points_sampler import BasePointSampler


class iSegBaseDataset(Dataset):
    def __init__(
        self,
        augmentator: Compose = None,
        points_sampler: BasePointSampler = None,
        min_object_area: float = 0,
        keep_background_prob: float = 0.0,
        with_image_info: bool = False,
        samples_scores_path: str = None,
        samples_scores_gamma: float = 1.0,
        sample_points: str = True,
        epoch_len: str = -1,
    ) -> None:
        super(iSegBaseDataset, self).__init__()
        self.epoch_len = epoch_len
        self.augmentator = augmentator
        self.min_object_area = min_object_area
        self.keep_background_prob = keep_background_prob
        self.points_sampler = points_sampler
        self.with_image_info = with_image_info
        self.samples_precomputed_scores = self._load_samples_scores(
            samples_scores_path, samples_scores_gamma
        )
        self.to_tensor = transforms.ToTensor()
        self.sample_points = sample_points

        self.dataset_samples = None

    def __getitem__(self, index: int) -> Dict:
        if self.samples_precomputed_scores is not None:
            index = np.random.choice(
                self.samples_precomputed_scores["indices"],
                p=self.samples_precomputed_scores["probs"],
            )
        else:
            if self.epoch_len > 0:
                index = random.randrange(0, len(self.dataset_samples))

        sample = self.get_sample(index)
        sample = self.augment_sample(sample)

        if sample.points is None:
            sample.remove_small_objects(self.min_object_area)
            self.points_sampler.sample_object(sample)
            if self.sample_points:
                points = np.array(self.points_sampler.sample_points())
            else:
                points = np.empty([self.points_sampler.max_num_points * 2, 3])
            mask = self.points_sampler.selected_mask
        else:
            points = sample.points
            mask = sample._encoded_masks.astype(np.float32)
            mask = mask.reshape([1, mask.shape[0], mask.shape[1]])

        output = {
            "images": self.to_tensor(sample.image),
            "points": points.astype(np.float32),
            "instances": mask,
        }

        if self.with_image_info:
            output["image_info"] = sample.sample_id

        return output

    def augment_sample(self, sample: DSample) -> DSample:
        if self.augmentator is None:
            return sample

        valid_augmentation = False
        while not valid_augmentation:
            sample.augment(self.augmentator)
            keep_sample = (
                self.keep_background_prob < 0.0
                or random.random() < self.keep_background_prob
            )
            valid_augmentation = len(sample) > 0 or keep_sample

        return sample

    def get_sample(self, index: int) -> DSample:
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return self.get_samples_number()

    def get_samples_number(self) -> int:
        return len(self.dataset_samples)

    @staticmethod
    def _load_samples_scores(
        samples_scores_path: str, samples_scores_gamma: str
    ) -> Dict:
        if samples_scores_path is None:
            return None

        with open(samples_scores_path, "rb") as f:
            images_scores = pickle.load(f)

        probs = np.array([(1.0 - x[2]) ** samples_scores_gamma for x in images_scores])
        probs /= probs.sum()
        samples_scores = {"indices": [x[0] for x in images_scores], "probs": probs}
        print(f"Loaded {len(probs)} weights with gamma={samples_scores_gamma}")
        return samples_scores
