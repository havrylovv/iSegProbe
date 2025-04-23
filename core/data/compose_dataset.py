"""Base classes for combined datasets."""

from math import isclose
from typing import List

import numpy as np

from .base_dataset import DSample, iSegBaseDataset


class ComposeDataset(iSegBaseDataset):
    def __init__(self, datasets: List[iSegBaseDataset], **kwargs) -> None:
        super(ComposeDataset, self).__init__(**kwargs)

        self._datasets = datasets
        self.dataset_samples = []
        for dataset_indx, dataset in enumerate(self._datasets):
            self.dataset_samples.extend(
                [(dataset_indx, i) for i in range(len(dataset))]
            )

    def get_sample(self, index: int) -> DSample:
        dataset_indx, sample_indx = self.dataset_samples[index]
        return self._datasets[dataset_indx].get_sample(sample_indx)


class ProportionalComposeDataset(iSegBaseDataset):
    def __init__(
        self, datasets: List[iSegBaseDataset], ratios: List[float], **kwargs
    ) -> None:
        super().__init__(**kwargs)

        assert len(ratios) == len(
            datasets
        ), "The number of datasets must match the number of ratios"
        assert isclose(sum(ratios), 1.0), "The sum of ratios must be equal to 1"

        self._ratios = ratios
        self._datasets = datasets
        self.dataset_samples = []
        for dataset_indx, dataset in enumerate(self._datasets):
            self.dataset_samples.extend(
                [(dataset_indx, i) for i in range(len(dataset))]
            )

    def get_sample(self, index: int) -> DSample:
        dataset_indx = np.random.choice(len(self._datasets), p=self._ratios)
        sample_indx = np.random.choice(len(self._datasets[dataset_indx]))

        return self._datasets[dataset_indx].get_sample(sample_indx)
