"""COCO + LVIS dataset."""

import json
import pickle
import random
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from core.data.base_dataset import iSegBaseDataset
from core.data.data_sample import DSample


class CocoLvisDataset(iSegBaseDataset):
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        stuff_prob: float = 0.0,
        allow_list_name=None,
        anno_file: str = "hannotation.pickle",
        **kwargs,
    ) -> None:
        super(CocoLvisDataset, self).__init__(**kwargs)
        dataset_path = Path(dataset_path)
        self._split_path = dataset_path / split
        self.split = split
        self._images_path = self._split_path / "images"
        self._masks_path = self._split_path / "masks"
        self.stuff_prob = stuff_prob

        with open(self._split_path / anno_file, "rb") as f:
            self.dataset_samples = sorted(pickle.load(f).items())

        if allow_list_name is not None:
            allow_list_path = self._split_path / allow_list_name
            with open(allow_list_path, "r") as f:
                allow_images_ids = json.load(f)
            allow_images_ids = set(allow_images_ids)

            self.dataset_samples = [
                sample
                for sample in self.dataset_samples
                if sample[0] in allow_images_ids
            ]

    def get_sample(self, index: int) -> DSample:
        image_id, sample = self.dataset_samples[index]
        image_path = self._images_path / f"{image_id}.jpg"

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        packed_masks_path = self._masks_path / f"{image_id}.pickle"
        with open(packed_masks_path, "rb") as f:
            encoded_layers, objs_mapping = pickle.load(f)
        layers = [cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in encoded_layers]
        layers = np.stack(layers, axis=2)

        instances_info = deepcopy(sample["hierarchy"])
        for inst_id, inst_info in list(instances_info.items()):
            if inst_info is None:
                inst_info = {"children": [], "parent": None, "node_level": 0}
                instances_info[inst_id] = inst_info
            inst_info["mapping"] = objs_mapping[inst_id]

        if self.stuff_prob > 0 and random.random() < self.stuff_prob:
            for inst_id in range(sample["num_instance_masks"], len(objs_mapping)):
                instances_info[inst_id] = {
                    "mapping": objs_mapping[inst_id],
                    "parent": None,
                    "children": [],
                }
        else:
            for inst_id in range(sample["num_instance_masks"], len(objs_mapping)):
                layer_indx, mask_id = objs_mapping[inst_id]
                layers[:, :, layer_indx][layers[:, :, layer_indx] == mask_id] = 0

        return DSample(image, layers, objects=instances_info)
