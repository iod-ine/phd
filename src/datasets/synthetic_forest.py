"""A torch_geometric wrapper for the synthetic forest dataset."""

import hashlib
import random
from typing import Literal

import laspy
import numpy as np
import torch
import torch_geometric
import tqdm

import src.clouds
from src.datasets.individual_trees import IndividualTreesBase


class SyntheticForest(IndividualTreesBase):
    """A generated synthetic forest from the individual trees."""

    def __init__(
        self,
        root,
        split: Literal["train", "val", "test"] = "train",
        random_seed: int = 42,
        train_samples: int = 100,
        val_samples: int = 20,
        test_samples: int = 20,
        trees_per_sample: int = 100,
        height_threshold: float = 2.0,
        dx: float = 2.0,
        dy: float = 2.0,
        xy_noise_mean: float = 0.0,
        xy_noise_std: float = 1.0,
        las_features=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """Create a new SyntheticForest instance."""
        self.split = split
        self.random_seed = random_seed
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.trees_per_sample = trees_per_sample
        self.height_threshold = height_threshold
        self.dx = dx
        self.dy = dy
        self.xy_noise_mean = xy_noise_mean
        self.xy_noise_std = xy_noise_std
        super().__init__(
            root=root,
            las_features=las_features,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        match split:
            case "train":
                self.load(self.processed_paths[0])
            case "val":
                self.load(self.processed_paths[1])
            case "test":
                self.load(self.processed_paths[2])

    @property
    def processed_file_names(self):
        """List of files that need to be found in processed_dir to skip processing."""
        param_set_id = self._generate_id()
        return [f"{split}_{param_set_id}" for split in ("train", "val", "test")]

    def process(self):
        """Process raw data and save it to processed_dir."""
        las_list = [laspy.read(path) for path in self.raw_paths]
        data_list = []
        total_samples = self.train_samples + self.val_samples + self.test_samples
        random.seed(self.random_seed)
        for i in tqdm.trange(total_samples):
            sample = random.sample(las_list, k=self.trees_per_sample)
            pos, x, y = src.clouds.create_regular_grid(
                las_list=sample,
                ncols=np.ceil(np.sqrt(self.trees_per_sample)),
                dx=self.dx,
                dy=self.dy,
                xy_noise_mean=self.xy_noise_mean,
                xy_noise_std=self.xy_noise_std,
                features_to_extract=self.las_features,
                height_threshold=self.height_threshold,
            )
            data = torch_geometric.data.Data(
                pos=torch.from_numpy(pos.astype(np.float32)),
                x=torch.from_numpy(x),
                y=torch.from_numpy(y),
            )
            data_list.append(data)

        train_data = data_list[: self.train_samples]
        val_data = data_list[self.train_samples : self.train_samples + self.val_samples]
        test_data = data_list[self.train_samples + self.val_samples :]

        self.save(train_data, self.processed_paths[0])
        self.save(val_data, self.processed_paths[1])
        self.save(test_data, self.processed_paths[2])

    def _generate_id(self) -> str:
        """Generate an ID that identifies the set of parameters used for generation."""
        params = [
            self.random_seed,
            self.train_samples,
            self.val_samples,
            self.test_samples,
            self.trees_per_sample,
            self.las_features,
        ]
        param_string = ",".join(map(str, params))
        return hashlib.md5(param_string.encode(), usedforsecurity=False).hexdigest()[:7]


if __name__ == "__main__":
    dataset = SyntheticForest(root="data/tmp")
