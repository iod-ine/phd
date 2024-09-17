"""A torch_geometric wrapper for the colored synthetic forest dataset."""

import functools
import itertools
import pathlib
import random
import zipfile
from typing import Literal, Optional

import kaggle
import laspy
import numpy as np
import rasterio
import torch
import torch_geometric

import src.clouds
import src.datasets.utils


class SyntheticForestRGBBase(torch_geometric.data.InMemoryDataset):
    """Base class for colored synthetic forest datasets."""

    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log: bool = True,
        force_reload: bool = False,
    ):
        """Create a new instance."""
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            log=log,
            force_reload=force_reload,
        )

    @functools.cached_property
    def raw_file_names(self):
        """List of files that need to be found in raw_dir to skip the download."""
        return list(
            itertools.chain.from_iterable(
                [
                    [f"Alder/alder_{i:>02}.las" for i in range(26)],
                    [f"Aspen/aspen_{i:>02}.las" for i in range(73)],
                    [f"Birch/birch_{i:>02}.las" for i in range(77)],
                    [f"Fir/fir_{i:>02}.las" for i in range(37)],
                    [f"Pine/pine_{i:>02}.las" for i in range(70)],
                    [f"Spruce/spruce_{i:>02}.las" for i in range(94)],
                    [f"Tilia/tilia_{i:>02}.las" for i in range(17)],
                    [f"plot_{i:>02}.tif" for i in range(1, 11)],
                ],
            )
        )

    def download(self):
        """Download raw data into raw_dir.

        Notes:
            Requires Kaggle API credentials in KAGGLE_USERNAME and KAGGLE_KEY
            environment variables or ~/.kaggle/kaggle.json. For details, see
            https://www.kaggle.com/docs/api#authentication.

        """
        kaggle.api.dataset_download_files(
            dataset="sentinel3734/uav-point-clouds-of-individual-trees",
            path=self.raw_dir,
            unzip=True,
        )
        for i in range(1, 11):
            file = f"plot_{i:>02}.tif"
            kaggle.api.dataset_download_file(
                dataset="sentinel3734/tree-detection-lidar-rgb",
                file_name=f"ortho/{file}",
                path=self.raw_dir,
            )
            archive = pathlib.Path(self.raw_dir) / f"{file}.zip"
            with zipfile.ZipFile(archive) as z:
                z.extractall(path=self.raw_dir)
            archive.unlink()


class SyntheticForestRGBGrid(SyntheticForestRGBBase):
    """Colored synthetic forest generated from individual trees by grid."""

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
        n_cols: Optional[int] = None,
        dx: float = 2.0,
        dy: float = 2.0,
        xy_noise_mean: float = 0.0,
        xy_noise_std: float = 1.0,
        las_features=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log: bool = True,
        force_reload: bool = False,
    ):
        """Create a new dataset instance."""
        self.split = split
        self.random_seed = random_seed
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.trees_per_sample = trees_per_sample
        self.height_threshold = height_threshold
        self.n_cols = n_cols or np.ceil(np.sqrt(trees_per_sample))
        self.dx = dx
        self.dy = dy
        self.xy_noise_mean = xy_noise_mean
        self.xy_noise_std = xy_noise_std
        self.las_features = las_features or ["red", "green", "blue"]
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            log=log,
            force_reload=force_reload,
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
        param_set_id = src.datasets.utils.generate_unique_id_for_parameter_set(
            self.random_seed,
            self.train_samples,
            self.val_samples,
            self.test_samples,
            self.trees_per_sample,
            self.height_threshold,
            self.n_cols,
            self.dx,
            self.dy,
            self.xy_noise_mean,
            self.xy_noise_std,
            self.las_features,
        )
        return [f"sfc_{split}_{param_set_id}" for split in ("train", "val", "test")]

    def process(self):
        """Process raw data and save it to processed_dir."""
        las_objects = [laspy.read(p) for p in self.raw_paths if p.endswith(".las")]
        ortho_files = [p for p in self.raw_paths if p.endswith(".tif")]

        xyzs, features = [], []

        for file in ortho_files:
            with rasterio.open(pathlib.Path(file)) as dataset:
                transformer = rasterio.transform.AffineTransformer(dataset.transform)
                ortho = dataset.read()

            for las in las_objects:
                if not dataset.bounds.left < las.xyz[0, 0] < dataset.bounds.right:
                    continue
                if not dataset.bounds.bottom < las.xyz[0, 1] < dataset.bounds.top:
                    continue

                row, col = transformer.rowcol(las.x, las.y)
                rgb = ortho[:, row, col].astype(np.float32)  # shape: (3, n_points)

                xyzs.append(las.xyz)
                features.append(np.swapaxes(rgb, 0, 1))

        data_list = []
        total_samples = self.train_samples + self.val_samples + self.test_samples
        random.seed(self.random_seed)
        for i in range(total_samples):
            indices = random.sample(range(len(xyzs)), k=self.trees_per_sample)
            pos, x, y = src.clouds.create_regular_grid(
                xyzs=[xyzs[i] for i in indices],
                features=[features[i] for i in indices],
                ncols=self.n_cols,
                dx=self.dx,
                dy=self.dy,
                xy_noise_mean=self.xy_noise_mean,
                xy_noise_std=self.xy_noise_std,
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


class SyntheticForestRGBPatch(SyntheticForestRGBBase):
    """Colored synthetic forest generated from individual trees by patch."""

    def __init__(
        self,
        root,
        split: Literal["train", "val", "test"] = "train",
        random_seed: int = 42,
        train_samples: int = 100,
        val_samples: int = 20,
        test_samples: int = 20,
        patch_width: float = 20.0,
        patch_height: float = 20.0,
        patch_overlap: float = 0.0,
        height_threshold: float = 2.0,
        las_features=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log: bool = True,
        force_reload: bool = False,
    ):
        """Create a new dataset instance."""
        self.split = split
        self.random_seed = random_seed
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.height_threshold = height_threshold
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patch_overlap = patch_overlap
        self.las_features = las_features or ["red", "green", "blue"]
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            log=log,
            force_reload=force_reload,
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
        param_set_id = src.datasets.utils.generate_unique_id_for_parameter_set(
            self.random_seed,
            self.train_samples,
            self.val_samples,
            self.test_samples,
            self.height_threshold,
            self.patch_height,
            self.patch_width,
            self.patch_overlap,
            self.las_features,
        )
        return [f"sfc_{split}_{param_set_id}" for split in ("train", "val", "test")]

    def process(self):
        """Process raw data and save it to processed_dir."""
        las_objects = [laspy.read(p) for p in self.raw_paths if p.endswith(".las")]
        ortho_files = [p for p in self.raw_paths if p.endswith(".tif")]

        xyzs, features = [], []

        for file in ortho_files:
            with rasterio.open(pathlib.Path(file)) as dataset:
                transformer = rasterio.transform.AffineTransformer(dataset.transform)
                ortho = dataset.read()

            for las in las_objects:
                if not dataset.bounds.left < las.xyz[0, 0] < dataset.bounds.right:
                    continue
                if not dataset.bounds.bottom < las.xyz[0, 1] < dataset.bounds.top:
                    continue

                row, col = transformer.rowcol(las.x, las.y)
                rgb = ortho[:, row, col].astype(np.float32)  # shape: (3, n_points)

                xyzs.append(las.xyz)
                features.append(np.swapaxes(rgb, 0, 1))

        data_list = []
        total_samples = self.train_samples + self.val_samples + self.test_samples
        random.seed(self.random_seed)
        for i in range(total_samples):
            pos, x, y = src.clouds.create_forest_patch(
                xyzs=xyzs,
                features=features,
                width=self.patch_width,
                height=self.patch_height,
                overlap=self.patch_overlap,
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


if __name__ == "__main__":
    dataset = SyntheticForestRGBPatch(root="data/tmp")
