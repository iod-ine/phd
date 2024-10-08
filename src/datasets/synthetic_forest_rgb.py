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
import skimage
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

    @property
    def processed_file_names(self):
        """List of files that need to be found in processed_dir to skip processing."""
        return [f"{self.__class__.__name__}_{split}" for split in ("train", "val")]


class SyntheticForestRGBGrid(SyntheticForestRGBBase):
    """Colored synthetic forest generated from individual trees by grid."""

    def __init__(
        self,
        root,
        split: Literal["train", "val"] = "train",
        random_seed: int = 42,
        train_samples: int = 100,
        val_samples: int = 20,
        trees_per_sample: int = 100,
        height_threshold: float = 2.0,
        n_cols: Optional[int] = None,
        dx: float = 2.0,
        dy: float = 2.0,
        xy_noise_mean: float = 0.0,
        xy_noise_std: float = 1.0,
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
        self.trees_per_sample = trees_per_sample
        self.height_threshold = height_threshold
        self.n_cols = n_cols or np.ceil(np.sqrt(trees_per_sample))
        self.dx = dx
        self.dy = dy
        self.xy_noise_mean = xy_noise_mean
        self.xy_noise_std = xy_noise_std
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
        total_samples = self.train_samples + self.val_samples
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
        val_data = data_list[self.train_samples :]

        self.save(train_data, self.processed_paths[0])
        self.save(val_data, self.processed_paths[1])


class SyntheticForestRGBPatch(SyntheticForestRGBBase):
    """Colored synthetic forest generated from individual trees by patch."""

    def __init__(
        self,
        root,
        split: Literal["train", "val"] = "train",
        random_seed: int = 42,
        train_samples: int = 100,
        val_samples: int = 20,
        patch_width: float = 20.0,
        patch_height: float = 20.0,
        patch_overlap: float = 0.0,
        height_threshold: float = 2.0,
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
        self.height_threshold = height_threshold
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patch_overlap = patch_overlap
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
        total_samples = self.train_samples + self.val_samples
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
        val_data = data_list[self.train_samples :]

        self.save(train_data, self.processed_paths[0])
        self.save(val_data, self.processed_paths[1])


class SyntheticForestRGBMBFPatch(SyntheticForestRGBBase):
    """RGB+MBF synthetic forest generated from individual trees by patch."""

    def __init__(
        self,
        root,
        split: Literal["train", "val"] = "train",
        intensity: bool = False,
        edges: bool = False,
        texture: bool = True,
        sigma_min: float = 0.5,
        sigma_max: float = 8,
        num_sigma: Optional[int] = None,
        random_seed: int = 42,
        train_samples: int = 100,
        val_samples: int = 20,
        patch_width: float = 20.0,
        patch_height: float = 20.0,
        patch_overlap: float = 0.0,
        height_threshold: float = 2.0,
        height_dropout_sigmoid_scale: Optional[float] = None,
        height_dropout_sigmoid_shift: Optional[float] = None,
        height_dropout_sigmoid_seed: Optional[int] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log: bool = True,
        force_reload: bool = False,
    ):
        """Create a new dataset instance."""
        self.split = split
        self.intensity = intensity
        self.edges = edges
        self.texture = texture
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_sigma = num_sigma
        self.random_seed = random_seed
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.height_threshold = height_threshold
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patch_overlap = patch_overlap
        self.height_dropout_sigmoid_scale = height_dropout_sigmoid_scale
        self.height_dropout_sigmoid_shift = height_dropout_sigmoid_shift
        self.height_dropout_sigmoid_seed = height_dropout_sigmoid_seed
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

    def process(self):
        """Process raw data and save it to processed_dir."""
        las_objects = [laspy.read(p) for p in self.raw_paths if p.endswith(".las")]
        ortho_files = [p for p in self.raw_paths if p.endswith(".tif")]

        xyzs, features = [], []

        for file in ortho_files:
            with rasterio.open(pathlib.Path(file)) as dataset:
                transformer = rasterio.transform.AffineTransformer(dataset.transform)
                ortho = dataset.read()
                multiscale_features = skimage.feature.multiscale_basic_features(
                    image=ortho,
                    channel_axis=0,
                    intensity=self.intensity,
                    edges=self.edges,
                    texture=self.texture,
                    sigma_min=self.sigma_min,
                    sigma_max=self.sigma_max,
                    num_sigma=self.num_sigma,
                )
                # Shapes:
                #   ortho: (3, H, W)
                #   multiscale_features: (H, W, n_features)

            for las in las_objects:
                if not dataset.bounds.left < las.xyz[0, 0] < dataset.bounds.right:
                    continue
                if not dataset.bounds.bottom < las.xyz[0, 1] < dataset.bounds.top:
                    continue

                if self.height_dropout_sigmoid_scale is not None:
                    xyz = src.clouds.dropout_low_points_sigmoid(
                        xyz=las.xyz,
                        scale=self.height_dropout_sigmoid_scale,
                        shift=self.height_dropout_sigmoid_shift,
                        seed=self.height_dropout_sigmoid_seed,
                    )
                else:
                    xyz = las.xyz

                row, col = transformer.rowcol(xyz[:, 0], xyz[:, 1])
                rgb = ortho[:, row, col].astype(np.float32)  # shape: (3, n_points)
                rgb = np.rollaxis(rgb, axis=1)  # shape: (n_points, 3)
                mbf = multiscale_features[row, col, :]  # shape: (n_points, n_features)

                xyzs.append(xyz)
                features.append(np.hstack([rgb, mbf]))

        data_list = []
        total_samples = self.train_samples + self.val_samples
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
        val_data = data_list[self.train_samples :]

        self.save(train_data, self.processed_paths[0])
        self.save(val_data, self.processed_paths[1])


if __name__ == "__main__":
    dataset = SyntheticForestRGBPatch(root="data/tmp")
