"""Preprocess Lysva point clouds.

Steps:
1. Normalize height.
2. Add color from the orthophoto.

"""

import pathlib

import laspy
import numpy as np
import rasterio
import tqdm

import src.clouds

if __name__ == "__main__":
    raw_data_dir = pathlib.Path("data/raw")

    interim_data_dir = pathlib.Path("data/interim/lysva")
    interim_als_dir = interim_data_dir / "als"
    interim_als_dir.mkdir(parents=True, exist_ok=True)

    las_files = sorted(raw_data_dir.glob("lysva/als/plot_*.las"))

    with rasterio.open(raw_data_dir / "full-lysva" / "ortho.tif") as dataset:
        for las_file in tqdm.tqdm(las_files):
            las = laspy.read(las_file)

            rgb = np.stack([c for c in dataset.sample(las.xyz[:, :2])])

            las.red[:] = rgb[:, 0]
            las.green[:] = rgb[:, 1]
            las.blue[:] = rgb[:, 2]

            normalized_coords = src.clouds.normalize_cloud_height(las)

            las.Z[:] = normalized_coords[:, 2] / las.header.scales[2]

            las.write(interim_als_dir / las_file.name)
