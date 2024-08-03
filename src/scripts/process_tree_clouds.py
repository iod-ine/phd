"""Add color to the individual tree point clouds."""

import pathlib

import laspy
import numpy as np
import rasterio
import tqdm

if __name__ == "__main__":
    raw_data_dir = pathlib.Path("data/raw")

    interim_data_dir = pathlib.Path("data/interim/trees")
    interim_data_dir.mkdir(exist_ok=True)

    las_files = raw_data_dir.glob("trees/*/*.las")
    # Pines are from another survey, not covered be the orthophoto
    las_files = filter(lambda x: x.parent.name != "Pine", las_files)
    las_files = sorted(las_files)

    with rasterio.open(raw_data_dir / "full-lysva" / "ortho.tif") as dataset:
        for las_file in tqdm.tqdm(las_files):
            las = laspy.read(las_file)
            las = laspy.convert(las, point_format_id=7)

            rgb = np.stack([c for c in dataset.sample(las.xyz[:, :2])])

            las.red[:] = rgb[:, 0]
            las.green[:] = rgb[:, 1]
            las.blue[:] = rgb[:, 2]

            las.write(interim_data_dir / las_file.name)
