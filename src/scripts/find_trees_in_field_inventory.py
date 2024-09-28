"""Match tree clouds from Lysva to the corresponding trees in the field inventory."""

import glob
import re

import geopandas as gpd
import laspy
import shapely

if __name__ == "__main__":
    fi = gpd.read_file("data/raw/lysva/field_survey.geojson")
    fi["las_file"] = None

    file_paths = glob.glob("data/interim/trees/*.las")
    assert len(file_paths) > 0
    species_regex = re.compile(r"/([a-z]+)_\d\d.las")

    for file in file_paths:
        species = species_regex.findall(file)[0].capitalize()
        las = laspy.read(file)
        centroid = shapely.Point(*las.xyz.mean(axis=0))
        distances = fi[fi["species"] == species].distance(centroid)
        fi.loc[distances.idxmin(), "las_file"] = file.split("/")[-1]

    fi.to_file("data/interim/lysva/field_survey.geojson")
