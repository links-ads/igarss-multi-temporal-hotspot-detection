import argparse
from pathlib import Path

import geopandas as gpd
import pyproj
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Point
from shapely.ops import transform


def read_tiff(path):
    with rasterio.open(path) as src:
        return src.read(), src.transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create dataset with landcover")
    parser.add_argument("--dataset", type=str, help="Path to the dataset file")
    parser.add_argument("--lc_2018",
                        type=str,
                        help="Path to the landcover 2018 file")
    parser.add_argument("--out_file", type=str, help="Path to the output file")
    return parser.parse_args()


args = parse_args()
trsf = pyproj.Transformer.from_crs(4326, 3035, always_xy=True)
df = gpd.read_file(args.dataset,
                   GEOM_POSSIBLE_NAMES="geometry",
                   KEEP_GEOM_COLUMNS="NO")
lc, lc_trsf = read_tiff(Path(args.lc_2018))
data = []
for row in df.iterrows():
    # print(row[1]["x"], row[1]["y"])
    p = Point(row[1]["x"], row[1]["y"])
    p = transform(trsf.transform, p)
    # print(p.x, p.y)
    r, c = rowcol(lc_trsf, p.x, p.y)
    data.append(lc[0, r, c])

df["lc_2018"] = data

map_class_esri = {
    1: 1,
    2: 2,
    4: 3,
    5: 4,
    7: 5,
    8: 6,
    9: 7,
    10: 8,
    11: 9,
    255: 0
}

df["lc_2018"] = df["lc_2018"].map(map_class_esri)
df.to_file(args.out_file, index=None, driver="GeoJSON")
