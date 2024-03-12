# script per creare il csv con tutti i pixels estratti dai dati di meteosat
import argparse
from pathlib import Path
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import xarray
from pyproj import CRS, Proj
from shapely.geometry import MultiPolygon, Polygon
from tqdm import tqdm

proj_dict = {
    'a': 6378169,
    'h': 35785831,
    'lon_0': 0,
    'no_defs': 'None',
    'proj': 'geos',
    'rf': 295.488065897014,
    'type': 'crs',
    'units': 'm',
    'x_0': 0,
    'y_0': 0
}
crs = CRS.from_dict(proj_dict)
proj = Proj(crs)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create dataset from meteosat data')
    parser.add_argument('--data_path',
                        type=Path,
                        help='Path to the data folder')
    parser.add_argument('--output_path',
                        type=Path,
                        default='data',
                        help='Path to the output folder')
    parser.add_argument('--output_name',
                        type=Path,
                        default='dataset.csv',
                        help='Name of the output csv file')
    parser.add_argument('--event_file',
                        type=Path,
                        help='Path to the  events file')
    parser.add_argument(
        '--margin',
        type=int,
        default=1,
        help='Margin to add to the pixels selection of the events')
    return parser.parse_args()


def extract_pixels_from_xarray(
    data: xarray,
    event_geometry: Union[Polygon, MultiPolygon],
    margin: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract pixels from the xarray data that are inside the event geometry

    Args:
    data: xarray
    event_geometry: Union[Polygon, MultiPolygon]
    margin: int

    Returns:
    positive data: np.ndarray
    negative data:np.ndarray
    positive points mask: np.ndarray
    negative points mask: np.ndarray
    data as numpy array: np.ndarray
    coordinate of positive points: np.ndarray
    coordinates of negative points: np.ndarray
    """

    minx = event_geometry.bounds["minx"].values[0]
    miny = event_geometry.bounds["miny"].values[0]
    maxx = event_geometry.bounds["maxx"].values[0]
    maxy = event_geometry.bounds["maxy"].values[0]

    correct_points_mask = data.where((data.x_geostationary >= minx)
                                     & (data.x_geostationary <= maxx)
                                     & (data.y_geostationary >= miny)
                                     & (data.y_geostationary <= maxy))
    # create a mask where nan is false and true otherwse
    mask = correct_points_mask.notnull().to_array().to_numpy()
    data_np = data.to_array().to_numpy()
    mask_pos = np.zeros_like(mask)
    mask_neg = np.zeros_like(mask)
    indexes = np.where(mask)

    indexes_x = indexes[3]
    indexes_y = indexes[2]

    indexes_x_right = indexes_x + margin
    indexes_y_right = indexes_y + margin
    indexes_x_left = indexes_x - margin
    indexes_y_left = indexes_y - margin

    indexes_y_right = indexes_y_right.clip(0, data_np.shape[2] - 1)
    indexes_x_right = indexes_x_right.clip(0, data_np.shape[3] - 1)
    indexes_y_left = indexes_y_left.clip(0, data_np.shape[2] - 1)
    indexes_x_left = indexes_x_left.clip(0, data_np.shape[3] - 1)

    mask_neg[:, :, indexes_y_right, indexes_x_right, :] = 1
    mask_neg[:, :, indexes_y_left, indexes_x_left, :] = 1
    mask_neg[:, :, indexes_y_left, indexes_x_right, :] = 1
    mask_neg[:, :, indexes_y_right, indexes_x_left, :] = 1
    mask_neg[:, :, indexes_y_right, indexes_x, :] = 1
    mask_neg[:, :, indexes_y_left, indexes_x, :] = 1
    mask_neg[:, :, indexes_y, indexes_x_right, :] = 1
    mask_neg[:, :, indexes_y, indexes_x_left, :] = 1
    mask_pos[:, :, indexes_y, indexes_x, :] = 1
    mask_neg[:, :, indexes_y, indexes_x, :] = 0

    x = data.x_geostationary.values
    y = data.y_geostationary.values

    points_pos = np.array([x[indexes_x], y[indexes_y]]).T

    indexes_neg = np.where(mask_neg)
    indexes = np.where(mask_neg)
    indexes_x_neg = indexes_neg[3]
    indexes_y_neg = indexes_neg[2]

    points_neg = np.array([x[indexes_x_neg], y[indexes_y_neg]]).T

    return data_np[mask_pos], data_np[
        mask_neg], mask_pos, mask_neg, data_np, points_pos, points_neg


def read_timeseries_xarray(folder: Path):
    return xarray.open_mfdataset(folder.glob("*NA.nc"), combine="by_coords")


def read_single_xarray(file: Path):
    return xarray.open_dataset(file)


def main():
    args = parse_args()

    # load events
    events_df = gpd.read_file(args.event_file,
                              GEOM_POSSIBLE_NAMES="geometry",
                              KEEP_GEOM_COLUMNS="NO")
    events_df["id"] = events_df["id"].astype(int)

    events_id = events_df["id"].unique()
    # load data
    pixels_df = gpd.GeoDataFrame(columns=[
        'x', 'y', 'class', 'event_id', 'IR_016', 'IR_039', 'IR_087', 'IR_097',
        'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073'
    ])
    df_id = 0

    for event in tqdm(events_id):
        folder = args.data_path / str(
            event) / "EO:EUM:DAT:MSG:HRSEVIRI" / "zarr"
        if not folder.exists():
            print(f"Folder {folder} does not exist")
            continue
        print(f"Processing event {event}")
        data = read_timeseries_xarray(folder)
        event_geometry = events_df[events_df["id"] == event].geometry
        data = extract_pixels_from_xarray(data, event_geometry, args.margin)
        positive_data, negative_data, positive_mask, negative_mask, data_np, positive_coords, negative_coords = data

        # save positive data

        for bands, point in zip(positive_data, positive_coords):
            x, y = point
            pixel_class = 1

            pixels_df[df_id] = [x, y, pixel_class, event] + [b for b in bands]

        # save negative data

        for bands, point in zip(negative_data, negative_coords):
            x, y = point
            pixel_class = 0

            pixels_df[df_id] = [x, y, pixel_class, event] + [b for b in bands]

        pixels_df.to_csv(args.output_path / args.output_name, index=False)


main()
