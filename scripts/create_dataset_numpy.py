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

    minx, miny, maxx, maxy = event_geometry.bounds
    # in the xarray x and y are swapped
    point_min = data.sel(x_geostationary=miny,
                         y_geostationary=minx,
                         method="nearest")
    point_max = data.sel(x_geostationary=maxy,
                         y_geostationary=maxx,
                         method="nearest")
    # IF THE GEOM IS TOO SMALL, ONLY 1 PIXEL IS SELECTED
    if point_min.x_geostationary.values == point_max.x_geostationary.values and point_min.y_geostationary.values == point_max.y_geostationary.values:
        correct_points_mask = data.where(
            (data.x_geostationary == point_min.x_geostationary.values)
            & (data.y_geostationary == point_min.y_geostationary.values))
    else:
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

    # recompute because sometimes mask misses some values. Who knows why
    indexes = np.where(mask_pos)

    indexes_x = indexes[3]
    indexes_y = indexes[2]

    points_pos = np.array([x[indexes_x], y[indexes_y]]).T

    indexes_neg = np.where(mask_neg)
    indexes = np.where(mask_neg)
    indexes_x_neg = indexes_neg[3]
    indexes_y_neg = indexes_neg[2]

    points_neg = np.array([x[indexes_x_neg], y[indexes_y_neg]]).T
    time = len(data["time"])
    num_points = round(mask.sum() / time / 11)
    num_points_neg = round(mask_neg.sum() / time / 11)
    return data_np[mask_pos].reshape(
        time, num_points, 11), data_np[mask_neg].reshape(
            time, num_points_neg,
            11), mask_pos, mask_neg, data_np, points_pos.reshape(
                time, num_points, 11,
                2)[0, :, 0, :], points_neg.reshape(time, num_points_neg, 11,
                                                   2)[0, :,
                                                      0, :], data.time.values


def read_timeseries_xarray(folder: Path):
    return xarray.open_mfdataset(folder.glob("*NA.nc"), combine="by_coords")


def read_single_xarray(file: Path):
    return xarray.open_dataset(file)


def store_points_data(data_array, coords, pixel_class, event_id, timestamps):
    data_df = []
    idx = 0
    for timeseries_data, point in zip(data_array, coords):
        x, y = point
        for idx, data in enumerate(timeseries_data):
            data_df.append({
                "x": x,
                "y": y,
                "class": pixel_class,
                "event_id": event_id,
                'IR_016': data[0],
                'IR_039': data[1],
                'IR_087': data[2],
                'IR_097': data[3],
                'IR_108': data[4],
                'IR_120': data[5],
                'IR_134': data[6],
                'VIS006': data[7],
                'VIS008': data[8],
                'WV_062': data[9],
                'WV_073': data[10],
                "time": timestamps[idx]
            })
    return data_df


def main():
    args = parse_args()

    # load events
    events_df = gpd.read_file(args.event_file,
                              GEOM_POSSIBLE_NAMES="geometry",
                              KEEP_GEOM_COLUMNS="NO")
    events_df["id"] = events_df["id"].astype(int)

    events_id = events_df["id"].unique()

    for event in tqdm(events_id):
        out_event_folder = args.output_path / str(event)
        if out_event_folder.exists() and (
                out_event_folder / "positive_data.npy").exists() and (
                    out_event_folder / "negative_data.npy").exists():
            print(f"Event {event} already processed")
            continue
        out_event_folder.mkdir(parents=True, exist_ok=True)
        folder = args.data_path / str(
            event) / "EO:EUM:DAT:MSG:HRSEVIRI" / "zarr"
        if not folder.exists():
            print(f"Folder {folder} does not exist")
            continue
        print(f"Processing event {event}")
        data = read_timeseries_xarray(folder)
        event_geometry = events_df[events_df["id"] == event].geometry
        data = extract_pixels_from_xarray(data,
                                          event_geometry.geometry.values[0],
                                          args.margin)

        positive_data, negative_data, positive_mask, negative_mask, data_np, positive_coords, negative_coords, timestamps = data

        # save positive data
        # TODO: rivedere il ciclo
        # positive_data shape 192, 1 ( numpoints) ,11 e positive_coords shape 192, 1 (numpoints), 11 (perchè? non dovrebbe esserci), 2
        positive_data = np.transpose(positive_data, (1, 0, 2))
        negative_data = np.transpose(negative_data, (1, 0, 2))

        np.save(out_event_folder / "positive_data.npy", positive_data)
        np.save(out_event_folder / "negative_data.npy", negative_data)


main()
