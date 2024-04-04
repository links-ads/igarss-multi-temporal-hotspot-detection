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
    parser.add_argument("--store_bands", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
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
    point_min = data.sel(lat=miny, lon=minx, method="nearest")
    point_max = data.sel(lat=maxy, lon=maxx, method="nearest")
    # IF THE GEOM IS TOO SMALL, ONLY 1 PIXEL IS SELECTED
    if point_min.lat.values == point_max.lat.values and point_min.lon.values == point_max.lon.values:
        correct_points_mask = data.where((data.lat == point_min.lat.values)
                                         & (data.lon == point_min.lon.values))
    else:
        correct_points_mask = data.where((data.lat >= minx)
                                         & (data.lat <= maxx)
                                         & (data.lon >= miny)
                                         & (data.lon <= maxy))
    # create a mask where nan is false and true otherwse
    mask = correct_points_mask.notnull().to_array().to_numpy()[1:, :, :, :]

    #  data_np = data.to_array().to_numpy()

    data_np = np.zeros((11, data.time.size, data.lat.size, data.lon.size))
    for i in range(1, 12):
        data_np[i - 1, :, :, :] = data[f"channel_{i}"].to_numpy()
    mask_pos = np.zeros_like(mask)
    mask_neg = np.zeros_like(mask)
    indexes = np.where(mask)

    indexes_x = indexes[2]
    indexes_y = indexes[3]

    indexes_x_right = indexes_x + margin
    indexes_y_right = indexes_y + margin
    indexes_x_left = indexes_x - margin
    indexes_y_left = indexes_y - margin

    indexes_y_right = indexes_y_right.clip(0, data_np.shape[3] - 1)
    indexes_x_right = indexes_x_right.clip(0, data_np.shape[2] - 1)
    indexes_y_left = indexes_y_left.clip(0, data_np.shape[3] - 1)
    indexes_x_left = indexes_x_left.clip(0, data_np.shape[2] - 1)

    mask_neg[:, :, indexes_x_right, indexes_y_right] = 1
    mask_neg[:, :, indexes_x_left, indexes_y_left] = 1
    mask_neg[:, :, indexes_x_left, indexes_y_right] = 1
    mask_neg[:, :, indexes_x_right, indexes_y_left] = 1
    mask_neg[:, :, indexes_x_right, indexes_y] = 1
    mask_neg[:, :, indexes_x_left, indexes_y] = 1
    mask_neg[:, :, indexes_x, indexes_y_right] = 1
    mask_neg[:, :, indexes_x, indexes_y_left] = 1
    mask_pos[:, :, indexes_x, indexes_y] = 1
    mask_neg[:, :, indexes_x, indexes_y] = 0

    x = data.lon.values
    y = data.lat.values

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
    final_data_pos = data_np[mask_pos].reshape(11, time, num_points)
    final_data_pos = np.transpose(final_data_pos, (1, 2, 0))
    final_data_neg = data_np[mask_neg].reshape(11, time, num_points_neg)
    final_data_neg = np.transpose(final_data_neg, (1, 2, 0))

    return final_data_pos, final_data_neg, mask_pos, mask_neg, data_np, points_pos.reshape(
        time, num_points, 11,
        2)[0, :, 0, :], points_neg.reshape(time, num_points_neg, 11,
                                           2)[0, :, 0, :], data.time.values


def read_timeseries_xarray(files: list):
    return xarray.open_mfdataset(files, concat_dim="time", combine="nested")


def read_single_xarray(file: Path):
    return xarray.open_dataset(file)


def store_points_data(data_array,
                      coords,
                      pixel_class,
                      event_id,
                      timestamps,
                      store_bands=False):
    data_df = []
    point_idx = 0
    for timeseries_data, point in zip(data_array, coords):
        x, y = point
        if store_bands:
            for idx, data in enumerate(timeseries_data):

                data_df.append({
                    "x": x,
                    "y": y,
                    "point_id": point_idx,
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
                    "time": timestamps[idx],
                    "geometry": f"POINT ({x} {y})"
                })
        else:
            data_df.append({
                "x": x,
                "y": y,
                "class": pixel_class,
                "event_id": event_id,
                "geometry": f"POINT ({x} {y})"
            })
        point_idx += 1
    return data_df


def main():
    args = parse_args()

    # load events
    events_df = gpd.read_file(args.event_file,
                              GEOM_POSSIBLE_NAMES="geometry",
                              KEEP_GEOM_COLUMNS="NO")
    events_df["id"] = events_df["id"].astype(int)

    events_id = events_df["id"].unique()

    it = 0
    if args.use_cache and (args.output_path / args.output_name).exists():
        cached_df = gpd.read_file(args.output_path / args.output_name)
        cached_events = cached_df["event_id"].unique()

    for folder in tqdm(args.data_path.iterdir()):

        event = str(folder).split("/")[-1]
        if args.use_cache and str(event) in cached_events:
            print(f"Event {event} already processed")
            continue
        # folder = args.data_path / str(event)
        if not folder.exists():
            print(f"Folder {folder} does not exist")
            continue
        print(f"Processing event {event}")

        files = [f for f in folder.glob("*.nc")]
        files.sort()
        timestamps = [f.stem.split("_")[1] for f in files]
        data = read_timeseries_xarray(files)
        event_geometry = events_df[events_df["id"] == int(event)].geometry
        data = extract_pixels_from_xarray(data,
                                          event_geometry.geometry.values[0],
                                          args.margin)

        positive_data, negative_data, positive_mask, negative_mask, data_np, positive_coords, negative_coords, _ = data

        positive_data = np.transpose(positive_data, (1, 0, 2))
        negative_data = np.transpose(negative_data, (1, 0, 2))

        data_df = []
        data_df.extend(
            store_points_data(positive_data, positive_coords, 1, event,
                              timestamps, args.store_bands))
        data_df.extend(
            store_points_data(negative_data, negative_coords, 0, event,
                              timestamps, args.store_bands))

        pixels_df = gpd.GeoDataFrame(data_df)
        if args.use_cache and len(cached_events) > 0:
            pixels_df.to_csv(args.output_path / args.output_name,
                             index=False,
                             mode='a',
                             header=False)
        else:
            pixels_df.to_csv(args.output_path / args.output_name,
                             index=False,
                             mode='a' if it > 0 else 'w',
                             header=it == 0)
        it += 1


main()
