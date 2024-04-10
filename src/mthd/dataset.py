from typing import OrderedDict

import geopandas as gpd
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# TODO: create dataset from csv with meteosat data and land cover

HRSEVIRI_BANDS = [
    "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134",
    "VIS006", "VIS008", "WV_062", "WV_073"
]

MINS = [
    -0.014845, -0.021301, -0.027130, 0.010035, 1.082570, 3.047131, 7.667633,
    16.565571, 18.289879, 24.567041, 33.154335
]

MAXS = [
    20.646154, 24.351849, 16.185978, 3.186471, 5.594554, 20.754074, 85.152573,
    56.398739, 132.604828, 145.506439, 98.822182
]

BANDS_GROUPS_IDX = OrderedDict([
    ("IR", [0, 1, 2, 3, 4, 5, 6]),
    ("VIS", [7, 8]),
    ("WV", [9, 10]),
])

NUM_LC_CLASSES = 10


class MeteosatDataset(Dataset):

    def __init__(self, csv_file):
        self.meteosat_dataframe = self.__load_data__(csv_file)
        self.mins = np.array(MINS)
        self.maxs = np.array(MAXS)

    def __load_data__(self, csv_file):
        self.meteosat_df = gpd.read_csv(csv_file)
        self.meteosat_df["time"] = pd.to_datetime(self.meteosat_df["time"])
        self.meteosat_df["event_id"] = self.meteosat_df["event_id"].astype(int)
        self.meteosat_df["point_id"] = self.meteosat_df["point_id"].astype(int)
        self.meteosat_df["class"] = self.meteosat_df["class"].astype(int)
        keys = self.meteosat_df.groupby(["event_id", "point_id",
                                         "class"]).groups.keys()
        self.data_idxs = list(keys)

    def __len__(self):
        return len(self.data_idxs)

    def __get_timestep_data__(self, sample):
        lon = sample["x"]
        lat = sample["y"]
        lc = sample["lc_2018"]
        month = sample["timestamp"].month

        data = np.zeros((len(HRSEVIRI_BANDS), 1))
        for b in HRSEVIRI_BANDS:
            data[HRSEVIRI_BANDS.index(b)] = sample[b]

        return data, lc, lon, lat, month

    def __get_timeseries__(self, event_id, point_id, class_point):
        data = self.meteosat_df[(self.meteosat_df["event_id"] == event_id)
                                & (self.meteosat_df["point_id"] == point_id) &
                                (self.meteosat_df["class"] == class_point)]
        timeseries_channels = []
        timeseries_lc = []
        timeseries_lon = []
        timeseries_lat = []
        timeseries_months = []
        for i in range(len(data)):
            sample = data.iloc[i]
            bands, lc, lon, lat, month = self.__get_timestep_data__(sample)
            timeseries_channels.append(bands)
            timeseries_lc.append(lc)
            timeseries_lon.append(lon)
            timeseries_lat.append(lat)
            timeseries_months.append(month)
        return np.array(timeseries_channels), np.array(
            timeseries_lc), np.array(timeseries_lon), np.array(
                timeseries_lat), np.array(timeseries_months)

    def __getitem__(self, idx):

        event_id, point_id, class_point = self.data_idxs[idx]

        channels, lc, lons, lats = self.__get_timeseries__(
            event_id, point_id, class_point)

        channels = (channels - self.mins) / (self.maxs - self.mins)

        return channels, lc, lons, lats
