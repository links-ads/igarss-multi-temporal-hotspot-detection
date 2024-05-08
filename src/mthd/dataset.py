from typing import OrderedDict

import geopandas as gpd
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Sampler

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

NUM_LC_CLASSES = 11


class MeteosatDataset(Dataset):

    def __init__(self, catalog_file, mask_params, max_timesteps):
        self.mins = np.array(MINS)
        self.maxs = np.array(MAXS)
        self.mask_params = mask_params
        self.max_timesteps = max_timesteps
        self.meteosat_dataframe = self.__load_data__(catalog_file)

    def __load_data__(self, catalog_file):
        self.meteosat_df = gpd.read_file(catalog_file,
                                         GEOM_POSSIBLE_NAMES="geometry",
                                         KEEP_GEOM_COLUMNS="NO")
        self.meteosat_df["time"] = pd.to_datetime(self.meteosat_df["time"])
        self.meteosat_df["event_id"] = self.meteosat_df["event_id"].astype(int)
        self.meteosat_df["point_id"] = self.meteosat_df["point_id"].astype(int)
        self.meteosat_df["class"] = self.meteosat_df["class"].astype(int)
        keys = self.meteosat_df.groupby(["event_id", "point_id",
                                         "class"]).groups.keys()
        self.data_idxs = list(keys)
        self.tot_timeseries = 0

        for event_id, point_id, class_point in self.data_idxs:

            channels, lc, lons, lats, months = self.__get_timeseries__(
                event_id, point_id, class_point)

            self.tot_timeseries += (channels.shape[0] - self.max_timesteps)
        return

    def __len__(self):
        return self.tot_timeseries

    def __get_timestep_data__(self, sample):
        lon = sample["x"]
        lat = sample["y"]
        lc = sample["lc_2018"]
        month = sample["time"].month

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
            timeseries_channels.append(bands.squeeze())
            timeseries_lc.append(lc if lc != 255 else 0)
            timeseries_lon.append(lon)
            timeseries_lat.append(lat)
            timeseries_months.append(month)
        return np.array(timeseries_channels), np.array(
            timeseries_lc), np.array(timeseries_lon), np.array(
                timeseries_lat), np.array(timeseries_months)

    def get_data(self, idx):

        event_id, point_id, class_point = self.data_idxs[idx]

        channels, lc, lons, lats, months = self.__get_timeseries__(
            event_id, point_id, class_point)

        channels = (channels - self.mins) / (self.maxs - self.mins)

        mask_eo, mask_lc, x_eo, y_eo, x_lc, y_lc, strategy = self.mask_params.mask_data(
            channels, lc, lc_missing_data_class=0)
        latlons = np.array([lats[0], lons[0]]).astype(float)

        # TODO: modify the method in order to use a fixed length of 96 (equals to 1 day) and for N multiple days, just pick N timeseries (move logic to sampler ?)
        padding = self.max_timesteps - channels.shape[0]

        if padding > 0:
            # pad at the end
            x_eo = np.pad(x_eo, ((0, padding), (0, 0)), mode="constant")
            y_eo = np.pad(y_eo, ((0, padding), (0, 0)), mode="constant")
            mask_eo = np.pad(mask_eo, ((0, padding), (0, 0)), mode="constant")

            x_lc = np.pad(x_lc, ((0, padding)), mode="constant")
            y_lc = np.pad(y_lc, ((0, padding)), mode="constant")
            mask_lc = np.pad(mask_lc, ((0, padding)), mode="constant")

            months = np.pad(months, (0, padding), mode="reflect")

            # latlons = np.pad(latlons, ((0, 0), (0, padding)), mode="constant")

            padding_mask_eo = np.zeros_like(x_eo)
            padding_mask_eo[x_eo.shape[0]:padding, :] = 1

            padding_mask_lc = np.zeros_like(x_lc)
            padding_mask_lc[x_lc.shape[0]:padding] = 1

        else:
            padding_mask_eo = np.zeros_like(x_eo)
            padding_mask_lc = np.zeros_like(x_lc)

        # cast class_point to array
        class_point = np.array([class_point])

        return mask_eo, mask_lc, x_eo.astype(np.float32), y_eo.astype(
            np.float32), x_lc.astype(int), y_lc.astype(int), latlons.astype(
                np.float32), months.astype(
                    int), class_point, padding_mask_eo, padding_mask_lc

    def __getitem__(self, idx):
        return idx


class CustomSampler(Sampler):

    def __init__(self, dataset: MeteosatDataset):

        super().__init__(data_source=dataset)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def sliding_window(self, eo, lc, months, stride):
        assert eo.shape[0] == lc.shape[0], "eo and lc have not same shape"
        for i in range(eo.shape[0] - stride):
            yield eo[i:min(i + stride, eo.shape[0])], lc[
                i:min(i + stride, lc.shape[0]
                      )], months[i:min(i + stride, months.shape[0])] - 1

    def __iter__(self):
        for idx in range(len(self.dataset.data_idxs)):
            event_id, point_id, class_point = self.dataset.data_idxs[idx]

            channels, lc, lons, lats, months = self.dataset.__get_timeseries__(
                event_id, point_id, class_point)

            channels = (channels - self.dataset.mins) / (self.dataset.maxs -
                                                         self.dataset.mins)

            # select sliding windows of 96 timesteps
            # cast class_point to array
            class_point = np.array([class_point])
            for channels_sw, lc_sw, months_sw, in self.sliding_window(
                    channels, lc, months, stride=self.dataset.max_timesteps):

                if channels_sw.shape[0] != self.dataset.max_timesteps:
                    channels_sw = np.pad(channels_sw,
                                         ((0, self.dataset.max_timesteps -
                                           channels_sw.shape[0]), (0, 0)),
                                         mode="constant")
                    lc_sw = np.pad(
                        lc_sw,
                        (0, self.dataset.max_timesteps - lc_sw.shape[0]),
                        mode="constant")
                    months_sw = np.pad(
                        months_sw,
                        (0, self.dataset.max_timesteps - months_sw.shape[0]),
                        mode="reflect")

                mask_eo, mask_lc, x_eo, y_eo, x_lc, y_lc, strategy = self.dataset.mask_params.mask_data(
                    channels_sw, lc_sw, lc_missing_data_class=0)
                latlons = np.array([lats[0], lons[0]]).astype(float)

                # # TODO:
                # padding = self.dataset.max_timesteps - channels_sw.shape[0]

                # if padding > 0:
                #     # pad at the end
                #     x_eo = np.pad(x_eo, ((0, padding), (0, 0)), mode="constant")
                #     y_eo = np.pad(y_eo, ((0, padding), (0, 0)), mode="constant")
                #     mask_eo = np.pad(mask_eo, ((0, padding), (0, 0)),
                #                     mode="constant")

                #     x_lc = np.pad(x_lc, ((0, padding)), mode="constant")
                #     y_lc = np.pad(y_lc, ((0, padding)), mode="constant")
                #     mask_lc = np.pad(mask_lc, ((0, padding)), mode="constant")

                #     months = np.pad(months, (0, padding), mode="reflect")

                #     # latlons = np.pad(latlons, ((0, 0), (0, padding)), mode="constant")

                #     padding_mask_eo = np.zeros_like(x_eo)
                #     padding_mask_eo[x_eo.shape[0]:padding, :] = 1

                #     padding_mask_lc = np.zeros_like(x_lc)
                #     padding_mask_lc[x_lc.shape[0]:padding] = 1

                # else:
                #     padding_mask_eo = np.zeros_like(x_eo)
                #     padding_mask_lc = np.zeros_like(x_lc)
                # print(f"masked elements: {mask_eo.sum()}")
                # if class_point > 0:
                #     print("positive point")
                yield mask_eo, mask_lc, x_eo.astype(np.float32), y_eo.astype(
                    np.float32), x_lc.astype(int), y_lc.astype(
                        int), latlons.astype(
                            np.float32), months_sw.astype(int), class_point
                # , padding_mask_eo, padding_mask_lc
                # yield row
