import argparse
from pathlib import Path

import geopandas as gpd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset in train, validation and test")

    parser.add_argument("--dataset_folder",
                        type=str,
                        help="Path to the dataset folder")
    parser.add_argument("--dataset_csv",
                        type=str,
                        help="Path to the dataset csv")
    parser.add_argument("--thraws_csv",
                        type=str,
                        help="Path to the thraws csv")
    parser.add_argument("--out_folder",
                        type=str,
                        help="Path to the output folder")
    return parser.parse_args()


def count_files(FOLDER):
    files_per_event = {}
    for event_folder in FOLDER.iterdir():
        if event_folder.is_dir():
            files = [f for f in event_folder.glob("*.nc")]
            files_per_event[event_folder.name] = len(files)
    return files_per_event


args = parse_args()
files_per_event = count_files(Path(args.dataset_folder))
files_per_event.pop("9021")
files_per_event.pop("12702")

thraws_in_df = {
    "France_0": 52897,
    "Greece_0": 54081,
    "Greece_1": 51932,
    "Greece_2": 51941,
    "Latvia_0": 51012,
    "Spain_2": 43538,
    "Greece_3": 42303,
    "Ukraine_0": 24806,
    "Greece_4": 16361,
    "Latvia_1": 13921,
    "Spain_5": 12998,
}

events_df = gpd.read_file(args.dataset_csv,
                          GEOM_POSSIBLE_NAMES="geometry",
                          KEEP_GEOM_COLUMNS="NO")

events_thraws_df = gpd.read_file(args.thraws_csv,
                                 GEOM_POSSIBLE_NAMES="geometry",
                                 KEEP_GEOM_COLUMNS="NO")

thraws_filtered_events = events_df[events_df["id"].isin(thraws_in_df.values())]

train, val = train_test_split(list(files_per_event.keys()),
                              test_size=0.2,
                              random_state=42)
# train, val = train_test_split(train,test_size=0.2, random_state=42)
test = [
    '52897', '54081', '51932', '51941', '51012', '43538', '42303', '24806',
    '16361', '13921', '12998'
]

# divide df basing on train, val, test
events_df["event_id"] = events_df["event_id"].astype(str)
events_thraws_df["event_id"] = events_thraws_df["event_id"].astype(str)
train_df = events_df[events_df["event_id"].isin(train)]
val_df = events_df[events_df["event_id"].isin(val)]
# test_df = events_df[events_df["event_id"].isin(test)]
test_df = events_thraws_df[events_thraws_df["event_id"].isin(test)]

train_df["event_id"] = train_df["event_id"].astype(int)
val_df["event_id"] = val_df["event_id"].astype(int)
test_df["event_id"] = test_df["event_id"].astype(int)

train_df.to_file(Path(args.out_folder) / "train.json",
                 index=None,
                 driver="GeoJSON")
val_df.to_file(Path(args.out_folder) / "val.json",
               index=None,
               driver="GeoJSON")
test_df.to_file(Path(args.out_folder) / "test.json",
                index=None,
                driver="GeoJSON")
