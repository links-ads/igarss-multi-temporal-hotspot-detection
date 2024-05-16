import argparse
import random
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        help="input csv/json file containg timeseries",
                        type=Path)
    parser.add_argument("--output",
                        help="output file",
                        type=Path,
                        default=Path("dataset_processed.json"))

    return parser.parse_args()


def read_file(catalog_file):

    df = gpd.read_file(catalog_file,
                       GEOM_POSSIBLE_NAMES="geometry",
                       KEEP_GEOM_COLUMNS="NO")
    df["time"] = pd.to_datetime(df["time"])
    df["event_id"] = df["event_id"].astype(int)
    df["point_id"] = df["point_id"].astype(int)
    df["class"] = df["class"].astype(int)

    return df


def main():
    args = parse_args()

    df = read_file(args.input)
    keys = df.groupby(["event_id", "point_id", "class"]).groups.keys()
    events = set([k[0] for k in keys])
    processed_df = gpd.GeoDataFrame(columns=df.columns)
    for e in tqdm(events):
        pos_event_df = df[(df["event_id"] == e) & (df["class"] == 1)]
        neg_event_df = df[(df["event_id"] == e) & (df["class"] == 0)]
        pos_points = pos_event_df["point_id"].unique()
        neg_points = neg_event_df["point_id"].unique()[:len(pos_points)]
        for p in pos_points:
            processed_df = processed_df.append(
                pos_event_df[pos_event_df["point_id"] == p])
        for p in neg_points:
            processed_df = processed_df.append(
                neg_event_df[neg_event_df["point_id"] == p])
    processed_df["time"] = processed_df["time"].astype('string')
    processed_df["event_id"] = processed_df["event_id"].astype(int)
    processed_df["point_id"] = processed_df["point_id"].astype(str)
    processed_df["class"] = processed_df["class"].astype(str)
    processed_df["lc_2018"] = processed_df["lc_2018"].astype(int)
    processed_df.to_file(args.output, index=None, driver="GeoJSON")


main()
