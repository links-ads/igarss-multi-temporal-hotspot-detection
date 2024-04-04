import argparse
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from epct import api
from tqdm import tqdm


def convert_folder(FILES: Path, output_folder: Path, chain_config: dict):
    output_folder.mkdir(exist_ok=True, parents=True)
    for f in tqdm(FILES):
        api.run_chain([str(f)],
                      chain_config=chain_config,
                      target_dir=str(output_folder))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert native files to netcdf')
    parser.add_argument('--event_file',
                        type=Path,
                        help='Input file with list of events')
    parser.add_argument('--input_folder',
                        type=Path,
                        help='Input folder with native files')
    parser.add_argument('--output_folder',
                        type=Path,
                        help='Output folder with netcdf files')
    parser.add_argument('--product',
                        type=str,
                        default='HRSEVIRI',
                        help='Product name')
    parser.add_argument('--format',
                        type=str,
                        default='netcdf4',
                        help='Output format')
    parser.add_argument('--projection',
                        type=str,
                        default='geographic',
                        help='Output projection')
    return parser.parse_args()


def main():
    args = parse_args()

    events = gpd.read_file(args.event_file,
                           GEOM_POSSIBLE_NAMES="geometry",
                           KEEP_GEOM_COLUMNS="NO")

    for folder in args.input_folder.iterdir():

        event = str(folder).split("/")[-1]
        native_files = [f for f in folder.glob("*/native/*.nat")]
        len_native_files = len(native_files)

        if (args.output_folder / event).exists() and len([
                f for f in (args.output_folder / event).glob("*.nc")
        ]) >= len_native_files:
            continue
        country = events[events["id"] == int(event)]["country"].values[0]
        chain_config = {
            'product': args.product,
            'format': args.format,
            'projection': args.projection,
            'roi': country
        }

        convert_folder(native_files, args.output_folder / event, chain_config)


main()
