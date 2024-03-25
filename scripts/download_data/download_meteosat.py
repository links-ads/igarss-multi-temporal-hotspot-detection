import argparse
import logging
import shutil
import threading
from datetime import datetime
from pathlib import Path

import eumdac
import eumdac.local_tailor
import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm

log_filename = f'logs/download_meteosat_{datetime.now()}'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    filename=f'{log_filename}.log',
                    filemode='w')


def parse_args():
    parser = argparse.ArgumentParser(description='Download meteosat images')

    parser.add_argument(
        '--event_list',
        type=str,
        help='Path to csv containing the details about the events. \
            There must be at least the columns "geometry", "initialdate",\
            "finaldate", "country"')

    parser.add_argument(
        '--time_range',
        type=int,
        help='Time range in days to download the timeseries of images in \
              terms of days before and after the event',
        default=0)

    parser.add_argument('--output',
                        type=Path,
                        help='Path to output folder',
                        default="data/meteosat")

    parser.add_argument('--key',
                        type=str,
                        help='Key for the meteosat api',
                        default="CKrgcmolhR7qugCoJRHCAq4_Z6ka")

    parser.add_argument('--secret',
                        type=str,
                        help='Secret for the meteosat api',
                        default="E6B0yVylTWg4rOA7L_GspDRlNtEa")

    parser.add_argument('--collection',
                        type=str,
                        help='Collection to search \
                                [EO:EUM:DAT:MSG:MSG15-RSS,\
                                EO:EUM:DAT:MSG:RSS-CLM,\
                                EO:EUM:DAT:MSG:HRSEVIRI, \
                                EO:EUM:DAT:MSG:CLM]',
                        default="EO:EUM:DAT:MSG:MSG15-RSS",
                        choices=[
                            "EO:EUM:DAT:MSG:MSG15-RSS",
                            "EO:EUM:DAT:MSG:RSS-CLM",
                            "EO:EUM:DAT:MSG:HRSEVIRI", "EO:EUM:DAT:MSG:CLM"
                        ])

    parser.add_argument('--threads',
                        type=int,
                        help='Number of threads to use',
                        default=1)

    return parser.parse_args()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_catalog(file: Path):
    df = gpd.read_file(file,
                       GEOM_POSSIBLE_NAMES="geometry",
                       KEEP_GEOM_COLUMNS="NO")
    df['initialdate'] = pd.to_datetime(df['initialdate'])
    df['finaldate'] = pd.to_datetime(df['finaldate'])
    return df[["id", "geometry", "initialdate", "finaldate", "country"]]


def search_meteosat_images(initial_date: datetime, final_date: datetime,
                           collection: str, key: str, secret: str):
    token = eumdac.AccessToken((key, secret))
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(collection)

    products = selected_collection.search(dtstart=initial_date,
                                          dtend=final_date)

    return [p for p in products]


def download_meteosat_product(event_id, product, country: str,
                              output_folder: Path):
    native_output_folder = output_folder / "native"
    native_output_folder.mkdir(parents=True, exist_ok=True)
    # zarr_output_folder = output_folder / "zarr"
    # zarr_output_folder.mkdir(parents=True, exist_ok=True)

    try:
        nat_filename = [e for e in product.entries if e.endswith(".nat")][0]
        out_nat_file = native_output_folder / nat_filename
        if not out_nat_file.is_file():
            logging.info(
                f'EVENT {event_id} - Downloading product {product}...')
            with product.open(entry=nat_filename) as fsrc, \
                    open(out_nat_file, mode='wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)

            logging.info(
                f'EVENT {event_id} - Download of product {product} finished.')
    except eumdac.product.ProductError as error:
        logging.error(
            f"EVENT {event_id} - Error related to the product '{product}' \
                while trying to download it: '{error.msg}'")
    except eumdac.collection.CollectionError as error:
        logging.error(f"EVENT {event_id} - Error related to the collection: \
                '{error.msg}'")
    except requests.exceptions.ConnectionError as error:
        logging.error(f"EVENT {event_id} - Error related to the connection: \
                '{error.msg}'")
    except requests.exceptions.RequestException as error:
        logging.error(f"EVENT {event_id} - Unexpected error: {error}")

    # logging.info(f'EVENT {event_id} - Converting product {product} to zarr...')
    # zarr_data, zarr_data_hrv = load_native_to_dataarray(out_nat_file,
    #                                                     area="EU",
    #                                                     temp_directory="temp")

    # zarr_data.to_netcdf(zarr_output_folder / f"{product}.nc")
    # zarr_data_hrv.to_netcdf(zarr_output_folder / f"{product}_hrv.nc")
    # logging.info(
    #     f'EVENT {event_id} - Conversion of product {product} to zarr finished.'
    # )


def is_product_already_downloaded(product, output_folder: Path):
    # logging.info(
    # f'Checking if product {product} is already downloaded in {output_folder / "native" / f"{product}.NAT"}...'
    # )
    return (output_folder / "native" / f"{product}.nat").is_file()


def downloader_thread(idx: int, event_list: pd.DataFrame, time_offset: int,
                      collection: str, key: str, secret: str,
                      output_folder: Path):
    log_file = open(f'{log_filename}_{idx}.log', 'w')
    for _, event in tqdm(event_list.iterrows(),
                         desc=f'Thread {idx}',
                         total=len(event_list),
                         file=log_file):

        initial_date = event["initialdate"] - pd.Timedelta(days=time_offset)
        final_date = event["finaldate"] + pd.Timedelta(days=time_offset)
        products = search_meteosat_images(initial_date=initial_date,
                                          final_date=final_date,
                                          collection=collection,
                                          key=key,
                                          secret=secret)

        for p in products:
            if is_product_already_downloaded(
                    p, output_folder / event["id"] / collection):
                logging.info(
                    f'EVENT {event["id"]} - Product {p} already downloaded')
                continue
            else:
                logging.info(
                    f'EVENT {event["id"]} - Downloading product {p}...')
                download_meteosat_product(event_id=event["id"],
                                          product=p,
                                          country=event["country"],
                                          output_folder=output_folder /
                                          event["id"] / collection)
        logging.info(f'EVENT {event["id"]} - Finished processing event')


def main(args):

    INPUT_FILE = Path(args.event_list)
    assert INPUT_FILE.is_file(), f'File {INPUT_FILE} does not exist'

    DEST_FOLDER = Path(args.output)
    DEST_FOLDER.mkdir(parents=True, exist_ok=True)

    logging.info(f'Loading event list from {INPUT_FILE}')
    df_effis = read_catalog(INPUT_FILE)
    logging.info(f'Loaded {len(df_effis)} events')

    ids = df_effis['id'].unique()
    threads = []
    for idx, sublist_ids in enumerate(chunks(ids, len(ids) // args.threads)):
        logging.info(f'Processing sublist {idx}')
        df_effis_subset = df_effis[df_effis['id'].isin(sublist_ids)]
        threads.append(
            threading.Thread(target=downloader_thread,
                             args=(idx, df_effis_subset, args.time_range,
                                   args.collection, args.key, args.secret,
                                   DEST_FOLDER)))
        threads[-1].start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    args = parse_args()
    main(args)
