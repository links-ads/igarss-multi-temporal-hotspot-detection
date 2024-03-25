import argparse
import datetime
import fnmatch
import logging
import os
import shutil
import threading
import time
from pathlib import Path

import eumdac
import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm

CONSUMER_KEY = "CKrgcmolhR7qugCoJRHCAq4_Z6ka"
CONSUMER_SECRET = "E6B0yVylTWg4rOA7L_GspDRlNtEa"

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def request_token(key=CONSUMER_KEY, secret=CONSUMER_SECRET):
    credentials = (key, secret)
    token = eumdac.AccessToken(credentials)
    return token


def request_products_list(token,
                          start,
                          end,
                          collection="EO:EUM:DAT:MSG:MSG15-RSS"):
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(collection)
    products = selected_collection.search(dtstart=start, dtend=end)

    return products


def download_meteosat_images(products: list,
                             output_folder: Path,
                             key=CONSUMER_KEY,
                             secret=CONSUMER_SECRET):

    credentials = (key, secret)

    chain = eumdac.tailor_models.Chain(
        # product='MSGCLMK',
        product="HRSEVIRI_RSS",
        name="Download geotiffs",
        format='geotiff',
        projection='geographic',
        roi='central_europe')

    for product in tqdm(products):
        # Defining the chain configuration
        token = eumdac.AccessToken(credentials)
        datatailor = eumdac.DataTailor(token)
        customisation = datatailor.new_customisation(product, chain)
        try:
            LOG.info(f"Customisation {customisation._id} started.")
        except eumdac.datatailor.DataTailorError as error:
            LOG.info(f"Error related to the Data Tailor: '{error.msg}'")
        except requests.exceptions.RequestException as error:
            LOG.info(f"Unexpected error: {error}")

        status = customisation.status
        sleep_time = 10  # seconds
        retry = True
        while retry:
            # Customisation Loop
            while status:
                # Get the status of the ongoing customisation
                status = customisation.status

                if "DONE" in status:
                    LOG.info(
                        f"Customisation {customisation._id} is \
                            successfully completed."
                    )
                    break
                elif status in [
                        "ERROR", "FAILED", "DELETED", "KILLED", "INACTIVE"
                ]:
                    LOG.info(
                        f"Customisation {customisation._id} was unsuccessful.\
                        Customisation log is LOG.infoed.\n"
                    )
                    LOG.info(customisation.logfile)
                    break
                elif "QUEUED" in status:
                    LOG.info(f"Customisation {customisation._id} is queued.")
                elif "RUNNING" in status:
                    LOG.info(f"Customisation {customisation._id} is running.")
                time.sleep(sleep_time)

            tif, = fnmatch.filter(customisation.outputs, '*.tif')
            LOG.info(tif)

            jobID = customisation._id
            LOG.info(f"Dowloading the tif output of the customisation {jobID}")
            try:
                with customisation.stream_output(tif,) as stream:
                    splits = stream.name.split("_")
                    name = "_".join(splits[0:-3]) + ".tif"
                    with open(output_folder / name, mode='wb') as fdst:
                        shutil.copyfileobj(stream, fdst)
                retry = False
                LOG.info(f"Dowloaded the tif output of the customisation {jobID}")
            except eumdac.datatailor.DataTailorError as error:
                LOG.info("Data Tailor Error", error)
            except requests.exceptions.RequestException as error:
                LOG.info(f"Unexpected error: {error}")

            # Delete customisation to free space,
            # The Data Tailor Web Service has a 20 Gb limit
            try:
                customisation.delete()
                retry = False
            except eumdac.datatailor.CustomisationError as exc:
                LOG.info("Customisation Error:", exc)
            except requests.exceptions.RequestException as error:
                LOG.info("Unexpected error:", error)


class DownloaderThread(threading.Thread):
    def __init__(self, products, output_folder, key, secret):
        threading.Thread.__init__(self)
        self.products = products
        self.output_folder = output_folder
        self.key = key
        self.secret = secret

    def run(self):
        download_meteosat_images(self.products, self.output_folder, self.key,
                                 self.secret)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parse_args():
    parser = argparse.ArgumentParser(description='Download meteosat images')
    parser.add_argument('--output',
                        type=Path,
                        help='Path to output folder',
                        default="data/meteosat_images")
    parser.add_argument('--effis', type=Path, help='Path to effis fires file')
    parser.add_argument('--effis_fire_id',
                        type=int,
                        help='Effis fire id',
                        default=0)
    parser.add_argument('--time_range',
                        type=int,
                        help='Time range in days',
                        default=5)

    return parser.parse_args()


def main():
    args = parse_args()

    assert args.effis.is_file(), f'File {args.effis} does not exist'
    DEST_FOLDER = args.output / f"{args.effis_fire_id}"
    DEST_FOLDER.mkdir(parents=True, exist_ok=True)

    LOG.info(f'Loading effis fires from {args.effis}')
    df_effis = gpd.read_file(args.effis,
                             GEOM_POSSIBLE_NAMES="geometry",
                             KEEP_GEOM_COLUMNS="NO")
    df_effis['initialdate'] = pd.to_datetime(df_effis['initialdate'],
                                             format='ISO8601')
    df_effis['finaldate'] = pd.to_datetime(df_effis['finaldate'],
                                           format='ISO8601')
    try:
        effis_fire = df_effis[df_effis['id'] == args.effis_fire_id]
    except Exception:
        LOG.info(f"Effis fire id {args.effis_fire_id} not found")
        exit(1)

    # start_date_hotspots = df_hotspots["timestamp"].min()
    # end_date_hotspots = df_hotspots["timestamp"].max()

    start = effis_fire["initialdate"] - datetime.timedelta(
        days=args.time_range)
    end = effis_fire["finaldate"] + datetime.timedelta(days=args.time_range)

    start = start.values[0]
    end = end.values[0]

    token = request_token(CONSUMER_KEY, CONSUMER_SECRET)

    try:
        LOG.info(f"This token '{token}' expires {token.expiration}")
    except requests.exceptions.HTTPError as error:
        LOG.info(f"Unexpected error: {error}")

    products = request_products_list(token,
                                     start=start,
                                     end=end,
                                     collection="EO:EUM:DAT:MSG:MSG15-RSS")
    threads = []
    LOG.info(f"Found {len(products)} products")
    size_chunks = len(products) // 10
    LOG.info(f"Dividing in chunks of size {size_chunks}")
    for i in chunks([p for p in products], size_chunks):
        thread = DownloaderThread(i, DEST_FOLDER, CONSUMER_KEY,
                                  CONSUMER_SECRET)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    
    LOG.info(f"Downloading products from {start} to {end}...")


if __name__ == '__main__':
    main()
