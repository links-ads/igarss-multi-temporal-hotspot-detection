import argparse
import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import eumdac
import eumdac.local_tailor
import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    filename=f'logs/download_meteosat_{datetime.now()}.log',
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

    parser.add_argument(
        '--collection',
        type=str,
        help=
        'Collection to search [EO:EUM:DAT:MSG:MSG15-RSS, EO:EUM:DAT:MSG:RSS-CLM, EO:EUM:DAT:MSG:HRSEVIRI, EO:EUM:DAT:MSG:CLM]',
        default="EO:EUM:DAT:MSG:MSG15-RSS",
        choices=[
            "EO:EUM:DAT:MSG:MSG15-RSS", "EO:EUM:DAT:MSG:RSS-CLM",
            "EO:EUM:DAT:MSG:HRSEVIRI", "EO:EUM:DAT:MSG:CLM"
        ])

    parser.add_argument(
        '--chain_product_family',
        type=str,
        help='Product family for the chain [HRSEVIRI, MSGCLMK]',
        default="HRSEVIRI",
        choices=["HRSEVIRI", "MSGCLMK"])

    parser.add_argument('--threads',
                        type=int,
                        help='Number of threads to use',
                        default=1)

    parser.add_argument('--output_file_format',
                        type=str,
                        help='Output file format ["tif", "png", "netcdf4"]',
                        default="netcdf4")

    parser.add_argument('--datatailor_data_folder',
                        type=str,
                        help='Data folder where datatailor stores data \
                    (see file ~/miniconda3/envs/epct-2.5/etc/epct/epct.yaml)',
                        default="/nfs/home/barco/epct/datatailor_results")

    parser.add_argument("--local_tailor",
                        help="Name of local tailor instance",
                        type=str,
                        default="dt_3_2")

    return parser.parse_args()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_catalog(file: Path):
    df = gpd.read_file(file,
                       GEOM_POSSIBLE_NAMES="geometry",
                       KEEP_GEOM_COLUMNS="NO")
    df['initialdate'] = pd.to_datetime(df['initialdate'], format='ISO8601')
    df['finaldate'] = pd.to_datetime(df['finaldate'], format='ISO8601')
    return df[["id", "geometry", "initialdate", "finaldate", "country"]]


def search_meteosat_images(initial_date: datetime, final_date: datetime,
                           collection: str, key: str, secret: str):
    token = eumdac.AccessToken((key, secret))
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(collection)

    products = selected_collection.search(dtstart=initial_date,
                                          dtend=final_date)

    return [p for p in products]


def download_meteosat_product(event_id, product, chain_product_family: str,
                              chain_name: str, chain_roi: str,
                              output_file_format: str, output_folder: Path,
                              local_data_tailor_instance: str):

    output_folder.mkdir(parents=True, exist_ok=True)
    # urls = URLs(inifile="local_endpoints.ini")
    # t = AnonymousAccessToken(urls=urls)
    # local_datatailor = eumdac.DataTailor(t)
    local_datatailor = eumdac.local_tailor.get_local_tailor(
        local_data_tailor_instance)
    chain = eumdac.tailor_models.Chain(
        product=chain_product_family,
        name=chain_name,
        format=output_file_format,
        projection='geographic',
        roi=chain_roi,
    )
    customisation = local_datatailor.new_customisation(product, chain)
    status = customisation.status
    sleep_time = 10  # seconds
    renew_token = False
    success = False
    # Customisation Loop
    while status:
        # Get the status of the ongoing customisation
        status = customisation.status
        if "DONE" in status:
            logging.info(
                f"Event {event_id} - Customisation {customisation._id} is successfully completed."
            )
            success = True
            break
        elif status in ["ERROR", "FAILED", "DELETED", "KILLED", "INACTIVE"]:
            logging.info(
                f"Event {event_id} - Customisation {customisation._id} was unsuccessful.\
                  Customisation log is printed.\n")
            logging.info(customisation.logfile)
            if "token" in customisation.logfile:
                # potrebbe essere scaduto il token
                logging.info(
                    f"Event {event_id} - Customisation {customisation._id} - Token expired. Renewing..."
                )
                # TODO: renew token and retry
                renew_token = True
            else:
                break
        elif "QUEUED" in status:
            logging.info(
                f"Event {event_id} - Customisation {customisation._id} is queued."
            )
        elif "RUNNING" in status:
            logging.info(
                f"Event {event_id} - Customisation {customisation._id} is running."
            )
        time.sleep(sleep_time)

    logging.info(
        f"Event {event_id} - Dowloading the tif output of the customisation {customisation._id}"
    )
    try:
        with customisation.stream_output(customisation.outputs[0], ) as stream:
            splits = customisation.outputs[0].split('.')[0].split("_")[0:3]
            ext = customisation.outputs[0].split('.')[-1]
            name = "_".join(splits)
            out_file = output_folder / f"{name}.{ext}"
            with open(out_file, mode='wb') as fdst:
                shutil.copyfileobj(stream, fdst)
    except eumdac.datatailor.DataTailorError as error:
        logging.error(f"Event {event_id} - Data Tailor Error: {error}")
        success = False
    except requests.exceptions.RequestException as error:
        logging.error(f"Event {event_id} - Unexpected error: {error}")
        success = False
    try:
        customisation.delete()
    except eumdac.datatailor.CustomisationError as exc:
        logging.error("Event {event_id} - Customisation Error:", exc)
        success = False
    except requests.exceptions.RequestException as error:
        logging.error("Event {event_id} - Unexpected error:", error)
        success = False

    return success, renew_token


def is_product_already_downloaded(product, output_folder: Path):
    d = str(product).split(".")[0].split("-")[-1]
    date = datetime.strptime(d, "%Y%m%d%H%M%S")
    filter_string = date.isoformat().replace("-", "").replace(":", "")
    files = [s for s in output_folder.glob(f"*{filter_string}*")]
    if len(files) > 0:
        return True
    else:
        return False


def downloader_thread(idx: int, event_list: pd.DataFrame, time_offset: int,
                      collection: str, chain_product_family: str, key: str,
                      secret: str, output_file_format: str,
                      output_folder: Path, local_data_tailor_instance: str):

    for _, event in tqdm(event_list.iterrows(),
                         desc=f'Thread {idx}',
                         total=len(event_list)):

        initial_date = event["initialdate"] - pd.Timedelta(days=time_offset)
        final_date = event["finaldate"] + pd.Timedelta(days=time_offset)
        products = search_meteosat_images(initial_date=initial_date,
                                          final_date=final_date,
                                          collection=collection,
                                          key=key,
                                          secret=secret)

        downloaded_counter = 0
        for p in products:
            if is_product_already_downloaded(
                    p, output_folder / event["id"] / chain_product_family /
                    collection):
                logging.info(f'Product {p} already downloaded')
                continue
            else:
                wait_for_download = True
                while wait_for_download:
                    success, renew_token = download_meteosat_product(
                        event_id=event["id"],
                        product=p,
                        chain_product_family=chain_product_family,
                        chain_name=
                        f'[{event["id"]}] - {p} to {output_file_format}',
                        chain_roi=event["country"],
                        output_file_format=output_file_format,
                        output_folder=output_folder / event["id"] /
                        chain_product_family / collection,
                        local_data_tailor_instance=local_data_tailor_instance)
                    if not success and renew_token:
                        products = search_meteosat_images(
                            initial_date=initial_date,
                            final_date=final_date,
                            collection=collection,
                            key=key,
                            secret=secret)
                        products = products[downloaded_counter:]
                    else:
                        wait_for_download = False
            if success:
                downloaded_counter += 1


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
                                   args.collection, args.chain_product_family,
                                   args.key, args.secret,
                                   args.output_file_format, DEST_FOLDER,
                                   args.local_tailor)))
        threads[-1].start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    args = parse_args()
    main(args)
