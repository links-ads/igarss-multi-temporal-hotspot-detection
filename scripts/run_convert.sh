
eval "$(/nfs/home/barco/miniconda3/bin/conda shell.bash hook)"
conda activate epct-2.5
python scripts/convert_native_with_datatailor.py --event_file /nfs/home/barco/projects/igarss-multi-temporal-hotspot-detection/data/hotspot/effis_burned_2012_2023_mapped_countries_geos.csv --input_folder /nfs/home/barco/projects/igarss-multi-temporal-hotspot-detection/data/hotspot/meteosat_satip --output_folder /nfs/home/barco/projects/igarss-multi-temporal-hotspot-detection/data/hotspot/meteosat_datatailor