# Greenness segregation shapes mental health racial inequality in the U.S.

## Introduction

This repo is the source code for paper: [Greenness segregation shapes mental health racial inequality in the U.S.]().


## Folder Structure
```none
├── MentalHealthInequity
│   ├── Data
│   │   ├── ACS
│   │   │   ├── CensusTract
│   │   │   │   ├── DP02-CT
│   │   │   │   │     ├── ACSDP5Y2019.DP02-Data.csv
│   │   │   │   ├── DP03-CT
│   │   │   │   ├── DP05-CT
│   │   │   ├── County
│   │   │   │   ├── DP02-County
│   │   │   │   ├── DP03-County
│   │   │   │   ├── DP05-County
│   │   ├── Boundary
│   │   │   ├── cb_2019_us_bg_500k
│   │   │   ├── cb_2019_us_county_5m
│   │   │   ├── cb_2019_us_tract_500k
│   │   │   ├── cb_2019_us_nation_5m
│   │   ├── PLACES (Please put the PLACES data here)
│   │   ├── Trust_for_Public_Land
│   │   │   ├── ParkServe_Shapefiles (Please put the ParkServe data here)
│   │   ├── WorldCover
│   │   │   ├── US (Please put the WorldCover data here)
│   │   ├── WorldPop
│   │   │   ├── usa_ppp_2019.tif (Please put the WorldPop data here)
│   ├── src
│   │   ├── data_download
│   │   ├── preprocess
│   │   ├── spark_for_safegraph
│   │   ├── Fig1_ABC.py
│   │   ├── Fig1_D.py
│   │   ├── Fig2_BC.py
│   │   ├── Fig2_EF.py
│   │   ├── Fig3.py
│   │   ├── Fig4.py
```

## System Requirement

### Installation Guide
Typically, a modern computer with fast internet can complete the installation within 10 mins.

1. Download Anaconda according to [Official Website](https://www.anaconda.com/products/distribution), which can be done by the following command (newer version of anaconda should also work)
``` bash
wget -c https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
```
2. Install Anaconda through the commandline guide. Permit conda init when asked.
``` bash
./Anaconda3-2023.09-0-Linux-x86_64.sh
```
3. Quit current terminal window and open a new one. You should be able to see (base) before your command line. 

4. Use the following command to install python environment
``` bash
conda create -n MentalHealth python=3.11
conda activate MentalHealth
pip install ipython pandas==2.1.3 matplotlib statsmodels plotly geopandas seaborn pathlib shapely rasterio scipy
```

(Optional) If you need to exit the environment for other project, use the following command.

``` bash
conda deactivate 
```

## Prepare the Data

### [Necessary] For reproducing the main results
We provide the necessary data in this [Google Drive](https://drive.google.com/file/d/1zrqdHmX9DTv0ENo27MQpc-sqwhioElLk/view?usp=sharing) link. Please download and put it in the root directory of this project.

### [Optional] Starting from the beginning
#### ACS data
Please download the 2019 ACS 5-year estimate for [DP02](https://data.census.gov/table/ACSDP5Y2019.DP02?q=DP02), [DP03](https://data.census.gov/table/ACSDP5Y2019.DP03?q=DP03), [DP04](https://data.census.gov/table/ACSDP5Y2019.DP04?q=DP04), [DP05](https://data.census.gov/table/ACSDP5Y2019.DP05?q=DP05) in both census tract and county level, uncompress and put the main data file in `Data/ACS` according to the direction of Folder Structure.

#### Boundary data
Please download the cartographic boundary files of census block groups, census tracts, counties and nation in 2019 from [United States Census Bureau](https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.2019.html#list-tab-1883739534).

#### PLACES data
Please download the [PLACES data](https://data.cdc.gov/500-Cities-Places/PLACES-Census-Tract-Data-GIS-Friendly-Format-2021-/mb5y-ytti/about_data) in census tract level. Note that there is a 2-year lag between the release date and the data sampling.

#### ParkServe data
Please download the [ParkServe data](https://www.tpl.org/park-data-downloads) shapefile.

#### WorldCover data
We provide a python script to automatically download the WorldCover data for the WorldCover data.

Run the following code from root directory:
``` bash
python ./src/data_download/download_worldcover_data.py
```

#### WorldPop data
Please download the [WorldPop data](https://hub.worldpop.org/geodata/listing?id=29) for United States in 2019 year.

#### SafeGraph data
The SafeGraph data can be purchased from [Dewey](https://www.deweydata.io/).

## Run the Code
### Preprocess the data
Note: this part is not necessary. We have provided the pre-processed data file described earlier.
``` bash
python ./src/preprocess/0_make_all_ct_data.py
python ./src/preprocess/1_process_park_data_census_tract.py
python ./src/preprocess/2_process_landuse_data_census_tract.py
python ./src/preprocess/3_post_process_dynamic_visit.py  # It requires processed SafeGraph data. We provide pyspark code of generating such processed SafeGraph data in ./src/spark_for_safegraph
```
After performing these steps, you will get the following files. These files are available in the above Google Drive link.

| File Name |
|---|
|`census_tract_data_all_with_park_2019.parquet`|
|`census_tract_data_all_with_park_with_landuse_2019.parquet`|
|`tract_visit_all_US_within_county_2019.parquet`|
|`tract_visit_selected_county_with_google_2019.parquet`|
|`park_visit_all_US_within_county_2019.parquet`| 
|`park_tract_bipart_all_us_within_county_2019.parquet`|

### Figure 1
``` bash
python ./src/Fig1_ABC.py
python ./src/Fig1_D.py
```

### Figure 2
``` bash
python ./src/Fig2_BC.py
python ./src/Fig2_EF.py
```

### Figure 3
``` bash
python ./src/Fig3.py
```

### Figure 4
``` bash
python ./src/Fig4.py
```