import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import geopandas as gpd
from datetime import datetime
from haversine import haversine
from math import ceil, sqrt
from collections import defaultdict
from multiprocessing import Pool
YEAR=2019
MULTI_PROCESS_NUM = 200

############################################################################################
# Data Path
SELECTED_CENSUS_TRACT_LOC = Path(f'./census_tract_data_all_{YEAR}.parquet')
PARK_SERVE_PRE_LOC = Path('./Data/Trust_for_Public_Land/ParkServe_Shapefiles/ParkServe_Parks.shp')
COUNTY_BOUNDARY_LOC = Path('./Data/Boundary/cb_2019_us_county_5m/cb_2019_us_county_5m.shp')
############################################################################################
# Park Data
park_serve = gpd.read_file(PARK_SERVE_PRE_LOC,dtype={'Park_Cou_1':'str','Park_Zip':'str','ParkID':'str'})[['ParkID','Park_Name','Park_Desig','Park_Acces','Park_Statu','Park_Cou_1','Park_Zip','Shape_Leng', 'Shape_Area','geometry']]
park_serve = park_serve.loc[(park_serve['Park_Acces'].isin([3,4])) & (park_serve['Park_Statu'].isin(['Open','OpenFee','OpenRestrict']))]
park_serve = park_serve.to_crs("EPSG:26918") 
############################################################################################
# Read Census Tract Data
census_tract_data_selected = gpd.read_parquet(SELECTED_CENSUS_TRACT_LOC)
census_tract_data_selected = census_tract_data_selected.set_crs("EPSG:4326",allow_override=True).to_crs("EPSG:26918")
census_tract_data_selected['geometry'] = census_tract_data_selected['geometry'].buffer(0)
census_tract_data_selected.loc[:,'buffered_geometry'] = census_tract_data_selected['geometry'].buffer(800)
############################################################################################
# Read County Boundary
county_boundary = gpd.read_file(COUNTY_BOUNDARY_LOC,dtype={'NAME':'str'})[[ 'STATEFP', 'COUNTYFP','NAME','geometry']].set_crs("EPSG:4326",allow_override=True).to_crs("EPSG:26918")

county_boundary.loc[:,'CountyFIPS'] = county_boundary['STATEFP'] + county_boundary['COUNTYFP']
county_boundary['centroid'] = county_boundary['geometry'].centroid

# 1. Select county
county_list = census_tract_data_selected['CountyFIPS'].drop_duplicates().to_list()
park_serve_selected = park_serve.loc[park_serve['Park_Cou_1'].isin(county_list)]
park_serve_selected['geometry'] = park_serve_selected['geometry'].buffer(0)  # fix self-intersection

# 2. Center of mess
census_tract_data_selected.loc[:,'centroid'] = census_tract_data_selected['geometry'].centroid
park_serve_selected.loc[:,'centroid'] = park_serve_selected['geometry'].centroid
census_tract_data_selected = census_tract_data_selected.dropna(subset=['centroid'])

park_serve_selected = park_serve_selected.sort_values(by='Park_Cou_1', ascending=False).reset_index(drop=True)

# 3. Multiprocess to calculate census tract intersection with park
def subprocess_cal_park_size(idx, total_process_num, park_df, census_tract_df, county_boundary, buffered):
    batch_num = int(ceil(park_df.shape[0] / total_process_num)) + 1
    park_df = park_df.iloc[idx * batch_num:(idx + 1) * batch_num, :].reset_index(drop=True)
    park_county_list = park_df['Park_Cou_1'].drop_duplicates().to_list()
    nearby_county_list = []
    for county in park_county_list:
        tmp = county_boundary.loc[county_boundary['CountyFIPS'] == county]
        if tmp.shape[0] == 0:
            continue
        else:
            target_county_centroid = tmp['centroid'].values[0]
            dist = county_boundary['centroid'].map(lambda x: sqrt((target_county_centroid.y - x.y)**2 + (target_county_centroid.x - x.x)**2))
            nearest_n_idx = dist.sort_values().index[:10]
            nearby_county_list += county_boundary.loc[nearest_n_idx, 'CountyFIPS'].to_list()

    total_county_list = list(set(park_county_list + nearby_county_list))
    census_tract_df_select = census_tract_df.loc[census_tract_df['CountyFIPS'].isin(total_county_list), :]

    if buffered:
        target_geometry = 'buffered_geometry'
    else:
        target_geometry = 'geometry'
    tmp_green_area = defaultdict(float)
    pk_fin_idx = 0
    for _, pk_row in park_df.iterrows():
        dist = census_tract_df_select['centroid'].map(lambda x: sqrt((pk_row['centroid'].y - x.y)**2 + (pk_row['centroid'].x - x.x)**2))
        nearest_n_idx = dist.sort_values().index[:20]
        nearest_n_ct = census_tract_df_select.loc[nearest_n_idx, :]
        for _, ct_row in nearest_n_ct.iterrows():
            tmp_green_area[ct_row['TractFIPS']] += ct_row[target_geometry].intersection(pk_row['geometry']).area
        pk_fin_idx += 1
        print('Process ' + str(idx) + ': finished ' + str(pk_fin_idx) + ' / ' + str(park_df.shape[0]) + ' parks. ' + str(pk_fin_idx/park_df.shape[0] * 100) + '%' + ' finished.')
    return tmp_green_area

def cal_park_size(park_df, census_tract_df, county_boundary, buffered=True):
    p = Pool(MULTI_PROCESS_NUM)
    result = []
    for i in range(MULTI_PROCESS_NUM):
        result.append(p.apply_async(subprocess_cal_park_size, args=(i, MULTI_PROCESS_NUM, park_df, census_tract_df, county_boundary, buffered)))

    tract_green_area = defaultdict(float)
    for i in result:
        tmp_dict = i.get()
        for k, v in tmp_dict.items():
            tract_green_area[k] += v
    p.close()
    p.join()

    tract_fips = list(tract_green_area.keys())
    green_area = list(tract_green_area.values())
    return tract_fips, green_area


buffered_tract_fips, buffered_green_area = cal_park_size(park_serve_selected, census_tract_data_selected,county_boundary, buffered=True)
buffered_green_space_df = pd.DataFrame({'TractFIPS':buffered_tract_fips, 'BufferedParkSize':buffered_green_area})
tract_fips, green_area = cal_park_size(park_serve_selected, census_tract_data_selected, county_boundary, buffered=False)
green_space_df = pd.DataFrame({'TractFIPS':tract_fips, 'ParkSize':green_area})

# Merge & Save
census_tract_data_selected = census_tract_data_selected.merge(buffered_green_space_df, on='TractFIPS', how='left')
census_tract_data_selected = census_tract_data_selected.merge(green_space_df, on='TractFIPS', how='left')
census_tract_data_selected['ParkSize'] = census_tract_data_selected['ParkSize'].fillna(0)
census_tract_data_selected['ParkPercentage'] = census_tract_data_selected['ParkSize'] / census_tract_data_selected['geometry'].area
census_tract_data_selected['BufferedParkSize'] = census_tract_data_selected['BufferedParkSize'].fillna(0)
census_tract_data_selected['BufferedParkPercentage'] = census_tract_data_selected['BufferedParkSize'] / census_tract_data_selected['geometry'].area

output = census_tract_data_selected.drop(['buffered_geometry', 'centroid'], axis=1).to_crs("EPSG:4326")
output.to_parquet(f'census_tract_data_all_with_park_{YEAR}.parquet')
