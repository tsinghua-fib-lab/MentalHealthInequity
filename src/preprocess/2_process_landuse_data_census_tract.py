import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import geopandas as gpd
from datetime import datetime
from math import ceil, sqrt
from collections import defaultdict
from multiprocessing import Pool
import rasterio
import rasterio.mask
from rasterio.plot import show
import shapely
from scipy.interpolate import RegularGridInterpolator

MULTI_PROCESS_NUM = 60
YEAR = 2019
############################################################################################
# Data Path
SELECTED_CENSUS_TRACT_LOC = Path(f"./census_tract_data_all_with_park_{YEAR}.parquet")
LANDUSE_PATH = Path('./Data/WorldCover/US/')
LANDUSE_GRID_LOC = Path('./Data/WorldCover/esa_worldcover_grid.geojson')
WORLDPOP_LOC = Path(f'./Data/WorldPop/usa_ppp_2019.tif')
OUTPUT_PATH = Path('./world_cover_us_tract')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
############################################################################################
def subprocess_cal_landuse(idx, total_process_num, buffered):
    landuse_grid = gpd.read_file(LANDUSE_GRID_LOC)
    census_tract_df = gpd.read_parquet(SELECTED_CENSUS_TRACT_LOC).set_crs("EPSG:4326",allow_override=True)
    census_tract_df['geometry'] = census_tract_df['geometry'].buffer(0)
    if buffered:
        census_tract_df = census_tract_df.to_crs("EPSG:26918")
        census_tract_df.loc[:, 'buffered_geometry'] = census_tract_df['geometry'].buffer(800).to_crs("EPSG:4326")
        census_tract_df = census_tract_df.to_crs("EPSG:4326")

    batch_num = int(ceil(census_tract_df.shape[0] / total_process_num)) + 1
    census_tract_df = census_tract_df.iloc[idx * batch_num:(idx + 1) * batch_num, :].reset_index(drop=True)
    world_pop_map = rasterio.open(str(WORLDPOP_LOC))
    landuse_origin_dict = {}
    landuse_pop_weighted_dict = {}
    ct_idx = 0
    total_no_worldpop_ct = 0
    for _, row in census_tract_df.iterrows():
        if buffered:
            target_geometry = row['buffered_geometry']
        else:
            target_geometry = row['geometry']
        tiles = landuse_grid.loc[landuse_grid.intersects(target_geometry), 'll_tile'].to_list()

        v_origin = {'10':0, '20':0, '30':0, '40':0, '50':0, '60':0, '70':0, '80':0, '90':0, '95':0, '100':0, 'TotalLandCount':0}
        v_pop_weighted = {'10':0, '20':0, '30':0, '40':0, '50':0, '60':0, '70':0, '80':0, '90':0, '95':0, '100':0, 'TotalPop':0}

        for t in tiles:
            with (rasterio.open(str(LANDUSE_PATH.joinpath(f"ESA_WorldCover_10m_2021_v200_{t}_Map.tif"))) as landuse_map):
                landuse_out_image, _ = rasterio.mask.mask(landuse_map, [target_geometry], filled=False, crop=True)
                landuse_data = landuse_out_image[~landuse_out_image.mask].data
                #world_pop_data = world_pop_out_image[~world_pop_out_image.mask].data

                if landuse_data.shape[0] < 1:
                    continue

                bounding_box = shapely.box(*target_geometry.bounds).intersection(shapely.box(*landuse_map.bounds))
                try:
                    bounding_geometry = target_geometry.intersection(shapely.box(*landuse_map.bounds))
                except:
                    print(row['TractFIPS'])
                    print(target_geometry)
                    print(landuse_map.bounds)
                world_pop_out_image, _ = rasterio.mask.mask(world_pop_map, [bounding_box], filled=False, crop=True)
                world_pop_out_image[world_pop_out_image < 0] = 0
                world_pop_out_image_precise, _ = rasterio.mask.mask(world_pop_map, [bounding_geometry], filled=False, crop=True)
                world_pop_out_image_precise[world_pop_out_image_precise < 0] = 0
                real_pop = world_pop_out_image_precise[~world_pop_out_image_precise.mask].sum()
                # unweighted land use count
                for k in v_origin.keys():
                    if k == 'TotalLandCount':
                        continue
                    v_origin[k] += (landuse_data == int(k)).sum()
                v_origin['TotalLandCount'] += landuse_data.shape[0]
                # interpolation
                try:
                    world_pop_data = world_pop_out_image.data[0, :, :]
                    x = np.linspace(0, 1, world_pop_data.shape[0])
                    y = np.linspace(0, 1, world_pop_data.shape[1])
                    x1 = np.linspace(0, 1, landuse_out_image.shape[1])
                    y1 = np.linspace(0, 1, landuse_out_image.shape[2])
                    x1g, y1g = np.meshgrid(x1, y1, indexing='ij')
                    world_pop_data_interp = RegularGridInterpolator((x, y), world_pop_data)((x1g, y1g))
                    world_pop_data_interp_masked = np.ma.masked_array(np.expand_dims(world_pop_data_interp,0), landuse_out_image.mask)
                    world_pop_data_interp_masked[world_pop_data_interp_masked<0] = 0
                    world_pop_data_interp_masked = world_pop_data_interp_masked / world_pop_data_interp_masked.sum() * real_pop

                    for k in v_pop_weighted.keys():
                        if k == 'TotalPop':
                            continue
                        landuse_mask = (landuse_out_image == int(k)).data
                        world_pop_data_interp_masked_k = world_pop_data_interp_masked * landuse_mask
                        v_pop_weighted[k] += world_pop_data_interp_masked_k.sum()
                    v_pop_weighted['TotalPop'] += real_pop
                except:
                    total_no_worldpop_ct += 1
                    v_pop_weighted['TotalPop'] += landuse_data.shape[0]
                    for k in v_pop_weighted.keys():
                        if k == 'TotalPop':
                            continue
                        v_pop_weighted[k] += (landuse_data == int(k)).sum() / landuse_data.shape[0]

        landuse_origin_dict[row['TractFIPS']] = v_origin
        landuse_pop_weighted_dict[row['TractFIPS']] = v_pop_weighted
        print('Process ' + str(idx) + ': finished ' + str(ct_idx + 1) + ' / ' + str(census_tract_df.shape[0]) + ' tracts. ' + str((ct_idx+1)/census_tract_df.shape[0] * 100) + '%' + ' finished.')
        ct_idx += 1
    return landuse_origin_dict, landuse_pop_weighted_dict

def cal_landuse(buffered=True):
    p = Pool(MULTI_PROCESS_NUM)
    result = []
    for i in range(MULTI_PROCESS_NUM):
        result.append(p.apply_async(subprocess_cal_landuse, args=(i, MULTI_PROCESS_NUM, buffered)))

    landuse_origin_dict = {}
    landuse_pop_weighted_dict = {}
    for i in result:
        tmp_landuse_origin_dict, tmp_landuse_pop_weighted_dict = i.get()
        landuse_origin_dict.update(tmp_landuse_origin_dict)
        landuse_pop_weighted_dict.update(tmp_landuse_pop_weighted_dict)
    p.close()
    p.join()

    return landuse_origin_dict, landuse_pop_weighted_dict

#############################################################################################################
landuse_lut = {'10':'TreeCover', '20':'Shrubland', '30':'Grassland', '40':'Cropland', '50':'Builtup', '60':'Bareland', '70':'SnowIce', '80':'Water', '90':'Wetland', '95':'Mangroves', '100':'MossLichen'}

# 1. buffered
buffered_landuse_origin_dict, buffered_landuse_pop_weighted_dict = cal_landuse(buffered=True)

buffered_landuse_origin_df = pd.DataFrame.from_dict(buffered_landuse_origin_dict, orient='index').reset_index().rename(columns={'index':'TractFIPS'})
buffered_landuse_origin_df = buffered_landuse_origin_df.rename(columns=landuse_lut)
buffered_landuse_origin_df['GreenSpaceBufferedOrigin'] = (buffered_landuse_origin_df['TreeCover'] + buffered_landuse_origin_df['Shrubland'] + buffered_landuse_origin_df['Grassland'] + buffered_landuse_origin_df['Wetland'] + buffered_landuse_origin_df['Mangroves']) / buffered_landuse_origin_df['TotalLandCount']

buffered_landuse_pop_weighted_df = pd.DataFrame.from_dict(buffered_landuse_pop_weighted_dict, orient='index').reset_index().rename(columns={'index':'TractFIPS'})
buffered_landuse_pop_weighted_df = buffered_landuse_pop_weighted_df.rename(columns=landuse_lut)
buffered_landuse_pop_weighted_df['GreenSpaceBufferedPopWeighted'] = (buffered_landuse_pop_weighted_df['TreeCover'] + buffered_landuse_pop_weighted_df['Shrubland'] + buffered_landuse_pop_weighted_df['Grassland'] + buffered_landuse_pop_weighted_df['Wetland'] + buffered_landuse_pop_weighted_df['Mangroves']) / buffered_landuse_pop_weighted_df['TotalPop']

# Save
buffered_landuse_origin_df.to_csv(os.path.join(OUTPUT_PATH, f'us_tract_buffered_landuse_origin.csv'), index=False)
buffered_landuse_pop_weighted_df.to_csv(os.path.join(OUTPUT_PATH, f'us_tract_buffered_landuse_pop_weighted.csv'), index=False)

#############################################################################################################
#2. unbuffered
landuse_origin_dict, landuse_pop_weighted_dict = cal_landuse(buffered=False)

landuse_origin_df = pd.DataFrame.from_dict(landuse_origin_dict, orient='index').reset_index().rename(columns={'index':'TractFIPS'})
landuse_origin_df = landuse_origin_df.rename(columns=landuse_lut)
landuse_origin_df['GreenSpaceOrigin'] = (landuse_origin_df['TreeCover'] + landuse_origin_df['Shrubland'] + landuse_origin_df['Grassland'] + landuse_origin_df['Wetland'] + landuse_origin_df['Mangroves']) / landuse_origin_df['TotalLandCount']

landuse_pop_weighted_df = pd.DataFrame.from_dict(landuse_pop_weighted_dict, orient='index').reset_index().rename(columns={'index':'TractFIPS'})
landuse_pop_weighted_df = landuse_pop_weighted_df.rename(columns=landuse_lut)
landuse_pop_weighted_df['GreenSpacePopWeighted'] = (landuse_pop_weighted_df['TreeCover'] + landuse_pop_weighted_df['Shrubland'] + landuse_pop_weighted_df['Grassland'] + landuse_pop_weighted_df['Wetland'] + landuse_pop_weighted_df['Mangroves']) / landuse_pop_weighted_df['TotalPop']

# Save
landuse_origin_df.to_csv(os.path.join(OUTPUT_PATH, f'us_tract_landuse_origin.csv'), index=False)
landuse_pop_weighted_df.to_csv(os.path.join(OUTPUT_PATH, f'us_tract_landuse_pop_weighted.csv'), index=False)


#############################################################################################################
# Read temp file
buffered_landuse_origin_df = pd.read_csv(os.path.join(OUTPUT_PATH, f'us_tract_buffered_landuse_origin.csv'),dtype={'TractFIPS':str})
buffered_landuse_pop_weighted_df = pd.read_csv(os.path.join(OUTPUT_PATH, f'us_tract_buffered_landuse_pop_weighted.csv'),dtype={'TractFIPS':str})
landuse_origin_df = pd.read_csv(os.path.join(OUTPUT_PATH, f'us_tract_landuse_origin.csv'),dtype={'TractFIPS':str})
landuse_pop_weighted_df = pd.read_csv(os.path.join(OUTPUT_PATH, f'us_tract_landuse_pop_weighted.csv'),dtype={'TractFIPS':str})

# Cal green land cover
greenspace_df = buffered_landuse_origin_df[['TractFIPS','GreenSpaceBufferedOrigin']].merge(buffered_landuse_pop_weighted_df[['TractFIPS','GreenSpaceBufferedPopWeighted']]).merge(landuse_origin_df[['TractFIPS','GreenSpaceOrigin']]).merge(landuse_pop_weighted_df[['TractFIPS','GreenSpacePopWeighted']])
greenspace_df.to_csv(os.path.join(OUTPUT_PATH, f'us_tract_green_space.csv'), index=False)


# Merge & Save
census_tract_data_selected = gpd.read_parquet(SELECTED_CENSUS_TRACT_LOC)
census_tract_data_selected = census_tract_data_selected.set_crs("EPSG:4326",allow_override=True)
census_tract_data_selected['geometry'] = census_tract_data_selected['geometry'].buffer(0)
census_tract_data_selected = census_tract_data_selected.merge(greenspace_df, on='TractFIPS', how='left')

for col in census_tract_data_selected.columns:
    if col.startswith('DP'):
        census_tract_data_selected[col] = census_tract_data_selected[col].astype(float)/100
    if col.endswith('CrudePrev'):
        census_tract_data_selected[col] = census_tract_data_selected[col].astype(float)/100

census_tract_data_selected.to_parquet(f'./census_tract_data_all_with_park_with_landuse_{YEAR}.parquet')
