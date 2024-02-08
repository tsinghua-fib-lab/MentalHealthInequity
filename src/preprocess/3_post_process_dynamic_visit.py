import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
from pathlib import Path
import pandas as pd
import geopandas as gpd
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import shapely
from multiprocessing import Pool
from math import ceil
import copy
sns.set_theme(style="white", palette=None)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13
YEAR=2019
MULTI_PROCESS_NUM = 60
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas==2.1.4

############################################################################################
# Data Path
CENSUS_TRACT_DATA_LOC = Path(f'./census_tract_data_all_with_park_with_landuse_{YEAR}.parquet')
CBG_BOUNDARY_LOC = Path('./Data/Boundary/cb_2019_us_bg_500k/cb_2019_us_bg_500k.shp')
CBG_CENSUS_LOC = Path(f'./Data/SafeGraph/cbg_census.csv')
PARK_VISIT_LOC = Path(f'./Data/SafeGraph/ALL_US_park_{YEAR}.csv')
CBG_VISIT_LOC = Path(f'./Data/SafeGraph/ALL_US_cbg_{YEAR}.csv')
PARK_CBG_BIPART_LOC = Path(f'./Data/SafeGraph/ALL_US_park_cbg_bipart_{YEAR}.csv')
HOME_PANEL_SUMMARY_LOC = Path('./Data/SafeGraph/monthly_home_panel_summary_merge_2018_2020.csv')
############################################################################################

# PART 1
# Read data
census_tract_data = gpd.read_parquet(CENSUS_TRACT_DATA_LOC).to_crs("EPSG:26918")

park_visit = (pd.read_csv(PARK_VISIT_LOC,dtype={'poi_cbg_new':'str'},).rename({'poi_cbg_new':'CBGFIPS'},axis=1))
park_visit['TractFIPS'] = park_visit['CBGFIPS'].str[:11]
park_visit = park_visit.drop(['CBGFIPS'],axis=1)
park_visit = park_visit.dropna()
park_visit['geometry'] = park_visit['geometry'].map(lambda x: shapely.wkt.loads(x))
park_visit = gpd.GeoDataFrame(park_visit).set_geometry('geometry').set_crs('EPSG:4326').to_crs("EPSG:26918")


cbg_visit = pd.read_csv(CBG_VISIT_LOC,dtype={'cbg':'str'}).rename({'cbg':'CBGFIPS'},axis=1)
cbg_visit['TractFIPS'] = cbg_visit['CBGFIPS'].str[:11]
cbg_visit = cbg_visit.drop(['CBGFIPS'],axis=1)
tract_visit = cbg_visit[['TractFIPS','visitor_count','park_num']].groupby('TractFIPS').sum().reset_index()
del cbg_visit

park_cbg_bipart = pd.read_csv(PARK_CBG_BIPART_LOC,dtype={'cbg':'str'}).rename({'cbg':'CBGFIPS'},axis=1)
park_cbg_bipart['TractFIPS'] = park_cbg_bipart['CBGFIPS'].str[:11]
park_tract_bipart = park_cbg_bipart[['placekey','TractFIPS','visitor_count']].groupby(['placekey','TractFIPS']).sum().reset_index()
del park_cbg_bipart

home_panel = pd.read_csv(HOME_PANEL_SUMMARY_LOC,dtype={'census_block_group':'str'}).rename({'census_block_group':'CBGFIPS'},axis=1)
home_panel['TractFIPS'] = home_panel['CBGFIPS'].str[:11]
if YEAR >= 2020:
    home_panel = home_panel.loc[(home_panel['year'] == 2020)]
else:
    home_panel = home_panel.loc[(home_panel['year'] == YEAR)]
home_panel = home_panel[['TractFIPS', 'number_devices_residing']].groupby('TractFIPS').mean().reset_index()


############################################################################################
# Cal RoG between Tract-Park
census_tract_essential = census_tract_data[['TractFIPS','geometry','TotalPopulation']]
tract_visit = census_tract_essential.merge(tract_visit, on='TractFIPS',how='inner').drop(['visitor_count'],axis=1)
park_tract_bipart = tract_visit.merge(park_tract_bipart,on='TractFIPS',how='inner').rename({'geometry':'TractGeometry'},axis=1).set_geometry('TractGeometry').merge(park_visit[['placekey','geometry']],on='placekey',how='inner').rename({'geometry':'ParkGeometry'},axis=1).set_geometry('ParkGeometry')
park_tract_bipart['TractCenter'] = park_tract_bipart['TractGeometry'].centroid
park_tract_bipart['ParkCenter'] = park_tract_bipart['ParkGeometry'].centroid
park_tract_bipart['Distance'] = park_tract_bipart.apply(lambda x: x['TractCenter'].distance(x['ParkCenter']),axis=1)
park_tract_bipart = park_tract_bipart.merge(home_panel,on='TractFIPS',how='inner')
park_tract_bipart['visitor_count'] = park_tract_bipart['visitor_count'] / (park_tract_bipart['number_devices_residing'] / park_tract_bipart['TotalPopulation'])
park_tract_bipart = park_tract_bipart.merge(park_visit[['placekey','TractFIPS']].rename({'TractFIPS':'ParkTractFIPS'},axis=1), on='placekey',how='inner')
park_tract_bipart['same_county'] = park_tract_bipart.apply(lambda x: 1 if x['TractFIPS'][:5]==x['ParkTractFIPS'][:5] else 0, axis=1,result_type='reduce')
park_tract_bipart = park_tract_bipart.loc[park_tract_bipart['same_county']==1]
park_tract_bipart = park_tract_bipart.drop(['same_county','ParkTractFIPS'],axis=1)

def subprocess_cal_park_radius_of_gyration(idx, total_process_num, park_tract_bipart):
    print(f'Process: {idx} start')
    begin_time = datetime.now()
    park_id = []
    park_radius_of_gyration = []
    for park in park_tract_bipart['placekey'].unique():
        tmp = park_tract_bipart.loc[park_tract_bipart['placekey'] == park]
        if tmp['visitor_count'].sum() <= 0:
            radius_of_gyration = -1
        else:
            rog = (tmp['Distance'] ** 2 * tmp['visitor_count']).sum() / tmp['visitor_count'].sum()
            if rog <= 0:
                radius_of_gyration = -1
            else:
                radius_of_gyration = np.sqrt(rog)
        park_id.append(park)
        park_radius_of_gyration.append(radius_of_gyration)
    end_time = datetime.now()
    print(f'Process: {idx} end time cost: ', (end_time - begin_time).total_seconds() / 60, ' min')
    return park_id, park_radius_of_gyration

def cal_park_radius_of_gyration(df):
    df = df.copy(deep=True)
    p = Pool(MULTI_PROCESS_NUM)
    result = []
    park_list = df['placekey'].unique().tolist()
    for i in range(MULTI_PROCESS_NUM):
        batch_num = int(ceil(len(park_list) / MULTI_PROCESS_NUM)) + 1
        park_list_tmp = copy.deepcopy(park_list[i * batch_num:(i + 1) * batch_num])
        park_tract_bipart_tmp = copy.deepcopy(df.loc[df['placekey'].isin(park_list_tmp)])


        result.append(p.apply_async(subprocess_cal_park_radius_of_gyration, args=(i, MULTI_PROCESS_NUM, park_tract_bipart_tmp)))

    park_id_list = []
    park_radius_of_gyration_list = []
    for i in result:
        tmp_park_id, tmp_park_rog = i.get()
        park_id_list += tmp_park_id
        park_radius_of_gyration_list += tmp_park_rog
    p.close()
    p.join()

    park_radius_of_gyration = pd.DataFrame({'placekey': park_id_list, 'radius_of_gyration': park_radius_of_gyration_list})
    park_radius_of_gyration.to_csv(f'./park_radius_of_gyration_all_us_within_county_{YEAR}.csv', index=False)



def subprocess_cal_tract_radius_of_gyration(idx, total_process_num, park_tract_bipart):
    print(f'Process: {idx} start')
    begin_time = datetime.now()
    tract_id = []
    tract_radius_of_gyration = []
    for tract in park_tract_bipart['TractFIPS'].unique():
        tmp = park_tract_bipart.loc[park_tract_bipart['TractFIPS'] == tract]
        if tmp['visitor_count'].sum() <= 0:
            radius_of_gyration = -1
        else:
            rog = (tmp['Distance'] ** 2 * tmp['visitor_count']).sum() / tmp['visitor_count'].sum()
            if rog <= 0:
                radius_of_gyration = -1
            else:
                radius_of_gyration = np.sqrt(rog)
        tract_id.append(tract)
        tract_radius_of_gyration.append(radius_of_gyration)
    end_time = datetime.now()
    print(f'Process: {idx} end time cost: ', (end_time - begin_time).total_seconds() / 60, ' min')
    return tract_id, tract_radius_of_gyration

def cal_tract_radius_of_gyration(df):
    df = df.copy(deep=True)
    p = Pool(MULTI_PROCESS_NUM)
    result = []
    tract_list = df['TractFIPS'].unique().tolist()
    for i in range(MULTI_PROCESS_NUM):
        batch_num = int(ceil(len(tract_list) / MULTI_PROCESS_NUM)) + 1
        tract_list_tmp = copy.deepcopy(tract_list[i * batch_num:(i + 1) * batch_num])
        park_tract_bipart_tmp = copy.deepcopy(df.loc[df['TractFIPS'].isin(tract_list_tmp)])
        result.append(p.apply_async(subprocess_cal_tract_radius_of_gyration, args=(i, MULTI_PROCESS_NUM, park_tract_bipart_tmp)))

    tract_id_list = []
    tract_radius_of_gyration_list = []
    for i in result:
        tmp_tract_id, tmp_tract_rog = i.get()
        tract_id_list += tmp_tract_id
        tract_radius_of_gyration_list += tmp_tract_rog
    p.close()
    p.join()

    tract_radius_of_gyration = pd.DataFrame({'TractFIPS': tract_id_list, 'radius_of_gyration': tract_radius_of_gyration_list})
    tract_radius_of_gyration.to_csv(f'./tract_radius_of_gyration_all_us_within_county_{YEAR}.csv', index=False)



begin_time = datetime.now()
cal_park_radius_of_gyration(park_tract_bipart)
end_time = datetime.now()
print('park radius of gyration time cost: ', (end_time - begin_time).total_seconds() / 60, ' min')

begin_time = datetime.now()
cal_tract_radius_of_gyration(park_tract_bipart)
end_time = datetime.now()
print('tract radius of gyration time cost: ', (end_time - begin_time).total_seconds() / 60, ' min')


############################################################################################
# Read Temp File
census_tract_data = gpd.read_parquet(CENSUS_TRACT_DATA_LOC)
park_radius_of_gyration = pd.read_csv(f'./park_radius_of_gyration_all_us_within_county_{YEAR}.csv', dtype={'placekey': 'str'})
tract_radius_of_gyration = pd.read_csv(f'./tract_radius_of_gyration_all_us_within_county_{YEAR}.csv', dtype={'TractFIPS': 'str'}).drop_duplicates()
park_tract_bipart = pd.read_parquet(f'./tmp_park_tract_bipart_all_us_within_county_{YEAR}.parquet')
park_visit = pd.read_parquet(f'./tmp_park_visit_all_us_within_county_{YEAR}.parquet')
tract_visit = pd.read_parquet(f'./tmp_tract_visit_all_us_within_county_{YEAR}.parquet')

############################################
percent_num = 4
percent_lut = {}
for i in range(percent_num):
    percent_lut[i] = str(int(100 / percent_num * i)) + '-' + str(int(100 / percent_num * (i + 1))) + '%'

census_tract_data['black_percentile'] = census_tract_data['DP05_0065PE'].map(lambda x: percent_lut[int(x / (1 / percent_num))] if x < 1 else percent_lut[percent_num - 1])
census_tract_data['black_percentile_no'] = census_tract_data['DP05_0065PE'].map(lambda x: int(x / (1 / percent_num)) if x < 1 else percent_num - 1)

park_tract_bipart = park_tract_bipart.merge(census_tract_data[['TractFIPS','black_percentile_no','DP05_0064PE', 'DP05_0065PE']].drop_duplicates(),on='TractFIPS',how='inner')
park_tract_bipart = park_tract_bipart.merge(park_visit[['placekey','TractFIPS']].rename(columns={'TractFIPS':'ParkTractFIPS'}), on='placekey',how='inner')
#########################################################################################################################
park_visit_by_black_percentile = park_tract_bipart[['placekey','black_percentile_no','TractFIPS','visitor_count']].groupby(['placekey','black_percentile_no']).sum().reset_index()
tmp_black_percentile = park_visit_by_black_percentile.pivot(index='placekey',columns='black_percentile_no',values='visitor_count').fillna(0).reset_index()

park_county = park_visit[['placekey','TractFIPS']].drop_duplicates()
park_county['CountyFIPS'] = park_county['TractFIPS'].str[:5]
tmp_black_percentile = tmp_black_percentile.merge(park_county[['placekey','CountyFIPS']],on='placekey',how='inner')

county_race_pop_dist = census_tract_data[['CountyFIPS','TotalPopulation','black_percentile_no']].groupby(['CountyFIPS','black_percentile_no']).sum().reset_index()
county_race_pop_dist = county_race_pop_dist.pivot(index='CountyFIPS',columns='black_percentile_no',values='TotalPopulation').fillna(0).reset_index()
county_race_pop_dist['CountyPop'] = county_race_pop_dist[0]
for i in range(percent_num-1):
    county_race_pop_dist['CountyPop'] += county_race_pop_dist[i+1]
for i in range(percent_num):
    county_race_pop_dist[str(i)+'_pop_dist'] = county_race_pop_dist[i] / county_race_pop_dist['CountyPop']
    county_race_pop_dist.drop([i],axis=1,inplace=True)

tmp_black_percentile['TotalVisit'] = tmp_black_percentile[0]
for i in range(percent_num - 1):
    tmp_black_percentile['TotalVisit'] += tmp_black_percentile[i + 1]
for i in range(percent_num):
    tmp_black_percentile[i] = tmp_black_percentile[i] / tmp_black_percentile['TotalVisit']

tmp_black_percentile = tmp_black_percentile.merge(county_race_pop_dist,on='CountyFIPS',how='inner')

def cal_js(x):
    visit = [x[i] for i in range(percent_num)]
    pop = [x[str(i)+'_pop_dist'] for i in range(percent_num)]
    return jensenshannon(visit,pop)

tmp_black_percentile['SegregationIndex'] = tmp_black_percentile.apply(cal_js,axis=1)
tmp_black_percentile = tmp_black_percentile[['placekey','SegregationIndex','CountyFIPS']]

dynamic_segregation_index = tmp_black_percentile.groupby(['CountyFIPS']).apply(lambda x: x.sort_values(by='SegregationIndex', ascending=False)['SegregationIndex'].quantile(0.75)).reset_index().rename({0:'HighSegCriteria'},axis=1)
tmp_black_percentile = tmp_black_percentile.merge(dynamic_segregation_index[['CountyFIPS','HighSegCriteria']], on=['CountyFIPS'], how='inner')

dynamic_segregation_index = tmp_black_percentile.groupby(['CountyFIPS']).apply(lambda x: x.sort_values(by='SegregationIndex', ascending=False)['SegregationIndex'].quantile(0.25)).reset_index().rename({0:'LowSegCriteria'},axis=1)
tmp_black_percentile = tmp_black_percentile.merge(dynamic_segregation_index[['CountyFIPS','LowSegCriteria']], on=['CountyFIPS'], how='inner')

def find_strong_segregation_park(x):
    if (x['SegregationIndex'] > x['HighSegCriteria']) and (x['SegregationIndex'] > 0):
        return True
    else:
        return False
tmp_black_percentile['IsStrongSegregationPark'] = tmp_black_percentile.apply(find_strong_segregation_park, axis=1)

def find_weak_segregation_park(x):
    if (x['SegregationIndex'] < x['LowSegCriteria']) and (x['SegregationIndex'] > 0):
        return True
    else:
        return False
tmp_black_percentile['IsWeakSegregationPark'] = tmp_black_percentile.apply(find_weak_segregation_park, axis=1)


park_tract_bipart = park_tract_bipart.merge(tmp_black_percentile[['placekey','IsStrongSegregationPark','SegregationIndex','IsWeakSegregationPark']],on='placekey',how='inner')
print(park_tract_bipart['TractFIPS'].drop_duplicates().shape[0])
#########################################################################################################################
criteria = 0.5

park_tract_bipart['WhiteMajorTract'] = park_tract_bipart['DP05_0064PE'].map(lambda x: 1 if x >= criteria else 0)
tmp_white_major_park = park_tract_bipart[['placekey','WhiteMajorTract','TractFIPS','visitor_count','TotalPopulation']].groupby(['placekey','WhiteMajorTract']).sum().reset_index()
tmp_white_major_park = tmp_white_major_park.pivot(index='placekey',columns='WhiteMajorTract',values='visitor_count').fillna(0).reset_index()

county_white_major_pop_dist = census_tract_data[['CountyFIPS','TotalPopulation','DP05_0065PE']]
county_white_major_pop_dist['WhiteMajorTract'] = county_white_major_pop_dist['DP05_0065PE'].map(lambda x: 1 if x < criteria else 0)
county_white_major_pop_dist = county_white_major_pop_dist.groupby(['CountyFIPS','WhiteMajorTract']).sum().reset_index()
county_white_major_pop_dist = county_white_major_pop_dist.pivot(index='CountyFIPS',columns='WhiteMajorTract',values='TotalPopulation').fillna(0).reset_index()
county_white_major_pop_dist['CountyPop'] = county_white_major_pop_dist[0] + county_white_major_pop_dist[1]
county_white_major_pop_dist['0_pop_dist'] = county_white_major_pop_dist[0] / county_white_major_pop_dist['CountyPop']
county_white_major_pop_dist['1_pop_dist'] = county_white_major_pop_dist[1] / county_white_major_pop_dist['CountyPop']
county_white_major_pop_dist.drop([0,1], axis=1, inplace=True)

tmp_white_major_park = tmp_white_major_park.merge(park_county[['placekey','CountyFIPS']],on='placekey',how='inner').merge(county_white_major_pop_dist,on='CountyFIPS',how='inner')
tmp_white_major_park['IsWhiteMajorVisitPark'] = (tmp_white_major_park[1] / tmp_white_major_park['1_pop_dist']) > (tmp_white_major_park[0] / tmp_white_major_park['0_pop_dist'])
park_tract_bipart = park_tract_bipart.merge(tmp_white_major_park[['placekey','IsWhiteMajorVisitPark']],on='placekey',how='inner')

park_tract_bipart['BlackMajorTract'] = park_tract_bipart['DP05_0065PE'].map(lambda x: 1 if x >= criteria else 0)
tmp_black_major_park = park_tract_bipart[['placekey','BlackMajorTract','TractFIPS','visitor_count','TotalPopulation']].groupby(['placekey','BlackMajorTract']).sum().reset_index()
tmp_black_major_park = tmp_black_major_park.pivot(index='placekey',columns='BlackMajorTract',values='visitor_count').fillna(0).reset_index()

county_black_major_pop_dist = census_tract_data[['CountyFIPS','TotalPopulation','DP05_0065PE']]
county_black_major_pop_dist['BlackMajorTract'] = county_black_major_pop_dist['DP05_0065PE'].map(lambda x: 1 if x >= criteria else 0)
county_black_major_pop_dist = county_black_major_pop_dist.groupby(['CountyFIPS','BlackMajorTract']).sum().reset_index()
county_black_major_pop_dist = county_black_major_pop_dist.pivot(index='CountyFIPS',columns='BlackMajorTract',values='TotalPopulation').fillna(0).reset_index()
county_black_major_pop_dist['CountyPop'] = county_black_major_pop_dist[0] + county_black_major_pop_dist[1]
county_black_major_pop_dist['0_pop_dist'] = county_black_major_pop_dist[0] / county_black_major_pop_dist['CountyPop']
county_black_major_pop_dist['1_pop_dist'] = county_black_major_pop_dist[1] / county_black_major_pop_dist['CountyPop']
county_black_major_pop_dist.drop([0,1], axis=1, inplace=True)

tmp_black_major_park = tmp_black_major_park.merge(park_county[['placekey','CountyFIPS']],on='placekey',how='inner').merge(county_black_major_pop_dist,on='CountyFIPS',how='inner')
tmp_black_major_park['IsBlackMajorVisitPark'] = (tmp_black_major_park[1] / tmp_black_major_park['1_pop_dist']) > (tmp_black_major_park[0] / tmp_black_major_park['0_pop_dist'])
tmp_black_major_park['IsBlackExtremeVisitPark'] = tmp_black_major_park[1] > tmp_black_major_park[0]   # 黑人访问量大于白人访问量，显著更强的指标
park_tract_bipart = park_tract_bipart.merge(tmp_black_major_park[['placekey','IsBlackMajorVisitPark', 'IsBlackExtremeVisitPark']],on='placekey',how='inner')
print(park_tract_bipart['TractFIPS'].drop_duplicates().shape[0])
#########################################################################################################################
tmp_black_located_park = census_tract_data[['TractFIPS','DP05_0065PE']].rename(columns={'TractFIPS':'ParkTractFIPS'})
tmp_black_located_park['IsBlackMajorLocatePark'] = tmp_black_located_park['DP05_0065PE'].map(lambda x: x >= criteria)
park_tract_bipart = park_tract_bipart.merge(tmp_black_located_park[['ParkTractFIPS','IsBlackMajorLocatePark']], on='ParkTractFIPS',how='inner')

tmp_white_located_park = census_tract_data[['TractFIPS','DP05_0064PE']].rename(columns={'TractFIPS':'ParkTractFIPS'})
tmp_white_located_park['IsWhiteMajorLocatePark'] = tmp_white_located_park['DP05_0064PE'].map(lambda x: x >= criteria)
park_tract_bipart = park_tract_bipart.merge(tmp_white_located_park[['ParkTractFIPS','IsWhiteMajorLocatePark']], on='ParkTractFIPS',how='inner')
print(park_tract_bipart['TractFIPS'].drop_duplicates().shape[0])
#########################################################################################################################
park_all_visit = park_tract_bipart[['placekey', 'visitor_count']].groupby('placekey').sum().reset_index().rename(
    columns={'visitor_count': 'total_visitor_count'})
park_all_visit = park_all_visit[park_all_visit['total_visitor_count'] != np.inf].reset_index(drop=True)  # 删除inf
park_tract_bipart = park_tract_bipart.merge(park_all_visit, on='placekey', how='inner')
tmp_bipart_segregation = park_tract_bipart[['placekey','visitor_count', 'total_visitor_count', 'TractFIPS', 'DP05_0065PE', 'TotalPopulation']]
tmp_bipart_segregation['CountyFIPS'] = tmp_bipart_segregation['TractFIPS'].str[:5]
#county_pop = tmp_bipart_segregation[['CountyFIPS','TractFIPS', 'TotalPopulation']].drop_duplicates()[['CountyFIPS', 'TotalPopulation']].groupby('CountyFIPS').sum().reset_index().rename({'TotalPopulation': 'CountyPop'}, axis=1)
county_pop = tmp_bipart_segregation[['CountyFIPS','TractFIPS', 'TotalPopulation']][['CountyFIPS', 'TotalPopulation']].groupby('CountyFIPS').sum().reset_index().rename({'TotalPopulation': 'CountyPop'}, axis=1)


tmp_bipart_segregation = tmp_bipart_segregation.merge(county_pop, on='CountyFIPS', how='inner')
tmp_bipart_segregation['visit_weight'] = tmp_bipart_segregation['visitor_count'] / tmp_bipart_segregation['total_visitor_count']
tmp_bipart_segregation['pop_weight'] = tmp_bipart_segregation['TotalPopulation'] / tmp_bipart_segregation['CountyPop']
tmp_bipart_segregation = tmp_bipart_segregation.dropna().reset_index(drop=True)
tmp_bipart_segregation['BipartBlackVisitation'] = tmp_bipart_segregation['visit_weight'] * tmp_bipart_segregation['DP05_0065PE'] * tmp_bipart_segregation['pop_weight']
tmp_bipart_segregation = tmp_bipart_segregation[['placekey', 'BipartBlackVisitation', 'CountyFIPS']].groupby(['placekey','CountyFIPS']).sum().reset_index()

county_avg_black_percent = census_tract_data[['TractFIPS','CountyFIPS','DP05_0065E','TotalPopulation']].groupby('CountyFIPS').sum().reset_index()
county_avg_black_percent['avg_black_percent'] = county_avg_black_percent['DP05_0065E'] / county_avg_black_percent['TotalPopulation']
county_avg_black_percent = county_avg_black_percent[['CountyFIPS','avg_black_percent']]
tmp_bipart_segregation = tmp_bipart_segregation.merge(county_avg_black_percent,on='CountyFIPS',how='inner')
tmp_bipart_segregation['OriBipartSegregationIndex'] = (tmp_bipart_segregation['BipartBlackVisitation'] - tmp_bipart_segregation['avg_black_percent']).abs()
tmp_bipart_segregation = tmp_bipart_segregation.drop(['avg_black_percent','BipartBlackVisitation'],axis=1)

dynamic_segregation_index = tmp_bipart_segregation.groupby(['CountyFIPS']).apply(lambda x: x.sort_values(by='OriBipartSegregationIndex', ascending=False)['OriBipartSegregationIndex'].quantile(0.75)).reset_index().rename({0:'HighSegCriteria'},axis=1)
tmp_bipart_segregation = tmp_bipart_segregation.merge(dynamic_segregation_index[['CountyFIPS','HighSegCriteria']], on=['CountyFIPS'], how='inner')

dynamic_segregation_index = tmp_bipart_segregation.groupby(['CountyFIPS']).apply(lambda x: x.sort_values(by='OriBipartSegregationIndex', ascending=False)['OriBipartSegregationIndex'].quantile(0.25)).reset_index().rename({0:'LowSegCriteria'},axis=1)
tmp_bipart_segregation = tmp_bipart_segregation.merge(dynamic_segregation_index[['CountyFIPS','LowSegCriteria']], on=['CountyFIPS'], how='inner')

def find_high_bipart_segregation_park(x):
    if (x['OriBipartSegregationIndex'] > x['HighSegCriteria']) and (x['OriBipartSegregationIndex'] > 0):
        return True
    else:
        return False
tmp_bipart_segregation['IsOriStrongBipartSegregation'] = tmp_bipart_segregation.apply(find_high_bipart_segregation_park, axis=1)

def find_weak_bipart_segregation_park(x):
    if (x['OriBipartSegregationIndex'] < x['LowSegCriteria']) and (x['OriBipartSegregationIndex'] > 0):
        return True
    else:
        return False
tmp_bipart_segregation['IsOriWeakBipartSegregation'] = tmp_bipart_segregation.apply(find_weak_bipart_segregation_park, axis=1)


park_tract_bipart = park_tract_bipart.merge(tmp_bipart_segregation[['placekey','OriBipartSegregationIndex', 'IsOriStrongBipartSegregation','IsOriWeakBipartSegregation']],on='placekey',how='inner').drop(['total_visitor_count'], axis=1)



#########################################################################################################################
park_all_visit = park_tract_bipart[['placekey', 'visitor_count']].groupby('placekey').sum().reset_index().rename(columns={'visitor_count': 'total_visitor_count'})
park_all_visit = park_all_visit[park_all_visit['total_visitor_count'] != np.inf].reset_index(drop=True)  # 删除inf
park_tract_bipart = park_tract_bipart.merge(park_all_visit, on='placekey', how='inner')
tmp_bipart_segregation = park_tract_bipart[['placekey','visitor_count', 'total_visitor_count', 'TractFIPS', 'DP05_0064PE', 'DP05_0065PE', 'TotalPopulation']]
tmp_bipart_segregation['CountyFIPS'] = tmp_bipart_segregation['TractFIPS'].str[:5]
# county_pop = tmp_bipart_segregation[['CountyFIPS','TractFIPS', 'TotalPopulation']].drop_duplicates()[['CountyFIPS', 'TotalPopulation']].groupby('CountyFIPS').sum().reset_index().rename({'TotalPopulation': 'CountyPop'}, axis=1)
# county_pop = tmp_bipart_segregation[['CountyFIPS','TractFIPS', 'TotalPopulation']][['CountyFIPS', 'TotalPopulation']].groupby('CountyFIPS').sum().reset_index().rename({'TotalPopulation': 'CountyPop'}, axis=1)
county_pop = census_tract_data[['TractFIPS','CountyFIPS','TotalPopulation']].groupby('CountyFIPS').sum().reset_index().rename({'TotalPopulation': 'CountyPop'}, axis=1)

tmp_bipart_segregation = tmp_bipart_segregation.merge(county_pop, on='CountyFIPS', how='inner')
tmp_bipart_segregation['visit_weight'] = tmp_bipart_segregation['visitor_count'] / tmp_bipart_segregation['total_visitor_count']
tmp_bipart_segregation['pop_weight'] = tmp_bipart_segregation['TotalPopulation'] / tmp_bipart_segregation['CountyPop']
tmp_bipart_segregation = tmp_bipart_segregation.dropna().reset_index(drop=True)
tmp_bipart_segregation['BipartWhiteVisitation'] = tmp_bipart_segregation['visit_weight'] * tmp_bipart_segregation['DP05_0064PE'] * tmp_bipart_segregation['pop_weight'] * 0.01
tmp_bipart_segregation['BipartBlackVisitation'] = tmp_bipart_segregation['visit_weight'] * tmp_bipart_segregation['DP05_0065PE'] * tmp_bipart_segregation['pop_weight'] * 0.01
tmp_bipart_segregation = tmp_bipart_segregation[['placekey','BipartWhiteVisitation', 'BipartBlackVisitation', 'CountyFIPS']].groupby(['placekey','CountyFIPS']).sum().reset_index()

county_avg_white_black_percent = census_tract_data[['TractFIPS','CountyFIPS','DP05_0064E','DP05_0065E','TotalPopulation']].groupby('CountyFIPS').sum().reset_index()
county_avg_white_black_percent['avg_white_percent'] = county_avg_white_black_percent['DP05_0064E'] / county_avg_white_black_percent['TotalPopulation']
county_avg_white_black_percent['avg_black_percent'] = county_avg_white_black_percent['DP05_0065E'] / county_avg_white_black_percent['TotalPopulation']
county_avg_white_black_percent = county_avg_white_black_percent[['CountyFIPS','avg_white_percent','avg_black_percent']]
tmp_bipart_segregation = tmp_bipart_segregation.merge(county_avg_white_black_percent,on='CountyFIPS',how='inner')
tmp_bipart_segregation['BlackBipartSegregationIndex'] = tmp_bipart_segregation['avg_black_percent'] - tmp_bipart_segregation['BipartBlackVisitation']  # 正代表黑人去的少，负代表黑人去得多
tmp_bipart_segregation['WhiteBipartSegregationIndex'] = tmp_bipart_segregation['avg_white_percent'] - tmp_bipart_segregation['BipartWhiteVisitation']  # 正代表白人去的少，负代表白人去得多
tmp_bipart_segregation['BlackBipartSegregationIndexAbs'] = (tmp_bipart_segregation['avg_black_percent'] - tmp_bipart_segregation['BipartBlackVisitation']).abs()  # 正代表黑人去的少，负代表黑人去得多
tmp_bipart_segregation['WhiteBipartSegregationIndexAbs'] = (tmp_bipart_segregation['avg_white_percent'] - tmp_bipart_segregation['BipartWhiteVisitation']).abs()  # 正代表白人去的少，负代表白人去得多
tmp_bipart_segregation = tmp_bipart_segregation.drop(['avg_black_percent','BipartBlackVisitation','avg_white_percent','BipartWhiteVisitation'],axis=1)

dynamic_segregation_index = tmp_bipart_segregation.groupby(['CountyFIPS']).apply(lambda x: x.loc[x['WhiteBipartSegregationIndex'] > 0].sort_values(by='WhiteBipartSegregationIndex', ascending=False)['WhiteBipartSegregationIndex'].quantile(0.75)).reset_index().rename({0:'WhiteHighSegCriteria'},axis=1)
tmp_bipart_segregation = tmp_bipart_segregation.merge(dynamic_segregation_index[['CountyFIPS','WhiteHighSegCriteria']], on=['CountyFIPS'], how='inner')
dynamic_segregation_index = tmp_bipart_segregation.groupby(['CountyFIPS']).apply(lambda x: x.loc[x['BlackBipartSegregationIndex'] > 0].sort_values(by='BlackBipartSegregationIndex', ascending=False)['BlackBipartSegregationIndex'].quantile(0.75)).reset_index().rename({0:'BlackHighSegCriteria'},axis=1)
tmp_bipart_segregation = tmp_bipart_segregation.merge(dynamic_segregation_index[['CountyFIPS','BlackHighSegCriteria']], on=['CountyFIPS'], how='inner')

dynamic_segregation_index = tmp_bipart_segregation.groupby(['CountyFIPS']).apply(lambda x: x.sort_values(by='WhiteBipartSegregationIndexAbs', ascending=False)['WhiteBipartSegregationIndexAbs'].quantile(0.25)).reset_index().rename({0:'WhiteLowSegCriteria'},axis=1)
tmp_bipart_segregation = tmp_bipart_segregation.merge(dynamic_segregation_index[['CountyFIPS','WhiteLowSegCriteria']], on=['CountyFIPS'], how='inner')
dynamic_segregation_index = tmp_bipart_segregation.groupby(['CountyFIPS']).apply(lambda x: x.sort_values(by='BlackBipartSegregationIndexAbs', ascending=False)['BlackBipartSegregationIndexAbs'].quantile(0.25)).reset_index().rename({0:'BlackLowSegCriteria'},axis=1)
tmp_bipart_segregation = tmp_bipart_segregation.merge(dynamic_segregation_index[['CountyFIPS','BlackLowSegCriteria']], on=['CountyFIPS'], how='inner')

def find_strong_white_bipart_segregation_park(x):
    if (x['WhiteBipartSegregationIndex'] > x['WhiteHighSegCriteria']) and (x['WhiteBipartSegregationIndex'] > 0):
        return True
    else:
        return False
tmp_bipart_segregation['IsWhiteStrongBipartSegregation'] = tmp_bipart_segregation.apply(find_strong_white_bipart_segregation_park, axis=1)
def find_strong_black_bipart_segregation_park(x):
    if (x['BlackBipartSegregationIndex'] > x['BlackHighSegCriteria']) and (x['BlackBipartSegregationIndex'] > 0):
        return True
    else:
        return False
tmp_bipart_segregation['IsBlackStrongBipartSegregation'] = tmp_bipart_segregation.apply(find_strong_black_bipart_segregation_park, axis=1)

def find_strong_overall_bipart_segregation_park(x):
    if ((x['WhiteBipartSegregationIndex'] > x['WhiteHighSegCriteria']) and (x['WhiteBipartSegregationIndex'] > 0)) or ((x['BlackBipartSegregationIndex'] > x['BlackHighSegCriteria']) and (x['BlackBipartSegregationIndex'] > 0)):
        return True
    else:
        return False
tmp_bipart_segregation['IsOverallStrongBipartSegregation'] = tmp_bipart_segregation.apply(find_strong_overall_bipart_segregation_park, axis=1)


def find_weak_white_bipart_segregation_park(x):
    if x['WhiteBipartSegregationIndexAbs'] < x['WhiteLowSegCriteria']:
        return True
    else:
        return False
tmp_bipart_segregation['IsWhiteWeakBipartSegregation'] = tmp_bipart_segregation.apply(find_weak_white_bipart_segregation_park, axis=1)
def find_weak_black_bipart_segregation_park(x):
    if x['BlackBipartSegregationIndexAbs'] < x['BlackLowSegCriteria']:
        return True
    else:
        return False
tmp_bipart_segregation['IsBlackWeakBipartSegregation'] = tmp_bipart_segregation.apply(find_weak_black_bipart_segregation_park, axis=1)

def find_weak_overall_bipart_segregation_park(x):
    if (x['BlackBipartSegregationIndexAbs'] < x['BlackLowSegCriteria']) or (x['WhiteBipartSegregationIndexAbs'] < x['WhiteLowSegCriteria']):
        return True
    else:
        return False
tmp_bipart_segregation['IsOverallWeakBipartSegregation'] = tmp_bipart_segregation.apply(find_weak_overall_bipart_segregation_park, axis=1)



park_tract_bipart = park_tract_bipart.merge(tmp_bipart_segregation[['placekey','WhiteBipartSegregationIndex','BlackBipartSegregationIndex',
                                                                    'WhiteBipartSegregationIndexAbs','BlackBipartSegregationIndexAbs',
                                                                    'IsWhiteStrongBipartSegregation','IsWhiteWeakBipartSegregation','IsOverallStrongBipartSegregation','IsOverallWeakBipartSegregation',
                                                                    'IsBlackStrongBipartSegregation','IsBlackWeakBipartSegregation']],on='placekey',how='inner').drop(['total_visitor_count'], axis=1)

print(park_tract_bipart['TractFIPS'].drop_duplicates().shape[0])


#########################################################################################################################
park_all_visit = park_tract_bipart[['placekey', 'visitor_count']].groupby('placekey').sum().reset_index().rename(columns={'visitor_count': 'total_visitor_count'})
park_all_visit = park_all_visit[park_all_visit['total_visitor_count'] != np.inf].reset_index(drop=True)  # 删除inf
park_tract_bipart = park_tract_bipart.merge(park_all_visit, on='placekey', how='inner')
tmp_bipart_ranking_segregation = park_tract_bipart[['placekey','visitor_count', 'total_visitor_count', 'TractFIPS', 'DP05_0065PE', 'TotalPopulation']]
tmp_bipart_ranking_segregation['CountyFIPS'] = tmp_bipart_ranking_segregation['TractFIPS'].str[:5]
#county_pop = tmp_bipart_ranking_segregation[['CountyFIPS', 'TotalPopulation']].groupby('CountyFIPS').sum().reset_index().rename({'TotalPopulation': 'CountyPop'}, axis=1)
county_pop = tmp_bipart_ranking_segregation[['CountyFIPS','TractFIPS', 'TotalPopulation']].drop_duplicates()[['CountyFIPS', 'TotalPopulation']].groupby('CountyFIPS').sum().reset_index().rename({'TotalPopulation': 'CountyPop'}, axis=1)

tmp_bipart_ranking_segregation = tmp_bipart_ranking_segregation.merge(county_pop, on='CountyFIPS', how='inner')
tmp_bipart_ranking_segregation['visit_weight'] = tmp_bipart_ranking_segregation['visitor_count'] / tmp_bipart_ranking_segregation['total_visitor_count']
tmp_bipart_ranking_segregation['pop_weight'] = tmp_bipart_ranking_segregation['TotalPopulation'] / tmp_bipart_ranking_segregation['CountyPop']
tmp_bipart_ranking_segregation = tmp_bipart_ranking_segregation.dropna().reset_index(drop=True)
tmp_bipart_ranking_segregation['BipartBlackVisitation'] = tmp_bipart_ranking_segregation['visit_weight'] * tmp_bipart_ranking_segregation['DP05_0065PE'] * tmp_bipart_ranking_segregation['pop_weight']
tmp_bipart_ranking_segregation = tmp_bipart_ranking_segregation[['placekey', 'BipartBlackVisitation', 'CountyFIPS']].groupby(['placekey','CountyFIPS']).sum().reset_index()

dynamic_ranking_segregation_criteria_strong_A = tmp_bipart_ranking_segregation.groupby(['CountyFIPS']).apply(lambda x: x.sort_values(by='BipartBlackVisitation', ascending=False)['BipartBlackVisitation'].quantile(0.875)).reset_index().rename({0:'StrongSegCriteriaA'},axis=1)
dynamic_ranking_segregation_criteria_strong_B = tmp_bipart_ranking_segregation.groupby(['CountyFIPS']).apply(lambda x: x.sort_values(by='BipartBlackVisitation', ascending=False)['BipartBlackVisitation'].quantile(0.125)).reset_index().rename({0:'StrongSegCriteriaB'},axis=1)
dynamic_ranking_segregation_criteria_weak_A = tmp_bipart_ranking_segregation.groupby(['CountyFIPS']).apply(lambda x: x.sort_values(by='BipartBlackVisitation', ascending=False)['BipartBlackVisitation'].quantile(0.625)).reset_index().rename({0:'WeakSegCriteriaA'},axis=1)
dynamic_ranking_segregation_criteria_weak_B = tmp_bipart_ranking_segregation.groupby(['CountyFIPS']).apply(lambda x: x.sort_values(by='BipartBlackVisitation', ascending=False)['BipartBlackVisitation'].quantile(0.375)).reset_index().rename({0:'WeakSegCriteriaB'},axis=1)

dynamic_ranking_segregation_criteria = dynamic_ranking_segregation_criteria_strong_A.merge(dynamic_ranking_segregation_criteria_strong_B[['CountyFIPS','StrongSegCriteriaB']],on='CountyFIPS',how='inner').merge(dynamic_ranking_segregation_criteria_weak_A[['CountyFIPS','WeakSegCriteriaA']],on='CountyFIPS',how='inner').merge(dynamic_ranking_segregation_criteria_weak_B[['CountyFIPS','WeakSegCriteriaB']],on='CountyFIPS',how='inner')
tmp_bipart_ranking_segregation = tmp_bipart_ranking_segregation.merge(dynamic_ranking_segregation_criteria, on=['CountyFIPS'], how='inner')

def find_strong_ranking_segregation_park(x):
    if (x['BipartBlackVisitation'] > x['StrongSegCriteriaA']) or (x['BipartBlackVisitation'] < x['StrongSegCriteriaB']):
        return True
    else:
        return False

def find_weak_ranking_segregation_park(x):
    if (x['BipartBlackVisitation'] > x['WeakSegCriteriaB']) and (x['BipartBlackVisitation'] < x['WeakSegCriteriaA']):
        return True
    else:
        return False
tmp_bipart_ranking_segregation['IsStrongBipartRankingSegregation'] = tmp_bipart_ranking_segregation.apply(find_strong_ranking_segregation_park, axis=1)
tmp_bipart_ranking_segregation['IsWeakBipartRankingSegregation'] = tmp_bipart_ranking_segregation.apply(find_weak_ranking_segregation_park, axis=1)

park_tract_bipart = park_tract_bipart.merge(tmp_bipart_ranking_segregation[['placekey', 'IsStrongBipartRankingSegregation','IsWeakBipartRankingSegregation']],on='placekey',how='inner')
print(park_tract_bipart['TractFIPS'].drop_duplicates().shape[0])
#########################################################################################################################
park_tract_bipart = park_tract_bipart.merge(park_visit[['placekey','median_dwell_week_mean']], on='placekey', how='inner')
park_tract_bipart['visit_time_weight'] = park_tract_bipart['median_dwell_week_mean'] * park_tract_bipart['visitor_count']
print(park_tract_bipart['TractFIPS'].drop_duplicates().shape[0])

############################################
park_tract_bipart.to_parquet(f'./park_tract_bipart_all_us_within_county_{YEAR}.parquet')
# Tract
tract_local_visitor_strong_segregation = park_tract_bipart.loc[park_tract_bipart['IsStrongSegregationPark']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_weak_segregation = park_tract_bipart.loc[park_tract_bipart['IsWeakSegregationPark']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_white_major_visit = park_tract_bipart.loc[park_tract_bipart['IsWhiteMajorVisitPark']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_black_major_visit = park_tract_bipart.loc[park_tract_bipart['IsBlackMajorVisitPark']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_black_extreme_visit = park_tract_bipart.loc[park_tract_bipart['IsBlackExtremeVisitPark']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_white_major_locate = park_tract_bipart.loc[park_tract_bipart['IsWhiteMajorLocatePark']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_black_major_locate = park_tract_bipart.loc[park_tract_bipart['IsBlackMajorLocatePark']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_ori_strong_bipart_segregation = park_tract_bipart.loc[park_tract_bipart['IsOriStrongBipartSegregation']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_ori_weak_bipart_segregation = park_tract_bipart.loc[park_tract_bipart['IsOriWeakBipartSegregation']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_overall_strong_bipart_segregation = park_tract_bipart.loc[park_tract_bipart['IsOverallStrongBipartSegregation']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_overall_weak_bipart_segregation = park_tract_bipart.loc[park_tract_bipart['IsOverallWeakBipartSegregation']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_white_strong_bipart_segregation = park_tract_bipart.loc[park_tract_bipart['IsWhiteStrongBipartSegregation']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_black_strong_bipart_segregation = park_tract_bipart.loc[park_tract_bipart['IsBlackStrongBipartSegregation']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_white_weak_bipart_segregation = park_tract_bipart.loc[park_tract_bipart['IsWhiteWeakBipartSegregation']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_black_weak_bipart_segregation = park_tract_bipart.loc[park_tract_bipart['IsBlackWeakBipartSegregation']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_strong_bipart_ranking_segregation = park_tract_bipart.loc[park_tract_bipart['IsStrongBipartRankingSegregation']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_weak_bipart_ranking_segregation = park_tract_bipart.loc[park_tract_bipart['IsWeakBipartRankingSegregation']==True][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()

tract_local_visitor_overall_strong_bipart_segregation_in_white_major_locate = park_tract_bipart.loc[(park_tract_bipart['IsOverallStrongBipartSegregation']==True) & (park_tract_bipart['IsWhiteMajorLocatePark']==True)][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_overall_strong_bipart_segregation_in_black_major_locate = park_tract_bipart.loc[(park_tract_bipart['IsOverallStrongBipartSegregation']==True) & (park_tract_bipart['IsBlackMajorLocatePark']==True)][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_overall_weak_bipart_segregation_in_white_major_locate = park_tract_bipart.loc[(park_tract_bipart['IsOverallWeakBipartSegregation']==True) & (park_tract_bipart['IsWhiteMajorLocatePark']==True)][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_overall_weak_bipart_segregation_in_black_major_locate = park_tract_bipart.loc[(park_tract_bipart['IsOverallWeakBipartSegregation']==True) & (park_tract_bipart['IsBlackMajorLocatePark']==True)][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()

tract_local_visitor_white_strong_bipart_segregation_in_white_major_locate = park_tract_bipart.loc[(park_tract_bipart['IsWhiteStrongBipartSegregation']==True) & (park_tract_bipart['IsWhiteMajorLocatePark']==True)][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_white_strong_bipart_segregation_in_black_major_locate = park_tract_bipart.loc[(park_tract_bipart['IsWhiteStrongBipartSegregation']==True) & (park_tract_bipart['IsBlackMajorLocatePark']==True)][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_black_strong_bipart_segregation_in_white_major_locate = park_tract_bipart.loc[(park_tract_bipart['IsBlackStrongBipartSegregation']==True) & (park_tract_bipart['IsWhiteMajorLocatePark']==True)][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_black_strong_bipart_segregation_in_black_major_locate = park_tract_bipart.loc[(park_tract_bipart['IsBlackStrongBipartSegregation']==True) & (park_tract_bipart['IsBlackMajorLocatePark']==True)][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()

tract_local_visitor_white_weak_bipart_segregation_in_white_major_locate = park_tract_bipart.loc[(park_tract_bipart['IsWhiteWeakBipartSegregation']==True) & (park_tract_bipart['IsWhiteMajorLocatePark']==True)][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_white_weak_bipart_segregation_in_black_major_locate = park_tract_bipart.loc[(park_tract_bipart['IsWhiteWeakBipartSegregation']==True) & (park_tract_bipart['IsBlackMajorLocatePark']==True)][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_black_weak_bipart_segregation_in_white_major_locate = park_tract_bipart.loc[(park_tract_bipart['IsBlackWeakBipartSegregation']==True) & (park_tract_bipart['IsWhiteMajorLocatePark']==True)][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()
tract_local_visitor_black_weak_bipart_segregation_in_black_major_locate = park_tract_bipart.loc[(park_tract_bipart['IsBlackWeakBipartSegregation']==True) & (park_tract_bipart['IsBlackMajorLocatePark']==True)][['TractFIPS','visitor_count']].groupby('TractFIPS').sum().reset_index()


tract_local_visitor = park_tract_bipart[['TractFIPS','visitor_count','visit_time_weight',
                                         'IsStrongSegregationPark','IsWeakSegregationPark', 
                                         'IsWhiteMajorVisitPark', 'IsBlackMajorVisitPark', 'IsBlackExtremeVisitPark',
                                         'IsWhiteMajorLocatePark', 'IsBlackMajorLocatePark',
                                         'IsOriStrongBipartSegregation', 'IsOriWeakBipartSegregation',
                                         'IsOverallStrongBipartSegregation', 'IsOverallWeakBipartSegregation',
                                         'IsWhiteStrongBipartSegregation', 'IsBlackStrongBipartSegregation','IsWhiteWeakBipartSegregation', 'IsBlackWeakBipartSegregation',
                                         'IsStrongBipartRankingSegregation', 'IsWeakBipartRankingSegregation']].groupby('TractFIPS').sum().drop_duplicates().reset_index()


tract_visit = tract_local_visitor.merge(tract_visit,on='TractFIPS',how='left').rename({'visitor_count':'total_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_strong_segregation,on='TractFIPS',how='left').rename({'visitor_count':'strong_segregation_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_weak_segregation,on='TractFIPS',how='left').rename({'visitor_count':'weak_segregation_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_white_major_visit,on='TractFIPS',how='left').rename({'visitor_count':'white_major_visit_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_black_major_visit,on='TractFIPS',how='left').rename({'visitor_count':'black_major_visit_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_black_extreme_visit,on='TractFIPS',how='left').rename({'visitor_count':'black_extreme_visit_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_white_major_locate,on='TractFIPS',how='left').rename({'visitor_count':'white_major_locate_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_black_major_locate,on='TractFIPS',how='left').rename({'visitor_count':'black_major_locate_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_ori_strong_bipart_segregation,on='TractFIPS',how='left').rename({'visitor_count':'ori_strong_bipart_segregation_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_ori_weak_bipart_segregation,on='TractFIPS',how='left').rename({'visitor_count':'ori_weak_bipart_segregation_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_overall_strong_bipart_segregation,on='TractFIPS',how='left').rename({'visitor_count':'overall_strong_bipart_segregation_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_overall_weak_bipart_segregation,on='TractFIPS',how='left').rename({'visitor_count':'overall_weak_bipart_segregation_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_white_strong_bipart_segregation,on='TractFIPS',how='left').rename({'visitor_count':'white_strong_bipart_segregation_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_black_strong_bipart_segregation,on='TractFIPS',how='left').rename({'visitor_count':'black_strong_bipart_segregation_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_white_weak_bipart_segregation,on='TractFIPS',how='left').rename({'visitor_count':'white_weak_bipart_segregation_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_black_weak_bipart_segregation,on='TractFIPS',how='left').rename({'visitor_count':'black_weak_bipart_segregation_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_strong_bipart_ranking_segregation,on='TractFIPS',how='left').rename({'visitor_count':'strong_bipart_ranking_segregation_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_weak_bipart_ranking_segregation,on='TractFIPS',how='left').rename({'visitor_count':'weak_bipart_ranking_segregation_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_overall_strong_bipart_segregation_in_white_major_locate,on='TractFIPS',how='left').rename({'visitor_count':'overall_strong_bipart_segregation_in_white_major_locate_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_overall_strong_bipart_segregation_in_black_major_locate,on='TractFIPS',how='left').rename({'visitor_count':'overall_strong_bipart_segregation_in_black_major_locate_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_overall_weak_bipart_segregation_in_white_major_locate,on='TractFIPS',how='left').rename({'visitor_count':'overall_weak_bipart_segregation_in_white_major_locate_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_overall_weak_bipart_segregation_in_black_major_locate,on='TractFIPS',how='left').rename({'visitor_count':'overall_weak_bipart_segregation_in_black_major_locate_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_white_strong_bipart_segregation_in_white_major_locate,on='TractFIPS',how='left').rename({'visitor_count':'white_strong_bipart_segregation_in_white_major_locate_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_white_strong_bipart_segregation_in_black_major_locate,on='TractFIPS',how='left').rename({'visitor_count':'white_strong_bipart_segregation_in_black_major_locate_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_black_strong_bipart_segregation_in_white_major_locate,on='TractFIPS',how='left').rename({'visitor_count':'black_strong_bipart_segregation_in_white_major_locate_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_black_strong_bipart_segregation_in_black_major_locate,on='TractFIPS',how='left').rename({'visitor_count':'black_strong_bipart_segregation_in_black_major_locate_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_white_weak_bipart_segregation_in_white_major_locate,on='TractFIPS',how='left').rename({'visitor_count':'white_weak_bipart_segregation_in_white_major_locate_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_white_weak_bipart_segregation_in_black_major_locate,on='TractFIPS',how='left').rename({'visitor_count':'white_weak_bipart_segregation_in_black_major_locate_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_black_weak_bipart_segregation_in_white_major_locate,on='TractFIPS',how='left').rename({'visitor_count':'black_weak_bipart_segregation_in_white_major_locate_visitor_count'},axis=1)
tract_visit = tract_visit.merge(tract_local_visitor_black_weak_bipart_segregation_in_black_major_locate,on='TractFIPS',how='left').rename({'visitor_count':'black_weak_bipart_segregation_in_black_major_locate_visitor_count'},axis=1)


tract_visit = tract_visit.fillna(0)
tract_visit['weekly_park_visit_total_per_people'] = tract_visit['total_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_time_total_per_people'] = tract_visit['visit_time_weight'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_strong_segregation_per_people'] = tract_visit['strong_segregation_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_weak_segregation_per_people'] = tract_visit['weak_segregation_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_white_major_visit_per_people'] = tract_visit['white_major_visit_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_black_major_visit_per_people'] = tract_visit['black_major_visit_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_black_extreme_visit_per_people'] = tract_visit['black_extreme_visit_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_white_major_locate_per_people'] = tract_visit['white_major_locate_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_black_major_locate_per_people'] = tract_visit['black_major_locate_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_ori_strong_bipart_segregation_per_people'] = tract_visit['ori_strong_bipart_segregation_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_ori_weak_bipart_segregation_per_people'] = tract_visit['ori_weak_bipart_segregation_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_overall_strong_bipart_segregation_per_people'] = tract_visit['overall_strong_bipart_segregation_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_overall_weak_bipart_segregation_per_people'] = tract_visit['overall_weak_bipart_segregation_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_white_strong_bipart_segregation_per_people'] = tract_visit['white_strong_bipart_segregation_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_black_strong_bipart_segregation_per_people'] = tract_visit['black_strong_bipart_segregation_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_white_weak_bipart_segregation_per_people'] = tract_visit['white_weak_bipart_segregation_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_black_weak_bipart_segregation_per_people'] = tract_visit['black_weak_bipart_segregation_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_strong_bipart_ranking_segregation_per_people'] = tract_visit['strong_bipart_ranking_segregation_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_weak_bipart_ranking_segregation_per_people'] = tract_visit['weak_bipart_ranking_segregation_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)

tract_visit['weekly_park_visit_overall_strong_bipart_segregation_in_white_major_locate_per_people'] = tract_visit['overall_strong_bipart_segregation_in_white_major_locate_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_overall_strong_bipart_segregation_in_black_major_locate_per_people'] = tract_visit['overall_strong_bipart_segregation_in_black_major_locate_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_overall_weak_bipart_segregation_in_white_major_locate_per_people'] = tract_visit['overall_weak_bipart_segregation_in_white_major_locate_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_overall_weak_bipart_segregation_in_black_major_locate_per_people'] = tract_visit['overall_weak_bipart_segregation_in_black_major_locate_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)

tract_visit['weekly_park_visit_white_strong_bipart_segregation_in_white_major_locate_per_people'] = tract_visit['white_strong_bipart_segregation_in_white_major_locate_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_white_strong_bipart_segregation_in_black_major_locate_per_people'] = tract_visit['white_strong_bipart_segregation_in_black_major_locate_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_black_strong_bipart_segregation_in_white_major_locate_per_people'] = tract_visit['black_strong_bipart_segregation_in_white_major_locate_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_black_strong_bipart_segregation_in_black_major_locate_per_people'] = tract_visit['black_strong_bipart_segregation_in_black_major_locate_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)

tract_visit['weekly_park_visit_white_weak_bipart_segregation_in_white_major_locate_per_people'] = tract_visit['white_weak_bipart_segregation_in_white_major_locate_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_white_weak_bipart_segregation_in_black_major_locate_per_people'] = tract_visit['white_weak_bipart_segregation_in_black_major_locate_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_black_weak_bipart_segregation_in_white_major_locate_per_people'] = tract_visit['black_weak_bipart_segregation_in_white_major_locate_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)
tract_visit['weekly_park_visit_black_weak_bipart_segregation_in_black_major_locate_per_people'] = tract_visit['black_weak_bipart_segregation_in_black_major_locate_visitor_count'] / tract_visit['TotalPopulation'] / (365/7)



tract_visit = tract_visit.merge(tract_radius_of_gyration.rename(columns={'radius_of_gyration':'TractRadiusOfGyration'}),on='TractFIPS',how='inner')

tract_visit = tract_visit.drop(['IsStrongSegregationPark', 'IsWeakSegregationPark',
                                'IsWhiteMajorVisitPark', 'IsBlackMajorVisitPark',
                                'IsBlackExtremeVisitPark', 'IsWhiteMajorLocatePark',
                                'IsBlackMajorLocatePark', 'IsOriStrongBipartSegregation',
                                'IsOriWeakBipartSegregation', 'IsOverallStrongBipartSegregation',
                                'IsOverallWeakBipartSegregation', 'IsWhiteStrongBipartSegregation',
                                'IsBlackStrongBipartSegregation', 'IsWhiteWeakBipartSegregation',
                                'IsBlackWeakBipartSegregation', 'IsStrongBipartRankingSegregation',
                                'IsWeakBipartRankingSegregation','TotalPopulation','total_visitor_count','visit_time_weight',
                                'strong_segregation_visitor_count', 'weak_segregation_visitor_count',
                                'white_major_visit_visitor_count','black_major_visit_visitor_count','black_extreme_visit_visitor_count',
                                'white_major_locate_visitor_count','black_major_locate_visitor_count',
                                'ori_strong_bipart_segregation_visitor_count','ori_weak_bipart_segregation_visitor_count',
                                'overall_strong_bipart_segregation_visitor_count','overall_weak_bipart_segregation_visitor_count',
                                'white_strong_bipart_segregation_visitor_count','black_strong_bipart_segregation_visitor_count',
                                'white_weak_bipart_segregation_visitor_count','black_weak_bipart_segregation_visitor_count',
                                'strong_bipart_ranking_segregation_visitor_count','weak_bipart_ranking_segregation_visitor_count',
                                'overall_strong_bipart_segregation_in_white_major_locate_visitor_count','overall_strong_bipart_segregation_in_black_major_locate_visitor_count',
                                'overall_weak_bipart_segregation_in_white_major_locate_visitor_count','overall_weak_bipart_segregation_in_black_major_locate_visitor_count',
                                'white_strong_bipart_segregation_in_white_major_locate_visitor_count','white_strong_bipart_segregation_in_black_major_locate_visitor_count',
                                'black_strong_bipart_segregation_in_white_major_locate_visitor_count','black_strong_bipart_segregation_in_black_major_locate_visitor_count',
                                'white_weak_bipart_segregation_in_white_major_locate_visitor_count','white_weak_bipart_segregation_in_black_major_locate_visitor_count',
                                'black_weak_bipart_segregation_in_white_major_locate_visitor_count','black_weak_bipart_segregation_in_black_major_locate_visitor_count',
                                'geometry'],axis=1)


def subprocess_cal_tract_radius_of_gyration_different_segregation(idx, total_process_num, park_tract_bipart):
    print(f'Process: {idx} start')
    begin_time = datetime.now()
    tract_id = []
    tract_radius_of_gyration = []
    for tract in park_tract_bipart['TractFIPS'].unique():
        tmp = park_tract_bipart.loc[park_tract_bipart['TractFIPS'] == tract]
        if tmp['visitor_count'].sum() <= 0:
            radius_of_gyration = -1
        else:
            rog = (tmp['Distance'] ** 2 * tmp['visitor_count']).sum() / tmp['visitor_count'].sum()
            if rog <= 0:
                radius_of_gyration = -1
            else:
                radius_of_gyration = np.sqrt(rog)
        tract_id.append(tract)
        tract_radius_of_gyration.append(radius_of_gyration)
    end_time = datetime.now()
    print(f'Process: {idx} end time cost: ', (end_time - begin_time).total_seconds() / 60, ' min')
    return tract_id, tract_radius_of_gyration

def cal_tract_radius_of_gyration_different_segregation(df):
    df = df.copy(deep=True)
    p = Pool(MULTI_PROCESS_NUM)
    result_strong_black_bipart_segregation = []
    tract_list = df['TractFIPS'].unique().tolist()
    for i in range(MULTI_PROCESS_NUM):
        batch_num = int(ceil(len(tract_list) / MULTI_PROCESS_NUM)) + 1
        tract_list_tmp = copy.deepcopy(tract_list[i * batch_num:(i + 1) * batch_num])
        park_tract_bipart_tmp = copy.deepcopy(df.loc[(df['TractFIPS'].isin(tract_list_tmp)) & df['IsBlackStrongBipartSegregation']])
        result_strong_black_bipart_segregation.append(p.apply_async(subprocess_cal_tract_radius_of_gyration_different_segregation, args=(i, MULTI_PROCESS_NUM, park_tract_bipart_tmp)))

    tract_id_list_strong_black_bipart_segregation = []
    tract_radius_of_gyration_list_strong_black_bipart_segregation = []
    for i in result_strong_black_bipart_segregation:
        tmp_tract_id, tmp_tract_rog = i.get()
        tract_id_list_strong_black_bipart_segregation += tmp_tract_id
        tract_radius_of_gyration_list_strong_black_bipart_segregation += tmp_tract_rog
    p.close()
    p.join()

    p = Pool(MULTI_PROCESS_NUM)
    result_not_strong_black_bipart_segregation = []
    tract_list = df['TractFIPS'].unique().tolist()
    for i in range(MULTI_PROCESS_NUM):
        batch_num = int(ceil(len(tract_list) / MULTI_PROCESS_NUM)) + 1
        tract_list_tmp = copy.deepcopy(tract_list[i * batch_num:(i + 1) * batch_num])
        park_tract_bipart_tmp = copy.deepcopy(df.loc[(df['TractFIPS'].isin(tract_list_tmp)) & (~df['IsBlackStrongBipartSegregation'])])
        result_not_strong_black_bipart_segregation.append(p.apply_async(subprocess_cal_tract_radius_of_gyration_different_segregation, args=(i, MULTI_PROCESS_NUM, park_tract_bipart_tmp)))

    tract_id_list_not_strong_black_bipart_segregation = []
    tract_radius_of_gyration_list_not_strong_black_bipart_segregation = []
    for i in result_not_strong_black_bipart_segregation:
        tmp_tract_id, tmp_tract_rog = i.get()
        tract_id_list_not_strong_black_bipart_segregation += tmp_tract_id
        tract_radius_of_gyration_list_not_strong_black_bipart_segregation += tmp_tract_rog
    p.close()
    p.join()


    tract_radius_of_gyration_strong_black_bipart_segregation = pd.DataFrame({'TractFIPS': tract_id_list_strong_black_bipart_segregation, 'TractRadiusOfGyrationStrongBlackBipartSegregation': tract_radius_of_gyration_list_strong_black_bipart_segregation})
    tract_radius_of_gyration_not_strong_black_bipart_segregation = pd.DataFrame({'TractFIPS': tract_id_list_not_strong_black_bipart_segregation, 'TractRadiusOfGyrationNotStrongBlackBipartSegregation': tract_radius_of_gyration_list_not_strong_black_bipart_segregation})

    tract_radius_of_gyration = tract_radius_of_gyration_strong_black_bipart_segregation.merge(tract_radius_of_gyration_not_strong_black_bipart_segregation, on='TractFIPS', how='outer')

    return tract_radius_of_gyration

begin_time = datetime.now()
tract_radius_of_gyration_different_segregation = cal_tract_radius_of_gyration_different_segregation(park_tract_bipart)
end_time = datetime.now()
print('tract radius of gyration different segregation time cost: ', (end_time - begin_time).total_seconds() / 60, ' min')

tract_visit = tract_visit.merge(tract_radius_of_gyration_different_segregation,on='TractFIPS',how='outer')
tract_visit.to_parquet(f'./tract_visit_all_US_within_county_{YEAR}.parquet',index=False)

# B01003e1: Total Population
# B02001e3: Black or African American alone
# B02001e2: White alone
# B19013e1: Median Household Income In The Past 12 Months (In 2019 Inflation-Adjusted Dollars)

# Park
park_local_visitor = park_tract_bipart[['placekey','visitor_count']].groupby('placekey').sum().reset_index()
park_visit = park_visit[['median_dwell_week_mean','placekey', 'location_name','TractFIPS']].merge(park_local_visitor,on='placekey',how='inner')
park_visit = park_visit.merge(park_radius_of_gyration.rename(columns={'radius_of_gyration':'ParkRadiusOfGyration'}), on='placekey',how='inner')
park_visit['visitor_count_weekly_mean'] = park_visit['visitor_count'] / (365/7)
park_visit = park_visit.drop(['visitor_count'],axis=1)
park_visit.to_parquet(f'./park_visit_all_US_within_county_{YEAR}.parquet',index=False)
