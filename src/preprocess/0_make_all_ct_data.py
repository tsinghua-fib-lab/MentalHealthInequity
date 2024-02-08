import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
from pathlib import Path
import pandas as pd
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white", palette=None)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13

YEAR = 2019
USE_SELECTED_ACS = False
############################################################################################
# Data Path
PLACES_CensusTract_LOC = Path(f'./Data/PLACES/PLACES__Census_Tract_Data__GIS_Friendly_Format___{YEAR+2}_release.csv')
PLACES_County_LOC = Path(f'./Data/PLACES/PLACES__County_Data__GIS_Friendly_Format___{YEAR+2}_release.csv')
CENSUS_TRACT_BOUNDARY_LOC = Path('./Data/Boundary/cb_2019_us_tract_500k/cb_2019_us_tract_500k.shp')
ACS_CensusTract_DP02_LOC = Path('./Data/ACS/CensusTract/DP02-CT/ACSDP5Y2019.DP02-Data.csv')
ACS_CensusTract_DP02_Selected_Indicator_LOC = Path('./Data/ACS/CensusTract/DP02_selection.csv')
ACS_CensusTract_DP03_LOC = Path('./Data/ACS/CensusTract/DP03-CT/ACSDP5Y2019.DP03-Data.csv')
ACS_CensusTract_DP03_Selected_Indicator_LOC = Path('./Data/ACS/CensusTract/DP03_selection.csv')
ACS_CensusTract_DP04_LOC = Path('./Data/ACS/CensusTract/DP04-CT/ACSDP5Y2019.DP04-Data.csv')
ACS_CensusTract_DP04_Selected_Indicator_LOC = Path('./Data/ACS/CensusTract/DP04_selection.csv')
ACS_CensusTract_DP05_LOC = Path('./Data/ACS/CensusTract/DP05-CT/ACSDP5Y2019.DP05-Data.csv')
ACS_CensusTract_DP05_Selected_Indicator_LOC = Path('./Data/ACS/CensusTract/DP05_selection.csv')
############################################################################################
# CDC Places
# Geolocation: Point(Longitude Latitude)
census_tract_data = pd.read_csv(PLACES_CensusTract_LOC,dtype={'CountyFIPS':'str','TractFIPS':'str'})[['StateAbbr','StateDesc','CountyName','CountyFIPS','TractFIPS','TotalPopulation','DEPRESSION_CrudePrev','MHLTH_CrudePrev','CHECKUP_CrudePrev','ACCESS2_CrudePrev','Geolocation']]
county_data = pd.read_csv(PLACES_County_LOC,dtype={'CountyFIPS':'str'})[['StateAbbr','StateDesc','CountyName','CountyFIPS','TotalPopulation','DEPRESSION_CrudePrev','MHLTH_CrudePrev','CHECKUP_CrudePrev','ACCESS2_CrudePrev','Geolocation']]

# ACS
DP02_Indicators = pd.read_csv(ACS_CensusTract_DP02_Selected_Indicator_LOC)
ACS_CensusTract_DP02 = pd.read_csv(ACS_CensusTract_DP02_LOC,dtype='str',encoding_errors='backslashreplace')[['GEO_ID']+DP02_Indicators['Column Name'].to_list()].iloc[1:,:]
DP03_Indicators = pd.read_csv(ACS_CensusTract_DP03_Selected_Indicator_LOC)
ACS_CensusTract_DP03 = pd.read_csv(ACS_CensusTract_DP03_LOC,dtype='str',encoding_errors='backslashreplace')[['GEO_ID']+DP03_Indicators['Column Name'].to_list()].iloc[1:,:]
DP04_Indicators = pd.read_csv(ACS_CensusTract_DP04_Selected_Indicator_LOC)
ACS_CensusTract_DP04 = pd.read_csv(ACS_CensusTract_DP04_LOC,dtype='str',encoding_errors='backslashreplace')[['GEO_ID']+DP04_Indicators['Column Name'].to_list()].iloc[1:,:]
DP05_Indicators = pd.read_csv(ACS_CensusTract_DP05_Selected_Indicator_LOC)
ACS_CensusTract_DP05 = pd.read_csv(ACS_CensusTract_DP05_LOC,dtype='str',encoding_errors='backslashreplace')[['GEO_ID']+DP05_Indicators['Column Name'].to_list()].iloc[1:,:]

# Select some demographics
if USE_SELECTED_ACS:
    ACS_CensusTract_DP02 = ACS_CensusTract_DP02[['GEO_ID','DP02_0060PE','DP02_0061PE','DP02_0062PE','DP02_0063PE','DP02_0064PE','DP02_0065PE','DP02_0066PE','DP02_0067PE','DP02_0068PE']]
    ACS_CensusTract_DP03 = ACS_CensusTract_DP03[['GEO_ID','DP03_0002PE','DP03_0007PE','DP03_0052PE','DP03_0062E','DP03_0088E','DP03_0128PE']] # In labor force, Not in labor force, Median household income (dollars), Per capita income (dollars), BELOW THE POVERTY LEVEL
    ACS_CensusTract_DP05 = ACS_CensusTract_DP05[['GEO_ID','DP05_0002PE','DP05_0003PE','DP05_0017PE','DP05_0018E','DP05_0019PE','DP05_0024PE','DP05_0064PE','DP05_0065PE','DP05_0071PE']] #Male, Female,85 years and over,Median age (years),Under 18 years,65 years and over, White, Black or African American, Hispanic or Latino (of any race)

# GEOID process
ACS_CensusTract_DP02.loc[:,"GEO_ID"] = ACS_CensusTract_DP02["GEO_ID"].str.replace('1400000US','')
ACS_CensusTract_DP03.loc[:,"GEO_ID"] = ACS_CensusTract_DP03["GEO_ID"].str.replace('1400000US','')
ACS_CensusTract_DP04.loc[:,"GEO_ID"] = ACS_CensusTract_DP04["GEO_ID"].str.replace('1400000US','')
ACS_CensusTract_DP05.loc[:,"GEO_ID"] = ACS_CensusTract_DP05["GEO_ID"].str.replace('1400000US','')

# Read Census tract boundary
census_tract_boundary = gpd.read_file(CENSUS_TRACT_BOUNDARY_LOC,dtype={'NAME':'str'})[['STATEFP','TRACTCE','COUNTYFP','NAME','geometry']]
census_tract_boundary.loc[:,'TractFIPS'] = census_tract_boundary['STATEFP'] + census_tract_boundary['COUNTYFP'] + census_tract_boundary['TRACTCE']
census_tract_boundary = census_tract_boundary.drop(['STATEFP','COUNTYFP','TRACTCE'], axis=1)

############################################################################################
census_tract_data_selected = census_tract_data
census_tract_data_selected = census_tract_data_selected.drop('Geolocation', axis=1)
# 2022
# census_tract_data_selected = census_tract_boundary.drop(['STUSPS', 'STATE_NAME'], axis=1).merge(census_tract_data_selected, on='TractFIPS', how='right')
census_tract_data_selected = census_tract_boundary.merge(census_tract_data_selected, on='TractFIPS', how='right')
ACS = ACS_CensusTract_DP02.merge(ACS_CensusTract_DP03, on='GEO_ID', how='inner')
ACS = ACS.merge(ACS_CensusTract_DP04, on='GEO_ID', how='inner')
ACS = ACS.merge(ACS_CensusTract_DP05, on='GEO_ID', how='inner')
def convert_dtype(x):
    try:
        tmp = float(x)
    except:
        tmp = np.nan
    return tmp
ACS.iloc[:,1:] = ACS.iloc[:,1:].applymap(convert_dtype)
ACS.iloc[:,1:] = ACS.iloc[:,1:].astype(float)
census_tract_data_selected = census_tract_data_selected.merge(ACS, left_on='TractFIPS', right_on='GEO_ID', how='inner')

# save parquet
census_tract_data_selected.to_crs("EPSG:4326").to_parquet(f'census_tract_data_all_{YEAR}.parquet')