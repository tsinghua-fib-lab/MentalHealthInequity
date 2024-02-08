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
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16

TARGET_METRIC = 'MHLTH_CrudePrev' #['MHLTH_CrudePrev','DEPRESSION_CrudePrev']
RACE_METRIC = 'DP05_0065E' # Black or African American
RACE_METRIC_PERCENT = 'DP05_0065PE' # Black or African American
# RACE_METRIC = 'DP05_0064E' # White
INCOME_METRIC = 'DP03_0088E' #Per capita income (dollars)
HEALTH_INSURANCE_METRIC = 'ACCESS2_CrudePrev'
MEDICAL_ROUTINE_METRIC = 'CHECKUP_CrudePrev'
EDUCATION_METRIC = 'DP02_0067E'

YEAR = 2019
############################################################################################
# Data Path
CENSUS_TRACT_DATA_PATH = Path(f'./census_tract_data_all_with_park_{YEAR}.parquet')
STATE_BOUNDARY_LOC = Path('./Data/Boundary/cb_2022_us_state_20m/cb_2022_us_state_20m.shp')
COUNTY_BOUNDARY_LOC = Path('./Data/Boundary/cb_2019_us_county_5m/cb_2019_us_county_5m.shp')


def load_census_tract_data(*args):
    census_tract_data = gpd.read_parquet(CENSUS_TRACT_DATA_PATH)
    census_tract_data_selecteds = [census_tract_data.loc[census_tract_data['CountyFIPS'].isin(args[i].values())] for i in range(len(args))]
    return census_tract_data_selecteds


def draw_overlay_map(name, census_tract_data_selected, target_metric, demographic_metric, draw_legend=False):
    census_tract_data_selected = census_tract_data_selected.to_crs("EPSG:3857")
    census_tract_data_selected[target_metric] = census_tract_data_selected[target_metric].astype(float)
    census_tract_data_selected[demographic_metric] = census_tract_data_selected[demographic_metric].astype(float)

    fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    census_tract_data_selected.plot(target_metric, ax=ax1, legend=draw_legend, cmap='OrRd', edgecolor='black', linewidth=0.1, alpha=0.8, vmin=5, vmax=25)
    census_tract_data_selected.plot(target_metric, ax=ax1, cmap='OrRd', edgecolor=None, linewidth=0, vmin=5, vmax=25)
    tmp = census_tract_data_selected.sort_values(by=demographic_metric, ascending=False).iloc[:int(census_tract_data_selected.shape[0]/4),:]
    general_boundary = gpd.GeoDataFrame(geometry=gpd.GeoSeries(tmp.unary_union)).set_crs("EPSG:3857")
    general_boundary.plot(ax=ax1, categorical=True, hatch='X', facecolor='None', edgecolor="black", linewidth=2)
    ax1.axis('off')
    plt.tight_layout()
    plt.savefig(f'Fig1D_mental_racial_demo_{name}_{YEAR}.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300, transparent=True)


if __name__ == "__main__":
    NYC = {'Bronx County': '36005', 'Kings County': '36047', 'New York County': '36061', 'Queens County': '36081', 'Richmond County': '36085'}
    COOK = {'Cook County':'17031'} # Chicago
    HARRIS = {'Harris County':'48201'} # Houston
    SEATTLE = {'King County, Washington': '53033'}

    census_tract_data_selecteds = load_census_tract_data(NYC, COOK, HARRIS, SEATTLE)
    county_names = ['New York County', 'Cook County', 'Harris County', 'Los Angeles County', 'King County']
    for i in range(len(census_tract_data_selecteds)):
        draw_overlay_map(county_names[i], census_tract_data_selecteds[i], TARGET_METRIC, RACE_METRIC_PERCENT, draw_legend=False)
