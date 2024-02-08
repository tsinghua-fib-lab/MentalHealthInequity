import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
from pathlib import Path
import pandas as pd
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
sns.set_theme(style="white", palette=None)
import palettable.matplotlib as palmpl
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


mpl.rcParams['hatch.linewidth'] = 0.1
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13

COLOR_LIST = sns.color_palette("Set2")


TARGET_METRIC = 'MHLTH_CrudePrev' #['MHLTH_CrudePrev','DEPRESSION_CrudePrev']
RACE_METRIC = 'DP05_0065E' # Black or African American
RACE_METRIC_PERCENT = 'DP05_0065PE' # Black or African American
INCOME_METRIC = 'DP03_0088E' #Per capita income (dollars)
HEALTH_INSURANCE_METRIC = 'ACCESS2_CrudePrev'
MEDICAL_ROUTINE_METRIC = 'CHECKUP_CrudePrev'
EDUCATION_METRIC = 'DP02_0067E'

YEAR=2019
############################################################################################
# Data Path
CENSUS_TRACT_DATA_PATH = Path(f'./census_tract_data_all_with_park_{YEAR}.parquet')
COUNTY_BOUNDARY_LOC = Path('./Data/Boundary/cb_2019_us_county_5m/cb_2019_us_county_5m.shp')
# Read Data
census_tract_data_selected = gpd.read_parquet(CENSUS_TRACT_DATA_PATH)
census_tract_data_selected[RACE_METRIC] = census_tract_data_selected[RACE_METRIC].astype(float)
census_tract_data_selected['NonWhite'] = 100 - census_tract_data_selected['DP05_0064PE'].astype(float)
census_tract_data_selected[INCOME_METRIC] = census_tract_data_selected[INCOME_METRIC].astype(float)
census_tract_data_selected[HEALTH_INSURANCE_METRIC] = 100 - census_tract_data_selected[HEALTH_INSURANCE_METRIC].astype(float)
census_tract_data_selected[MEDICAL_ROUTINE_METRIC] = census_tract_data_selected[MEDICAL_ROUTINE_METRIC].astype(float)
census_tract_data_selected[EDUCATION_METRIC] = census_tract_data_selected[EDUCATION_METRIC].astype(float)

county_boundary = gpd.read_file(COUNTY_BOUNDARY_LOC,dtype={'NAME':'str'})[[ 'STATEFP', 'COUNTYFP','NAME','geometry']].set_crs("EPSG:4326",allow_override=True).to_crs("EPSG:3857")
county_boundary.loc[:,'CountyFIPS'] = county_boundary['STATEFP'] + county_boundary['COUNTYFP']
county_boundary = county_boundary.drop(['STATEFP','COUNTYFP'], axis=1)

####################################################################
# Cal gini
census_tract_data_selected_cp = census_tract_data_selected.copy()
county_list = census_tract_data_selected_cp['CountyFIPS'].unique().tolist()
county_gini_list = []
R2_list = []
spearmanr_list = []
n = 10000
for county in tqdm(county_list):
    tmp = census_tract_data_selected_cp.loc[census_tract_data_selected_cp['CountyFIPS'] == county].sort_values(by=RACE_METRIC_PERCENT)
    tmp = tmp.loc[tmp['MHLTH_CrudePrev'].isna() == False]
    if tmp.shape[0] <= 1:
        county_gini_list.append(-1)
        R2_list.append(-1)
        spearmanr_list.append(-1)
        continue

    race = np.concatenate((np.array(0).reshape(-1),tmp[RACE_METRIC].to_numpy()))
    if race.sum() <=0:
        county_gini_list.append(-2)
        R2_list.append(-2)
        spearmanr_list.append(-2)
        continue
    race /= race.sum()
    race = np.cumsum(race)

    mental_health = np.concatenate((np.array(0).reshape(-1),tmp[TARGET_METRIC].to_numpy()/100 * tmp['TotalPopulation'].to_numpy()))
    mental_health /= mental_health.sum()
    mental_health = np.cumsum(mental_health)

    x_interp = np.linspace(0, 1, n)
    interp_result = np.interp(x=x_interp, xp=race, fp=mental_health)
    gini = 2*np.abs((x_interp - interp_result)).sum() / n
    county_gini_list.append(gini)

    m = sm.OLS.from_formula('MHLTH_CrudePrev ~ DP05_0065PE',data=tmp)
    results = m.fit()
    R2_list.append(results.rsquared_adj)

    spearmanr_list.append(spearmanr(tmp['MHLTH_CrudePrev'],tmp['DP05_0065PE'])[0])

inequity_df = pd.DataFrame({'CountyFIPS':county_list,'GINI':county_gini_list,'R2':R2_list,'SpearmanR':spearmanr_list})
inequity_df = county_boundary.merge(inequity_df, left_on='CountyFIPS',right_on='CountyFIPS',how='right')  # 采用right join


# Calculate counties with inequity
county_black_rate = pd.read_csv(f'./Data/ACS/County/DP05-County/ACSDP5Y2019.DP05-Data.csv',encoding_errors='backslashreplace',dtype={'CountyFIPS':str})[['GEO_ID', RACE_METRIC_PERCENT]].iloc[1:,:]
county_black_rate['CountyFIPS'] = county_black_rate['GEO_ID'].str.replace('0500000US','')
county_black_rate = county_black_rate.drop('GEO_ID',axis=1)
county_black_rate = county_black_rate.dropna()
county_black_rate = county_black_rate.rename(columns={RACE_METRIC_PERCENT: 'CountyAvgBlackRate'})

county_mh_rate = pd.read_csv(f'./Data/PLACES/PLACES__County_Data__GIS_Friendly_Format___{YEAR+2}_release.csv',dtype={'CountyFIPS':str})[['CountyFIPS','MHLTH_CrudePrev']].rename(columns={'MHLTH_CrudePrev': 'CountyAvgMentalHealthRate'})
county_mh_rate = county_mh_rate.dropna()

tmp = census_tract_data_selected[['TractFIPS', 'CountyFIPS',TARGET_METRIC, RACE_METRIC_PERCENT]].copy()
tmp = tmp.merge(county_black_rate, on='CountyFIPS', how='inner')
tmp = tmp.merge(county_mh_rate, on='CountyFIPS', how='inner')
tmp = tmp.dropna()
tmp['IsMoreBlack'] = tmp[RACE_METRIC_PERCENT] > tmp['CountyAvgBlackRate'].astype(float)
tmp['IsMoreMentalHealth'] = tmp[TARGET_METRIC] > tmp['CountyAvgMentalHealthRate'].astype(float)

county_list = tmp['CountyFIPS'].unique().tolist()
inequity_county = 0
inequity_tract = 0
for county in county_list:
    tmp2 = tmp.loc[tmp['CountyFIPS'] == county]
    if tmp2.shape[0] <= 1:
        continue

    x = tmp2.loc[tmp2['IsMoreBlack'] & tmp2['IsMoreMentalHealth']].shape[0]
    if x > 0 :
        inequity_county += 1
    inequity_tract += x
print('#County with inequity: ' + str(inequity_county) + ' , Percentage: ' + str(inequity_county / len(county_list)))
print('# Tracts with More Black Population: ' + str(tmp.loc[tmp['IsMoreBlack']].shape[0]))
print('#Census Tracts with inequity: ' + str(inequity_tract) + ' , Percentage: ' + str(inequity_tract / tmp.loc[tmp['IsMoreBlack']].shape[0]))



####################################################################
# Fig 1A

inequity_df['GINIforPlot'] = inequity_df['GINI']
inequity_df.loc[inequity_df['GINIforPlot'] < 0,'GINIforPlot'] = None
inequity_df['GINIforPlot'] = inequity_df['GINIforPlot'].astype(float)
inequity_df['MoreInequity'] = inequity_df['GINI'] >= 0.4

my_color_lists = ['#EBE730','#F8A93B','#E97A53','#CD4E74','#B93685','#422D88']
my_color_map = LinearSegmentedColormap.from_list("Jure", my_color_lists, N=256)

fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(10,6))
inequity_df.plot('GINIforPlot', ax=ax1,cmap=my_color_map, edgecolor='black', linewidth=0, alpha=0.9,legend=True, missing_kwds={'color': 'lightgrey'}, vmax=1, vmin=0)

ax1.axis('off')
plt.xlim(-1.4030e+07,-7.347e+06)
plt.ylim(2.638e+06,6.493e+06)
plt.savefig(f'Fig1A_MainLand_{YEAR}.pdf', dpi=300, bbox_inches='tight', transparent=True)

fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(10,6))
inequity_df.plot('GINIforPlot', ax=ax1,cmap=my_color_map,edgecolor='black',linewidth=0, alpha=0.9,legend=True,missing_kwds={'color': 'lightgrey'}, vmax=1, vmin=0)
ax1.axis('off')
plt.xlim(-2.0229e+07,-1.4431e+07)
plt.ylim(6.640e+06,1.1608e+07)
plt.savefig(f'Fig1A_Alaska_{YEAR}.pdf', dpi=300, bbox_inches='tight', transparent=True)


fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(10,6))
inequity_df.plot('GINIforPlot', ax=ax1,cmap=my_color_map,edgecolor='black',linewidth=0, alpha=0.9,legend=True,missing_kwds={'color': 'lightgrey'}, vmax=1, vmin=0)
ax1.axis('off')
plt.xlim(-1.78720e+07,-1.72184e+07)
plt.ylim(2.1195e+06,2.5757e+06)
plt.savefig(f'Fig1A_Hawaii_{YEAR}.pdf', dpi=300, bbox_inches='tight', transparent=True)


####################################################################
# Dist of Gini
print('High Inequity county:')
print((inequity_df['MoreInequity']).sum())
gini = inequity_df.loc[inequity_df['GINI']>0]['GINI'].to_numpy()

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(gini, ax=ax, bins=20, kde=True, stat='probability', color=colors.to_hex(COLOR_LIST[0]))
ax.set_xlabel('Gini Index', size=30)
ax.set_ylabel('Probability Density', size=27)
ax.tick_params(axis='y', labelsize=28)
ax.tick_params(axis='x', labelsize=24)
plt.tight_layout()
plt.savefig(f'SI_Fig1_GiniPDF_{YEAR}.pdf', dpi=300, bbox_inches='tight')


####################################################################
# Fig 1B
county_mental_health_race = census_tract_data_selected_cp[['CountyFIPS','TractFIPS','MHLTH_CrudePrev',RACE_METRIC_PERCENT, 'DP05_0064PE']]
def maj_white_or_maj_black(x):
    if x[RACE_METRIC_PERCENT] >= 50:
        return 'Majority Black'
    elif x['DP05_0064PE'] >= 50:
        return 'Majority White'
    else:
        return 'Others'
county_mental_health_race['Race'] = county_mental_health_race.apply(maj_white_or_maj_black, axis=1)
county_mental_health_race = county_mental_health_race.loc[county_mental_health_race['Race'] != 'Others']

fig, ax = plt.subplots(figsize=(10, 8))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
COLOR_LIST_VIOLIN = [sns.color_palette("Set2")[2], sns.color_palette("Set2")[1]]
g = sns.violinplot(county_mental_health_race, x='Race', y='MHLTH_CrudePrev', ax=ax, palette=COLOR_LIST_VIOLIN, inner_kws=dict(box_width=10, whis_width=2, color='#606060'))  # 修改为violin plot
ax.set_ylabel('Prevalence of Suboptimal Mental Health', size=30)
ax.set_xlabel('')
ori_y_ticks = ax.get_yticks()
new_y_ticks = [str(i) + '%' for i in ori_y_ticks]
ax.set_yticks(ori_y_ticks,new_y_ticks)
ax.set_ylim(bottom=4,top=31)
ax.tick_params(axis='y', labelsize=24)
ax.tick_params(axis='x', labelsize=28)

plt.tight_layout()
plt.savefig(f'Fig1B_MacroRiskWithRace_{YEAR}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)



###############################################################################################
# Fig 1C-F
def draw_single_dim(df, model_result, x_name, y_name, title=None, draw_y_title=False, draw_x_title=False, draw_y_axis=False, xvline=None, mannual_xlim=None):
    x = sm.add_constant(np.linspace(df[x_name].min(),
                                    df[x_name].max(), 100))
    pred_ols = model_result.get_prediction(x)
    iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    iv_u = pred_ols.summary_frame()["obs_ci_upper"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(df[x_name], df[y_name], ".",
            label="data", color='#FEECEB')
    ax.plot(x[:, 1], pred_ols.predicted, color='#B83B13', linewidth=3)
    if draw_x_title:
        if x_name == RACE_METRIC_PERCENT:
            ax.set_xlabel('Proportion of Black Population', size=22)
        else:
            ax.set_xlabel(x_name, size=24)
    else:
        ax.set_xlabel('')

    if draw_y_title:
        if y_name == 'MHLTH_CrudePrev':
            ax.set_ylabel('Prevalence of Suboptimal\nMental Health', size=22)
        else:
            ax.set_ylabel(y_name, size=24)
    else:
        ax.set_ylabel('')

    ori_y_ticks = ax.get_yticks()
    new_y_ticks = [str(int(i)) + '%' for i in ori_y_ticks]
    ax.set_yticks(ori_y_ticks, new_y_ticks)

    ori_x_ticks = ax.get_xticks()
    new_x_ticks = [str(int(i)) + '%' for i in ori_x_ticks]
    ax.set_xticks(ori_x_ticks, new_x_ticks)

    ax.set_ylim(5, 30)
    if not mannual_xlim:
        ax.set_xlim(0, 100)
    else:
        ax.set_xlim(mannual_xlim[0], mannual_xlim[1])

    if xvline:
        ax.axvline(x=xvline, color='black', linestyle='--', linewidth=1)

    if draw_y_axis == False:
        ax.set_yticks([])
    else:
        ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    if title:
        ax.set_title(title, size=22)
    plt.tight_layout()
    plt.savefig('Fig1C_' + y_name + '_' + x_name + f'_{title}.pdf', dpi=300, bbox_inches='tight')

    print(pearsonr(df[x_name], df[y_name]))
    #plt.show()


all_county_data = census_tract_data_selected.copy()
all_county_data[RACE_METRIC+'_Deciles'] = pd.qcut(all_county_data[RACE_METRIC], 10, labels=False)
all_county_data = all_county_data.reset_index(drop=True)
all_county_data[RACE_METRIC_PERCENT] = all_county_data[RACE_METRIC_PERCENT].astype(float)
all_county_data[TARGET_METRIC] = all_county_data[TARGET_METRIC].astype(float)
all_county_data = all_county_data.dropna(subset=[RACE_METRIC_PERCENT, TARGET_METRIC])

nyc = {'Bronx County': '36005', 'Kings County': '36047', 'New York County': '36061',
                      'Queens County': '36081', 'Richmond County': '36085'}
nyc_county_data = all_county_data.loc[all_county_data['CountyFIPS'].isin(nyc.values())]

cook = {'Cook County, Illinois': '17031'}
cook_county_data = all_county_data.loc[all_county_data['CountyFIPS'].isin(cook.values())]

bronx = {'Bronx County, New York': '36005'}
bronx_county_data = all_county_data.loc[all_county_data['CountyFIPS'].isin(bronx.values())]

harris = {'Harris County, Texas': '48201'}
harris_county_data = all_county_data.loc[all_county_data['CountyFIPS'].isin(harris.values())]

seattle = {'King County, Washington': '53033'}
seattle_county_data = all_county_data.loc[all_county_data['CountyFIPS'].isin(seattle.values())]


nyc_model = sm.OLS(nyc_county_data['MHLTH_CrudePrev'], sm.add_constant(nyc_county_data[['DP05_0065PE']]))
nyc_results = nyc_model.fit()
nyc_results.summary()
print(nyc_results.params)
nyc_xvline = nyc_county_data.sort_values(by=RACE_METRIC_PERCENT, ascending=False).iloc[int(nyc_county_data.shape[0]/4)][RACE_METRIC_PERCENT]
nyc_gini = np.around(inequity_df.loc[inequity_df['CountyFIPS']==list(nyc.values())[0]]['GINI'].iloc[0],3)
nyc_pearson = np.around(pearsonr(nyc_county_data['MHLTH_CrudePrev'], nyc_county_data['DP05_0065PE'])[0], 3)
draw_single_dim(nyc_county_data, nyc_results, RACE_METRIC_PERCENT, TARGET_METRIC, title=f"New York City, New York\n (Gini: {nyc_gini:.3f}, Pearson R: {nyc_pearson:.3f})", draw_y_title=True, draw_x_title=False, draw_y_axis=True, xvline=nyc_xvline)

cook_model = sm.OLS(cook_county_data['MHLTH_CrudePrev'], sm.add_constant(cook_county_data[['DP05_0065PE']]))
cook_results = cook_model.fit()
cook_results.summary()
print(cook_results.params)
cook_xvline = cook_county_data.sort_values(by=RACE_METRIC_PERCENT, ascending=False).iloc[int(cook_county_data.shape[0]/4)][RACE_METRIC_PERCENT]
cook_gini = np.around(inequity_df.loc[inequity_df['CountyFIPS']==list(cook.values())[0]]['GINI'].iloc[0],3)
cook_pearson = np.around(pearsonr(cook_county_data['MHLTH_CrudePrev'], cook_county_data['DP05_0065PE'])[0], 3)
draw_single_dim(cook_county_data, cook_results, RACE_METRIC_PERCENT, TARGET_METRIC, title=f"Cook County, Illinois\n (Gini: {cook_gini:.3f}, Pearson R: {cook_pearson:.3f})", draw_y_title=True, draw_x_title=False, draw_y_axis=True, xvline=cook_xvline)

bronx_model = sm.OLS(bronx_county_data['MHLTH_CrudePrev'], sm.add_constant(bronx_county_data[['DP05_0065PE']]))
bronx_results = bronx_model.fit()
bronx_results.summary()
print(bronx_results.params)
bronx_xvline = bronx_county_data.sort_values(by=RACE_METRIC_PERCENT, ascending=False).iloc[int(bronx_county_data.shape[0]/4)][RACE_METRIC_PERCENT]
bronx_gini = np.around(inequity_df.loc[inequity_df['CountyFIPS']==list(bronx.values())[0]]['GINI'].iloc[0],3)
bronx_pearson = np.around(pearsonr(bronx_county_data['MHLTH_CrudePrev'], bronx_county_data['DP05_0065PE'])[0], 3)
draw_single_dim(bronx_county_data, bronx_results, RACE_METRIC_PERCENT, TARGET_METRIC, title=f"Bronx County, New York\n (Gini: {bronx_gini:.3f}, Pearson R: {bronx_pearson:.3f})", draw_y_title=True, draw_x_title=False, draw_y_axis=True, xvline=bronx_xvline)

harris_model = sm.OLS(harris_county_data['MHLTH_CrudePrev'], sm.add_constant(harris_county_data[['DP05_0065PE']]))
harris_results = harris_model.fit()
harris_results.summary()
print(harris_results.params)
harris_xvline = harris_county_data.sort_values(by=RACE_METRIC_PERCENT, ascending=False).iloc[int(harris_county_data.shape[0]/4)][RACE_METRIC_PERCENT]
harris_gini = np.around(inequity_df.loc[inequity_df['CountyFIPS']==list(harris.values())[0]]['GINI'].iloc[0],3)
harris_pearson = np.around(pearsonr(harris_county_data['MHLTH_CrudePrev'], harris_county_data['DP05_0065PE'])[0], 3)
draw_single_dim(harris_county_data, harris_results, RACE_METRIC_PERCENT, TARGET_METRIC, title=f"Harris County, Texas\n (Gini: {harris_gini:.3f}, Pearson R: {harris_pearson:.3f})", draw_y_title=True, draw_x_title=False, draw_y_axis=True, xvline=harris_xvline)

seattle_model = sm.OLS(seattle_county_data['MHLTH_CrudePrev'], sm.add_constant(seattle_county_data[['DP05_0065PE']]))
seattle_results = seattle_model.fit()
seattle_results.summary()
print(seattle_results.params)
seattle_xvline = seattle_county_data.sort_values(by=RACE_METRIC_PERCENT, ascending=False).iloc[int(seattle_county_data.shape[0]/4)][RACE_METRIC_PERCENT]
seattle_gini = np.around(inequity_df.loc[inequity_df['CountyFIPS']==list(seattle.values())[0]]['GINI'].iloc[0],3)
seattle_pearson = np.around(pearsonr(seattle_county_data['MHLTH_CrudePrev'], seattle_county_data['DP05_0065PE'])[0], 3)
draw_single_dim(seattle_county_data, seattle_results, RACE_METRIC_PERCENT, TARGET_METRIC, title=f"King County, Washington\n (Gini: {seattle_gini:.3f}, Pearson R: {seattle_pearson:.3f})", draw_y_title=True, draw_x_title=False, draw_y_axis=True, xvline=seattle_xvline, mannual_xlim=(0,40))