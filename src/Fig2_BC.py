import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
from pathlib import Path
import pandas as pd
import geopandas as gpd
from scipy.stats import pearsonr, spearmanr
from datetime import datetime
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import matplotlib.colors as colors
from scipy.stats import f_oneway
from sklearn.decomposition import PCA

sns.set_theme(style="white", palette=None)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13

YEAR = 2019
INTERP_POINTS = 10000


COLOR_LIST = [sns.color_palette("Set2")[2], sns.color_palette("Set2")[1]]

TARGET_METRIC = 'MHLTH_CrudePrev' #['MHLTH_CrudePrev','DEPRESSION_CrudePrev']
TARGET_METRIC_LOG = 'MHLTH_CrudePrev_LOG'
RACE_METRIC = 'DP05_0065PE' # Black or African American
RACE_METRIC_DECILES = 'DP05_0065PE_Deciles'
INCOME_METRIC = 'DP03_0088E' #Per capita income (dollars)
INCOME_METRIC_DECILES = 'DP03_0088E_Deciles'
HEALTH_INSURANCE_METRIC = 'ACCESS2_CrudePrev'
HEALTH_INSURANCE_METRIC_DECILES = 'ACCESS2_CrudePrev_Deciles'
MEDICAL_ROUTINE_METRIC = 'CHECKUP_CrudePrev'
MEDICAL_ROUTINE_METRIC_DECILES = 'CHECKUP_CrudePrev_Deciles'
EDUCATION_METRIC = 'DP02_0067PE'
EDUCATION_METRIC_DECILES = 'DP02_0067PE_Deciles'

quantile_num = 10

def process_data():
    ############################################################################################
    CENSUS_TRACT_DATA_LOC = Path(f'./census_tract_data_all_with_park_with_landuse_{YEAR}.parquet')

    census_tract_data = gpd.read_parquet(CENSUS_TRACT_DATA_LOC)

    census_tract_data[TARGET_METRIC_LOG] = np.log(census_tract_data[TARGET_METRIC])
    census_tract_data[RACE_METRIC] = census_tract_data[RACE_METRIC].astype(float)
    census_tract_data[RACE_METRIC_DECILES] = pd.qcut(census_tract_data[RACE_METRIC], 10, labels=False)
    census_tract_data['NonWhite'] = 1 - census_tract_data['DP05_0064PE'].astype(float)

    census_tract_data[INCOME_METRIC] = census_tract_data[INCOME_METRIC].astype(float)
    census_tract_data[INCOME_METRIC_DECILES] = pd.qcut(census_tract_data[INCOME_METRIC], 10, labels=False)
    census_tract_data[HEALTH_INSURANCE_METRIC] = 1 - census_tract_data[HEALTH_INSURANCE_METRIC].astype(float)
    census_tract_data[HEALTH_INSURANCE_METRIC_DECILES] = pd.qcut(census_tract_data[HEALTH_INSURANCE_METRIC], 10,
                                                                 labels=False)
    census_tract_data[MEDICAL_ROUTINE_METRIC] = census_tract_data[MEDICAL_ROUTINE_METRIC].astype(float)
    census_tract_data[MEDICAL_ROUTINE_METRIC_DECILES] = pd.qcut(census_tract_data[MEDICAL_ROUTINE_METRIC], 10,
                                                                labels=False)
    census_tract_data[EDUCATION_METRIC] = census_tract_data[EDUCATION_METRIC].astype(float)
    census_tract_data[EDUCATION_METRIC_DECILES] = pd.qcut(census_tract_data[EDUCATION_METRIC], 10, labels=False)

    census_tract_data = census_tract_data[~census_tract_data['GreenSpaceBufferedOrigin'].isin(['--'])]
    census_tract_data = census_tract_data[~census_tract_data['GreenSpaceBufferedPopWeighted'].isin(['--'])]
    census_tract_data = census_tract_data[~census_tract_data['GreenSpaceOrigin'].isin(['--'])]
    census_tract_data = census_tract_data[~census_tract_data['GreenSpacePopWeighted'].isin(['--'])]

    census_tract_data['GreenSpaceBufferedPopWeighted'] = census_tract_data['GreenSpaceBufferedPopWeighted'].astype(
        float)
    census_tract_data['GreenSpaceBufferedOrigin'] = census_tract_data['GreenSpaceBufferedOrigin'].astype(float)
    census_tract_data['GreenSpacePopWeighted'] = census_tract_data['GreenSpacePopWeighted'].astype(float)
    census_tract_data['GreenSpaceOrigin'] = census_tract_data['GreenSpaceOrigin'].astype(float)
    census_tract_data = census_tract_data.dropna(subset=[RACE_METRIC])

    return census_tract_data




def draw_race_green_space_mental_health_relative_control_variable_regression_modify_violin(df, exog=None, percent_num=4, legend=True):
    percent_lut = {}
    for i in range(percent_num):
        percent_lut[i] = str(int(100 / percent_num * i)) + '-' + str(int(100 / percent_num * (i + 1))) + '%'
    tmp_df = df
    tmp_df['black_percentile'] = tmp_df[RACE_METRIC].map(lambda x: int(x / (1 / percent_num)) if x < 1 else percent_num - 1)


    ACS_selection = pd.read_csv('./Data/ACS/ACS_control_variables.csv')['Column Name'].to_list()
    tmp_df = tmp_df[ACS_selection + [TARGET_METRIC,RACE_METRIC,'GreenSpaceOrigin','GreenSpacePopWeighted','GreenSpaceBufferedOrigin','GreenSpaceBufferedPopWeighted',
                                     'ParkSize', 'BufferedParkSize','ParkPercentage','BufferedParkPercentage',
                                     'black_percentile','CHECKUP_CrudePrev','ACCESS2_CrudePrev','CountyFIPS','TotalPopulation']]

    county_avg_mhng = tmp_df.groupby('CountyFIPS').apply(
        lambda x: (x[TARGET_METRIC] * x['TotalPopulation']).sum() / x['TotalPopulation'].sum()).to_frame(
        name='CountyAvgMHNG').reset_index()
    tmp_df = tmp_df.merge(county_avg_mhng, on='CountyFIPS', how='inner')
    tmp_df['RelativeMHNG'] = tmp_df[TARGET_METRIC] / tmp_df['CountyAvgMHNG']
    tmp_df = tmp_df.dropna(subset=['RelativeMHNG'])
    tmp_df = tmp_df.drop(columns=['CountyAvgMHNG', 'CountyFIPS',TARGET_METRIC, 'TotalPopulation'], axis=1)

    for i in range(9):
        tmp_df.loc[:,'Edu_'+str(i).zfill(2)] = tmp_df.loc[:, f'DP02_006{i}PE']
        tmp_df = tmp_df.drop(columns=[f'DP02_006{i}PE'])

    for i in range(15):
        if i != 13:
            tmp_df.loc[:,'Age_'+str(i).zfill(2)] = tmp_df.loc[:, f'DP05_00{str(i+5).zfill(2)}PE']
            tmp_df = tmp_df.drop(columns=[f'DP05_00{str(i+5).zfill(2)}PE'])
        else:
            tmp_df.loc[:,'Age_'+str(i).zfill(2)] = tmp_df.loc[:, f'DP05_00{str(i+5).zfill(2)}E']
            tmp_df = tmp_df.drop(columns=[f'DP05_00{str(i+5).zfill(2)}E'])
    tmp_df = tmp_df.rename({'DP05_0024PE':'Age_15'}, axis=1)


    for i in range(4):
        tmp_df.loc[:,'Veh_'+str(i).zfill(2)] = tmp_df.loc[:, f'DP04_00{str(i+58).zfill(2)}PE']
        tmp_df = tmp_df.drop(columns=[f'DP04_00{str(i+58).zfill(2)}PE'])

    tmp_df = tmp_df.rename({'DP03_0007PE': 'Emp',
                    'DP05_0003PE': 'Sex',
                    'DP03_0088E':'Inc_00', 'DP03_0062E':"Inc_01",'DP03_0063E':"Inc_02", 'DP03_0086E':"Inc_03", 'DP03_0087E':"Inc_04", 'DP03_0119PE': "Inc_05",'DP03_0128PE':"Inc_06", 'DP04_0089E':"Inc_07",
                    'DP02_0006PE': "Mrg_00", 'DP02_0010PE': "Mrg_01", 'DP02_0026PE':'Mrg_02','DP02_0027PE':'Mrg_03','DP02_0028PE': "Mrg_04", 'DP02_0029PE': "Mrg_05", 'DP02_0030PE': "Mrg_06", 'DP02_0032PE':'Mrg_07','DP02_0033PE':'Mrg_08', 'DP02_0034PE': "Mrg_09", 'DP02_0035PE': "Mrg_10", 'DP02_0036PE': "Mrg_11",
                    }, axis=1)
    tmp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    tmp_df.replace(['--'], np.nan, inplace=True)
    tmp_df = tmp_df.dropna()
    tmp_df['Inc_00'] = tmp_df['Inc_00'] / tmp_df['Inc_00'].max() # Per capita income (dollars)
    tmp_df['Inc_01'] = tmp_df['Inc_01'] / tmp_df['Inc_01'].max() # Median household income (dollars)
    tmp_df['Inc_02'] = tmp_df['Inc_02'] / tmp_df['Inc_02'].max() # Mean household income (dollars)
    tmp_df['Inc_03'] = tmp_df['Inc_03'] / tmp_df['Inc_03'].max() # Median family income (dollars)
    tmp_df['Inc_04'] = tmp_df['Inc_04'] / tmp_df['Inc_04'].max() # Mean family income (dollars)
    tmp_df['Inc_07'] = tmp_df['Inc_07'] / tmp_df['Inc_07'].max() # Owner-occupied units!!Median (dollars)


    main_exog_list = ['GreenSpaceOrigin','GreenSpacePopWeighted','GreenSpaceBufferedOrigin','GreenSpaceBufferedPopWeighted','ParkSize', 'BufferedParkSize','ParkPercentage','BufferedParkPercentage',]
    control_exog_list = (['Emp','Sex','CHECKUP_CrudePrev','ACCESS2_CrudePrev'] + [f'Edu_{str(i).zfill(2)}' for i in range(8)] + [f'Age_{str(i).zfill(2)}' for i in range(16)] + [f'Inc_{str(i).zfill(2)}' for i in range(8)]
                         + [f'Mrg_{str(i).zfill(2)}' for i in range(12)] + [f'Veh_{str(i).zfill(2)}' for i in range(4)])

    if exog is not None:
        main_exog_index = main_exog_list.index(exog)
    else:
        main_exog_index = 3
    exog_list = [main_exog_list[main_exog_index]] + control_exog_list

    # PCA
    pca = PCA(n_components=0.95, svd_solver='full')
    pca.fit(tmp_df[exog_list[1:]])
    #pca.transform(tmp_df[exog_list[1:]])

    models = {}
    results = {}

    param_df = {'Independent Variable':[], 'Racial Groups':[],'VIF':[], 'VIF After PCA':[],'Beta':[],'95% CI':[],'P-value':[],'R2':[]}
    for i in range(percent_num):
        tmp_tmp_df = tmp_df.loc[tmp_df['black_percentile']==i].reset_index(drop=True)

        X = pca.transform(tmp_tmp_df[exog_list[1:]])
        X = np.concatenate([tmp_tmp_df[exog_list[0]].to_numpy().reshape(-1,1), X], axis=1)
        tmp_tmp_tmp_df = pd.DataFrame(X, columns=[exog] + [f'PC_{str(i).zfill(2)}' for i in range(X.shape[1]-1)])
        model = sm.OLS(tmp_tmp_df['RelativeMHNG'], sm.add_constant(tmp_tmp_tmp_df))
        result = model.fit()
        models[i] = model
        results[i] = result

        param_df['Independent Variable'].append(exog)
        param_df['Racial Groups'].append(percent_lut[i])
        param_df['Beta'].append(result.params[exog])
        ci = result.conf_int().loc[exog].to_numpy()
        param_df['95% CI'].append(f'[{ci[0]:.3f}, {ci[1]:.3f}]')
        param_df['P-value'].append(result.pvalues[exog])
        param_df['R2'].append(result.rsquared)


        print(result.params[main_exog_index])
        print('VIF')
        vif = [variance_inflation_factor(tmp_tmp_df[exog_list], 0)]
        print(vif)
        param_df['VIF'].append(vif[0])
        print('VIF After PCA')
        vif = [variance_inflation_factor(tmp_tmp_tmp_df, 0)]
        print(vif)
        param_df['VIF After PCA'].append(vif[0])

    fig, ax = plt.subplots(figsize=(8, 6))
    black_percent_list = []
    green_percent_list = []
    relative_mhng_list = []

    least_green_criteria = tmp_df[exog].quantile(0.1)
    most_green_criteria = tmp_df[exog].quantile(0.9)

    print(least_green_criteria)
    print(most_green_criteria)
    for i in range(percent_num):
        tmp_tmp_df = tmp_df.loc[tmp_df['black_percentile'] == i].reset_index(drop=True)

        X = pca.transform(tmp_tmp_df[exog_list[1:]])
        X = np.concatenate([tmp_tmp_df[exog_list[0]].to_numpy().reshape(-1,1), X], axis=1)
        tmp_tmp_tmp_df = pd.DataFrame(X, columns=[exog] + [f'PC_{str(i).zfill(2)}' for i in range(X.shape[1]-1)])

        tmp_tmp_tmp_df_least = sm.add_constant(tmp_tmp_tmp_df)
        tmp_tmp_tmp_df_most = sm.add_constant(tmp_tmp_tmp_df)

        tmp_tmp_tmp_df_least[exog] = least_green_criteria
        tmp_tmp_tmp_df_most[exog] = most_green_criteria

        pred_ols_least = results[i].get_prediction(tmp_tmp_tmp_df_least)
        pred_ols_most = results[i].get_prediction(tmp_tmp_tmp_df_most)
        black_percent_list.append([percent_lut[i]] * len(pred_ols_least.predicted) * 2)



        green_percent_list.append(['Lowest Decile'] * len(pred_ols_least.predicted) + ['Highest Decile'] * len(pred_ols_most.predicted))

        relative_mhng_list.append(list(pred_ols_least.predicted) + list(pred_ols_most.predicted))

    box_plot_df = pd.DataFrame({'Black Percentage': np.concatenate(black_percent_list),
                                'Green Space': np.concatenate(green_percent_list),
                                'Relative MHNG': np.concatenate(relative_mhng_list)})
    sns.violinplot(x='Black Percentage', y='Relative MHNG',split=True, hue='Green Space', data=box_plot_df, palette=COLOR_LIST, ax=ax, gap=0.1, inner='quartile')
    if legend:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), ncol=1, title=None, frameon=True, fontsize=12)
    else:
        plt.legend([], [], frameon=False)

    ax.set_ylabel('Relative Prevalence of\nSuboptimal Mental Health', size=20)
    ax.set_xlabel('Percentage of Black Population', size=20)
    if exog == 'GreenSpaceOrigin':
        ax.set_title('Proportion of Green Land Cover', size=20)
    elif exog == 'GreenSpacePopWeighted':
        ax.set_title('Proportion of Green Land Cover Weighted by Population', size=20)
    elif exog == 'GreenSpaceBufferedOrigin':
        ax.set_title('Proportion of Buffered Green Land Cover', size=20)
    elif exog == 'GreenSpaceBufferedPopWeighted':
        ax.set_title('Proportion of Buffered Green Land Cover Weighted by Population', size=18)
    elif exog == 'ParkSize':
        ax.set_title('Park Size', size=20)
    elif exog == 'BufferedParkSize':
        ax.set_title('Buffered Park Size', size=20)
    elif exog == 'ParkPercentage':
        ax.set_title('Proportion of Park Area', size=20)
    elif exog == 'BufferedParkPercentage':
        ax.set_title('Proportion of Buffered Park Area', size=20)
    else:
        ax.set_title(exog, size=20)

    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=18)

    plt.tight_layout()
    plt.savefig(f'./Fig2B_regression_' + str(exog) + f'_Rela_MentalHealth_{YEAR}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.3)

    param_df = pd.DataFrame(param_df)
    param_df.to_csv(f'./Fig2B_regression_' + str(exog) + f'_Rela_MentalHealth_{YEAR}.csv', index=False)


if __name__ == "__main__":
    census_tract_data = process_data()

    draw_race_green_space_mental_health_relative_control_variable_regression_modify_violin(census_tract_data, exog='GreenSpaceOrigin', percent_num=4, legend=True)  # 加入更多控制变量并利用PCA减小共线性
    draw_race_green_space_mental_health_relative_control_variable_regression_modify_violin(census_tract_data, exog='GreenSpacePopWeighted', percent_num=4, legend=True)
    draw_race_green_space_mental_health_relative_control_variable_regression_modify_violin(census_tract_data, exog='GreenSpaceBufferedOrigin', percent_num=4, legend=True)
    draw_race_green_space_mental_health_relative_control_variable_regression_modify_violin(census_tract_data, exog='GreenSpaceBufferedPopWeighted', percent_num=4, legend=False)
    draw_race_green_space_mental_health_relative_control_variable_regression_modify_violin(census_tract_data, exog='ParkSize', percent_num=4, legend=True)
    draw_race_green_space_mental_health_relative_control_variable_regression_modify_violin(census_tract_data, exog='BufferedParkSize', percent_num=4, legend=True)
    draw_race_green_space_mental_health_relative_control_variable_regression_modify_violin(census_tract_data, exog='ParkPercentage', percent_num=4, legend=True)
    draw_race_green_space_mental_health_relative_control_variable_regression_modify_violin(census_tract_data, exog='BufferedParkPercentage', percent_num=4, legend=False)

