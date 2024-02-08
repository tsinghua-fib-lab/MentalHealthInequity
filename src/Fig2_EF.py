import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
from pathlib import Path
import pandas as pd
import geopandas as gpd
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import shapely
sns.set_theme(style="white", palette=None)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13

COLOR_LIST = [sns.color_palette("Set2")[2], sns.color_palette("Set2")[1]]
YEAR=2019
TARGET_METRIC = 'MHLTH_CrudePrev'
RACE_METRIC = 'DP05_0065PE'
INCOME_METRIC = 'DP03_0088E'
############################################################################################
# Data Path
TRACT_VISIT_LOC = Path(f'./tract_visit_all_US_within_county_{YEAR}.parquet')
PARK_TRACT_VISIT_LOC = Path(f'./park_visit_all_US_within_county_{YEAR}.parquet')
CENSUS_TRACT_DATA_LOC = Path(f'./census_tract_data_all_with_park_with_landuse_{YEAR}.parquet')
############################################################################################


def load_and_process_data_tract():
    census_tract_data = gpd.read_parquet(CENSUS_TRACT_DATA_LOC)

    tract_visit = pd.read_parquet(TRACT_VISIT_LOC).dropna().reset_index(drop=True)
    census_tract_data = census_tract_data.merge(tract_visit,on='TractFIPS',how='inner').reset_index(drop=True)
    park_tract_visit = pd.read_parquet(PARK_TRACT_VISIT_LOC).dropna().reset_index(drop=True)

    return census_tract_data.copy(deep=True), tract_visit, park_tract_visit



def draw_tract_visit_mental_health_relative_control_variable_discrete_regression_modify_violin(df, exog=None, percent_num=4, xlim=None, ylim=None, legend=True):
    percent_lut = {}
    for i in range(percent_num):
        percent_lut[i] = str(int(100 / percent_num * i)) + '-' + str(int(100 / percent_num * (i + 1))) + '%'
    # df['black_percentile'] = df[RACE_METRIC].map(lambda x: percent_lut[int(x / (1 / percent_num))] if x < 1 else percent_lut[percent_num - 1])
    df['black_percentile'] = df[RACE_METRIC].map(lambda x: int(x / (1 / percent_num)) if x < 1 else percent_num - 1)

    ACS_selection = pd.read_csv('./Data/ACS/ACS_control_variables.csv')['Column Name'].to_list()
    df = df[ACS_selection + [TARGET_METRIC,RACE_METRIC,'GreenSpaceBufferedPopWeighted', 'black_percentile','CHECKUP_CrudePrev','ACCESS2_CrudePrev',
                             'CountyFIPS','TotalPopulation',
                             'weekly_park_visit_total_per_people','weekly_park_visit_time_total_per_people',
                             'weekly_park_visit_strong_segregation_per_people',
                             'weekly_park_visit_weak_segregation_per_people',
                             'weekly_park_visit_white_major_visit_per_people',
                             'weekly_park_visit_black_major_visit_per_people',
                             'weekly_park_visit_black_extreme_visit_per_people',
                             'weekly_park_visit_white_major_locate_per_people',
                             'weekly_park_visit_black_major_locate_per_people',
                             'weekly_park_visit_ori_strong_bipart_segregation_per_people',
                             'weekly_park_visit_ori_weak_bipart_segregation_per_people',
                             'weekly_park_visit_overall_strong_bipart_segregation_per_people',
                             'weekly_park_visit_overall_weak_bipart_segregation_per_people',
                             'weekly_park_visit_white_strong_bipart_segregation_per_people',
                             'weekly_park_visit_black_strong_bipart_segregation_per_people',
                             'weekly_park_visit_white_weak_bipart_segregation_per_people',
                             'weekly_park_visit_black_weak_bipart_segregation_per_people',
                             'weekly_park_visit_strong_bipart_ranking_segregation_per_people',
                             'weekly_park_visit_weak_bipart_ranking_segregation_per_people',
                             'weekly_park_visit_overall_strong_bipart_segregation_in_white_major_locate_per_people',
                             'weekly_park_visit_overall_strong_bipart_segregation_in_black_major_locate_per_people',
                             'weekly_park_visit_overall_weak_bipart_segregation_in_white_major_locate_per_people',
                             'weekly_park_visit_overall_weak_bipart_segregation_in_black_major_locate_per_people',
                             'weekly_park_visit_white_strong_bipart_segregation_in_white_major_locate_per_people',
                             'weekly_park_visit_white_strong_bipart_segregation_in_black_major_locate_per_people',
                             'weekly_park_visit_black_strong_bipart_segregation_in_white_major_locate_per_people',
                             'weekly_park_visit_black_strong_bipart_segregation_in_black_major_locate_per_people',
                             'weekly_park_visit_white_weak_bipart_segregation_in_white_major_locate_per_people',
                             'weekly_park_visit_white_weak_bipart_segregation_in_black_major_locate_per_people',
                             'weekly_park_visit_black_weak_bipart_segregation_in_white_major_locate_per_people',
                             'weekly_park_visit_black_weak_bipart_segregation_in_black_major_locate_per_people',
                             'park_num',
                             'TractRadiusOfGyration']]

    county_avg_mhng = df.groupby('CountyFIPS').apply(
        lambda x: (x[TARGET_METRIC] * x['TotalPopulation']).sum() / x['TotalPopulation'].sum()).to_frame(
            name='CountyAvgMHNG').reset_index()
    df = df.merge(county_avg_mhng, on='CountyFIPS', how='inner')
    df['RelativeMHNG'] = df[TARGET_METRIC] / df['CountyAvgMHNG']

    df = df.drop(columns=['CountyAvgMHNG', 'CountyFIPS',TARGET_METRIC, 'TotalPopulation'], axis=1)
    for i in range(9):
        df.loc[:,'Edu_'+str(i).zfill(2)] = df.loc[:, f'DP02_006{i}PE']
        df = df.drop(columns=[f'DP02_006{i}PE'])

    for i in range(15):
        if i != 13:
            df.loc[:,'Age_'+str(i).zfill(2)] = df.loc[:, f'DP05_00{str(i+5).zfill(2)}PE']
            df = df.drop(columns=[f'DP05_00{str(i+5).zfill(2)}PE'])
        else:
            df.loc[:,'Age_'+str(i).zfill(2)] = df.loc[:, f'DP05_00{str(i+5).zfill(2)}E']
            df = df.drop(columns=[f'DP05_00{str(i+5).zfill(2)}E'])
    df = df.rename({'DP05_0024PE':'Age_15'}, axis=1)


    for i in range(4):
        df.loc[:,'Veh_'+str(i).zfill(2)] = df.loc[:, f'DP04_00{str(i+58).zfill(2)}PE']
        df = df.drop(columns=[f'DP04_00{str(i+58).zfill(2)}PE'])

    df = df.rename({'DP03_0007PE': 'Emp',
                    'DP05_0003PE': 'Sex',
                    'DP03_0088E':'Inc_00', 'DP03_0062E':"Inc_01",'DP03_0063E':"Inc_02", 'DP03_0086E':"Inc_03", 'DP03_0087E':"Inc_04", 'DP03_0119PE': "Inc_05",'DP03_0128PE':"Inc_06", 'DP04_0089E':"Inc_07",
                    'DP02_0006PE': "Mrg_00", 'DP02_0010PE': "Mrg_01", 'DP02_0026PE':'Mrg_02','DP02_0027PE':'Mrg_03','DP02_0028PE': "Mrg_04", 'DP02_0029PE': "Mrg_05", 'DP02_0030PE': "Mrg_06", 'DP02_0032PE':'Mrg_07','DP02_0033PE':'Mrg_08', 'DP02_0034PE': "Mrg_09", 'DP02_0035PE': "Mrg_10", 'DP02_0036PE': "Mrg_11",
                    }, axis=1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.replace(['--'], np.nan, inplace=True)
    df = df.dropna()



    main_exog_list = ['weekly_park_visit_total_per_people','weekly_park_visit_time_total_per_people',
                             'weekly_park_visit_strong_segregation_per_people','weekly_park_visit_weak_segregation_per_people',
                             'weekly_park_visit_white_major_visit_per_people','weekly_park_visit_black_major_visit_per_people','weekly_park_visit_black_extreme_visit_per_people',
                             'weekly_park_visit_white_major_locate_per_people','weekly_park_visit_black_major_locate_per_people',
                             'weekly_park_visit_ori_strong_bipart_segregation_per_people','weekly_park_visit_ori_weak_bipart_segregation_per_people',
                             'weekly_park_visit_overall_strong_bipart_segregation_per_people','weekly_park_visit_overall_weak_bipart_segregation_per_people',
                             'weekly_park_visit_white_strong_bipart_segregation_per_people','weekly_park_visit_black_strong_bipart_segregation_per_people',
                             'weekly_park_visit_white_weak_bipart_segregation_per_people','weekly_park_visit_black_weak_bipart_segregation_per_people',
                             'weekly_park_visit_strong_bipart_ranking_segregation_per_people','weekly_park_visit_weak_bipart_ranking_segregation_per_people',
                             'weekly_park_visit_overall_strong_bipart_segregation_in_white_major_locate_per_people',
                             'weekly_park_visit_overall_strong_bipart_segregation_in_black_major_locate_per_people',
                             'weekly_park_visit_overall_weak_bipart_segregation_in_white_major_locate_per_people',
                             'weekly_park_visit_overall_weak_bipart_segregation_in_black_major_locate_per_people',
                             'weekly_park_visit_white_strong_bipart_segregation_in_white_major_locate_per_people',
                             'weekly_park_visit_white_strong_bipart_segregation_in_black_major_locate_per_people',
                             'weekly_park_visit_black_strong_bipart_segregation_in_white_major_locate_per_people',
                             'weekly_park_visit_black_strong_bipart_segregation_in_black_major_locate_per_people',
                             'weekly_park_visit_white_weak_bipart_segregation_in_white_major_locate_per_people',
                             'weekly_park_visit_white_weak_bipart_segregation_in_black_major_locate_per_people',
                             'weekly_park_visit_black_weak_bipart_segregation_in_white_major_locate_per_people',
                             'weekly_park_visit_black_weak_bipart_segregation_in_black_major_locate_per_people',
                             'park_num',
                             'TractRadiusOfGyration']
    control_exog_list = (['Emp','Sex','CHECKUP_CrudePrev','ACCESS2_CrudePrev'] + [f'Edu_{str(i).zfill(2)}' for i in range(8)] + [f'Age_{str(i).zfill(2)}' for i in range(16)] + [f'Inc_{str(i).zfill(2)}' for i in range(8)]
                         + [f'Mrg_{str(i).zfill(2)}' for i in range(12)] + [f'Veh_{str(i).zfill(2)}' for i in range(4)])


    if exog is not None:
        main_exog_index = main_exog_list.index(exog)
    else:
        main_exog_index = 9
    exog_list = [main_exog_list[main_exog_index]] + control_exog_list

    models = {}
    results = {}


    least_green_criteria = df['weekly_park_visit_total_per_people'].quantile(0.1)
    most_green_criteria = df['weekly_park_visit_total_per_people'].quantile(0.9)


    print(least_green_criteria)
    print(most_green_criteria)

    if exog == 'TractRadiusOfGyration':
        least_green_criteria = df['weekly_park_visit_total_per_people'].quantile(0.1)
        most_green_criteria = df['weekly_park_visit_total_per_people'].quantile(0.9)

    if exog == 'weekly_park_visit_time_total_per_people':
        least_green_criteria = df['weekly_park_visit_time_total_per_people'].quantile(0.1)
        most_green_criteria = df['weekly_park_visit_time_total_per_people'].quantile(0.9)


    print('Criteras:')
    print(least_green_criteria)
    print(most_green_criteria)
    param_df = {'Independent Variable':[], 'Racial Groups':[],'VIF':[],'Beta':[],'95% CI':[],'P-value':[],'R2':[]}
    for i in range(percent_num):
        tmp_df = df.loc[df['black_percentile']==i].reset_index(drop=True).astype(float)

        X = tmp_df[exog_list]
        model = sm.OLS(tmp_df['RelativeMHNG'], sm.add_constant(X))
        result = model.fit()
        models[i] = model
        results[i] = result

        param_df['Independent Variable'].append(exog)
        param_df['Racial Groups'].append(percent_lut[i] + ' Black')
        param_df['Beta'].append(result.params[exog])
        ci = result.conf_int().loc[exog].to_numpy()
        param_df['95% CI'].append(f'[{ci[0]:.3f}, {ci[1]:.3f}]')
        param_df['P-value'].append(result.pvalues[exog])
        param_df['R2'].append(result.rsquared)


        print(result.params[main_exog_list[main_exog_index]])
        print('VIF')
        vif = [variance_inflation_factor(tmp_df[exog_list], 0)]
        print(vif)
        param_df['VIF'].append(vif[0])



    fig, ax = plt.subplots(figsize=(8, 6))
    black_percent_list = []
    green_visitation_list = []
    relative_mhng_list = []

    for i in range(percent_num):
        tmp_df = df.loc[df['black_percentile'] == i].astype(float).reset_index(drop=True)
        tmp_tmp_df_least = sm.add_constant(tmp_df[exog_list])
        tmp_tmp_df_most = sm.add_constant(tmp_df[exog_list])

        tmp_tmp_df_least[exog] = least_green_criteria
        tmp_tmp_df_most[exog] = most_green_criteria

        pred_ols_least = results[i].get_prediction(tmp_tmp_df_least)
        pred_ols_most = results[i].get_prediction(tmp_tmp_df_most)

        black_percent_list.append([percent_lut[i]] * len(pred_ols_least.predicted) + [percent_lut[i]] * len(pred_ols_most.predicted))

        green_visitation_list.append(['Lowest Decile'] * len(pred_ols_least.predicted) + ['Highest Decile'] * len(pred_ols_most.predicted))
        relative_mhng_list.append(list(pred_ols_least.predicted) + list(pred_ols_most.predicted))


    box_plot_df = pd.DataFrame({'Black Percentage': np.concatenate(black_percent_list),
                                'Green Visit': np.concatenate(green_visitation_list),
                                'Relative MHNG': np.concatenate(relative_mhng_list)})
    sns.violinplot(x='Black Percentage', y='Relative MHNG',split=True, hue='Green Visit', data=box_plot_df, palette=COLOR_LIST, ax=ax, gap=0.1, inner='quartile')

    if legend:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), ncol=1, title=None, frameon=True, fontsize=12)
    else:
        plt.legend([], [], frameon=False)

    ax.set_ylabel('Relative Prevalence of\nSuboptimal Mental Health', size=20)
    ax.set_xlabel('Percentage of Black Population', size=20)

    if exog == 'weekly_park_visit_black_strong_bipart_segregation_per_people':
        ax.set_title('Weekly Visitation to Parks Affected by Segregation', size=20)
    elif exog == 'weekly_park_visit_black_major_locate_per_people':
        ax.set_title('Weekly Visitation to Parks Located in Majority Black Neighborhoods', size=18)
    elif exog == 'weekly_park_visit_black_strong_bipart_segregation_in_black_major_locate_per_people':
        ax.set_title('Weekly Visitation to Parks Affected by Segregation in Black Majority Neighborhoods', size=16)
    elif exog == 'TractRadiusOfGyration':
        ax.set_title('Radius of Gyration', size=20)
    elif exog == 'weekly_park_visit_total_per_people':
        ax.set_title('Weekly Visitation to All Parks', size=20)
    elif exog == 'weekly_park_visit_time_total_per_people':
        ax.set_title('Weekly Visitation Time to All Parks', size=20)
    else:
        ax.set_title(exog, size=20)

    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=18)

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig('./Fig2D_regression_' + str(exog) + f'_discrete_boxplot_{YEAR}.pdf', bbox_inches='tight', dpi=300, pad_inches=0.2)

    param_df = pd.DataFrame(param_df)
    param_df.to_csv(f'./Fig2D_regression_' + str(exog) + f'_discrete_boxplot_{YEAR}.csv', index=False)


 


if __name__ == "__main__":
    census_tract_data, tract_visit, park_visit = load_and_process_data_tract()

    draw_tract_visit_mental_health_relative_control_variable_discrete_regression_modify_violin(census_tract_data, exog='weekly_park_visit_total_per_people', percent_num=4)
    draw_tract_visit_mental_health_relative_control_variable_discrete_regression_modify_violin(census_tract_data, exog='weekly_park_visit_time_total_per_people', percent_num=4)
    draw_tract_visit_mental_health_relative_control_variable_discrete_regression_modify_violin(census_tract_data,exog='weekly_park_visit_black_strong_bipart_segregation_per_people',percent_num=4, legend=False)
    draw_tract_visit_mental_health_relative_control_variable_discrete_regression_modify_violin(census_tract_data,exog='weekly_park_visit_black_major_locate_per_people',percent_num=4, legend=False)
    draw_tract_visit_mental_health_relative_control_variable_discrete_regression_modify_violin(census_tract_data,exog='weekly_park_visit_black_strong_bipart_segregation_in_black_major_locate_per_people',percent_num=4)
    draw_tract_visit_mental_health_relative_control_variable_discrete_regression_modify_violin(census_tract_data,exog='TractRadiusOfGyration',percent_num=4)
