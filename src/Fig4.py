import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
from pathlib import Path
from math import ceil
import copy
import pandas as pd
import geopandas as gpd
from datetime import datetime
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import seaborn as sns
sns.set_theme(style="white")
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13
COLOR_LIST = [sns.color_palette("Set2")[1], sns.color_palette("Set2")[2]]


YEAR=2019
TARGET_METRIC = 'MHLTH_CrudePrev'
RACE_METRIC = 'DP05_0065PE'
MULTI_PROCESS_NUM = 10
############################################################################################
# Data Path

TRACT_VISIT_LOC = Path(f'./tract_visit_all_US_within_county_{YEAR}.parquet')
TRACT_VISIT_SELECTED_WITH_GOOGLE_LOC = Path(f'./tract_visit_selected_county_with_google_{YEAR}.parquet')
PARK_TRACT_VISIT_LOC = Path(f'./park_visit_all_US_within_county_{YEAR}.parquet')
CENSUS_TRACT_DATA_LOC = Path(f'./census_tract_data_all_with_park_with_landuse_{YEAR}.parquet')
PART_TRACT_BIPART_PROCESSED_LOC = Path(f'./park_tract_bipart_all_us_within_county_{YEAR}.parquet')

############################################################################################

def load_and_process_data_tract(google=False):
    census_tract_data = gpd.read_parquet(CENSUS_TRACT_DATA_LOC)
    if not google:
        tract_visit = pd.read_parquet(TRACT_VISIT_LOC).dropna().reset_index(drop=True)
    else:
        tract_visit = pd.read_parquet(TRACT_VISIT_SELECTED_WITH_GOOGLE_LOC).dropna().reset_index(drop=True).drop('CountyFIPS',axis=1)
    census_tract_data = census_tract_data.merge(tract_visit,on='TractFIPS',how='inner').reset_index(drop=True)

    return census_tract_data

def draw_tract_race_black_segregation_radius_of_gyration_density_no_facet(df,clip = (0,50000), percent_num=4):
    percent_lut = {}
    for i in range(percent_num):
        percent_lut[i] = str(int(100 / percent_num * i)) + '-' + str(int(100 / percent_num * (i + 1))) + '%'

    df['black_percentile'] = df[RACE_METRIC].map(lambda x: int(x / (1 / percent_num)) if x < 1 else percent_num - 1)
    df['black_quantile'] = pd.qcut(df[RACE_METRIC], percent_num, labels=False)
    df = df.sort_values(by=RACE_METRIC).reset_index(drop=True)

    if clip is not None:
        df['TractRadiusOfGyrationStrongBlackBipartSegregation'] = df['TractRadiusOfGyrationStrongBlackBipartSegregation'].clip(clip[0], clip[1])
        df['TractRadiusOfGyrationNotStrongBlackBipartSegregation'] = df['TractRadiusOfGyrationNotStrongBlackBipartSegregation'].clip(clip[0], clip[1])


    tract_id_list = []
    radius_of_gyration_list = []
    black_percentile_list = []
    black_percentile_readable_list = []
    is_strong_segregation_list = []

    for _, row in df.iterrows():
        tract_id_list.append(row['TractFIPS'])
        tract_id_list.append(row['TractFIPS'])
        radius_of_gyration_list.append(row['TractRadiusOfGyrationStrongBlackBipartSegregation'])
        radius_of_gyration_list.append(row['TractRadiusOfGyrationNotStrongBlackBipartSegregation'])
        black_percentile_readable_list.append(percent_lut[row['black_percentile']])
        black_percentile_readable_list.append(percent_lut[row['black_percentile']])
        black_percentile_list.append(row['black_percentile'])
        black_percentile_list.append(row['black_percentile'])
        is_strong_segregation_list.append('Parks Affected by Segregation')
        is_strong_segregation_list.append('Parks Not Affected by Segregation')

    df_longtable = pd.DataFrame({'TractFIPS': tract_id_list,
                                    'TractRadiusOfGyration': radius_of_gyration_list,
                                    'black_percentile': black_percentile_list,
                                    'Black Percentile': black_percentile_readable_list,
                                    'Segregation Type': is_strong_segregation_list})



    fig, axes = plt.subplots(percent_num, 1, figsize=(10, 8), sharex='row')
    for i in range(percent_num):
        sns.kdeplot(data=df_longtable.loc[df_longtable['black_percentile']==i], x='TractRadiusOfGyration', hue='Segregation Type',  palette=COLOR_LIST[:2], bw_adjust=.5, clip_on=False,fill=True, alpha=0.2 + i * 0.2, linewidth=1.5, ax=axes[i], hue_order=['Parks Affected by Segregation','Parks Not Affected by Segregation'],clip=clip)
        sns.kdeplot(data=df_longtable.loc[df_longtable['black_percentile']==i], x='TractRadiusOfGyration', hue='Segregation Type', clip_on=False, palette=['#FA5B1D','#6A85CD'],  lw=1.5, bw_adjust=.5, ax=axes[i], hue_order=['Parks Affected by Segregation','Parks Not Affected by Segregation'],clip=clip)

        x1 = df_longtable.loc[(df_longtable['black_percentile']==i) & (df_longtable['Segregation Type']=='Parks Affected by Segregation')]['TractRadiusOfGyration'].mean()
        x2 = df_longtable.loc[(df_longtable['black_percentile']==i) & (df_longtable['Segregation Type']=='Parks Not Affected by Segregation')]['TractRadiusOfGyration'].mean()
        axes[i].axvline(x=x1, color='#FA5B1D', linestyle='--', linewidth=1.5, ymax=0.95)
        axes[i].axvline(x=x2, color='#6A85CD', linestyle='--', linewidth=1.5, ymax=0.95)
        print(f"Race percentile: {i}, Mean of Segregation: {x1}, Mean of No Segregation {x2}")

        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        if i < (percent_num - 1):
            axes[i].spines['bottom'].set_visible(False)
            axes[i].set_xlabel('', size=1)
            axes[i].set_xticks([])
            a=1
        else:
            axes[i].spines['bottom'].set_visible(False)
            axes[i].tick_params(axis='x', labelsize=20)
            axes[i].set_xlabel('Radius of Gyration',size=24)

        axes[i].plot(0, linewidth=2, linestyle="-", color=None, clip_on=False)
        axes[i].set_ylabel('', size=14)
        axes[i].set_yticks([])

        axes[i].text(0.03, .2,  percent_lut[i], fontsize=18, fontweight="bold", color='k',
                ha="left", va="center", transform=axes[i].transAxes)

        if i > 0:
            axes[i].get_legend().remove()
            axes[i].set_facecolor((0, 0, 0, 0))
        else:
            sns.move_legend(
                axes[i], "lower center",
                bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False, fontsize=18
            )

    fig.subplots_adjust(hspace=-.3)
    plt.tight_layout()
    plt.savefig(f'./Fig4_radius_of_segregation_dist_{YEAR}.pdf', dpi=300, pad_inches=0.1, bbox_inches='tight', transparent=True)



def draw_tract_race_black_segregation_rating_density_no_facet(df, clip=None, percent_num=4):
    percent_lut = {}
    for i in range(percent_num):
        percent_lut[i] = str(int(100 / percent_num * i)) + '-' + str(int(100 / percent_num * (i + 1))) + '%'

    df['black_percentile'] = df[RACE_METRIC].map(lambda x: int(x / (1 / percent_num)) if x < 1 else percent_num - 1)
    df['black_quantile'] = pd.qcut(df[RACE_METRIC], percent_num, labels=False)
    df = df.sort_values(by=RACE_METRIC).reset_index(drop=True)

    if clip is not None:
        df['TractVisitedParkScoreStrongBlackBipartSegregation'] = df['TractVisitedParkScoreStrongBlackBipartSegregation'].clip(clip[0], clip[1])
        df['TractVisitedParkScoreNotStrongBlackBipartSegregation'] = df['TractVisitedParkScoreNotStrongBlackBipartSegregation'].clip(clip[0], clip[1])


    tract_id_list = []
    score_list = []
    black_percentile_list = []
    black_percentile_readable_list = []
    is_strong_segregation_list = []

    for _, row in df.iterrows():
        tract_id_list.append(row['TractFIPS'])
        tract_id_list.append(row['TractFIPS'])
        score_list.append(row['TractVisitedParkScoreStrongBlackBipartSegregation'])
        score_list.append(row['TractVisitedParkScoreNotStrongBlackBipartSegregation'])
        black_percentile_readable_list.append(percent_lut[row['black_percentile']])
        black_percentile_readable_list.append(percent_lut[row['black_percentile']])
        black_percentile_list.append(row['black_percentile'])
        black_percentile_list.append(row['black_percentile'])
        is_strong_segregation_list.append('Parks Affected by Segregation')
        is_strong_segregation_list.append('Parks Not Affected by Segregation')

    df_longtable = pd.DataFrame({'TractFIPS': tract_id_list,
                                    'Park Rating': score_list,
                                    'black_percentile': black_percentile_list,
                                    'Black Percentile': black_percentile_readable_list,
                                    'Segregation Type': is_strong_segregation_list})



    fig, axes = plt.subplots(percent_num, 1, figsize=(10, 8), sharex='row')
    for i in range(percent_num):
        sns.kdeplot(data=df_longtable.loc[df_longtable['black_percentile']==i], x='Park Rating', hue='Segregation Type', cut=0, palette=COLOR_LIST[:2], bw_adjust=.5, clip_on=False,fill=True, alpha=0.2 + i * 0.2, linewidth=1.5, ax=axes[i], hue_order=['Parks Affected by Segregation','Parks Not Affected by Segregation'],clip=clip)
        sns.kdeplot(data=df_longtable.loc[df_longtable['black_percentile']==i], x='Park Rating', hue='Segregation Type', cut=0, clip_on=False, palette=['#FA5B1D', '#6A85CD'],  lw=1.5, bw_adjust=.5, ax=axes[i], hue_order=['Parks Affected by Segregation','Parks Not Affected by Segregation'],clip=clip)


        x1 = df_longtable.loc[(df_longtable['black_percentile']==i) & (df_longtable['Segregation Type']=='Parks Affected by Segregation')]['Park Rating'].mean()
        x2 = df_longtable.loc[(df_longtable['black_percentile']==i) & (df_longtable['Segregation Type']=='Parks Not Affected by Segregation')]['Park Rating'].mean()
        axes[i].axvline(x=x1, color='#FA5B1D', linestyle='--', linewidth=1.5, ymax=0.95)
        axes[i].axvline(x=x2, color='#6A85CD', linestyle='--', linewidth=1.5, ymax=0.95)
        print(f"Race percentile: {i}, Mean of Segregation: {x1}, Mean of No Segregation {x2}")

        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        if not clip:
            axes[i].set_xlim(0,5)
        else:
            axes[i].set_xlim(clip)
        if i < (percent_num - 1):
            axes[i].spines['bottom'].set_visible(False)
            axes[i].set_xlabel('', size=1)
            axes[i].set_xticks([])
            a=1
        else:
            axes[i].spines['bottom'].set_visible(False)
            axes[i].tick_params(axis='x', labelsize=20)
            axes[i].set_xlabel('Park Rating',size=24)

        axes[i].plot(0, linewidth=2, linestyle="-", color=None, clip_on=False)
        axes[i].set_ylabel('', size=14)
        axes[i].set_yticks([])

        axes[i].text(0.03, .2,  percent_lut[i], fontsize=18, fontweight="bold", color='k',
                ha="left", va="center", transform=axes[i].transAxes)

        if i > 0:
            axes[i].get_legend().remove()
            axes[i].set_facecolor((0, 0, 0, 0))
        else:
            sns.move_legend(
                axes[i], "lower center",
                bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False, fontsize=18
            )

    fig.subplots_adjust(hspace=-.3)
    plt.tight_layout()
    plt.savefig(f'./Fig4_park_rating_dist_{YEAR}.pdf', dpi=300, pad_inches=0.1, bbox_inches='tight', transparent=True)


if __name__ == "__main__":

    census_tract_data = load_and_process_data_tract(google=False)
    draw_tract_race_black_segregation_radius_of_gyration_density_no_facet(census_tract_data, percent_num=4)

    selected_census_tract_data_google = load_and_process_data_tract(google=True)
    draw_tract_race_black_segregation_rating_density_no_facet(selected_census_tract_data_google, percent_num=4, clip=(3.8,4.8))


