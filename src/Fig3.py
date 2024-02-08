import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
from pathlib import Path
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import shapely
sns.set_theme(style="white", palette=None)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13
COLOR_LIST = sns.color_palette("Set2")
YEAR=2019
TARGET_METRIC = 'MHLTH_CrudePrev'
RACE_METRIC = 'DP05_0065PE'
INCOME_METRIC = 'DP03_0088E'
############################################################################################
# Data Path
TRACT_VISIT_LOC = Path(f'./tract_visit_all_US_within_county_{YEAR}.parquet')
PARK_TRACT_VISIT_LOC = Path(f'./park_visit_all_US_within_county_{YEAR}.parquet')
CENSUS_TRACT_DATA_LOC = Path(f'./census_tract_data_all_with_park_with_landuse_{YEAR}.parquet')
PART_TRACT_BIPART_LOC = Path(f'./park_tract_bipart_all_us_within_county_{YEAR}.parquet')
############################################################################################
def load_and_process_data_tract():
    census_tract_data = gpd.read_parquet(CENSUS_TRACT_DATA_LOC)

    tract_visit = pd.read_parquet(TRACT_VISIT_LOC).dropna().reset_index(drop=True)
    park_tract_visit = pd.read_parquet(PARK_TRACT_VISIT_LOC).dropna().reset_index(drop=True)
    park_tract_bipart = pd.read_parquet(PART_TRACT_BIPART_LOC)

    return census_tract_data.copy(deep=True), tract_visit, park_tract_visit, park_tract_bipart


def draw_bipart_network_nx_discrete(census_tract_data, park_tract_bipart, park_sample_rate=1, tract_sample_num=5, percent_num=20, selected_county={'Cook County': '17031'}, highlight_parks=None,draw_label=False):

    percent_lut = {}
    for i in range(percent_num):
        percent_lut[i] = str(int(100 / percent_num * i)) + '-' + str(int(100 / percent_num * (i + 1))) + '%'
    census_tract_data['black_percentile'] = census_tract_data[RACE_METRIC].map(lambda x: percent_lut[int(x / (1 / percent_num))] if x < 1 else percent_lut[percent_num - 1])
    census_tract_data['black_percentile_int'] = census_tract_data[RACE_METRIC].map(lambda x: int(x / (1 / percent_num)) if x < 1 else percent_num - 1)

    # Sample Parks
    park_list = park_tract_bipart['placekey'].drop_duplicates().sample(frac=park_sample_rate, random_state=0)
    park_tract_bipart = park_tract_bipart.loc[park_tract_bipart['placekey'].isin(park_list)].reset_index(drop=True)

    park_tract_bipart = park_tract_bipart.merge(census_tract_data[['TractFIPS', 'black_percentile', 'black_percentile_int']], on='TractFIPS', how='inner')
    park_tract_bipart = park_tract_bipart.loc[park_tract_bipart['TractFIPS'].str[:5].isin(selected_county.values())].reset_index(drop=True)

    if highlight_parks is not None:
        park_tract_bipart = park_tract_bipart[['TractFIPS', 'placekey', 'black_percentile',  'black_percentile_int', 'visitor_count', RACE_METRIC, highlight_parks]]
    else:
        park_tract_bipart = park_tract_bipart[
            ['TractFIPS', 'placekey', 'black_percentile', 'black_percentile_int', 'visitor_count', RACE_METRIC]]

    park_tract_bipart = park_tract_bipart.dropna().reset_index(drop=True)


    # Cal Park visit
    park_all_visit = park_tract_bipart[['placekey', 'visitor_count']].groupby('placekey').sum().reset_index().rename(
        columns={'visitor_count': 'total_visitor_count'})
    park_all_visit = park_all_visit[park_all_visit['total_visitor_count'] != np.inf].reset_index(drop=True)  # 删除inf

    if highlight_parks is not None:
        highlight_parks_placekey_list = park_tract_bipart.loc[park_tract_bipart[highlight_parks]]['placekey'].to_list()

    # Cal weight
    park_tract_bipart = park_tract_bipart[['placekey', 'visitor_count', 'TractFIPS', 'black_percentile', 'black_percentile_int',RACE_METRIC]]
    park_tract_bipart = park_tract_bipart.merge(park_all_visit, on='placekey', how='inner')
    park_tract_bipart['Weight'] = (park_tract_bipart['visitor_count'] / park_tract_bipart['total_visitor_count'])
    park_tract_bipart = park_tract_bipart.dropna().reset_index(drop=True)

    # Cal links
    most_linked_park_tract = park_tract_bipart[['placekey', 'TractFIPS', 'Weight']].groupby(['placekey']).apply(
        lambda x: x.sort_values(by='Weight', ascending=False).head(tract_sample_num))['TractFIPS'].reset_index().drop('level_1',axis=1)
    park_tract_bipart = park_tract_bipart.merge(most_linked_park_tract, on=['placekey', 'TractFIPS'], how='inner')


    # Node size
    park_node_size = park_tract_bipart[['placekey', 'total_visitor_count']].drop_duplicates().reset_index(drop=True).rename(
        columns={'total_visitor_count': 'NodeSize'})

    # Cal racial comp. as node color
    county_pop = census_tract_data[['CountyFIPS', 'TotalPopulation']].groupby('CountyFIPS').sum().reset_index().rename({'TotalPopulation': 'CountyPop'}, axis=1)
    park_tract_bipart['CountyFIPS'] = park_tract_bipart['TractFIPS'].str[:5]
    park_tract_bipart = park_tract_bipart.merge(county_pop, on='CountyFIPS', how='inner')
    park_tract_bipart = park_tract_bipart.merge(census_tract_data[['TractFIPS','TotalPopulation']], on='TractFIPS', how='inner')

    park_tract_bipart['TargetColor'] = park_tract_bipart['Weight'] * park_tract_bipart[RACE_METRIC] #* park_tract_bipart['TotalPopulation'] / park_tract_bipart['CountyPop']  # 这里会对权重进行调整，调整之后更合理但不好看了
    park_node_color = park_tract_bipart[['placekey', 'TargetColor']].groupby('placekey').sum().reset_index().rename(columns={'TargetColor': 'NodeColor'})

    park_node_info = park_node_size.merge(park_node_color, on='placekey', how='inner')

    park_tract_bipart = park_tract_bipart[['placekey', 'visitor_count', 'black_percentile', 'black_percentile_int',]].groupby(['placekey', 'black_percentile', 'black_percentile_int']).sum().reset_index()

    park_tract_bipart = park_tract_bipart[['placekey', 'visitor_count', 'black_percentile', 'black_percentile_int']]
    park_tract_bipart = park_tract_bipart.merge(park_all_visit, on='placekey', how='inner')
    park_tract_bipart['Weight'] = (park_tract_bipart['visitor_count'] / park_tract_bipart['total_visitor_count'])
    park_tract_bipart = park_tract_bipart.dropna().reset_index(drop=True)

    park_tract_bipart = park_tract_bipart[['black_percentile','black_percentile_int', 'placekey', 'Weight', 'total_visitor_count']].rename({'total_visitor_count': 'TargetVisit'}, axis=1)

    # Tract Node color
    tract_node_info = park_tract_bipart[['black_percentile','black_percentile_int']].drop_duplicates().reset_index(drop=True).rename(columns={'black_percentile': 'NodeID', 'black_percentile_int': 'NodeColor'})

    park_node_info = park_node_info.sort_values(by='NodeColor', ascending=True).reset_index(drop=True)
    park_node_info['NodeID'] = [i for i in np.arange(len(park_node_info))]
    park_node_info['NodeType'] = 'Park'

    tract_node_info = tract_node_info.sort_values(by='NodeColor', ascending=True).reset_index(drop=True)
    tract_node_info['NodeID'] = [i for i in np.arange(len(tract_node_info)) + len(park_node_info)]
    tract_node_info['NodeType'] = 'Tract'

    node_df = pd.concat([park_node_info[['NodeID', 'NodeType', 'NodeColor']],
                         tract_node_info[['NodeID', 'NodeType', 'NodeColor']]], axis=0).reset_index(drop=True)
    
    park_tract_bipart = park_tract_bipart.merge(tract_node_info[['NodeColor', 'NodeID']], left_on='black_percentile_int', right_on='NodeColor',
                                                how='left').rename(columns={'NodeID': 'Source'})
    park_tract_bipart = park_tract_bipart.merge(park_node_info[['placekey', 'NodeID']], on='placekey',
                                                how='left').rename(columns={'NodeID': 'Target'})

    if highlight_parks is not None:
        highlight_parks_ID_list = park_tract_bipart.loc[park_tract_bipart['placekey'].isin(highlight_parks_placekey_list)]['Target'].drop_duplicates().to_list()
    park_tract_bipart = park_tract_bipart[['Source', 'Target', 'Weight']].sort_values([ 'Source','Target'], ascending=True).reset_index(drop=True)

    G = nx.from_pandas_edgelist(park_tract_bipart, source='Source', target='Target', edge_attr='Weight')
    nx.set_node_attributes(G, node_df.set_index('NodeID').to_dict('index'))
    pos = nx.bipartite_layout(G, node_df.loc[node_df['NodeType'] == 'Tract']['NodeID'].to_list(), align='horizontal',
                              scale=30)

    typeA_boundary_x_min = min([pos[i][0] for i in node_df.loc[node_df['NodeType'] == 'Park']['NodeID'].to_list()])
    typeA_boundary_y_min = min([pos[i][1] for i in node_df.loc[node_df['NodeType'] == 'Park']['NodeID'].to_list()])
    typeA_boundary_x_max = max([pos[i][0] for i in node_df.loc[node_df['NodeType'] == 'Park']['NodeID'].to_list()])
    typeA_boundary_y_max = max([pos[i][1] for i in node_df.loc[node_df['NodeType'] == 'Park']['NodeID'].to_list()])

    typeB_boundary_x_min = min([pos[i][0] for i in node_df.loc[node_df['NodeType'] == 'Tract']['NodeID'].to_list()])
    typeB_boundary_y_min = min([pos[i][1] for i in node_df.loc[node_df['NodeType'] == 'Tract']['NodeID'].to_list()])
    typeB_boundary_x_max = max([pos[i][0] for i in node_df.loc[node_df['NodeType'] == 'Tract']['NodeID'].to_list()])
    typeB_boundary_y_max = max([pos[i][1] for i in node_df.loc[node_df['NodeType'] == 'Tract']['NodeID'].to_list()])


    pos_new = {}
    typeA_node_lists = node_df.loc[node_df['NodeType'] == 'Park']['NodeID'].to_list()
    typeB_node_lists = node_df.loc[node_df['NodeType'] == 'Tract']['NodeID'].to_list()
    for i in range(len(typeA_node_lists)):
        pos_new[typeA_node_lists[i]] = [typeA_boundary_x_min + (typeA_boundary_x_max - typeA_boundary_x_min) * i / len(typeA_node_lists), typeA_boundary_y_min + (typeA_boundary_y_max - typeA_boundary_y_min) * i / len(typeA_node_lists)]
    for i in range(len(typeB_node_lists)):
        pos_new[typeB_node_lists[i]] = [typeB_boundary_x_min + (typeB_boundary_x_max - typeB_boundary_x_min) * i / len(typeB_node_lists), typeB_boundary_y_min + (typeB_boundary_y_max - typeB_boundary_y_min) * i / len(typeB_node_lists)]

    if draw_label:
        pos_new_label = {}
        tract_label = {}
        for i in range(len(typeB_node_lists)):
            pos_new_label[typeB_node_lists[i]] = [pos_new[typeB_node_lists[i]][0], pos_new[typeB_node_lists[i]][1] - 2]
            tract_label[typeB_node_lists[i]] = percent_lut[node_df.loc[node_df['NodeID'] == typeB_node_lists[i]]['NodeColor'].to_list()[0]]

    options = {"edgecolors": None, "alpha": 0.9}
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos_new, cmap=sns.color_palette("rocket_r", as_cmap=True),
                           nodelist=node_df.loc[node_df['NodeType'] == 'Tract']['NodeID'].to_list(),
                           node_color=node_df.loc[node_df['NodeType'] == 'Tract']['NodeColor'].to_numpy(),
                           node_size=300,
                           node_shape='H', **options)
    if draw_label:
        nx.draw_networkx_labels(G, pos_new_label, labels=tract_label, font_size=15, font_color='black')
    if highlight_parks is None:
        nx.draw_networkx_nodes(G, pos_new, cmap=sns.color_palette("rocket_r", as_cmap=True),
                               nodelist=node_df.loc[node_df['NodeType'] == 'Park']['NodeID'].to_list(),
                               node_color=node_df.loc[node_df['NodeType'] == 'Park']['NodeColor'].to_numpy(),
                               node_size=300,#park_node_info['NodeSize'].to_numpy() * 300,
                               node_shape='o', **options)
    else:
        highlighted_parks_df = node_df.loc[(node_df['NodeType'] == 'Park') & (node_df['NodeID'].isin(highlight_parks_ID_list))]
        nx.draw_networkx_nodes(G, pos_new, cmap=sns.color_palette("rocket_r", as_cmap=True),
                               nodelist=highlighted_parks_df['NodeID'].to_list(),
                               node_color=highlighted_parks_df['NodeColor'].to_numpy(),
                               node_size=600,#park_node_info['NodeSize'].to_numpy() * 300,
                               node_shape='o', **options)
        nx.draw_networkx_nodes(G, pos_new, cmap=sns.color_palette("rocket_r", as_cmap=True),
                                 nodelist=node_df.loc[node_df['NodeType'] == 'Park']['NodeID'].to_list(),
                                 node_color=node_df.loc[node_df['NodeType'] == 'Park']['NodeColor'].to_numpy(),
                                 node_size=100,#park_node_info['NodeSize'].to_numpy() * 300,
                                 node_shape='o', **options)
    nx.draw_networkx_edges(G, pos_new, edge_color=park_tract_bipart['Weight'].to_numpy(), edge_cmap=plt.cm.Blues, width=1.5,
                           alpha=0.5)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    census_tract_data, tract_visit, park_visit, park_tract_bipart = load_and_process_data_tract()

    # selected_county = {'Bronx County':'36005','Kings County':'36047','New York County':'36061', 'Queens County':'36081','Richmond County':'36085'}
    # selected_county = {'Cook County': '17031'}  # Chicago
    # selected_county = {'Harris County': '48201'}  # Houston
    selected_county = {'King County': '53033'}
    draw_bipart_network_nx_discrete(census_tract_data, park_tract_bipart, park_sample_rate=1, tract_sample_num=10, percent_num=10,
                                     selected_county=selected_county, highlight_parks=None, draw_label=True)
