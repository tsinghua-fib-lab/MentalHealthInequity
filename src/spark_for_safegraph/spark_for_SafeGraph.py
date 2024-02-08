from posixpath import split
from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import lit, lpad
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
# import pyspark.pandas as pypd
from math import *
from pathlib import Path
import re

# spark-submit --master=yarn --num-executors 250 --executor-memory 5g --driver-memory 20g
# pyspark --master=yarn --num-executors 250 --executor-memory 5g --driver-memory 20g

YEAR=2019

spark = SparkSession\
        .builder\
        .appName("Safegraph") \
        .config("spark.default.parallelism", "300") \
        .config("spark.driver.maxResultSize",'10g') \
        .getOrCreate()

sc = spark.sparkContext

def file_path():
    data_path = {}
    data_path['weeklydata'] = '/user/usera/hanzhenyu/SafeGraph_old/SafeGraph_weekly_processed/*.csv'
    #data_path['weeklydata'] = '/user/usera/hanzhenyu/2018-12-31-weekly-patterns.csv'
    data_path['poidata'] = '/user/usera/hanzhenyu/SafeGraph_old/POI/*.csv'
    data_path['poidata_geo'] = '/user/usera/hanzhenyu/SafeGraph/SafeGraph_POI/POI_geometry.csv'
    data_path['home_panel_summary'] = '/user/usera/hanzhenyu/SafeGraph/SafeGraph_home_panel_summary/monthly_home_panel_summary_merge_2018_2020.csv'
    data_path['open_census'] = '/user/usera/hanzhenyu/SafeGraph/safegraph_open_census_data_2019/data/'
    data_path['temp_parquet'] = f'/user/usera/hanzhenyu/output/ALL_US_Park_Visit_Dataset_{YEAR}'
    data_path['output_park_visit'] = f'/user/usera/hanzhenyu/output/ALL_US_park_{YEAR}'
    data_path['output_park_cbg_bipart'] = f'/user/usera/hanzhenyu/output/ALL_US_park_cbg_bipart_{YEAR}'
    data_path['output_cbg_census'] = f'/user/usera/hanzhenyu/output/ALL_US_cbg_census_{YEAR}'
    data_path['output_cbg'] = f'/user/usera/hanzhenyu/output/ALL_US_cbg_{YEAR}'
    return data_path


data_path = file_path()


# B01003e1: Total Population
# B02001e3: Black or African American alone
# B02001e2: White alone
# B19013e1: Median Household Income In The Past 12 Months (In 2019 Inflation-Adjusted Dollars)
for i in ['cbg_b01.csv','cbg_b02.csv','cbg_b19.csv']:
    tmp = spark.read.options(delimiter=',').csv(data_path['open_census']+i, header=True, inferSchema=False)
    if i == 'cbg_b01.csv':
        open_census = tmp.select('census_block_group',F.col('B01003e1').cast('float'))
    elif i == 'cbg_b02.csv':
        open_census = open_census.join(tmp.select('census_block_group',F.col('B02001e3').cast('float'),F.col('B02001e2').cast('float')), on='census_block_group',how='inner')
    elif i == 'cbg_b19.csv':
        open_census = open_census.join(tmp.select('census_block_group',F.col('B19013e1').cast('float')), on='census_block_group',how='inner')
open_census.repartition(1).write.option('header', True).csv(data_path['output_cbg_census'])

poi_geo = spark.read.options(delimiter=',').csv(data_path['poidata_geo'], header=True, inferSchema=False).select(F.col('placekey'), F.col('polygon_wkt'))

poi = spark.read.options(delimiter=',').csv(data_path['poidata'], header=True, inferSchema=False)
poi = poi.select(poi.safegraph_place_id, poi.placekey,poi.location_name, poi.latitude, poi.longitude, poi.top_category, poi.sub_category, lpad(poi.naics_code.cast(StringType()), 6, '0').alias('naics_code'))
poi = poi.filter(poi.naics_code == '712190')

poi = poi.filter(poi.safegraph_place_id != '')
poi = poi.filter(poi.placekey != '')

#sub_category = 'Nature Parks and Other Similar Institutions'
# poi.describe().show()

data = spark.read.options(delimiter='\t').csv(data_path['weeklydata'], header=True, inferSchema=False)
data = data.withColumn("start_date", F.substring(data.date_range_start, pos=0, len=10))
data = data.withColumn("end_date", F.substring(data.date_range_end, pos=0, len=10))
data = data.select(data.safegraph_place_id, data.city, data.start_date, data.end_date, data.raw_visit_counts.cast('float'),
                   data.raw_visitor_counts.cast('float'), data.poi_cbg.cast('string'), data.visitor_home_cbgs.cast('string'),
                   data.visits_by_day.cast('string'), data.visits_by_each_hour.cast('string'),data.distance_from_home.cast('float'),
                   data.median_dwell.cast('float'),data.bucketed_dwell_times.cast('string'))
# data.show()


data = data.join(poi, data.safegraph_place_id==poi.safegraph_place_id, how='inner').drop(poi.safegraph_place_id)
data = data.join(poi_geo, data.placekey==poi_geo.placekey,how='inner').drop(poi_geo.placekey)
data.cache()
data.show()

data.write.parquet(data_path['temp_parquet'])


#====================================================================================
data = spark.read.parquet(data_path['temp_parquet'])

if YEAR == 2020:
    data = data.filter("start_date >= '2020-01-01' and start_date <= '2020-12-31'")
elif YEAR == 2019:
    data = data.filter("start_date >= '2019-01-01' and start_date <= '2019-12-31'")
data = data.filter("visitor_home_cbgs != '{}'")


# Park
if YEAR == 2019:
    park_info = data.select('placekey','location_name','poi_cbg','latitude','longitude',F.col('polygon_wkt').alias('geometry')).distinct()
    park_info = park_info.withColumnRenamed('poi_cbg','poi_cbg_new')
else:
    park_info = data.select('placekey','location_name','poi_cbg_new','latitude','longitude',F.col('polygon_wkt').alias('geometry')).distinct()

park_info.cache()
park_visit = data.select('placekey','raw_visit_counts','raw_visitor_counts', 'median_dwell','distance_from_home').groupby('placekey').agg(F.avg('raw_visit_counts').alias('raw_visit_counts_week_mean'),F.avg('raw_visitor_counts').alias('raw_visitor_counts_week_mean'), F.avg('median_dwell').alias('median_dwell_week_mean'), F.avg('distance_from_home').alias('median_distance_from_home_week_mean'))
park_visit = park_visit.join(park_info, park_visit.placekey==park_info.placekey,how='inner').drop(park_info.placekey).sort('raw_visit_counts_week_mean', ascending=False)
park_visit.repartition(1).write.option('header', True).csv(data_path['output_park_visit'])

# Bipartile Visit
def poi_cbg_bipart(x):
    home_cbgs = eval(x[1].replace('""','"')[1:-1])
    return [(x[0],k,v) for k,v in home_cbgs.items()]

poi_cbg_visit = (data.select('placekey','visitor_home_cbgs')).rdd.flatMap(poi_cbg_bipart).toDF(['placekey','cbg','visitor_count']).groupby('placekey','cbg').agg(F.sum('visitor_count').alias('visitor_count'))
poi_cbg_visit.repartition(1).write.option('header', True).csv(data_path['output_park_cbg_bipart'])


# CBG
cbg_visit = poi_cbg_visit.groupBy('cbg').agg(F.sum('visitor_count').alias('visitor_count'),F.count_distinct('placekey').alias('park_num'))
cbg_visit.repartition(1).write.option('header', True).csv(data_path['output_cbg'])
