import os
import time
from datetime import datetime
os.environ['USE_PYGEOS'] = '0'
from shapely.geometry import Polygon
from pathlib import Path
import geopandas as gpd

OUTPUT_PATH = Path('./Data/WorldCover/US')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

# load natural earth low res shapefile
# ne = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

#
# country = 'United States of America'
# geometry = ne[ne.name == country].iloc[0].geometry
#

US_BOUNDARY_LOC = Path('./Data/Boundary/cb_2019_us_nation_5m/cb_2019_us_nation_5m.shp')
geometry = gpd.read_file(US_BOUNDARY_LOC).iloc[0].geometry

print('load worldcover grid')
url = f'{s3_url_prefix}/v200/2021/esa_worldcover_grid.geojson'
grid = gpd.read_file(url)
print('load over')

# get grid tiles intersecting AOI
tiles = grid[grid.intersects(geometry)]

# use requests library to download them
import requests
from tqdm.auto import tqdm  # provides a progressbar

to_download = tiles.ll_tile.tolist()
total_num = len(to_download)
total_begin_time = datetime.now()
while to_download:
    tile = to_download.pop()
    begin_time = datetime.now()
    url = f"{s3_url_prefix}/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    while True:
        try:
            r = requests.get(url, allow_redirects=True, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36'})
            break
        except:
            time.sleep(3)
    out_fn = OUTPUT_PATH.joinpath(f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
    with open(out_fn, 'wb') as f:
        f.write(r.content)
    end_time = datetime.now()
    time_est = (end_time - total_begin_time).total_seconds() / 60 / (total_num - len(to_download)) * len(to_download)
    print(f"Downloaded {tile} in {(end_time - begin_time).total_seconds()}, ({total_num - len(to_download)} locs remaining with estimated time of {time_est} mins ")

