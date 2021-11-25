from rasterstats import zonal_stats
from pathlib import Path
from shapely.wkt import loads
import geopandas as gpd
import pandas as pd
import numpy as np
import datetime as dt
import re
import json


def main():
    compute_nuts3_stats()

def compute_nuts3_stats(index=SPI3):
    today = dt.datetime.strptime('2018-08-24', '%Y-%m-%d')
    drange = pd.date_range(today - dt.timedelta(days=365), today, freq='1D')

    # read shpfile
    nuts3 = gpd.read_file('/mnt/CEPH_PROJECTS/ADO/GIS/ERTS89_LAEA/alpinespace_eusalp_NUTS3_simple.shp')

    basepath = Path('/mnt/CEPH_PROJECTS/ADO/ARSO/ERA5_QM/SPI-3/')
    spi_list = list()
    for path in basepath.glob('*.tif'):
        spi_list.append(path)

    # sort list of paths
    spi_list = sorted(spi_list, key=lambda i: dt.datetime.strptime(i.name[6:14], '%Y%m%d'))

    # create a working copy
    tmp_nuts3 = nuts3.copy()

    # add SPI property
    dictlist = [dict() for i in range(nuts3.shape[0])]
    tmp_nuts3 = tmp_nuts3.assign(SPI3=dictlist)

    # calculate statistics
    for i_date in drange:
        for i in spi_list:
            if dt.datetime.strptime(i.name[6:14], '%Y%m%d') == i_date:
                i_path = i
                break
        r_stats = get_stats(i_path)
        date = dt.datetime.strptime(str(i_path.name[6:14]), '%Y%m%d')
        print(date)
        for ifeat, istat in zip(tmp_nuts3.iterfeatures(), r_stats):
            if np.isinf(istat['nanmean']):
                ifeat['properties']['SPI3'][date.strftime('%Y-%m-%d')] = None
            else:
                ifeat['properties']['SPI3'][date.strftime('%Y-%m-%d')] = round(istat['nanmean'].astype(float), 3)

    delrange = pd.date_range(today-dt.timedelta(days=365), today-dt.timedelta(days=335))
    # create time series per nuts region
    for ifeat in tmp_nuts3.iterfeatures():
        # check if file exists
        ts_outpath = '/mnt/CEPH_PROJECTS/ADO/JSON/timeseries/NUTS3_' + ifeat['properties']['NUTS_ID'] + '_tmpspi3.json'
        ts_list = list()
        for idate in ifeat['properties']['SPI3'].items():
            ts_list.append({'date': idate[0], 'spi3': idate[1]})
        # write time series
        with open(ts_outpath, 'w') as outfile:
            json.dump(ts_list, outfile)

        #remove dates from nuts3 maps
        for ddate in delrange:
            del ifeat['properties']['SPI3'][ddate.date().strftime('%Y-%m-%d')]

    tmp_nuts3_4325 = round_coordinates(tmp_nuts3.to_crs("EPSG:4326"))
    tmp_nuts3_4325.to_file('/mnt/CEPH_PROJECTS/ADO/JSON/spi3-latest.geojson', driver='GeoJSON', encoding='utf-8')


def get_stats(r_path):
    r_stats = zonal_stats("/mnt/CEPH_PROJECTS/ADO/GIS/ERTS89_LAEA/alpinespace_eusalp_NUTS3_simple.shp",
                          r_path,
                          add_stats={'nanmean': mymean})
    return r_stats


def mymean(x):
    return np.nanmean(x)


def round_coordinates(df):
    simpledec = re.compile(r"\d*\.\d+")

    def mround(match):
        return "{:.5f}".format(float(match.group()))

    df.geometry = df.geometry.apply(lambda x: loads(re.sub(simpledec, mround, x.wkt)))

    return df


if __name__ == "__main__":
    main()