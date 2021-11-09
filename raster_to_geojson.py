from rasterstats import zonal_stats
from pathlib import Path
import geopandas as gpd
import numpy as np
import datetime as dt

def main():
    # read shpfile
    nuts3 = gpd.read_file('/mnt/CEPH_PROJECTS/ADO/GIS/ERTS89_LAEA/alpinespace_eusalp_NUTS3.shp')

    basepath = Path('/mnt/CEPH_PROJECTS/ADO/ARSO/ERA5_QM/SPI-3/')
    spi_list = list()
    for path in basepath.glob('*.tif'):
        spi_list.append(path)

    # sort list of paths
    spi_list = sorted(spi_list, key=lambda i: dt.datetime.strptime(i.name[6:14], '%Y%m%d'))
    spi_list = np.array(spi_list)[[dt.datetime.strptime(x.name[6:14], '%Y%m%d').year == 2018 for x in spi_list]]

    # add SPI property
    dictlist = [dict() for i in range(272)]
    nuts3 = nuts3.assign(SPI3=dictlist)

    # calculate statistics
    for i_path in spi_list.tolist():
        r_stats = get_stats(i_path)
        datestring = i_path.name[6:14]
        print(datestring)
        for ifeat, istat in zip(nuts3.iterfeatures(), r_stats):
            if np.isinf(istat['nanmean']):
                ifeat['properties']['SPI3'][datestring] = None
            else:
                ifeat['properties']['SPI3'][datestring] = istat['nanmean'].astype(float)

    nuts3.to_file('/mnt/CEPH_PROJECTS/ADO/PNG/SPI_3/SPI3_2018_ts.geojson', driver='GeoJSON', encoding='utf-8')


def get_stats(r_path):
    r_stats = zonal_stats("/mnt/CEPH_PROJECTS/ADO/GIS/ERTS89_LAEA/alpinespace_eusalp_NUTS3.shp",
                          r_path,
                          add_stats={'nanmean': mymean})
    return r_stats


def mymean(x):
    return np.nanmean(x)


if __name__ == "__main__":
    main()