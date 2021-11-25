import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
import rioxarray
import rasterio
import os

from ado_tools import get_ado_extent


def main():
    # define paths
    vhi_path = '/mnt/CEPH_PROJECTS/ADO/VHI/03_results/vhi/'
    sma_path = '/mnt/CEPH_PROJECTS/ADO/SM/ERA5_ERA5l_QM/anomalies_long_ref/'
    spi3_path = '/mnt/CEPH_PROJECTS/ADO/ARSO/ERA5_QM_NEW/SPI-3/'
    spi1_path = '/mnt/CEPH_PROJECTS/ADO/ARSO/ERA5_QM_NEW/SPI-1/'

    cdi_path = '/mnt/CEPH_PROJECTS/ADO/CDI/'

    # iterate through time series
    drange = pd.date_range('2001-01-01', '2020-12-31', freq='1D')

    # iterate through days
    for iday in drange:
        # read data
        # vhi
        vhi = read_vhi(iday, vhi_path)
        spi1 = read_SPI(iday, spi1_path, prefix='spi30_')
        spi1_m1 = read_SPI(iday - dt.timedelta(days=30), spi1_path, prefix='spi30_')
        spi3 = read_SPI(iday, spi3_path, prefix='spi90_')
        spi3_m1 = read_SPI(iday - dt.timedelta(days=30), spi3_path, prefix='spi90_')
        sma = read_sma(iday, sma_path)

        cdi = calc_level(vhi, spi1, spi3, spi3_m1, spi1_m1, sma)

        outname = cdi_path + 'CDI_' + iday.strftime('%Y%m%d') + '.tif'
        cdi.rio.to_raster(outname, crs='epsg:3035', dtype=np.float32)


def calc_level(vhi, spi1, spi3, spi3_m1, spi1_m1, sma):
    level = xr.DataArray(data=0, coords=vhi.coords, name='CDI')
    # level.attrs = {'crs': '+init=epsg:3035'}
    # watch
    tmp = ((spi1 < -2) | (spi3 < - 1))
    level = level.where(tmp == False, other=1)
    # warning
    tmp = ((sma < - 1) & ((spi1 < -2) | (spi3 < - 1)))
    level = level.where(tmp == False, other=2)
    # alert
    tmp = ((vhi < 50) & ((spi1 < -2) | (spi3 < - 1)))
    level = level.where(tmp == False, other=3)
    # partial recovery
    tmp = (((vhi < 50) & ((spi3_m1 < -1) & (spi3 > -1))) |
           ((vhi < 50) & ((spi1_m1 < -2) & (spi1 > -2))))
    level = level.where(tmp == False, other=4)
    # full recovery
    tmp = (((spi3_m1 < -1) & (spi3 > -1)) |
           ((spi1_m1 < -2) & (spi1 > -1)))
    level = level.where(tmp == False, other=5)

    return level


def read_vhi(day, basepath):
    # calculate time step
    vhi_dpath = ''
    doy4step = (day.dayofyear // 4) * 4 + 4
    if doy4step > 364:
        doy4step = 364
    while not os.path.exists(vhi_dpath):
        vhi_dpath = basepath + 'vhi_4d_' + str(day.year) + '_' + f'{doy4step:03d}' + '.tif'
        # if vhi_dpath doesn't exist step to the previous acquisition
        doy4step = doy4step - 4
    vhi = xr.open_rasterio(vhi_dpath)
    # mask no data values
    vhi = vhi.where(vhi != 255)
    return resample_interpolate(vhi, method='downsample')


def read_SPI(day, basepath, prefix=''):
    spi_dpath = basepath + prefix + day.strftime('%Y%m%d') + '.tif'
    spi = xr.open_rasterio(spi_dpath)
    # mask no data value
    spi = spi.where(spi != -3.4e+38)
    return resample_interpolate(spi)


def read_sma(day, basepath):
    sma_path = basepath + 'era5qm_sm_anom_' + day.strftime('%Y%m%d') + '.tif'
    sma = xr.open_rasterio(sma_path)
    sma = sma.sel({'band': 2})
    # mask no data value
    sma = sma.where(sma != -9999)
    return resample_interpolate(sma)


def resample_interpolate(grid, method='interpolate'):
    # interpolate VHI to match to the defined grid and resolution

    # interpolate
    adoextent = get_ado_extent('LAEA')
    # define grid target coordinates
    x_target = np.arange(adoextent[0], adoextent[2], 5000) + 2500
    # y_target = np.arange(adoextent[1], adoextent[3], 5000) + 2500
    y_target = np.arange(adoextent[3], adoextent[1], -5000) + 2500
    # regrid
    if method == 'downsample':
        grid_out = grid.coarsen(x=20, y=20, boundary='pad').mean()
        grid_out = grid_out.interp(x=x_target, y=y_target, method='linear')
    elif method == 'interpolate':
        grid_out = grid.interp(x=x_target, y=y_target, method='linear')
    return grid_out


if __name__ == "__main__":
    main()
