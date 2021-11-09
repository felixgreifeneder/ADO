import xarray as xr
from ado_readers import get_ERA5_stack
from ado_readers import get_ERA5QM_stack
from ado_tools import compute_climatology_stack
import rioxarray
import rasterio.crs as CRS
from ado_tools import get_ado_extent
import numpy as np


def compute_climatology():
    # get ERA5
    #ERA5 = get_ERA5_stack()
    ERA5 = get_ERA5QM_stack()
    ERA5 = ERA5.sel(time=slice('1981-01-01', '2020-12-31'))

    # compute climatology
    clim_avg, clim_std = compute_climatology_stack(ERA5, dekad=True)
    clim_avg.to_netcdf('/mnt/CEPH_PROJECTS/ADO/SM/ERA5_ERA5l_QM/climatology/ERA5qm_clim_avg_1981_2020.nc')
    clim_std.to_netcdf('/mnt/CEPH_PROJECTS/ADO/SM/ERA5_ERA5l_QM/climatology/ERA5qm_clim_std_1981_2020.nc')
    clim_avg.close()
    clim_std.close()


def resample_interpolate_ERA5(grid):
    # interpolate ERA 5 to match to the defined grid and resolution
    # set projection
    grid.rio.write_crs("EPSG:4326", inplace=True)
    grid_laea = grid.rio.reproject('EPSG:3035')

    # interpolate
    adoextent = get_ado_extent('LAEA')
    # define grid target coordinates
    x_target = np.arange(adoextent[0], adoextent[2], 5000) + 2500
    #y_target = np.arange(adoextent[1], adoextent[3], 5000) + 2500
    y_target = np.arange(adoextent[3], adoextent[1], -5000) + 2500
    # interpolate
    grid_laea = grid_laea.interp(x=x_target, y=y_target, method='linear')

    return grid_laea


def compute_anomalies():
    # load climatology
    clim_avg = xr.open_dataset('/mnt/CEPH_PROJECTS/ADO/SM/ERA5_ERA5l_QM/climatology/ERA5qm_clim_avg_1981_2020.nc')
    clim_std = xr.open_dataset('/mnt/CEPH_PROJECTS/ADO/SM/ERA5_ERA5l_QM/climatology/ERA5qm_clim_std_1981_2020.nc')

    # iterate through all avaialable years and compute the anomaly
    for iy in range(1981, 2021):
        filename = '/mnt/CEPH_PROJECTS/ADO/ZAMG/QM/era5_era5l/volumetric_soil_water_layer/qm/' + str(iy) + \
                   '_swvl_daily_era5_eusalp.nc'
        era_stack = xr.open_dataset(filename, chunks={'time': 20})
        # anom_stack = (era_stack.groupby('time.dayofyear') - clim_avg).groupby('time.dayofyear') / clim_std
        anom_stack = xr.apply_ufunc(
            lambda x, m, s: (x - m) / s,
            era_stack.groupby('time.dayofyear'),
            clim_avg,
            clim_std,
            keep_attrs=True,
            dask='allowed'
        ) # in ther alternative calculation anoamlies are normalized with the mean instead of standard deviation

        # reproject, resample, and write to disk
        for id in anom_stack.time:
            outname = '/mnt/CEPH_PROJECTS/ADO/SM/ERA5_ERA5l_QM/anomalies_long_ref/era5qm_sm_anom_' + \
                      id.dt.strftime('%Y%m%d').values + '.tif'

            res_grid = resample_interpolate_ERA5(anom_stack.swvl.sel(time=id))
            res_grid.rio.to_raster(outname)

        era_stack.close()
        anom_stack.close()

