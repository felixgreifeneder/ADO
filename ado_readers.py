import xarray as xr
import os
import pandas as pd
import datetime as dt
import numpy as np
from pathlib import Path
from rasterio.warp import transform
from ado_tools import compute_anomaly


def get_t_label(x):
    if x == 2:
        depth_label = 'SWI_002'
    elif x == 5:
        depth_label = 'SWI_005'
    elif x == 10:
        depth_label = 'SWI_010'
    elif x == 15:
        depth_label = 'SWI_015'
    elif x == 20:
        depth_label = 'SWI_020'
    elif x == 40:
        depth_label = 'SWI_040'
    elif x == 60:
        depth_label = 'SWI_060'
    elif x == 100:
        depth_label = 'SWI_100'
    else:
        print("Depth not available")
        return None
    return depth_label


def extr_ts_copernicus_sm(x, y, depth=None, basepath='/mnt/CEPH_PROJECTS/ADO/SWI/', anom=False, return_pandas=True):
    # get depth label
    if depth is not None:
        depth_label = [get_t_label(x) for x in depth]
    else:
        depth_label = ['SWI_002', 'SWI_005', 'SWI_010', 'SWI_015', 'SWI_020', 'SWI_040', 'SWI_060', 'SWI_100']

    # multidataset method
    # get file paths
    sm_files = list()
    for path in Path(basepath).rglob('*_adoext.nc'):
        sm_files.append(path)

    sm_df = xr.open_mfdataset(sm_files,
                              combine='by_coords',
                              parallel=True)

    # extract time-series
    sm_df = sm_df.chunk({'time': 1934, 'lat': 1, 'lon': 1})
    sm_ts = sm_df[depth_label].sel(lon=x, lat=y, method='nearest')

    # save as temporary file
    tmppath = Path(basepath, 'tmpts.nc')
    sm_ts.to_netcdf(tmppath)
    sm_ts.close()
    sm_df.close()
    sm_ts = xr.open_dataset(tmppath)
    os.remove(tmppath)

    # mask invalid values
    sm_ts = sm_ts.where(sm_ts <= 100)

    if anom:
        sm_ts = _compute_climatology(sm_ts)
        return sm_ts

    if return_pandas:
        return sm_ts.to_dataframe()
    else:
        return sm_ts


def _ISMN_date_parser(x):
    datestr = x
    return dt.datetime.strptime(datestr, '%Y/%m/%d %H:%M')


def read_ISMN_data(network, station, basepath='/mnt/CEPH_PROJECTS/ADO/SM/reference_data/ISMN/'):
    # extract all measurements available for a station and store it in an xarray
    stationpath = Path(basepath, network, station)

    # create a list of all files in the directory
    stm_filelist = list()
    for path in stationpath.rglob('*.stm'):
        stm_filelist.append(path)
    csv_filelist = list()
    for path in stationpath.rglob('*.csv'):
        csv_filelist.append(path)

    # extract data from datafiles
    arraylist = list()
    for istm in stm_filelist:
        istm_attrs = istm.name.split('_')
        istm_var = istm_attrs[3]
        istm_depth = [istm_attrs[4], istm_attrs[5]]
        istm_sensor = istm_attrs[6]
        istm_data = pd.read_csv(istm,
                                header=0,
                                names=['nominal date', 'nominal time', 'actual date', 'actual time', 'CSE',
                                       'network', 'station', 'lat', 'lon',
                                       'elevation', 'depth from', 'depth to', 'value', 'ismn_qflag', 'dp_qflag'],
                                index_col=False,
                                delim_whitespace=True)
        tmp_date = [dt.datetime.strptime(x, '%Y/%m/%d %H:%M') for x in
                    istm_data['actual date'] + ' ' + istm_data['actual time']]
        istm_data.index = tmp_date
        # keep only samples with ISMN qflag 'G'
        istm_data = istm_data.where(istm_data['ismn_qflag'] == 'G', np.nan).dropna()
        tmp_xarray = xr.DataArray(data=istm_data['value'],
                                  coords={'time': istm_data.index},
                                  dims=['time'],
                                  attrs={'varname': istm_var,
                                         'depth': istm_depth,
                                         'sensor': istm_sensor,
                                         'lon': istm_data['lon'].iloc[0],
                                         'lat': istm_data['lat'].iloc[0]},
                                  name=istm_var + istm_depth[1] + istm_sensor)
        arraylist.append(tmp_xarray.copy())

    # merge in one dataset
    station_ds = xr.merge(arraylist)
    # read metadata
    if len(csv_filelist) > 0:
        metadata = pd.read_csv(csv_filelist[0], sep=';')
        station_ds.attrs['metadata'] = metadata
    station_ds.attrs['lon'] = arraylist[0].lon
    station_ds.attrs['lat'] = arraylist[0].lat

    return station_ds


def read_SWI_stack(basepath='/mnt/CEPH_PROJECTS/ADO/SM/SWI/', preprocess=False):

    def swi_preprocess(xrdat):
        SWI_labels = ['SWI_002', 'SWI_005', 'SWI_010', 'SWI_015',
                      'SWI_020', 'SWI_040', 'SWI_060', 'SWI_100']
        # apply masks
        xrdat = xrdat[SWI_labels].where(xrdat[SWI_labels] <= 100)
        xrdat = xrdat.where(xrdat.isin(xrdat['SWI_002'].flag_values * 0.5) == False)
        # crop SWI
        xrdat = xrdat.where((xrdat.lat >= 45.8238) & (xrdat.lat <= 47.8024) &
                            (xrdat.lon >= 5.9202) & (xrdat.lon <= 10.5509),
                            drop=True)

        return xrdat

    sm_files = list()
    for path in Path(basepath).rglob('*_adoext.nc'):
        sm_files.append(path)

    sm_df = xr.open_mfdataset(sm_files,
                              concat_dim='time',
                              parallel=True,
                              #engine='h5netcdf',
                              preprocess=swi_preprocess if preprocess else None)

    return sm_df


def get_SWISS_ref_grid(basepath='/mnt/CEPH_PROJECTS/ADO/SM/reference_data/swiss_model/'):
    # initiate sm stack
    tmp = xr.open_rasterio(basepath + 'fcp_20190101.tif')
    # del tmp['band']
    # coordinate transformation
    tmp.attrs['crs'] = 'EPSG:21781'
    ny, nx = len(tmp['y']), len(tmp['x'])
    x, y = np.meshgrid(tmp['x'], tmp['y'])
    lon, lat = transform(tmp.crs, {'init': 'EPSG:4326'},
                         x.flatten(), y.flatten())
    lon = np.asarray(lon).reshape((ny, nx))
    lat = np.asarray(lat).reshape((ny, nx))
    tmp.coords['lon'] = (('y', 'x'), lon)
    tmp.coords['lat'] = (('y', 'x'), lat)

    ref_grid = xr.DataArray(np.full((len(lat[:, 0]), len(lon[0, :])), -9999.0),
                            coords=[('lat', lat[:, 0]), ('lon', lon[0, :])],
                            name='ref_grid')
    tmp.close()
    return ref_grid


def read_SWISS_SM(basepath='/mnt/CEPH_PROJECTS/ADO/SWI/reference_data/swiss_model/', format='tif'):
    # multidataset method
    # get file paths
    sm_files = list()
    date_list = list()

    if format == 'tif':
        for path in Path(basepath).rglob('*.tif'):
            sm_files.append(path)
            date_list.append(dt.datetime.strptime(path.name[4:12], '%Y%m%d'))
    elif format == 'asc':
        for path in Path(basepath).rglob('*.asc'):
            sm_files.append(path)
            date_list.append(dt.datetime.strptime(path.name[4:12], '%Y%m%d'))

    # initiate sm stack
    tmp = xr.open_rasterio(basepath + 'fcp_20190101.tif')
    #del tmp['band']
    # coordinate transformation
    tmp.attrs['crs'] = 'EPSG:21781'
    ny, nx = len(tmp['y']), len(tmp['x'])
    x, y = np.meshgrid(tmp['x'], tmp['y'])
    lon, lat = transform(tmp.crs, {'init': 'EPSG:4326'},
                         x.flatten(), y.flatten())
    lon = np.asarray(lon).reshape((ny, nx))
    lat = np.asarray(lat).reshape((ny, nx))
    tmp.coords['lon'] = (('y', 'x'), lon)
    tmp.coords['lat'] = (('y', 'x'), lat)

    stack = xr.DataArray(np.full((365, len(lat[:, 0]), len(lon[0, :])), -9999.0),
                         coords=[('time', date_list), ('lat', lat[:, 0]), ('lon', lon[0, :])],
                         name='SM_2019')
    tmp.close()
    tmp = None

    # read all sm files
    sm_data_list = list()
    for (path, date) in zip(sm_files, date_list):

        if format == 'tif':
            tmp = xr.open_rasterio(path)
            stack.loc[dict(time=date)] = tmp.values.squeeze()
            tmp.close()
        elif format == 'asc':
            tmp = pd.read_csv(path,
                              sep=" ",
                              header=None,
                              skiprows=6)
            stack.loc[dict(time=date)] = tmp.values

    return stack


def get_SWISS_anomalies(basepath='/mnt/CEPH_PROJECTS/ADO/SM/reference_data/swiss_model_anomalies/',
                         zid=None, start='01-01-2014', stop='31-12-2018', monthly=False, return_absolutes=False):
    # read the time-series for all points
    fcp = pd.read_csv(basepath + 'ch_500_eval.FCP',
                      header=None, sep=" ", skipinitialspace=True, parse_dates=True,
                      date_parser=lambda x: dt.datetime.strptime(x, '%Y%m%d'), index_col=0)
    fcp.index.rename('time', inplace=True)

    # temporal subset
    fcp = fcp.loc[fcp.index.isin(pd.date_range(start, stop))]

    if monthly:
        fcp = fcp.rolling(30).mean()

    # calulate anomalies
    anomalies = compute_anomaly(fcp)

    if zid is None:
        if return_absolutes:
            return anomalies, fcp
        else:
            return anomalies
    else:
        if return_absolutes:
            return anomalies[zid], fcp[zid]
        else:
            return anomalies[zid]

def get_ERA5Land_stack():
    era_path = '/mnt/CEPH_PROJECTS/ADO/ZAMG/QM/era5_era5l/volumetric_soil_water_layer/era5l/'

    sm_files = list()
    for path in Path(era_path).rglob('*.nc'):
        sm_files.append(path)

    era_stack = xr.open_mfdataset(sm_files,
                                  concat_dim='time',
                                  parallel=True)

    era_stack = era_stack.rename({'longitude': 'lon', 'latitude': 'lat'})
    # era_stack = era_stack['swvl2_0001']
    era_stack = era_stack.sortby('time')

    # mask 1
    era_stack = era_stack.where((era_stack >= 0) & (era_stack <= 1))

    return era_stack


def get_ERA5QM_stack():
    era_path = '/mnt/CEPH_PROJECTS/ADO/ZAMG/QM/era5_era5l/volumetric_soil_water_layer/qm/'

    sm_files = list()
    for path in Path(era_path).rglob('*.nc'):
        sm_files.append(path)

    era_stack = xr.open_mfdataset(sm_files,
                                  concat_dim='time',
                                  parallel=True)

    # era_stack = era_stack.rename({'longitude': 'lon', 'latitude': 'lat'})
    # era_stack = era_stack['swvl2_0001']

    # mask 1
    era_stack = era_stack.where((era_stack >= 0) & (era_stack <= 1))

    return era_stack


def get_ERA5_stack():
    era_path = '/mnt/CEPH_PROJECTS/ADO/ZAMG/QM/era5_era5l/volumetric_soil_water_layer/era5/'

    sm_files = list()
    for path in Path(era_path).rglob('*.nc'):
        sm_files.append(path)

    era_stack = xr.open_mfdataset(sm_files,
                                  concat_dim='time',
                                  parallel=True)

    era_stack = era_stack.rename({'longitude': 'lon', 'latitude': 'lat'})
    era_stack = era_stack.sortby('time')
    #era_stack = era_stack['swvl2_0001']

    # mask 1
    era_stack = era_stack.where((era_stack >= 0) & (era_stack <= 1))

    return era_stack


def get_LISFLOOD_stack():
    lisflood_path = '/mnt/CEPH_PROJECTS/ADO/SM/LISFLOOD/'

    sm_files = list()
    for path in Path(lisflood_path).glob('*.nc'):
        sm_files.append(path)

    lisflood_stack = xr.open_mfdataset(sm_files,
                                       concat_dim='time',
                                       parallel=True)
    lisflood_stack = lisflood_stack.sortby('time')

    return lisflood_stack.resample(time='1D').mean()


def get_UERRA_stack():
    uerra_path = '/mnt/CEPH_PROJECTS/ADO/ZAMG/UERRA/derived/full/volumetric_soil_moisture/'

    sm_files = list()
    for path in Path(uerra_path).rglob('*.nc'):
        sm_files.append(path)

    uerra_stack = xr.open_mfdataset(sm_files,
                                    concat_dim='time',
                                    parallel=True)
    uerra_stack = uerra_stack.sortby('time')

    return uerra_stack.resample(time='1D').mean()


def get_CCI_stack():
    cci_path = '/mnt/CEPH_PROJECTS/ADO/SM/CCI/combined/'

    sm_files = list()
    for path in Path(cci_path).glob('*.nc'):
        sm_files.append(path)

    cci_stack = xr.open_mfdataset(sm_files,
                                  concat_dim='time',
                                  parallel=True)
    cci_stack = cci_stack.sortby('time')
    cci_stack = cci_stack.where(cci_stack.flag == 0)

    return cci_stack


def get_PREVAH_point_ts(site_code, basepath='/mnt/CEPH_PROJECTS/ADO/SM/reference_data/prevah/hydromodell_smex_idealized/'):

    filename = 'mem_' + site_code + '_FCP.dat'
    prevah_ts = pd.read_csv(basepath + filename, header=None, skiprows=1, usecols=[1, 2], names=['date', 'FCP'],
                            sep=" ", index_col=0, date_parser=lambda x: dt.datetime.strptime(str(x), '%Y%m%d'),
                            parse_dates=True)
    return prevah_ts


def get_SwissSMEX_ts(site_code, basepath='/mnt/CEPH_PROJECTS/ADO/SM/reference_data/SwissSMEX_sm _int/SwissSMEX_sm _int/'):
    for path in Path(basepath).glob(site_code + '*'):
        filename = path

    smexts = pd.read_csv(filename, sep=' ', skipinitialspace=True, skiprows=7, parse_dates=True,
                         date_parser=lambda x: dt.datetime.strptime(x, '%Y-%m-%d'), header=0, index_col=0)

    return smexts


def get_Mazia_ts(site_code, basepath='/mnt/CEPH_PROJECTS/ADO/SM/reference_data/Mazia/'):
    pos = pd.read_csv(basepath + 'Permanent_stations.csv', sep=',', skipinitialspace=True, header=0, index_col=0)
    lon = pos.loc[site_code].lon
    lat = pos.loc[site_code].lat

    tss = pd.read_csv(basepath + 'SWC5_day.csv', sep=',', skipinitialspace=True,
                      parse_dates=True,
                      date_parser=lambda x: dt.datetime.strptime(x, '%Y-%m-%d'),
                      header=0,
                      index_col=0)

    tsout = tss[site_code]

    return lon, lat, tsout