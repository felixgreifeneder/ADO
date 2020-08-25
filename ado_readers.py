import xarray as xr
import os
import pandas as pd
import datetime as dt
import numpy as np
from pathlib import Path


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


def read_ISMN_data(network, station, basepath='/mnt/CEPH_PROJECTS/ADO/SWI/reference_data/'):
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


def _compute_climatology(xrds):
    r = xrds.rolling(time=30).median().dropna("time")
    # compute daily anomaly
    med_anom = (r.groupby("time.dayofyear") - r.groupby("time.dayofyear").median("time")) / \
               r.groupby("time.dayofyear").std("time")
    return med_anom.to_dataframe().droplevel(0)
