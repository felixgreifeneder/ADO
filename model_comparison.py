import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import datetime as dt
import ado_readers
import numpy as np

import geopandas as gp
from ado_readers import get_SWISS_anomalies
from ado_readers import get_ERA5_stack
from ado_readers import get_ERA5Land_stack
from ado_readers import get_ERA5QM_stack
from ado_readers import get_LISFLOOD_stack
from ado_readers import get_UERRA_stack
from ado_tools import mask_array
from ado_tools import compute_anomaly
from ado_tools import shp_to_raster
from ado_tools import compute_anomaly_stack
from ado_tools import transform_to_custom
from ado_readers import get_PREVAH_point_ts
from ado_readers import get_SwissSMEX_ts
from pytesmo.metrics import tcol_metrics

def sm_swiss_vs_swi(valpath='/mnt/CEPH_PROJECTS/ADO/SWI/validation/switzerland/'):
    # initiation
    swiss_stack = ado_readers.read_SWISS_SM(format='asc')
    swi_stack = ado_readers.read_SWI_stack()

    depth_label = ['SWI_002', 'SWI_005', 'SWI_010', 'SWI_015', 'SWI_020', 'SWI_040', 'SWI_060', 'SWI_100']
    selected = ['SWI_002']

    # greater regions
    region_mask = xr.open_rasterio('/mnt/CEPH_PROJECTS/ADO/SWI/validation/switzerland/masks/z.tif')

    # mask 1
    for i in depth_label:
        swi_stack[i] = swi_stack[i].where(swi_stack[i] <= 100)
        swi_stack[i] = swi_stack[i].where(swi_stack[i].isin(swi_stack[i].flag_values * 0.5) == False)

    swiss_stack = swiss_stack.where(swiss_stack > -9999)
    swiss_stack = swiss_stack.where(swiss_stack <= 1)

    # crop and interpolate swi dataset
    # swi_stack = swi_stack.where((swi_stack.lat >= swiss_stack.lat.min()) & (swi_stack.lat <= swiss_stack.lat.max()) &
    #                             (swi_stack.lon >= swiss_stack.lon.min()) & (swi_stack.lon <= swiss_stack.lon.max()),
    #                             drop=True)
    # apply regional mask
    # region_mask = region_mask.interp(y=swiss_stack.lat, x=swiss_stack.lon, method='nearest').squeeze()
    # swiss_stack = swiss_stack.where(region_mask == 1, drop=True)
    swi_stack = swi_stack.interp(lat=swiss_stack.lat, lon=swiss_stack.lon)
    swi_stack = swi_stack.sel(time=slice('2019-05-01', '2019-09-30'))
    swiss_stack = swiss_stack.interp(time=swi_stack.time)
    # swiss_stack = swiss_stack.sel(time=slice('2019-05-01', '2019-09-30'))
    # swiss_stack = swiss_stack.interp_like(swi_stack)

    # mask 2
    for i in depth_label:
        swi_stack[i] = swi_stack[i].where(swiss_stack > -9999)
        swi_stack[i] = swi_stack[i].where(swiss_stack <= 1)

    swiss_stack = swiss_stack.where(swi_stack['SWI_002'] <= 100)
    swiss_stack = swiss_stack.where(swi_stack['SWI_002'].isin(swi_stack['SWI_002'].flag_values * 0.5) == False)

    # combine
    merged = xr.merge([swiss_stack, swi_stack])

    # plots
    corrmap = merged.copy()
    for i in depth_label:
        corrmap[i] = xr.corr(merged[i], merged['SM_2019'], dim='time')
    print('aa')

    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    nplot = 0
    for i in range(2):
        for j in range(4):
            corrmap[depth_label[nplot]].plot.hist(ax[i, j])
            nplot = nplot + 1
    plt.tight_layout()
    plt.savefig(valpath + 'temporal_cor_pixel.png')
    plt.close()

    spatial_cor = pd.DataFrame(data=None, index=swi_stack.time.values)
    for i in depth_label:
        spatial_cor[i] = xr.corr(merged[i], merged['SM_2019'], dim=('lat', 'lon'))

    spatial_cor.plot()
    plt.savefig(valpath + 'spatial_cor.png')
    plt.close()

    corlist = list()

    for i in depth_label:
        swi_stack[i].median(dim='time').plot(vmin=0, vmax=100)
        plt.savefig(valpath + i + '/mean_swi_2019.png')
        plt.close()
        swi_stack[i].std(dim='time').plot(vmin=0, vmax=20)
        plt.savefig(valpath + i + '/std_swi_2019.png')
        plt.close()
        swi_stack[i].min(dim='time').plot(vmin=0, vmax=100)
        plt.savefig(valpath + i + '/min_swi_2019.png')
        plt.close()
        swi_stack[i].max(dim='time').plot(vmin=0, vmax=100)
        plt.savefig(valpath + i + '/max_swi_2019.png')
        plt.close()

        swiss_stack.median(dim='time').plot(vmin=0, vmax=1)
        plt.savefig(valpath + i + '/mean_swiss_2019.png')
        plt.close()
        swiss_stack.std(dim='time').plot(vmin=0, vmax=0.2)
        plt.savefig(valpath + i + '/std_swiss_2019.png')
        plt.close()
        swiss_stack.min(dim='time').plot(vmin=0, vmax=1)
        plt.savefig(valpath + i + '/min_swiss_2019.png')
        plt.close()
        swiss_stack.max(dim='time').plot(vmin=0, vmax=1)
        plt.savefig(valpath + i + '/max_swiss_2019.png')
        plt.close()

        corrmap[i].plot(vmin=-0.6, vmax=0.6)
        plt.savefig(valpath + i + '/corr_2019.png')
        plt.close()

        ts1 = swi_stack[i].mean(dim=('lat', 'lon'), skipna=True)
        ts2 = swiss_stack.mean(dim=('lat', 'lon'), skipna=True)
        ts1.plot()
        plt.savefig(valpath + i + '/swits_2019.png')
        plt.close()
        ts2.plot()
        plt.savefig(valpath + i + '/swissts_2019.png')
        plt.close()

        merged.mean(dim=('lat', 'lon'), skipna=True).plot.scatter(x=i, y='SM_2019')
        plt.savefig(valpath + i + '/meanscatter_2019.png')
        plt.close()

        avgcor = xr.corr(ts1, ts2, dim='time').values
        print(avgcor)
        corlist.append(avgcor)

        # for (i, name) in zip(range(5, 10), ['may', 'jun', 'jul', 'aug', 'sep']):
        #     swiss_stack.sel(time=slice('2019-0' + str(i) + '-01', '2019-0' + str(i) + '-30')).plot(x="lon", y="lat",
        #                                                                                            col='time', col_wrap=5)
        #     plt.savefig('/mnt/CEPH_PROJECTS/ADO/SWI/validation/switzerland/swiss_' + name + '.tif')
        #     plt.close()
        #     swi_002_crop.sel(time=slice('2019-0' + str(i) + '-01', '2019-0' + str(i) + '-30')).plot(x="lon", y="lat",
        #                                                                                             col='time', col_wrap=5)
        #     plt.savefig('/mnt/CEPH_PROJECTS/ADO/SWI/validation/switzerland/swi_' + name + '.tif')
        #     plt.close()

    # create a plot giving the correlation related to t
    tvalues = [2, 5, 10, 15, 20, 40, 60, 100]
    print(corlist)
    plt.plot(tvalues, corlist)
    plt.xlabel('t-value')
    plt.ylabel('R')
    plt.savefig(valpath + 'RbyT.png')
    plt.close()


def sm_swiss_vs_era5land(valpath='/mnt/CEPH_PROJECTS/ADO/SWI/validation/switzerland/'):
    # initiation
    swiss_stack = ado_readers.read_SWISS_SM(format='asc')
    era_stack = xr.open_dataset('/mnt/CEPH_PROJECTS/ADO/SWI/reference_data/era5land2019.nc')
    era_stack = era_stack.rename({'longitude': 'lon', 'latitude': 'lat'})
    era_stack = era_stack['swvl2']

    # greater regions
    region_mask = xr.open_rasterio('/mnt/CEPH_PROJECTS/ADO/SWI/validation/switzerland/masks/z.tif')

    # mask 1
    era_stack = era_stack.where((era_stack >= 0) & (era_stack <= 1))
    swiss_stack = swiss_stack.where(swiss_stack > -9999)
    swiss_stack = swiss_stack.where(swiss_stack <= 1)

    # apply regional mask
    # region_mask = region_mask.interp(y=swiss_stack.lat, x=swiss_stack.lon, method='nearest').squeeze()
    # swiss_stack = swiss_stack.where(region_mask == 1, drop=True)
    era_stack = era_stack.interp(lat=swiss_stack.lat, lon=swiss_stack.lon)
    era_stack = era_stack.sel(time=slice('2019-05-01', '2019-09-30'))
    swiss_stack = swiss_stack.interp(time=era_stack.time)

    # mask 2
    era_stack = era_stack.where(era_stack > -9999)
    era_stack = era_stack.where(era_stack <= 1)

    swiss_stack = swiss_stack.where((era_stack >= 0) & (era_stack <= 1))

    # combine
    merged = xr.merge([swiss_stack, era_stack])

    # plots
    corrmap = xr.corr(merged['swvl2'], merged['SM_2019'], dim='time')
    corrmap.plot.hist()
    plt.savefig(valpath + 'temporal_cor_pixel.png')
    plt.close()

    spatial_cor = xr.corr(merged['swvl2'], merged['SM_2019'], dim=('lat', 'lon'))
    spatial_cor.plot()
    plt.savefig(valpath + 'spatial_cor.png')
    plt.close()

    corlist = list()

    era_stack.median(dim='time').plot(vmin=0, vmax=1, cmap='viridis_r')
    plt.savefig(valpath + '/mean_era_2019.png')
    plt.close()
    era_stack.std(dim='time').plot(vmin=0, vmax=0.2, cmap='viridis_r')
    plt.savefig(valpath + '/std_era_2019.png')
    plt.close()
    era_stack.min(dim='time').plot(vmin=0, vmax=1, cmap='viridis_r')
    plt.savefig(valpath + '/min_era_2019.png')
    plt.close()
    era_stack.max(dim='time').plot(vmin=0, vmax=1, cmap='viridis_r')
    plt.savefig(valpath + '/max_era_2019.png')
    plt.close()

    swiss_stack.median(dim='time').plot(vmin=0, vmax=1, cmap='viridis_r')
    plt.savefig(valpath + '/mean_swiss_2019.png')
    plt.close()
    swiss_stack.std(dim='time').plot(vmin=0, vmax=0.2, cmap='viridis_r')
    plt.savefig(valpath + '/std_swiss_2019.png')
    plt.close()
    swiss_stack.min(dim='time').plot(vmin=0, vmax=1, cmap='viridis_r')
    plt.savefig(valpath + '/min_swiss_2019.png')
    plt.close()
    swiss_stack.max(dim='time').plot(vmin=0, vmax=1, cmap='viridis_r')
    plt.savefig(valpath + '/max_swiss_2019.png')
    plt.close()

    corrmap.plot(vmin=-1, vmax=1, cmap='RdBu')
    plt.savefig(valpath + '/corr_2019.png')
    plt.close()

    ts1 = era_stack.mean(dim=('lat', 'lon'), skipna=True)
    ts2 = swiss_stack.mean(dim=('lat', 'lon'), skipna=True)
    ts1.plot()
    plt.savefig(valpath + '/erats_2019.png')
    plt.close()
    ts2.plot()
    plt.savefig(valpath + '/swissts_2019.png')
    plt.close()

    merged.mean(dim=('lat', 'lon'), skipna=True).plot.scatter(x='swvl2', y='SM_2019')
    plt.savefig(valpath + '/meanscatter_2019.png')
    plt.close()


def era5_downscaling(outpath='/mnt/CEPH_PROJECTS/ADO/SWI/validation/switzerland/', eps=0.0001):
    from sklearn import preprocessing
    from sklearn.svm import LinearSVR
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LinearRegression, BayesianRidge
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split

    # initiation
    # modelled
    swiss_stack = ado_readers.read_SWISS_SM(format='asc')
    # era5
    era_stack = xr.open_dataset('/mnt/CEPH_PROJECTS/ADO/SWI/reference_data/era52019.nc')
    era_stack = era_stack.rename({'longitude': 'lon', 'latitude': 'lat'})
    era_stack = era_stack['swvl2']
    # dem
    elev = xr.open_rasterio('/mnt/CEPH_PROJECTS/ADO/EU_DEM/eudem_WGS84_adoext.tif')
    elev = elev.rename({'x': 'lon', 'y': 'lat'})
    slop = xr.open_rasterio('/mnt/CEPH_PROJECTS/ADO/EU_DEM/eudem_slope_WGS84_adoext.tif')
    slop = slop.rename({'x': 'lon', 'y': 'lat'})
    aspt = xr.open_rasterio('/mnt/CEPH_PROJECTS/ADO/EU_DEM/eudem_aspect_WGS84_adoext.tif')
    aspt = aspt.rename({'x': 'lon', 'y': 'lat'})
    # soil type
    soil = xr.open_rasterio('/mnt/CEPH_PROJECTS/ADO/GIS/EUSR5000/eusr5000.tif')
    soil = soil.rename({'x': 'lon', 'y': 'lat'})

    # mask 1
    era_stack = era_stack.where((era_stack >= 0) & (era_stack <= 1))
    swiss_stack = swiss_stack.where(swiss_stack > -9999)
    swiss_stack = swiss_stack.where(swiss_stack <= 1)

    # interpolate
    era_stack = era_stack.interp(lat=swiss_stack.lat, lon=swiss_stack.lon)
    era_stack = era_stack.sel(time=slice('2019-05-01', '2019-09-30'))
    swiss_stack = swiss_stack.interp(time=era_stack.time)
    elev = elev.interp(lat=swiss_stack.lat, lon=swiss_stack.lon)
    slop = slop.interp(lat=swiss_stack.lat, lon=swiss_stack.lon)
    aspt = aspt.interp(lat=swiss_stack.lat, lon=swiss_stack.lon)
    soil = soil.interp(lat=swiss_stack.lat, lon=swiss_stack.lon, method='nearest')

    # mask 2
    era_stack = era_stack.where(swiss_stack > -9999)
    era_stack = era_stack.where(swiss_stack <= 1)

    swiss_stack = swiss_stack.where((era_stack >= 0) & (era_stack <= 1))

    # construct training data
    training_set = pd.DataFrame({'elev': np.tile(elev.data.ravel(), 153),
                                 'slop': np.tile(slop.data.ravel(), 153),
                                 'aspt': np.tile(aspt.data.ravel(), 153),
                                 'soil': np.tile(soil.data.ravel(), 153),
                                 'era5': era_stack.data.ravel()})
    training_set = training_set.dropna()
    target = swiss_stack.data.ravel()
    target = target[training_set.index]

    # testing on a different date
    test_set = pd.DataFrame({'elev': elev.data.ravel(),
                             'slop': slop.data.ravel(),
                             'aspt': aspt.data.ravel(),
                             'soil': soil.data.ravel(),
                             'era5': era_stack.sel(time='2019-08-01').data.ravel()})
    training_target = swiss_stack.sel(time='2019-08-01').data.ravel()

    # scaling
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(training_set)
    x_training = scaler.transform(training_set)

    grid = {'alpha': 10.0 ** -np.arange(1, 7)}
    model = MLPRegressor(random_state=12,
                         epsilon=eps)
    for i, j in zip(np.split(x_training, range(100000, x_training.shape[0] - 100000, 100000), axis=0),
                    np.split(target, range(100000, x_training.shape[0] - 100000, 100000))):
        # fitting the downscaling model
        model.partial_fit(i, j)
    print(model.best_loss_)

    # testing
    nrows = test_set.shape[0]
    test_set = test_set.dropna()
    training_target = training_target[test_set.index]

    # prediction
    x_test = scaler.transform(test_set)
    sm_pred = model.predict(x_test)
    pred_score = model.score(x_test, training_target)
    print(pred_score)
    # reshaping
    sm_pred_full = np.full(nrows, np.nan)
    sm_pred_full[test_set.index] = sm_pred
    sm_pred_array = np.reshape(sm_pred_full, (1, era_stack.shape[1], era_stack.shape[2]))
    # applying scaling to era5
    era5_downscaled = swiss_stack.sel(time='2019-08-01')
    era5_downscaled.data = sm_pred_array

    # ploting the results
    era_stack.sel(time='2019-08-01').plot(vmin=0, vmax=1, cmap='viridis_r')
    plt.savefig(outpath + 'era5.png')
    plt.close()
    swiss_stack.sel(time='2019-08-01').plot(vmin=0, vmax=1, cmap='viridis_r')
    plt.savefig(outpath + 'swiss.png')
    plt.close()
    era5_downscaled.plot(vmin=0, vmax=1, cmap='viridis_r')
    plt.savefig(outpath + 'downscaled.png')
    plt.close()
    diff = swiss_stack.sel(time='2019-08-01') - era5_downscaled
    diff.plot(vmin=-0.2, vmax=0.2, cmap='RdBu')
    plt.savefig(outpath + 'diff.png')
    plt.close()
    diff.mean(dim=('lat', 'lon'))
    xr.corr(swiss_stack, era5_downscaled, dim=('lat', 'lon'))

    return model


def sm_swiss_ts_comparison(year, monthly=False, anomalies=True, includeSWI=False,
                           outpath='/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/noSWI_long_1995_2018/time_series/absolute/'):
    # get shape file
    areas = gp.read_file('/mnt/CEPH_PROJECTS/ADO/SM/reference_data/swiss_model_anomalies/shapefile_307_regions'
                         '/Regions_307_wgs84.shp')
    areas3035 = gp.read_file('/mnt/CEPH_PROJECTS/ADO/SM/reference_data/swiss_model_anomalies/shapefile_307_regions'
                             '/shapefile_307_regions_3035.shp')
    areasUERRA = gp.read_file('/mnt/CEPH_PROJECTS/ADO/SM/reference_data/swiss_model_anomalies/shapefile_307_regions'
                         '/Regions_307_uerra.shp')

    # get SWISS anomalies
    SWISS, SWISSabs = get_SWISS_anomalies(start='2011-01-01', stop='31-12-2018', monthly=monthly, return_absolutes=True)

    # get SWI
    if includeSWI:
        SWI = ado_readers.read_SWI_stack()
        SWI = SWI.sel(time=slice('2015-01-01', '2018-12-31'))
        SWI_labels = ['SWI_002', 'SWI_005', 'SWI_010', 'SWI_015',
                      'SWI_020', 'SWI_040', 'SWI_060', 'SWI_100']
        # apply masks
        SWI = SWI[SWI_labels].where(SWI[SWI_labels] <= 100)
        SWI = SWI.where(SWI.isin(SWI['SWI_002'].flag_values * 0.5) == False)
        # crop SWI
        SWI = SWI.where((SWI.lat >= 45.8238) & (SWI.lat <= 47.8024) &
                        (SWI.lon >= 5.9202) & (SWI.lon <= 10.5509),
                        drop=True)

        SWI = SWI.compute()

    # get ERA5
    ERA5 = get_ERA5_stack()
    ERA5 = ERA5.sel(time=slice('2011-01-01', '2018-12-31')).swvl
    # scale to SWISS model
    ERA5 = ERA5.sel(depthBelowLandLayer=0) * 0.108 + ERA5.sel(depthBelowLandLayer=7) * 0.323 + ERA5.sel(
        depthBelowLandLayer=28) * 0.569
    # ERA5 = ERA5.interp(lat=SWI.lat, lon=SWI.lon)

    # get ERA5 land
    ERA5l = get_ERA5Land_stack()
    ERA5l = ERA5l.sel(time=slice('2011-01-01', '2018-12-31')).swvl
    # scale to SWISS model
    ERA5l = ERA5l.sel(depthBelowLandLayer=0) * 0.108 + ERA5l.sel(depthBelowLandLayer=7) * 0.323 + ERA5l.sel(
        depthBelowLandLayer=28) * 0.569
    # ERA5l = ERA5l.interp(lat=SWI.lat, lon=SWI.lon)

    # get ERA5 QM
    ERA5qm = get_ERA5QM_stack()
    ERA5qm = ERA5qm.sel(time=slice('2011-01-01', '2018-12-31')).swvl
    ERA5qm = ERA5qm.sel(depthBelowLandLayer=0) * 0.108 + ERA5qm.sel(depthBelowLandLayer=7) * 0.323 + ERA5qm.sel(
        depthBelowLandLayer=28) * 0.569
    # ERA5qm = ERA5qm.interp(lat=SWI.lat, lon=SWI.lon)

    # get LISFLOOD
    LISFLOOD = get_LISFLOOD_stack()
    LISFLOOD = LISFLOOD.sel(time=slice('2011-01-01', '2018-12-31')).vsw
    LISFLOOD = LISFLOOD.sel(soilLayer=0) * 0.077 + LISFLOOD.sel(soilLayer=1) * 0.923

    # uerra
    UERRA = get_UERRA_stack()
    UERRA = UERRA.sel(time=slice('2011-01-01', '2018-12-31')).vsw
    UERRA = UERRA.isel(step=0)
    UERRA = UERRA.isel(soilLayer=0) * 0.025 + UERRA.isel(soilLayer=1)*0.075 + UERRA.isel(soilLayer=2)*0.15 + UERRA.isel(
        soilLayer=3)*0.25 + UERRA.isel(soilLayer=4)*0.5

    # get SIWSSMEX locations
    swmx = pd.read_csv('/mnt/CEPH_PROJECTS/ADO/SM/reference_data/swissmex_w_prevah_regions.csv')

    # cycle through area ids
    # for i in range(1, 307):
    for i, swmxname in zip(swmx['join_ID'], swmx['field_1']):
        aoi = areas.query("ID == " + str(i)).reset_index(drop=True)
        aoi3035 = areas3035.query("ID == " + str(i)).reset_index(drop=True)
        aoiUERRA = areasUERRA.query('ID ==' + str(i)).reset_index(drop=True)
        # apply masks
        if includeSWI:
            SWI_masked = mask_array(aoi, SWI)
            SWI_ts = SWI_masked.mean(dim=('lat', 'lon'))
        ERA5_masked = mask_array(aoi, ERA5)
        ERA5_ts = ERA5_masked.mean(dim=('lat', 'lon'))
        ERA5l_masked = mask_array(aoi, ERA5l)
        ERA5l_ts = ERA5l_masked.mean(dim=('lat', 'lon'))
        ERA5qm_masked = mask_array(aoi, ERA5qm)
        ERA5qm_ts = ERA5qm_masked.mean(dim=('lat', 'lon'))
        LISFLOOD_masked = mask_array(aoi3035, LISFLOOD)
        LISFLOOD_ts = LISFLOOD_masked.mean(dim=('x', 'y'))
        UERRA_masked = mask_array(aoiUERRA, UERRA)
        UERRA_ts = UERRA_masked.mean(dim=('x', 'y'))

        SWISSabs_ts = SWISSabs.loc[:, i]
        SWISS_ts = SWISS.loc[:, i]

        if monthly:
            SWISSabs_ts = SWISSabs_ts.rolling('10D').mean()
            if includeSWI:
                SWI_ts = SWI_ts.chunk({'time': SWI_ts.shape[0]})
                SWI_ts = SWI_ts.rolling(time=10).mean()
            ERA5_ts = ERA5_ts.chunk({'time': ERA5_ts.shape[0]})
            ERA5_ts = ERA5_ts.rolling(time=10).mean()
            ERA5l_ts = ERA5l_ts.chunk({'time': ERA5l_ts.shape[0]})
            ERA5l_ts = ERA5l_ts.rolling(time=10).mean()
            ERA5qm_ts = ERA5qm_ts.chunk({'time': ERA5qm_ts.shape[0]})
            ERA5qm_ts = ERA5qm_ts.rolling(time=10).mean()
            LISFLOOD_ts = LISFLOOD_ts.chunk({'time': LISFLOOD_ts.shape[0]})
            LISFLOOD_ts = LISFLOOD_ts.rolling(time=10).mean()
            UERRA_ts = UERRA_ts.chunk({'time': UERRA_ts.shape[0]})
            UERRA_ts = UERRA_ts.rolling(time=10).mean()

        if anomalies:
            SWISS_clim, SWISS_clim_std, SWISS_ts = compute_anomaly(SWISS_ts, return_clim=True)
            ERA5_anomaly = compute_anomaly(ERA5_ts.to_series())
            ERA5_anomaly.index = pd.DatetimeIndex(ERA5_anomaly.index.date)
            ERA5l_anomaly = compute_anomaly(ERA5l_ts.to_series())
            ERA5l_anomaly.index = pd.DatetimeIndex(ERA5l_anomaly.index.date)
            ERA5qm_anomaly = compute_anomaly(ERA5qm_ts.to_series())
            ERA5qm_anomaly.index = pd.DatetimeIndex(ERA5qm_anomaly.index.date)
            LISFLOOD_anomaly = compute_anomaly(LISFLOOD_ts.to_series())
            LISFLOOD_anomaly.index = pd.DatetimeIndex(LISFLOOD_anomaly.index.date)
            UERRA_anomaly = compute_anomaly(UERRA_ts.to_series())
            UERRA_anomaly.index = pd.DatetimeIndex(UERRA_anomaly.index.date)

            if includeSWI:
                SWI_anomaly = compute_anomaly(SWI_ts.to_series())
                SWI_anomaly.index = pd.DatetimeIndex(SWI_anomaly.index.date)
                merged = pd.concat({'SWISS': SWISS_ts,
                                    'SWI': SWI_anomaly,
                                    'ERA5': ERA5_anomaly,
                                    'ERA5l': ERA5l_anomaly,
                                    # 'ERA5qm': ERA5qm_anomaly,
                                    'LISFLOOD': LISFLOOD_anomaly}, axis=1)
            else:
                merged = pd.concat({'SWISS': SWISS_ts,
                                    'ERA5': ERA5_anomaly,
                                    'ERA5l': ERA5l_anomaly,
                                    # 'ERA5qm': ERA5qm_anomaly,
                                    'LISFLOOD': LISFLOOD_anomaly,
                                    'UERRA': UERRA_anomaly}, axis=1)
        else:
            SWISS_clim, SWISS_clim_std, _ = compute_anomaly(SWISSabs_ts, return_clim=True)
            ERA5_ts = ERA5_ts.to_series()
            ERA5_ts.index = pd.DatetimeIndex(ERA5_ts.index.date)
            ERA5l_ts = ERA5l_ts.to_series()
            ERA5l_ts.index = pd.DatetimeIndex(ERA5l_ts.index.date)
            ERA5qm_ts = ERA5qm_ts.to_series()
            ERA5qm_ts.index = pd.DatetimeIndex(ERA5qm_ts.index.date)
            LISFLOOD_ts = LISFLOOD_ts.to_series()
            LISFLOOD_ts.index = pd.DatetimeIndex(LISFLOOD_ts.index.date)
            UERRA_ts = UERRA_ts.to_series()
            UERRA_ts.index = pd.DatetimeIndex(UERRA_ts.index.date)
            if includeSWI:
                SWI_ts = SWI_ts.to_series()
                SWI_ts.index = pd.DatetimeIndex(SWI_ts.index.date)
                merged = pd.concat({'SWISS': SWISSabs_ts,
                                    'SWI': SWI_ts / 100,
                                    'ERA5': ERA5_ts,
                                    'ERA5l': ERA5l_ts,
                                    # 'ERA5qm': ERA5qm_ts,
                                    'LISFLOOD': LISFLOOD_ts,
                                    'UERRA': UERRA_ts}, axis=1)
            else:
                merged = pd.concat({'SWISS': SWISSabs_ts,
                                    'ERA5': ERA5_ts,
                                    'ERA5l': ERA5l_ts,
                                    # 'ERA5qm': ERA5qm_ts,
                                    'LISFLOOD': LISFLOOD_ts,
                                    'UERRA': UERRA_ts}, axis=1)

        create_ts_plots(merged, SWISS_clim, SWISS_clim_std, year,
                        outpath, swmxname, anomalies)


def sm_swiss_ts_avg_plot(year,
                         basepath='/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/noSWI_long_1995_2018/time_series/absolute/',
                         outpath='/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/noSWI_long_1995_2018/time_series/absolute/',
                         anomaly=False):
    # get SIWSSMEX locations
    swmx = pd.read_csv('/mnt/CEPH_PROJECTS/ADO/SM/reference_data/swissmex_w_prevah_regions.csv')

    cntr = 0
    for i, swmxname in zip(swmx['join_ID'], swmx['field_1']):
        cntr = cntr + 1
        # load csv
        i_ts = pd.read_csv(basepath + swmxname + '_' + str(year) + '.csv', header=0, index_col=0, parse_dates=True)
        if cntr == 1:
            avg_ts = i_ts
        else:
            avg_ts = avg_ts + i_ts

    avg_ts = avg_ts / cntr

    if anomaly:
        SWISS_clim, SWISS_clim_std, SWISS_ts = compute_anomaly(avg_ts['SWISS'], return_clim=True)
        ERA5_anomaly = compute_anomaly(avg_ts['ERA5'])
        ERA5_anomaly.index = pd.DatetimeIndex(ERA5_anomaly.index.date)
        ERA5l_anomaly = compute_anomaly(avg_ts['ERA5l'])
        ERA5l_anomaly.index = pd.DatetimeIndex(ERA5l_anomaly.index.date)
        LISFLOOD_anomaly = compute_anomaly(avg_ts['LISFLOOD'])
        LISFLOOD_anomaly.index = pd.DatetimeIndex(LISFLOOD_anomaly.index.date)
        UERRA_anomaly = compute_anomaly(avg_ts['UERRA'])
        UERRA_anomaly.index = pd.DatetimeIndex(UERRA_anomaly.index.date)
        avg_ts = pd.concat({'SWISS': SWISS_ts,
                            'ERA5': ERA5_anomaly,
                            'ERA5l': ERA5l_anomaly,
                            'LISFLOOD': LISFLOOD_anomaly,
                            'UERRA': UERRA_anomaly}, axis=1)
    else:

        # compute SWISS climatology
        SWISS_clim, SWISS_clim_std, _ = compute_anomaly(avg_ts['SWISS'], return_clim=True)

    create_ts_plots(avg_ts, SWISS_clim, SWISS_clim_std, year,
                    outpath, 'swmx_avg',
                    anomaly)


def create_ts_plots(merged, SWISS_clim, SWISS_clim_std, year, outpath, name, anomalies=False):
    # statistics
    pearsonr = merged.corr()
    spearman = merged.corr('spearman')

    # create plots
    fig = plt.figure(figsize=(15.5, 5))
    gs0 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs0[0])
    gs01 = gs0[1].subgridspec(2, 2)

    ax2 = fig.add_subplot(gs01[0, 0])
    ax3 = fig.add_subplot(gs01[1, 1])
    ax4 = fig.add_subplot(gs01[0, 1])
    ax5 = fig.add_subplot(gs01[1, 0])

    # plot the climatology
    SWISS_clim.index = pd.DatetimeIndex(
        [dt.datetime(year=year, month=1, day=1) + dt.timedelta(days=int(dix) - 1) for dix in
         SWISS_clim.index.values])
    SWISS_clim_std.index = pd.DatetimeIndex(
        [dt.datetime(year=year, month=1, day=1) + dt.timedelta(days=int(dix) - 1) for dix in
         SWISS_clim_std.index.values])
    SWISS_clim.index.freq = 'D'
    SWISS_clim_std.freq = 'D'
    SWISS_clim_neg_2 = SWISS_clim - 2 * SWISS_clim_std
    SWISS_clim_pos_2 = SWISS_clim + 2 * SWISS_clim_std
    if anomalies:
        SWISS_clim[:] = 0
        SWISS_clim_neg_2[:] = -2
        SWISS_clim_pos_2[:] = 2

        SWISS_climatology = pd.concat({'avg': SWISS_clim[str(year) + '-01-01':str(year) + '-12-31'],
                                       '2std+': SWISS_clim_pos_2[str(year) + '-01-01':str(year) + '-12-31'],
                                       '2std-': SWISS_clim_neg_2[str(year) + '-01-01':str(year) + '-12-31']},
                                      axis=1)

        allmerged = pd.concat([merged[str(year) + '-01-01':str(year) + '-12-31'], SWISS_climatology], axis=1)
        allmerged.plot(ax=ax1,
                       style=['b-', 'r-',
                              'm-', 'c-',
                              'y-', 'k-',
                              'k--', 'k--'])
    else:
        SWISS_climatology = pd.concat({'SWISS_clim': SWISS_clim[str(year) + '-01-01':str(year) + '-12-31'],
                                       'SWISS_clim_2std+': SWISS_clim_pos_2[str(year) + '-01-01':str(year) + '-12-31'],
                                       'SWISS_clim_2std-': SWISS_clim_neg_2[str(year) + '-01-01':str(year) + '-12-31']},
                                      axis=1)

        allmerged = pd.concat([merged[str(year) + '-01-01':str(year) + '-12-31'], SWISS_climatology], axis=1)
        allmerged.plot(ax=ax1, secondary_y=['SWISS', 'SWISS_clim', 'SWISS_clim_2std+', 'SWISS_clim_2std-'],
                       style=['b-', 'r-',
                              'm-', 'c-',
                              'y-', 'k-',
                              'k--', 'k--'])

    if anomalies:
        scatterlimmin = -2
        scatterlimmax = 2
    else:
        scatterlimmin = 0
        scatterlimmax = 1
    merged[str(year) + '-01-01':str(year) + '-12-31'].plot.scatter('SWISS', 'ERA5', ax=ax2)
    ax2.set_xlim(scatterlimmin, scatterlimmax)
    ax2.set_ylim(scatterlimmin, scatterlimmax)
    ax2.set_aspect('equal')
    ax2.text(0.1, 0.1, "R: " + "{:10.3f}".format(pearsonr.iloc[0, 2]) + '\n' + "Sp: " + "{:10.3f}".format(
        spearman.iloc[0, 2]), transform=ax2.transAxes)
    merged[str(year) + '-01-01':str(year) + '-12-31'].plot.scatter('SWISS', 'ERA5l', ax=ax3)
    ax3.set_xlim(scatterlimmin, scatterlimmax)
    ax3.set_ylim(scatterlimmin, scatterlimmax)
    ax3.set_aspect('equal')
    ax3.text(0.1, 0.1, "R: " + "{:10.3f}".format(pearsonr.iloc[0, 3]) + '\n' + "Sp: " + "{:10.3f}".format(
        spearman.iloc[0, 3]), transform=ax3.transAxes)
    merged.plot.scatter('SWISS', 'UERRA', ax=ax4)
    ax4.set_xlim(scatterlimmin, scatterlimmax)
    ax4.set_ylim(scatterlimmin, scatterlimmax)
    ax4.set_aspect('equal')
    ax4.text(0.1, 0.1, "R: " + "{:10.3f}".format(pearsonr.iloc[0, 4]) + '\n' + "Sp: " + "{:10.3f}".format(
        spearman.iloc[0, 4]), transform=ax4.transAxes)
    merged[str(year) + '-01-01':str(year) + '-12-31'].plot.scatter('SWISS', 'LISFLOOD', ax=ax5)
    ax5.set_xlim(scatterlimmin, scatterlimmax)
    ax5.set_ylim(scatterlimmin, scatterlimmax)
    ax5.set_aspect('equal')
    ax5.text(0.1, 0.1, "R: " + "{:10.3f}".format(pearsonr.iloc[0, 1]) + '\n' + "Sp: " + "{:10.3f}".format(
        spearman.iloc[0, 1]), transform=ax5.transAxes)
    fig.tight_layout()
    plt.savefig(
        outpath + name + '_' + str(year) + '.png',
        dpi=600)
    plt.close()
    merged.to_csv(
        outpath + name + '_' + str(year) + '.csv')


def sm_swiss_anomalies_spatial_cor(correct_bias=False, monthly=False, includeSWI=True):
    from bias_correction import BiasCorrection
    from ado_readers import get_SWISS_ref_grid

    # get shape file
    areas = gp.read_file('/mnt/CEPH_PROJECTS/ADO/SM/reference_data/swiss_model_anomalies/shapefile_307_regions'
                         '/Regions_307_wgs84.shp')
    areas3035 = gp.read_file('/mnt/CEPH_PROJECTS/ADO/SM/reference_data/swiss_model_anomalies/shapefile_307_regions'
                             '/shapefile_307_regions_3035.shp')
    areasUERRA = gp.read_file('/mnt/CEPH_PROJECTS/ADO/SM/reference_data/swiss_model_anomalies/shapefile_307_regions'
                              '/Regions_307_uerra.shp')

    # get SWISS anomalies
    SWISS, SWISSabs = get_SWISS_anomalies(start='1995-01-01', stop='31-12-2018', monthly=monthly, return_absolutes=True)

    # get SWI
    if includeSWI:
        SWI = ado_readers.read_SWI_stack()
        SWI = SWI.sel(time=slice('2015-01-01', '2018-12-31'))
        SWI_labels = ['SWI_002', 'SWI_005', 'SWI_010', 'SWI_015',
                      'SWI_020', 'SWI_040', 'SWI_060', 'SWI_100']
        # apply masks
        SWI = SWI[SWI_labels].where(SWI[SWI_labels] <= 100)
        SWI = SWI.where(SWI.isin(SWI['SWI_002'].flag_values * 0.5) == False)
        # crop SWI
        SWI = SWI.where((SWI.lat >= 45.8238) & (SWI.lat <= 47.8024) &
                        (SWI.lon >= 5.9202) & (SWI.lon <= 10.5509),
                        drop=True)

        SWI = SWI.compute()

    # get ERA5
    ERA5 = get_ERA5_stack()
    ERA5 = ERA5.sel(time=slice('1995-01-01', '2018-12-31')).swvl
    # scale to SWISS model
    ERA5 = ERA5.sel(depthBelowLandLayer=0) * 0.108 + ERA5.sel(depthBelowLandLayer=7) * 0.323 + ERA5.sel(
        depthBelowLandLayer=28) * 0.569
    # ERA5 = ERA5.interp(lat=SWI.lat, lon=SWI.lon)

    # get ERA5 land
    ERA5l = get_ERA5Land_stack()
    ERA5l = ERA5l.sel(time=slice('1995-01-01', '2018-12-31')).swvl
    # scale to SWISS model
    ERA5l = ERA5l.sel(depthBelowLandLayer=0) * 0.108 + ERA5l.sel(depthBelowLandLayer=7) * 0.323 + ERA5l.sel(
        depthBelowLandLayer=28) * 0.569
    # ERA5l = ERA5l.interp(lat=SWI.lat, lon=SWI.lon)

    # get ERA5 QM
    ERA5qm = get_ERA5QM_stack()
    ERA5qm = ERA5qm.sel(time=slice('1995-01-01', '2018-12-31')).swvl
    # scale to SWISS model
    ERA5qm = ERA5qm.sel(depthBelowLandLayer=0) * 0.108 + ERA5qm.sel(depthBelowLandLayer=7) * 0.323 + ERA5qm.sel(
        depthBelowLandLayer=28) * 0.569
    # ERA5qm = ERA5qm.interp(lat=SWI.lat, lon=SWI.lon)

    # get LISFLOOS
    LISFLOOD = get_LISFLOOD_stack()
    LISFLOOD = LISFLOOD.sel(time=slice('1995-01-01', '2018-12-31')).vsw
    LISFLOOD = LISFLOOD.sel(soilLayer=0) * 0.077 + LISFLOOD.sel(soilLayer=1) * 0.923

    # uerra
    UERRA = get_UERRA_stack()
    UERRA = UERRA.sel(time=slice('2011-01-01', '2018-12-31')).vsw
    UERRA = UERRA.isel(step=0)
    UERRA = UERRA.isel(soilLayer=0) * 0.025 + UERRA.isel(soilLayer=1) * 0.075 + UERRA.isel(
        soilLayer=2) * 0.15 + UERRA.isel(
        soilLayer=3) * 0.25 + UERRA.isel(soilLayer=4) * 0.5

    # rasterize shp
    ref_grid = get_SWISS_ref_grid()
    areas_raster = shp_to_raster(areas, ref_grid)
    pearsonmap = xr.Dataset(coords={'lon': ref_grid.lon, 'lat': ref_grid.lat})
    sd0map = xr.Dataset(coords={'lon': ref_grid.lon, 'lat': ref_grid.lat})
    sd05map = xr.Dataset(coords={'lon': ref_grid.lon, 'lat': ref_grid.lat})
    sd1map = xr.Dataset(coords={'lon': ref_grid.lon, 'lat': ref_grid.lat})
    if includeSWI:
        mapnames = ['SWISSvERA5', 'SWISSvERA5l', 'SWISSvERA5qm',
                    'SWISSvSWI_002', 'SWISSvSWI_005', 'SWISSvSWI_010',
                    'SWISSvSWI_015', 'SWISSvSWI_020', 'SWISSvSWI_040',
                    'SWISSvSWI_060', 'SWISSvSWI_100']
    else:
        mapnames = ['SWISSvERA5', 'SWISSvERA5l', 'SWISSvERA5qm',
                    'SWISSvLISFLOOD', 'SWISSvUERRA']

    for iname in mapnames:
        pearsonmap[iname] = areas_raster.copy(data=np.full(areas_raster.shape, np.nan))
        sd0map[iname] = areas_raster.copy(data=np.full(areas_raster.shape, np.nan))
        sd05map[iname] = areas_raster.copy(data=np.full(areas_raster.shape, np.nan))
        sd1map[iname] = areas_raster.copy(data=np.full(areas_raster.shape, np.nan))

    for i in areas.ID:
        print('Processing area id ' + str(i) + '...')
        # LISFLOOD
        print('Retrieving LISFLOOD data')
        aoi3035 = areas3035.query("ID == " + str(i)).reset_index(drop=True)
        # apply mask
        LISFLOOD_masked = mask_array(aoi3035, LISFLOOD)
        LISFLOOD_ts = LISFLOOD_masked.mean(dim=('x', 'y'))
        LISFLOOD_ts = LISFLOOD_ts.to_series()
        # calculate anomalies
        LISFLOOD_ts_anomalies = compute_anomaly(LISFLOOD_ts, monthly=monthly)
        LISFLOOD_ts_anomalies.index = pd.DatetimeIndex(LISFLOOD_ts.index.date)

        try:
            merged = pd.concat({'SWISS': SWISS.loc[:, i],
                                'LISFLOOD': LISFLOOD_ts_anomalies}, axis=1)
            success = 1
        except:
            success = 0
            continue

        pearsonr = merged.corr()

        # SWISS vs LISFLOOD
        if success == 1:
            pearsonmap['SWISSvLISFLOOD'] = xr.where(areas_raster == i,
                                                    pearsonr.iloc[0, 1],
                                                    pearsonmap['SWISSvLISFLOOD'])
            sd1map['SWISSvLISFLOOD'] = xr.where(areas_raster == i,
                                                sum((merged['SWISS'] < -1) & (
                                                        merged['LISFLOOD'] < -1)) / sum(
                                                    merged['SWISS'] < -1),
                                                sd1map['SWISSvLISFLOOD'])
            sd05map['SWISSvLISFLOOD'] = xr.where(areas_raster == i,
                                                 sum((merged['SWISS'] < -0.5) & (
                                                         merged['LISFLOOD'] < -0.5)) / sum(
                                                     merged['SWISS'] < -0.5),
                                                 sd05map['SWISSvLISFLOOD'])
            sd0map['SWISSvLISFLOOD'] = xr.where(areas_raster == i,
                                                sum((merged['SWISS'] < 0) & (
                                                        merged['LISFLOOD'] < 0)) / sum(
                                                    merged['SWISS'] < 0),
                                                sd0map['SWISSvLISFLOOD'])

        # UERRA
        print('Retrieving UERRA data')
        aoiUERRA = areasUERRA.query("ID == " + str(i)).reset_index(drop=True)
        # apply mask
        UERRA_masked = mask_array(aoiUERRA, UERRA)
        UERRA_ts = UERRA_masked.mean(dim=('x', 'y'))
        UERRA_ts = UERRA_ts.to_series()
        # calculate anomalies
        UERRA_ts_anomalies = compute_anomaly(LISFLOOD_ts, monthly=monthly)
        UERRA_ts_anomalies.index = pd.DatetimeIndex(LISFLOOD_ts.index.date)

        try:
            merged = pd.concat({'SWISS': SWISS.loc[:, i],
                                'UERRA': UERRA_ts_anomalies}, axis=1)
            success = 1
        except:
            success = 0
            continue

        pearsonr = merged.corr()

        # SWISS vs UERRA
        if success == 1:
            pearsonmap['SWISSvUERRA'] = xr.where(areas_raster == i,
                                                 pearsonr.iloc[0, 1],
                                                 pearsonmap['SWISSvUERRA'])
            sd1map['SWISSvUERRA'] = xr.where(areas_raster == i,
                                             sum((merged['SWISS'] < -1) & (
                                                  merged['UERRA'] < -1)) / sum(
                                                  merged['SWISS'] < -1),
                                             sd1map['SWISSvUERRA'])
            sd05map['SWISSvUERRA'] = xr.where(areas_raster == i,
                                              sum((merged['SWISS'] < -0.5) & (
                                                   merged['UERRA'] < -0.5)) / sum(
                                                  merged['SWISS'] < -0.5),
                                              sd05map['SWISSvUERRA'])
            sd0map['SWISSvUERRA'] = xr.where(areas_raster == i,
                                             sum((merged['SWISS'] < 0) & (
                                                  merged['UERRA'] < 0)) / sum(
                                                 merged['SWISS'] < 0),
                                             sd0map['SWISSvUERRA'])

        # ERA5
        print('Processing ERA')
        # compute ERA anomaly for subset
        # ERA5_masked = ERA5.sel(depthBelowLandLayer=ERAdepth).where(areas_raster == i)
        # ERA5_ts = ERA5_masked.mean(dim=('lat', 'lon'), skipna=True)
        # ERA5l_masked = ERA5l.sel(depthBelowLandLayer=ERAdepth).where(areas_raster == i)
        # ERA5l_ts = ERA5l_masked.mean(dim=('lat', 'lon'), skipna=True)
        # ERA5qm_masked = ERA5qm.sel(depthBelowLandLayer=ERAdepth).where(areas_raster == i)
        # ERA5qm_ts = ERA5qm_masked.mean(dim=('lat', 'lon'), skipna=True)
        aoi = areas.query("ID == " + str(i)).reset_index(drop=True)
        # apply mask
        ERA5_masked = mask_array(aoi, ERA5)
        ERA5_ts = ERA5_masked.mean(dim=('lat', 'lon'))
        ERA5l_masked = mask_array(aoi, ERA5l)
        ERA5l_ts = ERA5l_masked.mean(dim=('lat', 'lon'))
        ERA5qm_masked = mask_array(aoi, ERA5qm)
        ERA5qm_ts = ERA5qm_masked.mean(dim=('lat', 'lon'))
        # if ERA5_ts.isnull().all():
        #     print('out of here')
        #     continue
        ERA5_ts = ERA5_ts.to_series()
        if correct_bias:
            bc = BiasCorrection(SWISSabs.loc[:, i], ERA5_ts, ERA5_ts)
            ERA5_ts = bc.correct('basic_quantile')
        ERA5_ts_anomalies = compute_anomaly(ERA5_ts, monthly=monthly)
        ERA5_ts_anomalies.index = pd.DatetimeIndex(ERA5_ts.index.date)

        ERA5l_ts = ERA5l_ts.to_series()
        ERA5l_ts_anomalies = compute_anomaly(ERA5l_ts, monthly=monthly)
        ERA5l_ts_anomalies.index = pd.DatetimeIndex(ERA5l_ts.index.date)

        ERA5qm_ts = ERA5qm_ts.to_series()
        ERA5qm_ts_anomalies = compute_anomaly(ERA5qm_ts, monthly=monthly)
        ERA5qm_ts_anomalies.index = pd.DatetimeIndex(ERA5qm_ts.index.date)

        try:
            merged = pd.concat({'SWISS': SWISS.loc[:, i],
                                'ERA5': ERA5_ts_anomalies,
                                'ERA5l': ERA5l_ts_anomalies,
                                'ERA5qm': ERA5qm_ts_anomalies}, axis=1)
            success = 1
        except:
            success = 0
            continue

        pearsonr = merged.corr()

        # SWISS vs ERA
        if success == 1:
            pearsonmap['SWISSvERA5'] = xr.where(areas_raster == i,
                                                pearsonr.iloc[0, 1],
                                                pearsonmap['SWISSvERA5'])
            pearsonmap['SWISSvERA5l'] = xr.where(areas_raster == i,
                                                 pearsonr.iloc[0, 2],
                                                 pearsonmap['SWISSvERA5l'])
            pearsonmap['SWISSvERA5qm'] = xr.where(areas_raster == i,
                                                  pearsonr.iloc[0, 3],
                                                  pearsonmap['SWISSvERA5qm'])
            sd1map['SWISSvERA5'] = xr.where(areas_raster == i,
                                            sum((merged['SWISS'] < -1) & (
                                                    merged['ERA5'] < -1)) / sum(
                                                merged['SWISS'] < -1),
                                            sd1map['SWISSvERA5'])
            sd05map['SWISSvERA5'] = xr.where(areas_raster == i,
                                             sum((merged['SWISS'] < -0.5) & (
                                                     merged['ERA5'] < -0.5)) / sum(
                                                 merged['SWISS'] < -0.5),
                                             sd05map['SWISSvERA5'])
            sd0map['SWISSvERA5'] = xr.where(areas_raster == i,
                                            sum((merged['SWISS'] < 0) & (
                                                    merged['ERA5'] < 0)) / sum(
                                                merged['SWISS'] < 0),
                                            sd0map['SWISSvERA5'])
            sd1map['SWISSvERA5l'] = xr.where(areas_raster == i,
                                             sum((merged['SWISS'] < -1) & (
                                                     merged['ERA5l'] < -1)) / sum(
                                                 merged['SWISS'] < -1),
                                             sd1map['SWISSvERA5l'])
            sd05map['SWISSvERA5l'] = xr.where(areas_raster == i,
                                              sum((merged['SWISS'] < -0.5) & (
                                                      merged['ERA5l'] < -0.5)) / sum(
                                                  merged['SWISS'] < -0.5),
                                              sd05map['SWISSvERA5l'])
            sd0map['SWISSvERA5l'] = xr.where(areas_raster == i,
                                             sum((merged['SWISS'] < 0) & (
                                                     merged['ERA5l'] < 0)) / sum(
                                                 merged['SWISS'] < 0),
                                             sd0map['SWISSvERA5l'])
            sd1map['SWISSvERA5qm'] = xr.where(areas_raster == i,
                                              sum((merged['SWISS'] < -1) & (
                                                      merged['ERA5qm'] < -1)) / sum(
                                                  merged['SWISS'] < -1),
                                              sd1map['SWISSvERA5qm'])
            sd05map['SWISSvERA5qm'] = xr.where(areas_raster == i,
                                               sum((merged['SWISS'] < -0.5) & (
                                                       merged['ERA5qm'] < -0.5)) / sum(
                                                   merged['SWISS'] < -0.5),
                                               sd05map['SWISSvERA5qm'])
            sd0map['SWISSvERA5qm'] = xr.where(areas_raster == i,
                                              sum((merged['SWISS'] < 0) & (
                                                      merged['ERA5qm'] < 0)) / sum(
                                                  merged['SWISS'] < 0),
                                              sd0map['SWISSvERA5qm'])

        if includeSWI:
            # compute SWI anomaly per subset per depth
            for ilabel in SWI_labels:
                print('...' + ilabel)
                SWI_masked = SWI[ilabel].where(areas_raster == i)
                SWI_ts = SWI_masked.mean(dim=('lat', 'lon'), skipna=True)
                SWI_ts = SWI_ts.to_series()
                if correct_bias:
                    bc = BiasCorrection(SWISSabs.loc[:, i], SWI_ts, SWI_ts)
                    SWI_ts = bc.correct('basic_quantile')
                SWI_ts_anomalies = compute_anomaly(SWI_ts, monthly=monthly)
                SWI_ts_anomalies.index = pd.DatetimeIndex(SWI_ts_anomalies.index.date)
                # compute correlations
                try:
                    merged = pd.concat({'SWISS': SWISS.loc[:, i],
                                        'SWI': SWI_ts_anomalies}, axis=1)
                    success = 1
                except:
                    success = 0
                    continue

                # statistics
                pearsonr = merged.corr()
                outlabelSWISS = 'SWISSv' + ilabel
                pearsonmap[outlabelSWISS] = xr.where(areas_raster == i, pearsonr.iloc[0, 1], pearsonmap[outlabelSWISS])
                sd1map[outlabelSWISS] = xr.where(areas_raster == i,
                                                 sum((merged['SWISS'] < -1) & (merged['SWI'] < -1)) / sum(
                                                     merged['SWISS'] < -1),
                                                 sd1map[outlabelSWISS])
                sd05map[outlabelSWISS] = xr.where(areas_raster == i,
                                                  sum((merged['SWISS'] < -0.5) & (merged['SWI'] < -0.5)) / sum(
                                                      merged['SWISS'] < -0.5),
                                                  sd05map[outlabelSWISS])
                sd0map[outlabelSWISS] = xr.where(areas_raster == i,
                                                 sum((merged['SWISS'] < 0) & (merged['SWI'] < 0)) / sum(
                                                     merged['SWISS'] < 0),
                                                 sd0map[outlabelSWISS])

                # plot time series for SWI_002
                # if ilabel == 'SWI_002':
                #     # create plots
                #     fig = plt.figure(figsize=(15.5, 5))
                #     gs0 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
                #     ax1 = fig.add_subplot(gs0[0])
                #     gs01 = gs0[1].subgridspec(2, 2)
                #
                #     ax2 = fig.add_subplot(gs01[0, 0])
                #     ax3 = fig.add_subplot(gs01[1, 1])
                #     ax4 = fig.add_subplot(gs01[0, 1])
                #     ax5 = fig.add_subplot(gs01[1, 0])
                #     # fig, axs = plt.subplots(1, 2, squeeze=True, figsize=(15, 5), gridspec_kw={'width_ratios': [2, 1]})
                #     merged['2017-01-01':'2018-12-31'].plot(ax=ax1)
                #     ax1.hlines(0, dt.datetime(2015, 1, 1), dt.datetime(2018, 12, 31), linestyles='dashed', colors='k')
                #
                #     scatterlimmin = -2
                #     scatterlimmax = 2
                #
                #     merged.plot.scatter('SWISS', 'ERA5', ax=ax2)
                #     ax2.set_xlim(scatterlimmin, scatterlimmax)
                #     ax2.set_ylim(scatterlimmin, scatterlimmax)
                #     ax2.set_aspect('equal')
                #     ax2.text(0.1, 0.1, "R: " + "{:10.3f}".format(pearsonr.iloc[0, 2]) + '\n' + "Sp: " + "{:10.3f}".format(
                #         spearman.iloc[0, 2]), transform=ax2.transAxes)
                #     merged.plot.scatter('SWISS', 'ERA5l', ax=ax3)
                #     ax3.set_xlim(scatterlimmin, scatterlimmax)
                #     ax3.set_ylim(scatterlimmin, scatterlimmax)
                #     ax3.set_aspect('equal')
                #     ax3.text(0.1, 0.1, "R: " + "{:10.3f}".format(pearsonr.iloc[0, 3]) + '\n' + "Sp: " + "{:10.3f}".format(
                #         spearman.iloc[0, 3]), transform=ax3.transAxes)
                #     merged.plot.scatter('SWISS', 'ERA5qm', ax=ax4)
                #     ax4.set_xlim(scatterlimmin, scatterlimmax)
                #     ax4.set_ylim(scatterlimmin, scatterlimmax)
                #     ax4.set_aspect('equal')
                #     ax4.text(0.1, 0.1, "R: " + "{:10.3f}".format(pearsonr.iloc[0, 4]) + '\n' + "Sp: " + "{:10.3f}".format(
                #         spearman.iloc[0, 4]), transform=ax4.transAxes)
                #     merged.plot.scatter('SWISS', 'SWI', ax=ax5)
                #     ax5.set_xlim(scatterlimmin, scatterlimmax)
                #     ax5.set_ylim(scatterlimmin, scatterlimmax)
                #     ax5.set_aspect('equal')
                #     ax5.text(0.1, 0.1, "R: " + "{:10.3f}".format(pearsonr.iloc[0, 1]) + '\n' + "Sp: " + "{:10.3f}".format(
                #         spearman.iloc[0, 1]), transform=ax5.transAxes)
                #     fig.tight_layout()
                #     plt.savefig('/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/switzerland_anom_ts/id' + str(i) + '.png',
                #                 dpi=600)
                #     plt.close()
                #     merged.to_csv('/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/switzerland_anom_ts/id' + str(i) + '.csv')

    # save maps to disk
    outpath = '/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/noSWI_long_1995_2018_dinterpol/corrmaps/'
    for ilabel in pearsonmap.keys():
        plt.rcParams.update({'font.size': 8})
        fig = plt.figure(figsize=(2.91, 2.18))
        ax = pearsonmap[ilabel].plot(vmin=-1, vmax=1, cmap='RdBu')
        plt.text(0.1, 0.1, "Average: " + "{:10.3f}".format(
            pearsonmap[ilabel].mean().values) + '\n' + "Stddev.: " + "{:10.3f}".format(
            pearsonmap[ilabel].std().values), transform=ax.axes.transAxes)
        plt.tight_layout()
        plt.savefig(outpath + 'pearson/' + ilabel + '.png', dpi=600)
        plt.close()

        fig = plt.figure(figsize=(2.91, 2.18))
        pearsonmap[ilabel].plot.hist(bins=np.linspace(-1, 1, 20), density=True)
        plt.title('')
        plt.ylim(0, 3)
        plt.tight_layout()
        plt.savefig(outpath + 'pearson/' + ilabel + '_hist.png', dpi=600)
        plt.close()

    outpath = '/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/noSWI_long_1995_2018_dinterpol/probamaps/'
    for ilabel in pearsonmap.keys():
        fig = plt.figure(figsize=(2.91, 2.18))
        ax = sd1map[ilabel].plot(vmin=0, vmax=1)
        plt.text(0.1, 0.1, "Average: " + "{:10.3f}".format(
            sd1map[ilabel].mean().values) + '\n' + "Stddev.: " + "{:10.3f}".format(
            sd1map[ilabel].std().values), transform=ax.axes.transAxes)
        plt.tight_layout()
        plt.savefig(outpath + 'sd1_' + ilabel + '.png', dpi=600)
        plt.close()

        fig = plt.figure(figsize=(2.91, 2.18))
        ax = sd05map[ilabel].plot(vmin=0, vmax=1)
        plt.text(0.1, 0.1, "Average: " + "{:10.3f}".format(
            sd05map[ilabel].mean().values) + '\n' + "Stddev.: " + "{:10.3f}".format(
            sd05map[ilabel].std().values), transform=ax.axes.transAxes)
        plt.tight_layout()
        plt.savefig(outpath + 'sd05_' + ilabel + '.png', dpi=600)
        plt.close()

        fig = plt.figure(figsize=(2.91, 2.18))
        ax = sd0map[ilabel].plot(vmin=0, vmax=1)
        plt.text(0.1, 0.1, "Average: " + "{:10.3f}".format(
            sd0map[ilabel].mean().values) + '\n' + "Stddev.: " + "{:10.3f}".format(
            sd0map[ilabel].std().values), transform=ax.axes.transAxes)
        plt.tight_layout()
        plt.savefig(outpath + 'sd0_' + ilabel + '.png', dpi=600)
        plt.close()

    pearsonmap.to_netcdf('/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/noSWI_long_1995_2018_dinterpol/corrmaps/pearson/corrmaps.nc')
    sd1map.to_netcdf('/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/noSWI_long_1995_2018_dinterpol/probamaps/sd1.nc')
    sd05map.to_netcdf('/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/noSWI_long_1995_2018_dinterpol/probamaps/sd05.nc')
    sd0map.to_netcdf('/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/noSWI_long_1995_2018_dinterpol/probamaps/sd0.nc')


def corrmaps_replot():
    pearsonmap = xr.open_dataset('/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/corrmaps/pearson/corrmaps.nc')
    sd1map = xr.open_dataset('/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/probabmaps/sd1.nc')
    sd05map = xr.open_dataset('/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/probabmaps/sd05.nc')
    sd0map = xr.open_dataset('/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/probabmaps/sd0.nc')

    # save maps to disk
    outpath = '/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/corrmaps/'
    for ilabel in pearsonmap.keys():
        plt.rcParams.update({'font.size': 8})
        fig = plt.figure(figsize=(2.91, 2.18))
        ax = pearsonmap[ilabel].plot(vmin=-1, vmax=1, cmap='RdBu')
        plt.text(0.1, 0.1, "Average: " + "{:10.3f}".format(
            pearsonmap[ilabel].mean().values) + '\n' + "Stddev.: " + "{:10.3f}".format(
            pearsonmap[ilabel].std().values), transform=ax.axes.transAxes)
        plt.tight_layout()
        plt.savefig(outpath + 'pearson/' + ilabel + '.png', dpi=600)
        plt.close()

        fig = plt.figure(figsize=(2.91, 2.18))
        pearsonmap[ilabel].plot.hist(bins=np.linspace(-1, 1, 20), density=True)
        plt.title('')
        plt.tight_layout()
        plt.savefig(outpath + 'pearson/' + ilabel + '_hist.png', dpi=600)
        plt.close()

    outpath = '/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/probabmaps/'
    for ilabel in pearsonmap.keys():
        fig = plt.figure(figsize=(2.91, 2.18))
        ax = sd1map[ilabel].plot(vmin=0, vmax=1)
        plt.text(0.1, 0.1, "Average: " + "{:10.3f}".format(
            sd1map[ilabel].mean().values) + '\n' + "Stddev.: " + "{:10.3f}".format(
            sd1map[ilabel].std().values), transform=ax.axes.transAxes)
        plt.tight_layout()
        plt.savefig(outpath + 'sd1_' + ilabel + '.png', dpi=600)
        plt.close()

        fig = plt.figure(figsize=(2.91, 2.18))
        ax = sd05map[ilabel].plot(vmin=0, vmax=1)
        plt.text(0.1, 0.1, "Average: " + "{:10.3f}".format(
            sd05map[ilabel].mean().values) + '\n' + "Stddev.: " + "{:10.3f}".format(
            sd05map[ilabel].std().values), transform=ax.axes.transAxes)
        plt.tight_layout()
        plt.savefig(outpath + 'sd05_' + ilabel + '.png', dpi=600)
        plt.close()

        fig = plt.figure(figsize=(2.91, 2.18))
        ax = sd0map[ilabel].plot(vmin=0, vmax=1)
        plt.text(0.1, 0.1, "Average: " + "{:10.3f}".format(
            sd0map[ilabel].mean().values) + '\n' + "Stddev.: " + "{:10.3f}".format(
            sd0map[ilabel].std().values), transform=ax.axes.transAxes)
        plt.tight_layout()
        plt.savefig(outpath + 'sd0_' + ilabel + '.png', dpi=600)
        plt.close()


def SwissSMEX_ts_comp_points(year=None, monthly=False, anomalies=True, includeSWI=False,
                             outpath='', interval=1):

    # get SWI
    if includeSWI:
        SWI = ado_readers.read_SWI_stack()
        SWI = SWI.sel(time=slice('2015-01-01', '2018-12-31'))
        SWI_labels = ['SWI_002', 'SWI_005', 'SWI_010', 'SWI_015',
                      'SWI_020', 'SWI_040', 'SWI_060', 'SWI_100']
        # apply masks
        SWI = SWI[SWI_labels].where(SWI[SWI_labels] <= 100)
        SWI = SWI.where(SWI.isin(SWI['SWI_002'].flag_values * 0.5) == False)
        # crop SWI
        SWI = SWI.where((SWI.lat >= 45.8238) & (SWI.lat <= 47.8024) &
                        (SWI.lon >= 5.9202) & (SWI.lon <= 10.5509),
                        drop=True)

        SWI = SWI.compute()

    # get ERA5
    ERA5 = get_ERA5_stack()
    ERA5 = ERA5.sel(time=slice('2009-01-01', '2018-12-31')).swvl
    # scale to SWISS model
    ERA5 = ERA5.sel(depthBelowLandLayer=0) * 0.127 + ERA5.sel(depthBelowLandLayer=7) * 0.382 + ERA5.sel(
        depthBelowLandLayer=28) * 0.491
    # ERA5 = ERA5.interp(lat=SWI.lat, lon=SWI.lon)

    # get ERA5 land
    ERA5l = get_ERA5Land_stack()
    ERA5l = ERA5l.sel(time=slice('2009-01-01', '2018-12-31')).swvl
    # scale to SWISS model
    ERA5l = ERA5l.sel(depthBelowLandLayer=0) * 0.127 + ERA5l.sel(depthBelowLandLayer=7) * 0.382 + ERA5l.sel(
        depthBelowLandLayer=28) * 0.491
    # ERA5l = ERA5l.interp(lat=SWI.lat, lon=SWI.lon)

    # get ERA5 QM
    ERA5qm = get_ERA5QM_stack()
    ERA5qm = ERA5qm.sel(time=slice('2009-01-01', '2018-12-31')).swvl
    ERA5qm = ERA5qm.sel(depthBelowLandLayer=0) * 0.127 + ERA5qm.sel(depthBelowLandLayer=7) * 0.382 + ERA5qm.sel(
        depthBelowLandLayer=28) * 0.491
    # ERA5qm = ERA5qm.interp(lat=SWI.lat, lon=SWI.lon)

    # get LISFLOOD
    LISFLOOD = get_LISFLOOD_stack()
    LISFLOOD = LISFLOOD.sel(time=slice('2009-01-01', '2018-12-31')).vsw
    LISFLOOD = LISFLOOD.sel(soilLayer=0) * 0.091 + LISFLOOD.sel(soilLayer=1) * 0.909

    # uerra
    UERRA = get_UERRA_stack()
    UERRA = UERRA.sel(time=slice('2009-01-01', '2018-12-31')).vsw
    UERRA = UERRA.isel(step=0)
    UERRA = UERRA.isel(soilLayer=0) * 0.025 + UERRA.isel(soilLayer=1) * 0.075 + UERRA.isel(
        soilLayer=2) * 0.15 + UERRA.isel(
        soilLayer=3) * 0.25 + UERRA.isel(soilLayer=4) * 0.5

    # get station data
    swmx = pd.read_csv('/mnt/CEPH_PROJECTS/ADO/SM/reference_data/SiteInfo_SwissSMEX0713_grassland_selected.csv', skiprows=1)

    for index, i_swmx in swmx.iterrows():
        # get coordinates of the current stations
        i_lat = i_swmx['lat']
        i_lon = i_swmx['lon']
        i_x = i_swmx['x coord']
        i_y = i_swmx['y coord']


        ERA5_ts = ERA5.interp(lat=i_lat, lon=i_lon)
        ERA5l_ts = ERA5l.interp(lat=i_lat, lon=i_lon)
        ERA5qm_ts = ERA5qm.interp(lat=i_lat, lon=i_lon)

        x3035, y3035 = transform_to_custom(i_lon, i_lat, targetproj=3035)
        LISFLOOD_ts = LISFLOOD.interp(x=x3035, y=y3035)
        xUERRA, yUERRA = transform_to_custom(i_lon, i_lat,
                                             targetproj='+proj=lcc +lat_1=50 +lat_2=50 +lat_0=50 +lon_0=8 +x_0=2937018.5829291 +y_0=2937031.41074803 +a=6371229 +b=6371229 +units=m +no_defs',
                                             epsg=False)
        UERRA_ts = UERRA.interp(x=xUERRA, y=yUERRA)

        # load PREVAH
        PREVAH_ts = get_PREVAH_point_ts(i_swmx['Code'])
        PREVAH_ts = PREVAH_ts['FCP']['2009-01-01':'2018-12-31']
        SWMX_ts = get_SwissSMEX_ts(i_swmx['Code'])
        SWMX_ts = SWMX_ts[i_swmx['Code'] + '_nc_sm_mm'] / 500#['2011-01-01':'2018-12-31']
        SWMX_ts = SWMX_ts['2010-01-01'::]


        # if monthly:
        #     PREVAH_ts = PREVAH_ts.rolling('10D').mean()
        #     SWMX_ts = SWMX_ts.rolling('10D').mean()
        #     if includeSWI:
        #         SWI_ts = SWI_ts.chunk({'time': SWI_ts.shape[0]})
        #         SWI_ts = SWI_ts.rolling(time=10).mean()
        #     ERA5_ts = ERA5_ts.chunk({'time': ERA5_ts.shape[0]})
        #     ERA5_ts = ERA5_ts.rolling(time=10).mean()
        #     ERA5l_ts = ERA5l_ts.chunk({'time': ERA5l_ts.shape[0]})
        #     ERA5l_ts = ERA5l_ts.rolling(time=10).mean()
        #     ERA5qm_ts = ERA5qm_ts.chunk({'time': ERA5qm_ts.shape[0]})
        #     ERA5qm_ts = ERA5qm_ts.rolling(time=10).mean()
        #     LISFLOOD_ts = LISFLOOD_ts.chunk({'time': LISFLOOD_ts.shape[0]})
        #     LISFLOOD_ts = LISFLOOD_ts.rolling(time=10).mean()
        #     UERRA_ts = UERRA_ts.chunk({'time': UERRA_ts.shape[0]})
        #     UERRA_ts = UERRA_ts.rolling(time=10).mean()

        SWMX_ts.index = pd.DatetimeIndex(SWMX_ts.index.date)
        PREVAH_ts.index = pd.DatetimeIndex(PREVAH_ts.index.date)
        ERA5_ts = ERA5_ts.to_series()
        ERA5_ts.index = pd.DatetimeIndex(ERA5_ts.index.date)
        ERA5l_ts = ERA5l_ts.to_series()
        ERA5l_ts.index = pd.DatetimeIndex(ERA5l_ts.index.date)
        ERA5qm_ts = ERA5qm_ts.to_series()
        ERA5qm_ts.index = pd.DatetimeIndex(ERA5qm_ts.index.date)
        LISFLOOD_ts = LISFLOOD_ts.to_series()
        LISFLOOD_ts.index = pd.DatetimeIndex(LISFLOOD_ts.index.date)
        UERRA_ts = UERRA_ts.to_series()
        UERRA_ts.index = pd.DatetimeIndex(UERRA_ts.index.date)
        if includeSWI:
            SWI_ts = SWI_ts.to_series()
            SWI_ts.index = pd.DatetimeIndex(SWI_ts.index.date)
            merged = pd.concat({'PREVAH': PREVAH_ts,
                                'SWMX': SWMX_ts,
                                'SWI': SWI_ts / 100,
                                'ERA5': ERA5_ts,
                                'ERA5l': ERA5l_ts,
                                # 'ERA5qm': ERA5qm_ts,
                                'LISFLOOD': LISFLOOD_ts,
                                'UERRA': UERRA_ts}, axis=1)
        else:
            merged = pd.concat({'PREVAH': PREVAH_ts,
                                'SWMX': SWMX_ts,
                                'ERA5': ERA5_ts,
                                'ERA5l': ERA5l_ts,
                                # 'ERA5qm': ERA5qm_ts,
                                'LISFLOOD': LISFLOOD_ts,
                                'UERRA': UERRA_ts}, axis=1)

        if interval > 1:
            merged = merged.groupby(merged.index.year).resample(str(interval) + 'D').mean().droplevel(0)

        anom_merged = plot_only_ts(merged, outpath, i_swmx['Code'], year, anomalies, monthly, interval=interval)

        # create scatterplot
        # create scatterplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        collist = ['PREVAH', 'ERA5', 'ERA5l', 'LISFLOOD', 'UERRA']
        if anomalies:
            scattermerged = anom_merged
        else:
            scattermerged = merged.copy()
        # scattermerged[scattermerged.isnull().any(axis=1)] = np.nan
        scattermerged = scattermerged.dropna(axis=0)

        pearsonr = scattermerged.corr()

        for i in range(5):
            if anomalies:
                plotlims = [-2.5, 2.5]
            else:
                plotlims = [0.1, 0.7]
            scattermerged.plot.scatter(x='SWMX', y=collist[i], ax=axs[np.unravel_index(i, (2, 3))],
                                title='SwissSMEX vs ' + collist[i],
                                xlim=plotlims, ylim=plotlims)
            # calculate rmse
            rmse = ((scattermerged[collist[i]] - scattermerged['SWMX']) ** 2).mean() ** .5
            axs[np.unravel_index(i, (2, 3))].text(0.1, 0.1,
                                                  "R: " + "{:10.3f}".format(pearsonr['SWMX'][collist[i]]) + '\n' +
                                                  "RMSE: " + "{:10.3f}".format(rmse),
                                                  transform=axs[np.unravel_index(i, (2, 3))].transAxes)



        fig.tight_layout()

        plt.savefig(
            outpath + i_swmx['Code'] + '_SCATTER.png',
            dpi=600)
        plt.close()


def initialize_stacks():
    # get ERA5
    ERA5 = get_ERA5_stack()
    ERA5 = ERA5.sel(time=slice('2009-01-01', '2018-12-31')).swvl
    # scale to SWISS model
    ERA5 = ERA5.sel(depthBelowLandLayer=0) * 0.127 + ERA5.sel(depthBelowLandLayer=7) * 0.382 + ERA5.sel(
        depthBelowLandLayer=28) * 0.491
    # ERA5 = ERA5.interp(lat=SWI.lat, lon=SWI.lon)

    # get ERA5 land
    ERA5l = get_ERA5Land_stack()
    ERA5l = ERA5l.sel(time=slice('2009-01-01', '2018-12-31')).swvl
    # scale to SWISS model
    ERA5l = ERA5l.sel(depthBelowLandLayer=0) * 0.127 + ERA5l.sel(depthBelowLandLayer=7) * 0.382 + ERA5l.sel(
        depthBelowLandLayer=28) * 0.491
    # ERA5l = ERA5l.interp(lat=SWI.lat, lon=SWI.lon)

    # get LISFLOOD
    LISFLOOD = get_LISFLOOD_stack()
    LISFLOOD = LISFLOOD.sel(time=slice('2009-01-01', '2018-12-31')).vsw
    LISFLOOD = LISFLOOD.sel(soilLayer=0) * 0.091 + LISFLOOD.sel(soilLayer=1) * 0.909

    # uerra
    UERRA = get_UERRA_stack()
    UERRA = UERRA.sel(time=slice('2009-01-01', '2018-12-31')).vsw
    UERRA = UERRA.isel(step=0)
    UERRA = UERRA.isel(soilLayer=0) * 0.025 + UERRA.isel(soilLayer=1) * 0.075 + UERRA.isel(
        soilLayer=2) * 0.15 + UERRA.isel(
        soilLayer=3) * 0.25 + UERRA.isel(soilLayer=4) * 0.5

    return ERA5, ERA5l, LISFLOOD, UERRA


def SwissSMEX_ts_create_plots(ERA5, ERA5l,
                              LISFLOOD,
                              UERRA,
                              i_swmx, interval, monthly=True, anomalies=True, year=None):
    # for index, i_swmx in swmx_sites.iterrows():
    # get coordinates of the current stations
    i_lat = i_swmx['lat']
    i_lon = i_swmx['lon']
    i_x = i_swmx['x coord']
    i_y = i_swmx['y coord']

    # extract time-series
    ERA5_ts = ERA5.interp(lat=i_lat, lon=i_lon)
    ERA5l_ts = ERA5l.interp(lat=i_lat, lon=i_lon)

    # coordinate tranformations
    x3035, y3035 = transform_to_custom(i_lon, i_lat, targetproj=3035)
    LISFLOOD_ts = LISFLOOD.interp(x=x3035, y=y3035)
    xUERRA, yUERRA = transform_to_custom(i_lon, i_lat,
                                         targetproj='+proj=lcc +lat_1=50 +lat_2=50 +lat_0=50 +lon_0=8 +x_0=2937018.5829291 +y_0=2937031.41074803 +a=6371229 +b=6371229 +units=m +no_defs',
                                         epsg=False)
    UERRA_ts = UERRA.interp(x=xUERRA, y=yUERRA)

    # load PREVAH time-series
    PREVAH_ts = get_PREVAH_point_ts(i_swmx['Code'])
    PREVAH_ts = PREVAH_ts['FCP']['2009-01-01':'2018-12-31']
    SWMX_ts = get_SwissSMEX_ts(i_swmx['Code'])
    SWMX_ts = SWMX_ts[i_swmx['Code'] + '_nc_sm_mm'] / 500  # ['2011-01-01':'2018-12-31']
    SWMX_ts = SWMX_ts['2010-01-01'::]

    SWMX_ts.index = pd.DatetimeIndex(SWMX_ts.index.date)
    PREVAH_ts.index = pd.DatetimeIndex(PREVAH_ts.index.date)
    ERA5_ts = ERA5_ts.to_series()
    ERA5_ts.index = pd.DatetimeIndex(ERA5_ts.index.date)
    ERA5l_ts = ERA5l_ts.to_series()
    ERA5l_ts.index = pd.DatetimeIndex(ERA5l_ts.index.date)
    LISFLOOD_ts = LISFLOOD_ts.to_series()
    LISFLOOD_ts.index = pd.DatetimeIndex(LISFLOOD_ts.index.date)
    UERRA_ts = UERRA_ts.to_series()
    UERRA_ts.index = pd.DatetimeIndex(UERRA_ts.index.date)

    # create a common data frame
    merged = pd.concat({'SWMX': SWMX_ts,
                        'PREVAH': PREVAH_ts,
                        'ERA5': ERA5_ts,
                        'ERA5l': ERA5l_ts,
                        'LISFLOOD': LISFLOOD_ts,
                        'UERRA': UERRA_ts}, axis=1)

    if interval > 1:
        merged = merged.groupby(merged.index.year).resample(str(interval) + 'D').mean().droplevel(0)

    anom_merged = plot_only_ts(merged, '', i_swmx['Code'], year, anomalies, monthly, interval=interval)

    # create scatterplot
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    collist = ['PREVAH', 'ERA5', 'ERA5l', 'LISFLOOD', 'UERRA']
    if anomalies:
        scattermerged = anom_merged
    else:
        scattermerged = merged.copy()
    # scattermerged[scattermerged.isnull().any(axis=1)] = np.nan
    scattermerged = scattermerged.dropna(axis=0)

    pearsonr = scattermerged.corr()

    for i in range(5):
        if anomalies:
            plotlims = [-2.5, 2.5]
        else:
            plotlims = [0.1, 0.7]
        scattermerged.plot.scatter(x='SWMX', y=collist[i], ax=axs[np.unravel_index(i, (2, 3))],
                                   title='SwissSMEX vs ' + collist[i],
                                   xlim=plotlims, ylim=plotlims)
        # calculate rmse
        rmse = ((scattermerged[collist[i]] - scattermerged['SWMX']) ** 2).mean() ** .5
        axs[np.unravel_index(i, (2, 3))].text(0.1, 0.1,
                                              "R: " + "{:10.3f}".format(pearsonr['SWMX'][collist[i]]) + '\n' +
                                              "RMSE: " + "{:10.3f}".format(rmse),
                                              transform=axs[np.unravel_index(i, (2, 3))].transAxes,
                                              size='large')

    fig.tight_layout()
    return anom_merged, merged


def SwissSMEX_ts_avg_plot(year,
                          anomalies=False):

    if anomalies:
        basepath = '/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/no_SWI_SwissSMEX_points/absolute/'
        outpath = '/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/no_SWI_SwissSMEX_points/anomalies/'
    else:
        basepath = '/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/no_SWI_SwissSMEX_points/absolute/'
        outpath = '/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/no_SWI_SwissSMEX_points/absolute/'

    # get station data
    swmx = pd.read_csv('/mnt/CEPH_PROJECTS/ADO/SM/reference_data/SiteInfo_SwissSMEX0713_grassland.csv', skiprows=1)

    for icol in ['PREVAH', 'SWMX', 'ERA5', 'ERA5l', 'LISFLOOD', 'UERRA']:
        cntr = 0
        for index, i_swmx in swmx.iterrows():
            # get coordinates of the current stations
            i_lat = i_swmx['lat']
            i_lon = i_swmx['lon']
            i_x = i_swmx['x coord']
            i_y = i_swmx['y coord']
            i_name = i_swmx['Code']

            cntr = cntr + 1
            # load csv
            i_ts = pd.read_csv(basepath + i_name + '.csv', header=0, index_col=0, parse_dates=True)
            i_ts = i_ts[icol]
            if cntr == 1:
                avg_ts = i_ts
            else:
                avg_ts = pd.concat([avg_ts, i_ts], axis=1)

        avg_ts = avg_ts.mean(axis=1, skipna=True)
        avg_ts.name = icol

        if icol == 'PREVAH':
            merged = avg_ts
        else:
            merged = pd.concat([merged, avg_ts], axis=1)

    plot_only_ts(merged, year,
                 outpath, 'AVG', anomalies=anomalies, monthly=True)


def SWISSsmex_scatterplot(anomalies=False):

    if anomalies:
        basepath = '/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/no_SWI_SwissSMEX_points/anomalies/'
        outpath = '/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/no_SWI_SwissSMEX_points/anomalies/'
    else:
        basepath = '/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/no_SWI_SwissSMEX_points/absolute/'
        outpath = '/mnt/CEPH_PROJECTS/ADO/SM/model_comparison/no_SWI_SwissSMEX_points/absolute/'


    # get station data
    swmx = pd.read_csv('/mnt/CEPH_PROJECTS/ADO/SM/reference_data/SiteInfo_SwissSMEX0713_grassland_selected.csv', skiprows=1)

    #for icol in ['PREVAH', 'SWMX', 'ERA5', 'ERA5l', 'LISFLOOD', 'UERRA']:
    cntr = 0
    for index, i_swmx in swmx.iterrows():
        i_name = i_swmx['Code']

        cntr = cntr + 1
        # load csv
        i_ts = pd.read_csv(basepath + i_name + '.csv', header=0, index_col=0, parse_dates=True)

        if anomalies:
            for ikey in i_ts.keys():
                tmp_anom = compute_anomaly(i_ts[ikey], return_clim=False, monthly=True)
                i_ts[ikey] = tmp_anom

        if cntr == 1:
            all_ts = i_ts
        else:
            all_ts = pd.concat([all_ts, i_ts], axis=0, ignore_index=True)

    # create scatterplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    collist = ['PREVAH', 'ERA5', 'ERA5l', 'LISFLOOD', 'UERRA']

    if anomalies:
        axisrange = [-3, 3]
    else:
        axisrange = [0, 1]

    pearsonr = all_ts.corr()

    for i in range(5):
        all_ts.plot.scatter(x='SWMX', y=collist[i], ax=axs[np.unravel_index(i, (2, 3))], title='SwissSMEX vs ' + collist[i],
                            xlim=axisrange, ylim=axisrange)
        # calculate rmse
        rmse = ((all_ts[collist[i]] - all_ts['SWMX']) ** 2).mean() ** .5
        axs[np.unravel_index(i, (2, 3))].text(0.1, 0.1, "R: " + "{:10.3f}".format(pearsonr['SWMX'][collist[i]]) + '\n' + "RMSE: " + "{:10.3f}".format(rmse), transform=axs[np.unravel_index(i, (2, 3))].transAxes)

    fig.tight_layout()
    plt.savefig(outpath + 'SCATTER_ALL.png', dpi=600)
    plt.close()


def plot_only_ts(merged, outpath, name, year=None, anomalies=False, monthly=False, interval=1, plot_full=True):

    #
    #merged = merged.dropna()
    # set all values to nan, per row, if any nan

    merged = merged[merged['SWMX'].index.min(): merged['SWMX'].index.max()]
    merged = merged.interpolate(axis=0)
    merged[merged.isnull().any(axis=1)] = np.nan
    merged = merged.dropna()

    if not anomalies:
        merged.to_csv(
            outpath + name + '.csv')
    anom_merged = merged.copy()

    # create plots

    cntr = 0
    if anomalies:
        tmp_clim_ext = pd.Series(index=merged.index)
        tmp_clim_neg_2 = pd.Series(index=merged.index)
        tmp_clim_pos_2 = pd.Series(index=merged.index)
        tmp_clim_ext[:] = 0
        tmp_clim_neg_2[:] = -2
        tmp_clim_pos_2[:] = 2

        tmp_climatology = pd.concat({'avg': tmp_clim_ext,
                                     '2std+': tmp_clim_pos_2,
                                     '2std-': tmp_clim_neg_2},
                                    axis=1)

        for ikey in merged.keys():
            tmp_clim, tmp_clim_std, tmp_anom = compute_anomaly(merged[ikey], return_clim=True, monthly=monthly)
            anom_merged[ikey] = tmp_anom

        allmerged = pd.concat([anom_merged, tmp_climatology], axis=1)
        allmerged.plot(figsize=(15,5), title='Full time-series',
                       style=['C0-', 'C1-', 'C4-',
                              'C6-', 'y-', 'C9-',
                              'k--', 'b--', 'r--'])
        plt.tight_layout()

    else:
        fig, axs = plt.subplots(6, 1, sharex=True, figsize=(15, 15), squeeze=True)
        for ikey in merged.keys():
            tmp_clim, tmp_clim_std, tmp_anom = compute_anomaly(merged[ikey], return_clim=True, monthly=monthly)
            anom_merged[ikey] = tmp_anom
            tmp_clim_ext = pd.Series(index=merged.index)
            tmp_clim_std_ext = pd.Series(index=merged.index)
            if interval == 1:
                fillrange = range(1, 367)
            else:
                fillrange = range(1, 367, 10)

            for i in fillrange:
                tmp_clim_ext[tmp_clim_ext.index.dayofyear == i] = tmp_clim.loc[i]
                tmp_clim_std_ext[tmp_clim_std_ext.index.dayofyear == i] = tmp_clim_std.loc[i]

            tmp_clim_neg_2 = tmp_clim_ext - 2 * tmp_clim_std_ext
            tmp_clim_pos_2 = tmp_clim_ext + 2 * tmp_clim_std_ext

            tmp_climatology = pd.concat({'avg': tmp_clim_ext,
                                         '2std+': tmp_clim_pos_2,
                                         '2std-': tmp_clim_neg_2},
                                        axis=1)

            allmerged = pd.concat([merged[ikey], tmp_climatology], axis=1)

            allmerged.plot(ax=axs[cntr],
                           style=['k-', 'k--', 'b--', 'r--'],
                           title=ikey)

            cntr = cntr + 1
        fig.suptitle('Full time-series')
        fig.tight_layout()
    if outpath != '':
        plt.savefig(
            outpath + name + '_FULL.png',
            dpi=600)
    else:
        plt.show()
    plt.close()

    if year is not None:
        for iyear in year:
            if anomalies:
                allmerged[allmerged.index.year == iyear].plot(figsize=(15, 5), title=str(iyear),
                                                              style=['C0-', 'C1-', 'C4-',
                                                                     'C6-', 'y-', 'C9-',
                                                                     'k--', 'b--', 'r--'])
                plt.tight_layout()
            else:
                # create plots
                fig, axs = plt.subplots(6, 1, sharex=True, figsize=(10, 15), squeeze=True)

                # plot the climatoloies
                cntr = 0
                for ikey in merged.keys():
                    tmp_clim, tmp_clim_std, tmp_anom = compute_anomaly(merged[ikey], return_clim=True, monthly=monthly)
                    anom_merged[ikey] = tmp_anom
                    tmp_clim.index = pd.DatetimeIndex(
                        [dt.datetime(year=iyear, month=1, day=1) + dt.timedelta(days=int(dix) - 1) for dix in
                         tmp_clim.index.values])
                    tmp_clim_std.index = pd.DatetimeIndex(
                        [dt.datetime(year=iyear, month=1, day=1) + dt.timedelta(days=int(dix) - 1) for dix in
                         tmp_clim_std.index.values])
                    if interval > 1:
                        tmp_clim.index.freq = str(interval) + 'D'
                        tmp_clim_std.freq = str(interval) + 'D'
                    else:
                        tmp_clim.index.freq = 'D'
                        tmp_clim_std.freq = 'D'
                    tmp_clim_neg_2 = tmp_clim - 2 * tmp_clim_std
                    tmp_clim_pos_2 = tmp_clim + 2 * tmp_clim_std
                    if anomalies:
                        tmp_clim[:] = 0
                        tmp_clim_neg_2[:] = -2
                        tmp_clim_pos_2[:] = 2

                    tmp_climatology = pd.concat({'avg': tmp_clim[tmp_clim.index.year == iyear],
                                                 '2std+': tmp_clim_pos_2[tmp_clim.index.year == iyear],
                                                 '2std-': tmp_clim_neg_2[tmp_clim.index.year == iyear]},
                                                axis=1)

                    if anomalies:
                        allmerged = pd.concat([tmp_anom[tmp_anom.index.year == iyear], tmp_climatology], axis=1)
                    else:
                        allmerged = pd.concat([merged[ikey][merged.index.year == iyear], tmp_climatology], axis=1)

                    allmerged.plot(ax=axs[cntr],
                                   style=['k-', 'k--', 'b--', 'r--'],
                                   title=ikey)
                    if anomalies:
                        axs[cntr].set_ylim([-3, 3])

                    cntr = cntr + 1

                fig.suptitle(str(iyear))
                fig.tight_layout()
            if outpath != '':
                plt.savefig(
                    outpath + name + '_' + str(iyear) + '.png',
                    dpi=600)
            else:
                plt.show()
            plt.close()

    if anomalies and (outpath != ''):
        anom_merged.to_csv(
            outpath + name + '.csv')

    return anom_merged


def sma_threshold(threshold, anom_merged):
    metrics = pd.DataFrame(data=None,
                                 index=['TPR', 'FNR', 'FPR', 'TNR', 'ACC'],
                                 columns=['ERA5', 'ERA5l',
                                          'LISFLOOD', 'PREVAH', 'UERRA'])

    # true positive
    drought_class = pd.DataFrame(data=None,
                                 index=anom_merged.index,
                                 columns=anom_merged.columns)
    for ind, irow in anom_merged.iterrows():

        labels = ['ERA5', 'ERA5l', 'LISFLOOD', 'PREVAH', 'UERRA']

        for ilabel in labels:
            if (irow['SWMX'] < threshold) and (irow[ilabel] < threshold):
                drought_class.at[ind, ilabel] = 1
            else:
                drought_class.at[ind, ilabel] = 0
        drought_class.at[ind, 'SWMX'] = 1 if irow['SWMX'] < threshold else 0

    tps = drought_class.sum(axis=0)
    for ilabel in labels:
        metrics.at['TPR', ilabel] = tps[ilabel] / tps['SWMX']

    # false negative
    drought_class = pd.DataFrame(data=None,
                                 index=anom_merged.index,
                                 columns=anom_merged.columns)
    for ind, irow in anom_merged.iterrows():

        labels = ['ERA5', 'ERA5l', 'LISFLOOD', 'PREVAH', 'UERRA']

        for ilabel in labels:
            if (irow['SWMX'] < threshold) and (irow[ilabel] >= threshold):
                drought_class.at[ind, ilabel] = 1
            else:
                drought_class.at[ind, ilabel] = 0
        drought_class.at[ind, 'SWMX'] = 1 if irow['SWMX'] < threshold else 0

    fns = drought_class.sum(axis=0)
    for ilabel in labels:
        metrics.at['FNR', ilabel] = fns[ilabel] / fns['SWMX']

    # false positive
    drought_class = pd.DataFrame(data=None,
                                 index=anom_merged.index,
                                 columns=anom_merged.columns)
    for ind, irow in anom_merged.iterrows():

        labels = ['ERA5', 'ERA5l', 'LISFLOOD', 'PREVAH', 'UERRA']

        for ilabel in labels:
            if (irow['SWMX'] >= threshold) and (irow[ilabel] < threshold):
                drought_class.at[ind, ilabel] = 1
            else:
                drought_class.at[ind, ilabel] = 0
        drought_class.at[ind, 'SWMX'] = 1 if irow['SWMX'] >= threshold else 0

    fps = drought_class.sum(axis=0)
    for ilabel in labels:
        metrics.at['FPR', ilabel] = fps[ilabel] / fps['SWMX']

    # true negative
    drought_class = pd.DataFrame(data=None,
                                 index=anom_merged.index,
                                 columns=anom_merged.columns)
    for ind, irow in anom_merged.iterrows():

        labels = ['ERA5', 'ERA5l', 'LISFLOOD', 'PREVAH', 'UERRA']

        for ilabel in labels:
            if (irow['SWMX'] >= threshold) and (irow[ilabel] >= threshold):
                drought_class.at[ind, ilabel] = 1
            else:
                drought_class.at[ind, ilabel] = 0
        drought_class.at[ind, 'SWMX'] = 1 if irow['SWMX'] >= threshold else 0

    tns = drought_class.sum(axis=0)
    for ilabel in labels:
        metrics.at['TNR', ilabel] = tns[ilabel] / tns['SWMX']

    # accuracy
    for ilabel in labels:
        metrics.at['ACC', ilabel] = (tps[ilabel] + tns[ilabel]) / (tps['SWMX'] + tns['SWMX'])

    return metrics