import xarray as xr
from pathlib import Path
import os
import matplotlib.pyplot as plt
import urllib.request
import urllib.error
import pandas as pd
import datetime as dt
import shutil
import ado_readers
import numpy as np
from ado_tools import get_ado_extent
from ado_tools import get_cop_sm_depths
from ado_tools import get_subdirs
from dask.diagnostics import ProgressBar


def urlexists(site, path):
    url = site + path

    # create a password manager
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()

    # Add the username and password.
    # If we knew the realm, we could use it instead of None.
    top_level_url = site
    password_mgr.add_password(None, top_level_url, 'felixgreifeneder', 'nacktmul20')

    handler = urllib.request.HTTPBasicAuthHandler(password_mgr)

    # create "opener" (OpenerDirector instance)
    opener = urllib.request.build_opener(handler)

    # use the opener to fetch a URL
    for _ in range(3):
        try:
            opener.open(url)
            # Install the opener.
            # Now all calls to urllib.request.urlopen use our opener.
            urllib.request.install_opener(opener)

            return urllib.request.urlopen(url, timeout=10).getcode() == 200
        except urllib.error.HTTPError as error:
            print('Could not open ' + url + ' due to Error: ' + str(error))
        except urllib.error.URLError as error:
            print('URLError: ' + str(error))
    else:
        print('Download failed')
        handler.close()
        return False


def downloadfile(site, path, dest):
    url = site + path

    # create a password manager
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()

    # Add the username and password.
    # If we knew the realm, we could use it instead of None.
    top_level_url = site
    password_mgr.add_password(None, top_level_url, 'felixgreifeneder', 'nacktmul20')

    handler = urllib.request.HTTPBasicAuthHandler(password_mgr)

    # create "opener" (OpenerDirector instance)
    opener = urllib.request.build_opener(handler)

    # use the opener to fetch a URL
    for _ in range(3):
        try:
            opener.open(url)
            break
        except urllib.error.HTTPError as error:
            print('Could not open ' + url + ' due to Error: ' + str(error))
        except urllib.error.URLError as error:
            print('URLError: ' + str(error))
    else:
        print('Download failed')
        handler.close()
        return 0

    # Install the opener.
    # Now all calls to urllib.request.urlopen use our opener.
    urllib.request.install_opener(opener)

    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=10) as response, open(dest, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            handler.close()
            return 1
        except urllib.error.HTTPError as error:
            print('Data not retrieved because from ' + url + ' due to Error: ' + str(error))
        except urllib.error.URLError as error:
            print('URLError: ' + str(error))
    else:
        print('Download failed')
        handler.close()
        return 0


def download_SWI(basepath):
    vitosite = 'https://land.copernicus.vgt.vito.be/'
    baseurlpath = 'PDF/datapool/Vegetation/Soil_Water_Index/Daily_SWI_1km_Europe_V1/'
    file2 = open(basepath + r"errorlog.txt", "w+")

    # iterate through each year and day, download and crop the corresponding SWI files
    for di in pd.date_range('2020-02-24', dt.date.today(), freq='D'):
        download_success_nc = 0
        download_success_xml = 0
        for version in ['V1.0.1', 'V1.0.2']:
            try:
                # build the url'
                di_url = di.strftime('%Y') + '/' + di.strftime('%m') + '/' + di.strftime('%d') + '/SWI1km_' + \
                         di.strftime('%Y%m%d') + '1200_CEURO_SCATSAR_' + version + '/'
                nc_url = 'c_gls_SWI1km_' + di.strftime('%Y%m%d') + '1200_CEURO_SCATSAR_' + version + '.nc'
                meta_url = 'c_gls_SWI1km_PROD-DESC_' + di.strftime('%Y%m%d') + '1200_CEURO_SCATSAR_' + version + '.xml'

                # build download destination path
                di_dest = basepath + di.strftime('%Y') + '/' + di.strftime('%m') + '/' + di.strftime('%d') + '/'

                if not os.path.exists(di_dest):
                    Path(di_dest).mkdir(parents=True, exist_ok=True)

                # check if files were downloaded before
                if os.path.exists(di_dest + nc_url) or os.path.exists(di_dest + nc_url[0:-3] + '_adoext.nc'):
                    break

                # download files
                download_success_nc = downloadfile(vitosite, baseurlpath + di_url + nc_url, di_dest + nc_url)
                download_success_xml = downloadfile(vitosite, baseurlpath + di_url + meta_url, di_dest + meta_url)

                # test read data
                testread = xr.open_dataset(di_dest + nc_url)
                testread.close()
                break
            except OSError:
                print('Testing V1.0.2')
                os.remove(di_dest + nc_url) if os.path.exists(di_dest + nc_url) else print(nc_url + ' not existing')
                os.remove(di_dest + meta_url) if os.path.exists(di_dest + meta_url) else print(nc_url + ' not existing')
                download_success_nc = 0
                download_success_xml = 0
        if (download_success_nc == 1) and (download_success_xml == 1):
            # crop to ado extent
            crop_SWI_to_ado(di_dest + nc_url)
        else:
            file2.write(baseurlpath + di_url + nc_url + '\n')
            file2.write(baseurlpath + di_url + meta_url + '\n')
    file2.close()


def re_arrange_folder_structure():
    import shutil
    basepath = '/mnt/CEPH_PROJECTS/ADO/SWI/2019/'
    for di in pd.date_range('2019-01-01', '2019-12-31', freq='D'):
        oldfolder = 'SWI1km_' + di.strftime('%Y%m%d') + '1200_CEURO_SCATSAR_V1.0.1/'
        newfolder = di.strftime('%m') + '/' + di.strftime('%d') + '/'
        Path(basepath + newfolder).mkdir(parents=True, exist_ok=True)

        # copy files
        version = 'V1.0.1'
        nc_file = 'c_gls_SWI1km_' + di.strftime('%Y%m%d') + '1200_CEURO_SCATSAR_' + version + '_adoext.nc'
        meta_file = 'c_gls_SWI1km_PROD-DESC_' + di.strftime('%Y%m%d') + '1200_CEURO_SCATSAR_V1.0.1.xml'
        quicklook = 'c_gls_SWI1km_QL_' + di.strftime('%Y%m%d') + '1200_CEURO_SCATSAR_V1.0.1.tiff'
        if os.path.exists(basepath + oldfolder + nc_file) == False:
            # try with different version
            version = 'V1.0.2'
            nc_file = 'c_gls_SWI1km_' + di.strftime('%Y%m%d') + '1200_CEURO_SCATSAR_' + version + '_adoext.nc'
            meta_file = 'c_gls_SWI1km_PROD-DESC_' + di.strftime('%Y%m%d') + '1200_CEURO_SCATSAR_' + version + '.xml'
            quicklook = 'c_gls_SWI1km_QL_' + di.strftime('%Y%m%d') + '1200_CEURO_SCATSAR_' + version + '.tiff'
            oldfolder = 'SWI1km_' + di.strftime('%Y%m%d') + '1200_CEURO_SCATSAR_V1.0.2/'
            if os.path.exists(basepath + oldfolder + nc_file) == False:
                print('File does not exist: ' + nc_file)

        shutil.copyfile(basepath + oldfolder + nc_file, basepath + newfolder + nc_file)
        shutil.copyfile(basepath + oldfolder + meta_file, basepath + newfolder + meta_file)
        shutil.copyfile(basepath + oldfolder + quicklook, basepath + newfolder + quicklook)
        # remove old directory
        shutil.rmtree(basepath + oldfolder)


def crop_SWI_to_ado(inpath, outpath=None, deloriginal=True):
    # create a list of all files
    # filelist = list()
    # for path in Path('/mnt/CEPH_PROJECTS/ADO/SWI').rglob('*V1.0.1.nc'):
    #     filelist.append(path)
    # for path in Path('/mnt/CEPH_PROJECTS/ADO/SWI').rglob('*V1.0.2.nc'):
    #     filelist.append(path)
    inpath = Path(inpath)

    # get the extent
    adoext = get_ado_extent()

    # define outpath if not specified explicitly
    if outpath == None:
        outpath = Path(inpath.parent, inpath.name[0:-3] + '_adoext.nc')

    dstmp = xr.open_dataset(inpath)
    # crop the original file
    dstmp_cropped = dstmp.where((dstmp.lon > adoext[0]) & (dstmp.lon < adoext[2]) & (dstmp.lat > adoext[1]) &
                                (dstmp.lat < adoext[3]), drop=True)
    # correct attributes
    dstmp_cropped.attrs['geospatial_lon_min'] = adoext[0]
    dstmp_cropped.attrs['geospatial_lat_min'] = adoext[1]
    dstmp_cropped.attrs['geospatial_lon_max'] = adoext[2]
    dstmp_cropped.attrs['geospatial_lat_max'] = adoext[3]

    # save to cropped file to disk
    dstmp_cropped.to_netcdf(outpath,
                            unlimited_dims=['time'],
                            encoding={'SSF': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                              'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'SWI_002': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                  'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'QFLAG_002': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                    'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'SWI_005': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                  'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'QFLAG_005': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                    'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'SWI_010': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                  'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'QFLAG_010': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                    'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'SWI_015': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                  'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'QFLAG_015': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                    'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'SWI_020': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                  'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'QFLAG_020': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                    'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'SWI_040': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                  'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'QFLAG_040': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                    'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'SWI_060': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                  'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'QFLAG_060': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                    'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'SWI_100': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                  'contiguous': False, 'chunksizes': (1, 283, 503)},
                                      'QFLAG_100': {'zlib': True, 'shuffle': True, 'complevel': 4, 'fletcher32': False,
                                                    'contiguous': False, 'chunksizes': (1, 283, 503)}})

    dstmp.close()
    dstmp_cropped.close()

    # delete the original file
    if deloriginal:
        os.remove(inpath)


def valid_pixels():
    # create a list of all files
    filelist = list()
    for path in Path('/mnt/CEPH_PROJECTS/ADO/SWI/2019').rglob('*.nc'):
        filelist.append(path)

    # concatenate all files into one array
    # open the first dataset
    first = xr.open_dataset(filelist[0])
    # first = first['SWI_002'].where((first.lon < 12.76) & (first.lon > 9.6) & (first.lat < 48.4) & (first.lat > 45.27),
    # drop=True)
    # first = first['SWI_002'].where((first.lon < 17.4) & (first.lon > 5) & (first.lat < 49.1) & (first.lat > 43.4),
    #                               drop=True)
    first = first['SWI_005']
    filelist.pop(0)

    # iterate through all remaining files and stack them
    for fi in filelist:
        dstmp = xr.open_dataset(fi)
        # dstmp = dstmp['SWI_002'].where((first.lon < 12.76) & (first.lon > 9.6) & (first.lat < 48.4) & (first.lat > 45.27),
        #                                drop=True)
        # dstmp = dstmp['SWI_002'].where((first.lon < 17.4) & (first.lon > 5) & (first.lat < 49.1) & (first.lat > 43.4),
        #                               drop=True)
        dstmp = dstmp['SWI_005']
        if ('smstack' in locals()) or ('smstack' in globals()):
            smstack = xr.concat([smstack, dstmp.copy()], dim='time')
        else:
            smstack = first.copy()
        dstmp.close()

    first.close()

    # count the valid pixels
    sm_masked = smstack.where(smstack.isin(smstack.flag_values * 0.5) == False)
    sm_masked = sm_masked.where(sm_masked < 100)
    sm_val_count = sm_masked.count(dim='time')
    sm_val_count.plot()
    plt.savefig('/mnt/CEPH_PROJECTS/ADO/SWI/SWI_005_count.png')
    sm_val_count.to_netcdf('/mnt/CEPH_PROJECTS/ADO/SWI/SWI_005_count.nc')
    smstack.close()
    sm_val_count.close()
    sm_masked.close()


def validate_ismn(ismnpath='/mnt/CEPH_PROJECTS/ADO/SWI/reference_data/',
                  valpath='/mnt/CEPH_PROJECTS/ADO/SWI/validation/'):
    # get the available copernicus sm layers
    cop_sm_depths = get_cop_sm_depths()

    # initiate dataframe to collect correlations
    corr_df = pd.DataFrame(columns=['network', 'station', 'lon', 'lat', 'depth', 'SWI_002', 'SWI_005',
                                    'SWI_010', 'SWI_015', 'SWI_020', 'SWI_040', 'SWI_060', 'SWI_100'])

    # compile a list of all available ISMN networks
    ismn_networks = get_subdirs(ismnpath)

    # iterate through all networks
    for i_net in ismn_networks:
        i_net_stations = get_subdirs(i_net)

        # iterate through all stations in current network
        for i_station in i_net_stations:
            # get ismn data
            st_data = ado_readers.read_ISMN_data(i_net.name, i_station.name)
            # create folder for plots
            plotfolder = Path(valpath, i_net.name)
            if not os.path.exists(plotfolder):
                os.mkdir(plotfolder)
            # iterate through all time-series
            for i_ts in list(st_data.data_vars.keys()):
                # only check sm measurements
                if 'sm' in i_ts:
                    # get copernicus sm data
                    cop_sm_ts = ado_readers.extr_ts_copernicus_sm(st_data.lon, st_data.lat,
                                                                  depth=cop_sm_depths)
                    # merge cop sm and ismn ts
                    combo_ts = pd.concat([st_data[i_ts].to_dataframe(), cop_sm_ts.where(cop_sm_ts <= 100)], axis=1,
                                         join='inner')
                    # create plot
                    combo_ts.interpolate().plot(secondary_y=i_ts)
                    plt.savefig(Path(plotfolder, i_station.name + '_' + i_ts + '.png'))
                    plt.close()
                    # calculate correlations
                    i_corrs = combo_ts.dropna().corr()
                    tmp_df = {'network': i_net.name, 'station': i_station.name, 'lon': st_data.lon, 'lat': st_data.lat,
                              'depth': st_data[i_ts].depth,
                              'SWI_002': i_corrs.iloc[1, 0], 'SWI_005': i_corrs.iloc[2, 0],
                              'SWI_010': i_corrs.iloc[3, 0], 'SWI_015': i_corrs.iloc[4, 0],
                              'SWI_020': i_corrs.iloc[5, 0], 'SWI_040': i_corrs.iloc[6, 0],
                              'SWI_060': i_corrs.iloc[7, 0], 'SWI_100': i_corrs.iloc[8, 0]}
                    corr_df = corr_df.append(tmp_df, ignore_index=True)

    corr_df.to_pickle(Path(valpath, 'corr_summary.pds'))
    return 'Validaton complete'


def create_network_boxplots(valpath='/mnt/CEPH_PROJECTS/ADO/SWI/validation/'):
    # create boxplots per ISMN network
    corr_summary = pd.read_pickle(Path(valpath, 'corr_summary.pds'))
    swi_labels = ['SWI_002', 'SWI_005', 'SWI_010', 'SWI_015', 'SWI_020', 'SWI_040', 'SWI_060', 'SWI_100']
    # iterate through all networks
    for inet in pd.unique(corr_summary['network']):
        tmp = corr_summary.loc[corr_summary['network'] == inet]
        # iterate through all sensing depths
        ud, idx, cnt = np.unique(tmp['depth'], return_index=True, return_counts=True)
        for i in range(len(ud)):
            tmp2 = tmp.iloc[idx[i]:idx[i] + cnt[i], ]
            bptitle = inet + ' ' + ud[i][0] + '-' + ud[i][1]
            ax = tmp2[swi_labels].boxplot()
            ax.set_title(bptitle)
            ax.set_ylabel('R')
            ax.set_ylim(-1, 1)
            plt.savefig(Path(valpath, inet + '_' + ud[i][0] + '-' + ud[i][1] + '.png'))
            plt.close()


def compute_daily_climatology(swipath='/mnt/CEPH_PROJECTS/ADO/SWI/',
                              climpath='/mnt/CEPH_PROJECTS/ADO/SWI/climatology/'):
    # get depth label
    depth_label = ['SWI_002', 'SWI_005', 'SWI_010', 'SWI_015', 'SWI_020', 'SWI_040', 'SWI_060', 'SWI_100']

    # multidataset method
    # get file paths
    sm_files = list()
    for path in Path(swipath).rglob('*_adoext.nc'):
        sm_files.append(path)

    sm_df = xr.open_mfdataset(sm_files,
                              combine='by_coords',
                              parallel=True,
                              engine='h5netcdf')

    # set chunk size
    sm_df = sm_df.chunk({'time': 1934, 'lat': 10, 'lon': 10})

    # select SWI fields and mask
    sm_df = sm_df[depth_label[0]]
    sm_df = sm_df.where(sm_df <= 100)

    # compute climatology
    r = sm_df.rolling(time=30).median().dropna("time")
    # compute daily anomaly
    sm_med_clim = r.groupby("time.dayofyear").median("time")
    sm_std_clim = r.groupby("time.dayofyear").std("time")

    # store datset
    sm_med_clim.to_netcdf(Path(climpath, 'daily_median_climatology.nc'),
                          encoding={ilabel: {'zlib': True,
                                             'shuffle': True,
                                             'complevel': 4,
                                             'fletcher32': False,
                                             'contiguous': False,
                                             'chunksizes': (1, 283, 503)}
                                    for ilabel in depth_label},
                          compute=False)

    # print('Export median SWI climatology')
    # with ProgressBar():
    #     results = delayed_obj.compute()

    sm_std_clim.to_netcdf(Path(climpath, 'daily_stddev_climatology.nc'),
                          encoding={ilabel: {'zlib': True,
                                             'shuffle': True,
                                             'complevel': 4,
                                             'fletcher32': False,
                                             'contiguous': False,
                                             'chunksizes': (1, 283, 503)}
                                    for ilabel in depth_label},
                          compute=False)

    # print('Export stddev SWI climatology')
    # with ProgressBar():
    #     results = delayed_obj.compute()

    sm_df.close()
    sm_med_clim.close()
    sm_std_clim.close()
