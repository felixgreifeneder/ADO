import cdsapi
import xarray as xr
import os
from ado_tools import get_ado_extent


def download_lisflood_ado():

    minlon, minlat, maxlon, maxlat = get_ado_extent(proj='LAEA')

    c = cdsapi.Client()
    months = ['january', 'february',
              'march', 'april',
              'may', 'june',
              'july', 'august',
              'september', 'october',
              'november', 'december']

    for iyear in range(2009, 2021):

        for imonth in months:

            print(imonth)
            print(iyear)
            outpath = '/mnt/CEPH_PROJECTS/ADO/SM/LISFLOOD/SM_' + str(iyear) + str(imonth) + '.nc'
            outpath_cropped = '/mnt/CEPH_PROJECTS/ADO/SM/LISFLOOD/SM_' + str(iyear) + str(imonth) + '_SUB.nc'
            print(outpath)

            if not os.path.exists(outpath) and not os.path.exists(outpath_cropped):
                c.retrieve(
                    'efas-historical',
                    {
                        'format': 'netcdf',
                        'system_version': 'version_4_0',
                        'variable': 'volumetric_soil_moisture',
                        'model_levels': 'soil_levels',
                        'soil_level': [
                            '1', '2'
                        ],
                        'hyear': str(iyear),
                        'hmonth': imonth,
                        'hday': [
                                '01', '02', '03',
                                '04', '05', '06',
                                '07', '08', '09',
                                '10', '11', '12',
                                '13', '14', '15',
                                '16', '17', '18',
                                '19', '20', '21',
                                '22', '23', '24',
                                '25', '26', '27',
                                '28', '29', '30',
                                '31',
                            ],
                        'time': [
                                '00:00', '06:00', '12:00',
                                '18:00',
                        ]
                    },
                    outpath)

            if not os.path.exists(outpath_cropped):
                # open file and crop to switzerland
                print('Cropping to AOI ...')
                efasimg = xr.open_dataset(outpath)
                # crop the original file
                efasimg_cropped = efasimg.where((efasimg.x > minlon) & (efasimg.x < maxlon) & (efasimg.y > minlat) &
                                                (efasimg.y < maxlat), drop=True)
                efasimg_cropped.to_netcdf(outpath_cropped)
                os.remove(outpath)


def download_lisflood_switzerland():

    c = cdsapi.Client()
    months = ['january', 'february',
              'march', 'april',
              'may', 'june',
              'july', 'august',
              'september', 'october',
              'november', 'december']

    for iyear in range(2009, 2021):

        for imonth in months:

            print(imonth)
            print(iyear)
            outpath = '/mnt/CEPH_PROJECTS/ADO/SM/LISFLOOD/Switzerland/SM_' + str(iyear) + '_' + \
                      imonth + '.nc'
            outpath_cropped = '/mnt/CEPH_PROJECTS/ADO/SM/LISFLOOD/Switzerland/SM_' + str(iyear) + '_' + \
                      imonth + '_SUB.nc'
            print(outpath)

            c.retrieve(
                'efas-historical',
                {
                    'format': 'netcdf',
                    'system_version': 'version_4_0',
                    'variable': 'volumetric_soil_moisture',
                    'model_levels': 'soil_levels',
                    'soil_level': [
                        '1', '2', '3'
                    ],
                    'hyear': str(iyear),
                    'hmonth': imonth,
                    'hday': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '06:00', '12:00',
                        '18:00',
                    ],
                    #'area': [47.90, 5.90, 45.70, 10.60],
                },
                outpath)

            # open file and crop to switzerland
            print('Cropping to AOI ...')
            efasimg = xr.open_dataset(outpath)
            # crop the original file
            efasimg_cropped = efasimg.where((efasimg.x > 4000171) & (efasimg.x < 4365848) & (efasimg.y > 2513134) &
                                        (efasimg.y < 2746352), drop=True)
            efasimg_cropped.to_netcdf(outpath_cropped)
            os.remove(outpath)


def download_era5land_soil_temp():
    c = cdsapi.Client()
    months = [
        '01', '02', '03', '04', '05', '06',
        '07', '08', '09', '10', '11', '12'
        ]

    for iyear in range(1982, 2021):

        print(iyear)
        outpath = '/mnt/CEPH_PROJECTS/ADO/SM/ERA5L/raw/soil_temperature/soil_temperature_' + str(iyear) + '.nc'
        outpath_cropped = '/mnt/CEPH_PROJECTS/ADO/SM/ERA5L/raw/soil_temperature/soil_temperature_' + str(iyear) + '_SUB.nc'
        print(outpath)

        c.retrieve(
            'reanalysis-era5-land',
            {
                'format': 'netcdf',
                'variable': [
                    'soil_temperature_level_1', 'soil_temperature_level_2', 'soil_temperature_level_3',
                    'soil_temperature_level_4'
                    ],
                'year': str(iyear),
                'month': months,
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00', '03:00', '04:00',
                    '05:00', '06:00', '07:00', '08:00', '09:00',
                    '10:00', '11:00', '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00', '18:00', '19:00',
                    '20:00', '21:00', '22:00', '23:00'
                ],
                'area': [
                    50.85, 2.85, 42.35,
                    18.05,
                ],
            },
            outpath)

        # open file and crop to switzerland
        print('Cropping to AOI ...')
        efasimg = xr.open_dataset(outpath)
        # crop the original file
        # efasimg_cropped = efasimg.where((efasimg.x > 2.85) & (efasimg.x < 18.05) & (efasimg.y > 42.35) &
        #                                 (efasimg.y < 50.85), drop=True)
        efasimg_cropped_daily = efasimg.resample(time='1D').min()
        efasimg_cropped_daily.to_netcdf(outpath_cropped)
        efasimg.close()
        os.remove(outpath)
        efasimg_cropped_daily.close()