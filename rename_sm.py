import os
import glob
import datetime as dt
import pathlib


def main():
    main_dir = pathlib.Path('/mnt/CEPH_PROJECTS/ADO/SM/ERA5_ERA5l_QM/anomalies/')
    # iterate through all files
    for ifile in main_dir.glob('*.tif'):
        idate = dt.datetime(int(ifile.name[0:4]), 1, 1) + dt.timedelta(int(ifile.name[5:8]) - 1)
        newname = 'era5qm_sm_anom_' + idate.strftime('%Y%m%d') + '.tif'
        ifile.rename(pathlib.Path(ifile.parent, newname))


if __name__ == '__main__':
    main()