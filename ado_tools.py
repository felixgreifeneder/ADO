from pathlib import Path
import geopandas
from rasterio import features
from affine import Affine
import numpy as np
import xarray as xr
from IPython.display import HTML
import random
import os


# collection of useful functions
def get_ado_extent(proj='WGS84'):
    # returns the bounding box for the ADO extent: minlon, minlat, maxlon, maxlat
    if proj == 'WGS84':
        return 3.6846960000000046, 42.9910929945153200, 17.1620089999999941, 50.5645599970407318
    elif proj == 'LAEA':
        return 3830000, 2215000, 4855000, 3055000
    else:
        print('Not supported')
        return None


def get_cop_sm_depths():
    return [2, 5, 10, 15, 20, 40, 60, 100]


def get_subdirs(parentdir):
    # compile a list of all available ISMN networks
    subdirs = list()
    for path in Path(parentdir).glob('*'):
        if path.is_dir():
            subdirs.append(path)

    return subdirs


def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def transform_to_custom(lon, lat, targetproj='', epsg=True):
    from pyproj import CRS, Transformer
    if epsg:
        targetcrs = CRS.from_epsg(targetproj)
    else:
        targetcrs = CRS.from_proj4(targetproj)
    srccrs = CRS.from_epsg(4326)

    ctrans = Transformer.from_crs(srccrs, targetcrs, always_xy=True)

    return ctrans.transform(lon, lat)


def rasterize(shapes, coords, fill=np.nan, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.
    """
    if 'lon' in coords._names:
        transform = transform_from_latlon(coords['lat'], coords['lon'])
        out_shape = (len(coords['lat']), len(coords['lon']))
        raster = features.rasterize(shapes, out_shape=out_shape,
                                    fill=fill, transform=transform,
                                    dtype=float, all_touched=True, **kwargs)
        return xr.DataArray(raster, coords=coords, dims=('lat', 'lon'))
    elif 'x' in coords._names:
        transform = transform_from_latlon(coords['y'], coords['x'])
        out_shape = (len(coords['y']), len(coords['x']))
        raster = features.rasterize(shapes, out_shape=out_shape,
                                    fill=fill, transform=transform,
                                    dtype=float, all_touched=True, **kwargs)
        return xr.DataArray(raster, coords=coords, dims=('y', 'x'))


def mask_array(shapeobj, xrobj):
    if 'lon' in xrobj.coords._names:
        ds = xr.Dataset(coords={'lon': xrobj.lon, 'lat': xrobj.lat})
    elif 'x' in xrobj.coords._names:
        ds = xr.Dataset(coords={'x': xrobj.x, 'y': xrobj.y})
    else:
        print('Unknown coordinate system')
        return
    ds['mask'] = rasterize(shapeobj.geometry, ds.coords)

    # example of applying a mask
    return xrobj.where(ds.mask == 1)


def shp_to_raster(shapeobj, xrobj):
    ds = xr.Dataset(coords={'lon': xrobj.lon, 'lat': xrobj.lat})
    ds['raster'] = rasterize(zip(shapeobj.geometry, shapeobj.ID), ds.coords)
    return ds.raster


def compute_anomaly(pddf, monthly=True, return_clim=False, avgnorm=False):
    if monthly:
        pddf_clim_in = pddf.rolling(10).mean()
    else:
        pddf_clim_in = pddf
    # calculate daily climatology
    clim = pddf_clim_in.groupby(pddf.index.dayofyear).mean()
    clim_std = pddf_clim_in.groupby(pddf.index.dayofyear).std()
    # calulate anomalies
    anomalies = pddf.copy()
    # for i in range(1, 367, 10):
    for i in np.unique(pddf.index.dayofyear):
        if avgnorm:
            anomalies.loc[anomalies.index.dayofyear == i] = pddf.loc[pddf.index.dayofyear == i] / clim.loc[i]
        else:
            anomalies.loc[anomalies.index.dayofyear == i] = (pddf.loc[pddf.index.dayofyear == i] - clim.loc[i]) / \
                                                            clim_std.loc[i]

    if return_clim:
        return clim, clim_std, anomalies
    else:
        return anomalies


def compute_climatology_stack(stack, monthly=False, dekad=False):
    # calculate climatology
    if monthly:
        # stack = stack.chunk(chunks={'time': 100})
        stack = stack.rolling(time=30).mean()
    if dekad:
        # stack = stack.chunk(chunks={'time': 100})
        stack = stack.rolling(time=10).mean()
    clim_avg = stack.groupby('time.dayofyear').mean("time", skipna=True)
    clim_std = stack.groupby('time.dayofyear').std("time", skipna=True)
    return clim_avg, clim_std


def compute_anomaly_stack(stack, monthly=False, dekad=False):
    # calculate climatology
    if monthly:
        stack = stack.chunk(chunks={'time': 100})
        stack = stack.rolling(time=30).mean()
    if dekad:
        stack = stack.chunk(chunks={'time': 100})
        stack = stack.rolling(time=10).mean()
    clim = stack.groupby('time.dayofyear').mean(skipna=True)
    return stack.groupby('time.dayofyear') / clim


def hide_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1, 2 ** 64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current,
        toggle_text=toggle_text
    )

    return HTML(html)


def crop_CCI():
    cci_path = '/mnt/CEPH_PROJECTS/ADO/SM/CCI/combined/'
    minlon, minlat, maxlon, maxlat = get_ado_extent()

    sm_files = list()
    for path in Path(cci_path).rglob('*.nc'):
        sm_files.append(path)

    for ipath in sm_files:
        outpath_cropped = str(ipath.parent) + '/' + ipath.name[:-3] + '_SUB.nc'

        img = xr.open_dataset(ipath)
        # crop the original file
        img_cropped = img.where((img.lon > minlon) & (img.lon < maxlon) & (img.lat > minlat) &
                                (img.lat < maxlat), drop=True)
        img_cropped.to_netcdf(outpath_cropped)
        img.close()
        os.remove(ipath)


def CCI_annual_stacks():
    cci_path = '/mnt/CEPH_PROJECTS/ADO/SM/CCI/combined/'
    for i in range(1978, 2021):
        sm_files = list()
        for path in Path(cci_path + str(i) + '/').rglob('*.nc'):
            sm_files.append(path)

        cci_stack = xr.open_mfdataset(sm_files,
                                      concat_dim='time',
                                      parallel=True)
        cci_stack = cci_stack.sortby('time')
        cci_stack.to_netcdf(cci_path + 'CCI_' + str(i) + '.nc')
        cci_stack.close()
