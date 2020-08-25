from pathlib import Path

# collection of useful functions
def get_ado_extent(proj='WGS84'):
    # returns the bounding box for the ADO extent: minlon, minlat, maxlon, maxlat
    if proj == 'WGS84':
        return 3.6846960000000046, 42.9910929945153200, 17.1620089999999941, 50.5645599970407318
    else:
        print('Only WGS84 supported at the moment')
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
