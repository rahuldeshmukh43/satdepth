#!/bin/python3
import os
from osgeo import gdal, osr
import socket
import numpy as np

def basename(full_path:str):
    base = os.path.split(full_path)[1].split('.')[0]
    return base

def check_if_exists(path:str):
    if os.path.exists(path):
        return True
    else:
        raise ValueError("\n\nERROR: %s does not exist.\n"%(path))
        return False

def convert_to_epsg4326(dem_file:str):
    """
    Convert dem from utm to epsg4326
    """
    dem_ds = gdal.Open(dem_file)
    dem_srs = osr.SpatialReference(wkt=dem_ds.GetProjectionRef())
    dem_projcs = dem_srs.GetAttrValue('PROJCS')

    # Get dem projcs. i.e. utm code
    dem_srs_epsg = '%s:%s'%(dem_srs.GetAttrValue('AUTHORITY',0),
                            dem_srs.GetAttrValue('AUTHORITY', 1))

    dem_ext = os.path.splitext(dem_file)[-1]
    dem_file_epsg4326 = dem_file.replace(dem_ext, '_epsg4326.tif')

    if dem_projcs is not None and 'UTM' in dem_projcs:
        if not os.path.isfile(dem_file_epsg4326):
            print("Converting file from %s to EPSG4326"%(dem_srs_epsg))

            dem_nodataval = dem_ds.GetRasterBand(1).GetNoDataValue()
            if dem_nodataval is None:
                dem_nodataval = np.nan

            dem_nodataval = str(dem_nodataval)

            command = 'gdalwarp -t_srs EPSG:4326 -r near -srcnodata ' + \
                      dem_nodataval + ' -dstnodata ' + dem_nodataval + ' ' + \
                      dem_file + ' ' + dem_file_epsg4326
            os.system(command)
        else:
            raise ValueError("\nAre you sure if DEM %s is in UTM projection? Stopping"%(dem_file))

    dem_ds = None
    return dem_file_epsg4326

def get_vm_ip_address() -> str:
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    return ip