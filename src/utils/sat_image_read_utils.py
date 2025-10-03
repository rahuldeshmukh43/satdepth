import numpy as np
try:
    from osgeo import gdal
    IMPORTED_GDAL=True
except ModuleNotFoundError:
    IMPORTED_GDAL=False
    import rasterio

NODATA = -9999. # nodata value for images 

#------------------------#
## Image reading utils
#------------------------#
def img_to_uInt8(img):
    """
    convert 0-1 img to uint8
    Args:
        img: np.array [H, W] PAN image
    Returns:
        img_uint8: np.array [H, W] PAN image
    """
    img_uint8 = np.uint8(np.clip(255.*img, 0, 255))
    return img_uint8

def repeat_channels(img, r=3):
    """
    repeats the grayscale PAN image to make a 3 channel image
    Args:
        img: np.array [H, W] PAN image
        r: int number of channels to repeat
    Returns:
        img: np.array [H, W, 3] PAN image
    """
    assert len(img.shape)==2
    return np.repeat(img[:,:,np.newaxis],r, axis=2) #[H,W,3]


def ReadRaster(tif_name:str, 
               x_off:int=None, 
               y_off:int=None, 
               x_size:int=None, 
               y_size:int=None, 
               normalize:bool=False)-> np.array:
    """
    Read PAN image: reads raster and returns raster in range 0-1
    Args:
        tif_name: str path to tif file
        x_off: int x offset
        y_off: int y offset
        x_size: int x size
        y_size: int y size 
        normalize: bool normalize the image to 0-1
    Returns:
        img: np.array [H, W] PAN image
    """
    eps = 1e-6
    if IMPORTED_GDAL:
        ds = gdal.Open(tif_name, gdal.GA_ReadOnly)
        if x_off != None:
            # read only portion of image
            img = ds.ReadAsArray(x_off, y_off, x_size, y_size)#.astype(np.float32)
        else:
            img = ds.ReadAsArray()#.astype(np.float32)
    else:
        ds = rasterio.open(tif_name)
        if x_off != None:
            img = ds.read(1, window=rasterio.windows.Window(x_off, y_off, x_size, y_size))
        else:
            img = ds.read(1)

    nodata_mask = img==NODATA

    if normalize:
        data_mask = img!=NODATA
        vMin = np.min(img[data_mask])
        # vMax = np.nanpercentile(img[data_mask], IMG_MAX_PERCENTILE) # percentile wouldnt work well when we have patches with majority of shadows (resulting in 95 p being the shadow value which is less than the max)
        vMax = np.max(img[data_mask])
        img[nodata_mask] = vMin
        img = np.clip(1. * (img - vMin) / (vMax - vMin + eps), 0., 1.)  # nodata gets turned to zero
    ds = None
    return img.astype(np.float32), nodata_mask

def ReadLatLonHt(tif_name:str, 
                 x_off:int=None, 
                 y_off:int=None, 
                 x_size:int=None, 
                 y_size:int=None)->np.array:
    """
    Read Lat/Lon/Ht image: reads Float64 raster and returns it
    Args:
        tif_name: str path to tif file
        x_off: int x offset
        y_off: int y offset
        x_size: int x size
        y_size: int y size
    Returns:
        img: np.array [H, W] lat/lon/ht image
    """
    if IMPORTED_GDAL:
        if x_off != None:
            # read only portion of image
            img = gdal.Open(tif_name, gdal.GA_ReadOnly).ReadAsArray(x_off, y_off, x_size, y_size)#.astype(np.float32)
        else:
            # read the whole image
            img = gdal.Open(tif_name, gdal.GA_ReadOnly).ReadAsArray()#.astype(np.float32)
    else:
        if x_off != None:
            with rasterio.open(tif_name) as src:
                img = src.read(1, window=rasterio.windows.Window(x_off, y_off, x_size, y_size))
        else:
            img = rasterio.open(tif_name).read(1)
    return img