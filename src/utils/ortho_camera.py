import os
import numpy as np

try:
    IMPORTED_GDAL=True
    from osgeo import gdal
except ModuleNotFoundError:
    IMPORTED_GDAL=False
    import rasterio

from satdepth.src.utils.useful_methods import get_basename, get_img_path, get_file_ext
from satdepth.src.utils.sat_image_read_utils import ReadRaster

class Ortho_Camera(object):
    "Camera model for single band Ortho-rectified images"
    def __init__(self):
        self.ds = None        
    
    @classmethod
    def from_file(cls, img_name:str):
        """
        Create an Ortho_Camera object from an image file
        """
        img_file_ext = ['tif','jp2', 'TIF']
        ortho_cam = cls()
        ortho_cam.filepath = get_img_path(img_name)
        ortho_cam.ext = get_file_ext(img_name)
        ortho_cam.image_basename  = get_basename(img_name)
        if ortho_cam.ext in img_file_ext:
            ortho_cam.from_tif(img_name)
        else:
            print('cannot read %s image file type'%(img_name))
            raise IOError
        return ortho_cam
        
    def from_tif(self, img_name:str):
        """
        Read image file and extract metadata
        """
        if IMPORTED_GDAL:
            self.ds = gdal.Open(img_name, gdal.GA_Update)
            self.geotransform = self.ds.GetGeoTransform()
            self.nrows, self.ncols = self.ds.ReadAsArray().shape
        else:
            self.ds = rasterio.open(img_name)
            self.geotransform = self.ds.transform
            self.nrows, self.ncols = self.ds.height, self.ds.width

        self.originX = self.geotransform[0]
        self.originY = self.geotransform[3]
        self.pixWd = self.geotransform[1]
        self.pixHt = self.geotransform[5]
                
        self.ds = None
        
    
    def backproject(self,line:float, samp:float) -> list:
        """
        Backproject a pixel to a world point
        Args: 
            line (float|list|np.array) : line, row, y
            samp (float|list|np.array) : sample, col, x
        Returns:
            X (float|list|np.array) : lon
            Y (float|list|np.array) : lat 
        """
        X = self.originX + self.pixWd*samp
        Y = self.originY + self.pixHt*line
        return X, Y
           
    def project(self, X:float, Y:float)-> list:
        """
        Project a world point to its corresponding pixel location
        Args: 
            X (float|list|np.array) : lon
            Y (float|list|np.array) : lat
        Returns:
            r (float|list|np.array) : line, row, y
            c (float|list|np.array) : sample, col, x
        """
        r = (Y - self.originY)/self.pixHt # line, y
        c = (X - self.originX)/self.pixWd # samp, x       
        return r, c #line, samp or y,x
    
    def ReadImg(self, 
                x_off:int=None,
                y_off:int=None,
                x_size:int=None,
                y_size:int=None,
                normalize:bool=False)->np.array:
        """
        Read image file and extract metadata
        Args:
            x_off (int) : x offset
            y_off (int) : y offset
            x_size (int) : x size
            y_size (int) : y size
            normalize (bool) : normalize image to 0-1
        Returns:
            img (np.array) : image
        """
        tif_name = os.path.join(self.filepath, self.image_basename +'.'+ self.ext)
        img, self.nodata_mask = ReadRaster(tif_name, 
                                            x_off=x_off, y_off=y_off, x_size=x_size, y_size=y_size,
                                            normalize=normalize)
        return img