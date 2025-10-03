import os
import numpy as np
import cv2
from datetime import datetime

import utm

try:
    from osgeo import gdal
    IMPORTED_GDAL=True
except ModuleNotFoundError:
    IMPORTED_GDAL=False
    import rasterio

from satdepth.src.utils.sat_image_read_utils import ReadRaster, repeat_channels, ReadLatLonHt, img_to_uInt8
from satdepth.src.utils.rpc import RPC, load_imd
from satdepth.src.utils.ortho_camera import Ortho_Camera

IMG_MAX_PERCENTILE = 80 # 95 # percentile for image normalization
NODATA = -9999. # nodata value for images 
MAX_ALLOWABLE_NODATA_PERCENT = 20.0 # % percentage of image with nodata=0

def image_name_to_short_name(image_name:str)->str:
    base = os.path.basename(image_name)
    sn =  base.split('.')[0][:7]
    return sn

def short_name_to_date(sn: str) -> datetime:
    return datetime.strptime(sn, "%d%b%y")

def time_difference(sn1:str, sn2:str)->int:
    """
    Calculate the difference in days between two worldview image names
    dates are in format DDMMMYY
    Args:
        sn1: str worldview image name
        sn2: str worldview image name
    Returns:
        days_difference: int abs difference in days between two images
    """    
    # # Define the format for parsing
    # date_format = "%d%b%y"

    # # Parse the dates into datetime objects
    # date1 = datetime.strptime(sn1, date_format)
    # date2 = datetime.strptime(sn2, date_format)

    date1 = short_name_to_date(sn1)
    date2 = short_name_to_date(sn2)

    # Calculate the difference in days
    difference = date2 - date1
    days_difference = difference.days
    return abs(days_difference)

def compute_relative_track_angle(img0_ds,
                                img1_ds,
                                num_pts:int=10,
                                method:str='dot')->float:
    """
    Function for computing the relative track angle between two images
    Args:
        img0_ds: datasample object
        img1_ds: datasample object
        num_pts (int): number of points to sample along the center line scan
        method (str): method for computing the relative track angle ['dot', 'ccw']
    Returns:
        theta (float): relative track angle between two images
    """
    assert method == "dot" or method == "ccw", "wrong method for computing relative track angle" # ccw computes angle between 0 and 1 in ccw direction 

    def estimate_line_scan_dir(img_ds, num_pts:int)->np.array:
        """
        Function for estimating the line scan direction of the center line scan
        Args:
            img_ds: datasample object
            num_pts (int): number of points to sample along the center line scan
        Returns:
            scan_dir (np.array): line scan direction
        """
        # get img extents, 
        nrows, ncols = img_ds.ReadRasterSize()
        # create linspace of pts along center row
        cols = np.linspace(0, ncols, num_pts).astype(np.int16).clip(0,ncols-1)
        # rows = np.linspace(0, nrows, num_pts).astype(np.int).clip(0,nrows-1)
        # read lat, lon for the grid of pts
        lat, lon, ht = img_ds.ReadLatLonHt()
        lats = lat[nrows//2, cols]
        lons = lon[nrows//2, cols]
        # lats = lat[rows, ncols//2]
        # lons = lon[rows, ncols//2]
        # filter no data
        lats, lons = zip(*filter(lambda pair: pair[0] != NODATA, zip(lats, lons)))
        # print(lats, lons)
        # convert lat, lon to x,y in meters
        X, Y, _utm_zone_int, _utm_zone_letter = utm.from_latlon(np.array(lats), np.array(lons))
        num_x = X.shape[0]
        # normalize X and Y using centroid
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)
        x = X - X_mean
        y = Y - Y_mean
        # scaling 
        d = np.mean(np.sqrt(x**2 + y**2))
        s = np.sqrt(2)/d
        x = s * x
        y = s * y
        # estimate line equation using x,y -- linear regression
        D = np.ones((num_x, 3)).astype(np.float32)
        D[:,0] = x
        D[:,1] = y
        # Perform SVD
        U, S, Vt = np.linalg.svd(D)
        line_hc = Vt[-1]  # The last row of Vt corresponds to the smallest singular value
        a, b, c = line_hc
        # return np.array([a,b,c])
        A = a*s
        B = b*s
        C = c - a*s*X_mean - b*s*Y_mean
        # compute  left to right dir of line
        if np.abs(B) > 1e-8 :
            X0 = X[0]
            Y0 = - (A*X0 + C)/ B
            X1 = X[-1]
            Y1 = - (A*X1 + C)/ B
            scan_dir = np.array([X1 - X0, Y1 - Y0])
        else:
            # B is nearly 0 ie vertical line
            Y0 = Y[0]
            Y1 = Y[-1]
            scan_dir = np.array([0., Y1 - Y0])
        return scan_dir
    
    # compute line equation of center line scan for both images
    L0 = estimate_line_scan_dir(img0_ds, num_pts)
    L1 = estimate_line_scan_dir(img1_ds, num_pts)

    # compute the intersection angle
    if method=="dot":
        theta = np.arccos(np.dot(L0, L1)/(np.linalg.norm(L0)*np.linalg.norm(L1)))
        theta = np.rad2deg(theta)
        return theta
    elif method == "ccw":
        theta = np.arccos(np.dot(L0, L1)/(np.linalg.norm(L0)*np.linalg.norm(L1))) # cos theta
        sin_theta = np.cross(L0, L1)  # sin theta
        if sin_theta < 0:
            # clockwise
            theta *= -1        
        theta = np.rad2deg(theta)
        return theta

def ImageIsUseable(img:np.array)->bool:
    "img: (np.array) [h,w,c]"
    # compute percentage of nodata=0
    try:
        h, w, _ = img.shape
        nodata_raster = img[:,:,0]==0
    except:
        h, w = img.shape
        nodata_raster = img[:, :] == 0
    p = (np.sum(nodata_raster)/(h*w)) * 100.
    if p > MAX_ALLOWABLE_NODATA_PERCENT:
        return False
    return True

#------------------------#
## File reading utils
#------------------------#
class DataSample():
    "Container for easy storage of paths for each sat img sample"
    def __init__(self,
                img_path:str,
                rpc_path:str,
                lat_path:str,
                lon_path:str,
                ht_path:str,
                dsm_path:str):
        self.img_path = img_path
        self.rpc_path = rpc_path
        self.lat_path = lat_path
        self.lon_path = lon_path
        self.ht_path = ht_path
        self.dsm_path = dsm_path

    def ReadRasterSize(self):
        if IMPORTED_GDAL:
            ds = gdal.Open(self.img_path, gdal.GA_ReadOnly)
            nrows = ds.RasterYSize
            ncols = ds.RasterXSize
            ds = None
        else:
            ds = rasterio.open(self.img_path)
            nrows = ds.height
            ncols = ds.width
            ds = None
        return nrows, ncols

    def ReadImg(self,
                x_off=None,
                y_off=None,
                x_size=None,
                y_size=None,
                repeat=True):
        img, _nodata_mask = ReadRaster(self.img_path, x_off, y_off, x_size, y_size, normalize=True)
        if repeat:
            return repeat_channels(img)
        return img

    def ReadRPC(self):
        return ReadRPC(self.rpc_path)

    def ReadDSM(self):
        dsm_cam = ReadDSM(self.dsm_path)
        return dsm_cam

    def ReadLatLonHt(self, x_off=None, y_off=None, x_size=None, y_size=None):
        lat = ReadLatLonHt(self.lat_path,  x_off, y_off, x_size, y_size)
        lon = ReadLatLonHt(self.lon_path, x_off, y_off, x_size, y_size)
        ht = ReadLatLonHt(self.ht_path, x_off, y_off, x_size, y_size)
        return lat, lon, ht

def ReadDSM(dsm_path:str):
    dsm_cam = Ortho_Camera.from_file(dsm_path)
    return dsm_cam

def ReadRPC(rpc_path:str):
    rpc = RPC.from_file(rpc_path) 
    return rpc

def GetRPCAffine(rpc,
                lat_center:float,
                lon_center:float,
                ht_center:float,
                x_off:int=0,
                y_off:int=0):
    """
    Function to compute affine matrix for a given rpc camera object
    NOTE: be careful with input rpc and x_off, y_off. If you translated the rpc to x_off, y_off before calling this
     function then most probably you dont need to give x_off, y_off to this function call
    Args:
        rpc: rpc camera object of entire image
        lat_center: np array of size N
        lon_center: np array of size N
        ht_center: np array of size N
        x_off: upper left x coordinate of the patch
        y_off: upper left y coordinate of the patch

    Return:
        - P (np.array) [3,4] camera matrix such that [samp, line, 1] = P * [lon, lat, ht, 1]
             eqv [x, y, 1] = P * [X, Y, Z, 1]
    """
    rpc_affine, img0_rpc_affine_mat = rpc.affine_approx(lat_center, lon_center, ht_center, flagReturnMatrix=True)
    Scale_XYZ = np.array([[1./rpc_affine.lon_scale, 0., 0., -rpc_affine.lon_off/rpc_affine.lon_scale],
                          [0., 1./rpc_affine.lat_scale, 0., -rpc_affine.lat_off/rpc_affine.lat_scale],
                          [0., 0., 1./rpc_affine.height_scale, -rpc_affine.height_off/rpc_affine.height_scale],
                          [0., 0., 0., 1.]])
    affine_mat_3x4 = np.vstack((img0_rpc_affine_mat, np.array([0., 0., 0., 1.])))
    Scale_xy = np.array([[rpc.samp_scale, 0., rpc.samp_off],
                         [0., rpc.line_scale, rpc.line_off],
                         [0., 0., 1.]])
    translate_to_patch = np.array([[1., 0., -x_off],
                                   [0., 1., -y_off],
                                   [0., 0., 1.]])
    return translate_to_patch @ (Scale_xy @ (affine_mat_3x4 @ Scale_XYZ))

#------------------------#
## geometry utils
#------------------------#
def latlonh_to_ecef(lat:float, 
                    lon:float, 
                    h:float) -> np.array:
    """
    Converts lat, lon, ht to ECEF coordinates in order
    to compute euclidean distance between two points
    Args:
        lat: latitude in degrees
        lon: longitude in degrees
        h: height in meters
    Returns:
        x, y, z: ECEF coordinates in meters
    """
    a = 6378137.0
    b = 6356752.314245
    lat_rad = lat * np.pi / 180.
    lon_rad = lon * np.pi / 180.
    N_lat = a**2 / np.sqrt((a**2)*np.cos(lat_rad)**2 +
                           (b**2)*np.sin(lat_rad)**2)
    x = (N_lat + h)*np.cos(lat_rad)*np.cos(lon_rad)
    y = (N_lat + h)*np.cos(lat_rad)*np.sin(lon_rad)
    z = ((b**2)/(a**2)*N_lat + h)*np.sin(lat_rad)
    return x, y, z

def make_view_vector(el:float, 
                     az:float)->np.array:
    """
    Args:        
        el (float) elevation angle in radians
        az (float) azimuth angle in radians
    Returns:
        view_vector (np.array) [3] view vector in ECEF coordinates
    """
    view_vector = np.array([ np.cos(el) * np.sin(az),
                             np.cos(el) * np.cos(az),
                             np.sin(el)])
    return view_vector

def compute_intersection_angle(imd0_file:str, imd1_file:str)->float:
    """
    Args
        imd0_file: (str) path to imd fle
        imd1_file: (str) path to imd fle
    Returns:
        - angle (float) viewing intersection angle for two images  in degrees
    """
    imd0 = load_imd(imd0_file)
    imd1 = load_imd(imd1_file)
    # make sat view vector
    v0 = make_view_vector(np.deg2rad(float(imd0["IMAGE_1"]["meanSatEl"])),
                          np.deg2rad(float(imd0["IMAGE_1"]["meanSatAz"])))
    v1 = make_view_vector(np.deg2rad(float(imd1["IMAGE_1"]["meanSatEl"])),
                          np.deg2rad(float(imd1["IMAGE_1"]["meanSatAz"])))
    # compute dot product and angle
    angle = np.rad2deg(np.arccos(v0.dot(v1)))
    return angle

def compute_distance(lat0:float, 
                     lon0:float, 
                     ht0:float, 
                     lat1:float, 
                     lon1:float, 
                     ht1:float)->float:
    x0, y0, z0 = latlonh_to_ecef(lat0, lon0, ht0)
    x1, y1, z1 = latlonh_to_ecef(lat1, lon1, ht1)
    d = np.sqrt( (x0 - x1)**2 + (y0 - y1)**2 + (z0 - z1)**2 )
    return d

def get_random_pt_on_dsm(img0_ds,
                        patch_size:int,
                        dsm_picking_buffer:int)->tuple:
    """
    Function to get a random point on the dsm
    Args:
        img0_ds: datasample object
        patch_size: int patch size
        dsm_picking_buffer: int buffer for picking point
    Returns:
        lat, lon, ht: tuple of float values
    """
    dsm_cam = img0_ds.ReadDSM()
    dsm = dsm_cam.ReadImg()

    #draw a random point on dsm
    ht = -9999 #nodata
    while(ht==-9999):
        pt_row, pt_col = np.random.rand(2)
        pt_row *= (dsm_cam.nrows -2*dsm_picking_buffer - patch_size); pt_row += dsm_picking_buffer + patch_size//2
        pt_row = int(pt_row)
        pt_col *= (dsm_cam.ncols-2*dsm_picking_buffer - patch_size); pt_col += dsm_picking_buffer + patch_size//2
        pt_col = int(pt_col)
        ht = dsm[pt_row, pt_col]

    lon, lat = dsm_cam.backproject(pt_row, pt_col)
    return lat, lon, ht

def get_random_shifted_lat_lon_ht(img_ds, lat:float,lon:float,ht:float, shift_window_size:int=0):
    dsm_cam = img_ds.ReadDSM()
    dsm = dsm_cam.ReadImg()

    # find row, col for lat, lon
    row, col = dsm_cam.project(lon, lat)
    row, col = int(row), int(col)
    # randomly shift row, col wihin the extents such that the new point has a valid ht
    ht_out = -9999 #nodata
    while(ht_out==-9999):
        row_shift = np.random.randint(-shift_window_size//2, shift_window_size//2)
        col_shift = np.random.randint(-shift_window_size//2, shift_window_size//2)
        new_row = row + row_shift
        new_col = col + col_shift
        ht_out = dsm[new_row, new_col]

    # get lat, lon for the new point
    lon_out, lat_out = dsm_cam.backproject(new_row, new_col)
    return lat_out, lon_out, ht_out

def get_corresponding_patch_sizes(img0_ds, 
                                  img0_rpc, 
                                  img1_ds, 
                                  img1_rpc, 
                                  out_img_size:int, 
                                  lat:float, 
                                  lon:float, 
                                  ht:float):
    """
    Args:
        img0_ds: datasample object
        img0_rpc:  rpc object
        img1_rpc: rpc object
        out_img_size (int): output image size in pixels
    Return:
        x0_off (int): x coord of offset for patch on original image
        y0_off (int): y coord of offset for patch on original image
        x0_size (int): xsize of patch
        y0_size (int): ysize of patch
        x1_off (int): x coord of offset for patch on original image
        y1_off (int): y coord of offset for patch on original image
        x1_size (int): xsize of patch
        y1_size (int): ysize of patch
    """
    #project 3d point into images and get coordinates for image patches
    row0, col0 = img0_rpc.rpc(lat, lon, ht)
    row1, col1 = img1_rpc.rpc(lat, lon, ht)
    row0, col0, row1, col1 = list(map(lambda x: int(np.round(x)), [row0, col0, row1, col1]))

    x0_off = col0 - (out_img_size//2)
    y0_off = row0 - (out_img_size // 2)
    x0_size = out_img_size
    y0_size = out_img_size

    x1_off = col1 - (out_img_size // 2)
    y1_off = row1 - (out_img_size // 2)
    x1_size = out_img_size
    y1_size = out_img_size

    #check if patches are withing valid bounds
    img0_nrows, img0_ncols = img0_ds.ReadRasterSize()
    img1_nrows, img1_ncols = img1_ds.ReadRasterSize()

    if (np.any(np.array([x0_off, y0_off, x1_off, y1_off]) < 0 ) or 
        np.any( np.array([x0_off, y0_off, x1_off, y1_off]) + np.array([x0_size, y0_size, x1_size, y1_size]) >
        np.array([ img0_ncols, img0_nrows, img1_ncols, img1_nrows]))):
        print("patch out of bounds")
        return None
    #TODO: clean up the return -- we dont need to return lat, lon, ht
    return x0_off, y0_off, x0_size, y0_size, x1_off, y1_off, x1_size, y1_size, lat, lon , ht

def get_valid_patch(img0_ds, 
                    img0_rpc, 
                    out_img_size:int, 
                    lat:float, 
                    lon:float, 
                    ht:float):
    """
    Args:
        img0_ds: datasample object
        img0_rpc:  rpc object
        out_img_size: integer in pixels
    Returns:
        x0_off: int x coord of offset for patch on original image
        y0_off: int y coord of offset for patch on original image
        x0_size: int xsize of patch
        y0_size: int ysize of patch
    """
    #project 3d point into images and get coordinates for image patches
    row0, col0 = img0_rpc.rpc(lat, lon, ht)
    row0, col0= list(map(lambda x: int(np.round(x)), [row0, col0]))

    x0_off = col0 - (out_img_size//2)
    y0_off = row0 - (out_img_size // 2)
    x0_size = out_img_size
    y0_size = out_img_size

    #check if patches are withing valid bounds
    img0_nrows, img0_ncols = img0_ds.ReadRasterSize()

    if np.any(np.array([x0_off, y0_off]) < 0 ) or\
        np.any( np.array([x0_off, y0_off]) + np.array([x0_size, y0_size]) >
        np.array([ img0_ncols, img0_nrows]) ):
        print("patch0 out of bounds")
        return None

    return x0_off, y0_off, x0_size, y0_size

def get_lat_lon_ht(x0:list,
                   y0:list,
                   img0_lat:np.array,
                   img0_lon:np.array,
                   img0_ht:np.array,
                   nodata_value:int= -9999):
    """
    Function to get lat, lon, ht from image lat/lon/ht rasters given x0, y0
    Args:
        x0: list of x coordinates
        y0: list of y coordinates
        img0_lat: np.array of latitudes
        img0_lon: np.array of longitudes
        img0_ht: np.array of heights
        nodata_value: int nodata value
    Returns:    
        x0: list of valid x coordinates
        y0: list of valid y coordinates
        lat0: list of valid latitudes
        lon0: list of valid longitudes
        ht0: list of valid heights
        mask0: boolean mask for valid points
    """
    latlonh0 = np.array([[img0_lat[r, c], img0_lon[r, c], img0_ht[r, c]] for c, r in zip(x0, y0)])
    mask0 = latlonh0[:,2] != nodata_value
    lat0, lon0, ht0 = latlonh0[mask0,0], latlonh0[mask0,1], latlonh0[mask0,2]
    y0, x0 = y0[mask0], x0[mask0]
    return x0, y0, lat0, lon0, ht0, mask0

def affine_cam_project(affine_cam:np.array, 
                       lat:np.array, 
                       lon:np.array, 
                       ht:np.array):
    """
    Project lat, lon, ht to pixel coordinates using affine camera matrix
    Args:
        affine_cam (np.array) [3,4] -- projects lon lat, ht to pixel coordinate
        lat, lon, ht (np.array) [N] -- N world pts
    Returns:
        x, y (np.array) [N] -- N pixel coordinates
    """
    xy = affine_cam @ np.vstack((lon, lat, ht, np.ones_like(ht)))
    xy = xy / xy[-1, :]
    x= xy[0, :]
    y = xy[1, :]
    return x, y


#------------------------#
## Matching Point utils -- useful for CAPS, DualRC Training
##------------------------#
def warp_kpts(kpts0:np.array,
            lat0:np.array,
            lon0:np.array,
            ht0:np.array,
            lat1:np.array, 
            lon1:np.array, 
            ht1:np.array,
            affine_cam0:np.array,
            affine_cam1:np.array,
            nodata_value:int=-9999, 
            distance_th:float=1.0):
    """
    Warp kpts0 from I0 to I1 with lat, lon, h
    Also check covisibility and lat, lon , ht consistency.
    world point  is consistent if distance < distance_th (hard-coded).

    Args:
        kpts0 (np.array): [L, 2] - <x, y>,
        lat0 (np.array): [H, W] lat for img0
        lon0 (np.array): [H, W] lon for img0
        ht0 (np.array): [H, W] ht for img0

        lat1 (np.array): [H, W] lat for img1
        lon1 (np.array): [H, W] lon for img1
        ht1 (np.array): [H, W] ht for img1

        affine_cam0 (np.array): [3, 4] affine camera for img0
        affine_cam1 (np.array): [3, 4] affine camera for img1
        nodata_value : -9999 default
        distance_th : threshold for true matching points (in meters)
    Returns:
        calculable_mask (np.array): [L]
        warped_keypoints0 (np.array): [L, 2] <x0_hat, y1_hat>
    """
    
    kpts0_int = kpts0.round().astype(np.int)

    # Sample lat0, lon0, ht0, get calculable_mask on depth != 0
    kpts0_lat = lat0[ kpts0_int[:, 1], kpts0_int[:, 0]]  # (L)
    kpts0_lon = lon0[ kpts0_int[ :, 1], kpts0_int[ :, 0]] # (L)
    kpts0_ht  = ht0[ kpts0_int[:, 1], kpts0_int[ :, 0]] # (L)
    valid_mask0 = kpts0_lat != nodata_value

    # Project sampled lat0, lon0, ht0 to img1 using affine_cam1
    w_kpts0_h = np.stack([kpts0_lon, kpts0_lat, kpts0_ht, np.ones_like(kpts0_lat)], axis=-1) # (L, 4)
    kpts1 = (affine_cam1 @ w_kpts0_h.transpose(1,0)).transpose(1,0) # (L, 3)
    # kpts1 = kpts1[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4) # (N, L, 2) for regular transformation we need to normalize hc
    kpts1 = kpts1[:, :2] # (L, 2) but for affine camera the third hc is always 1

    # covisible check
    h,w = lat1.shape
    covisible_mask = (kpts1[:, 0] > 0) * (kpts1[:, 0] < w - 1) * \
                     (kpts1[:, 1] > 0) * (kpts1[:, 1] < h - 1)
    kpts1_int = kpts1.round().astype(np.int)
    kpts1_int[~covisible_mask, :] = 0

    # Read true lat1, lon1, ht1 at projected points
    kpts1_lat = lat1[kpts1_int[:, 1],  kpts1_int[:, 0]]  # (L)
    kpts1_lon = lon1[kpts1_int[:, 1],  kpts1_int[:, 0]] # (L)
    kpts1_ht  = ht1[ kpts1_int[:, 1],  kpts1_int[:, 0]] # (L)
    valid_mask1 = kpts1_lat != nodata_value

    # compute consistency using ecef distance between two sets of kpts lat, lon, ht
    # consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    distance = compute_distance(kpts0_lat, kpts0_lon, kpts0_ht,
                                kpts1_lat, kpts1_lon, kpts1_ht)
    consistent_mask = distance < distance_th # meters
    valid_mask = valid_mask0 * valid_mask1 * covisible_mask * consistent_mask

    return valid_mask, kpts1

def get_matches(mode:str,
                pct_sift:float,
                num_pts:int,
                distance_thresh:float,
                nodata_value:int,
                img0:np.array,
                affine_cam0:np.array,
                lat0:np.array, 
                lon0:np.array, 
                ht0:np.array,
                img1:np.array,
                affine_cam1:np.array,
                lat1:np.array, 
                lon1:np.array, 
                ht1:np.array,
                x0_size:int,
                y0_size:int,
                x1_size:int, 
                y1_size:int):
    """
    Function to get matches between two images
    Args:
        mode: str mode for generating keypoints ['random', 'sift', 'mixed']
        pct_sift: float percentage of sift keypoints
        num_pts: int number of keypoints to generate
        distance_thresh: float threshold f            # print("read whole")
        lon0: np.array [H, W] lon for img0
        ht0: np.array [H, W] ht for img0
        img1: np.array [H, W] PAN image
        affine_cam1: np.array [3, 4] affine camera for img1
        lat1: np.array [H, W] lat for img1
        lon1: np.array [H, W] lon for img1
        ht1: np.array [H, W] ht for img1
        x0_size: int size of patch for img0
        y0_size: int size of patch for img0
        x1_size: int size of patch for img1
        y1_size: int size of patch for img1
    Returns:
        x0: np.array [N] x coordinates for img0
        y0: np.array [N] y coordinates for img0
        x1: np.array [N] x coordinates for img1
        y1: np.array [N] y coordinates for img1
    """
    # if img has 3 channels, convert to grayscale
    if len(img0.shape) == 3:
        img0 = img0[:,:,0]
        img1 = img1[:,:,0]
    
    kpts0 = generate_query_kpt(img0, mode, num_pts, y0_size, x0_size, pct_sift = pct_sift, nodata_value=nodata_value) #[N, 2]

    valid_mask, kpts1 = warp_kpts(kpts0,
                                    lat0, lon0, ht0,
                                    lat1, lon1, ht1,
                                    affine_cam0,
                                    affine_cam1,
                                    nodata_value=nodata_value, 
                                    distance_th=distance_thresh)
    
    # filter out invalid matches
    kpts0 = kpts0[valid_mask]
    kpts1 = kpts1[valid_mask]

    x0, y0, x1, y1 = kpts0[:, 0], kpts0[:, 1], kpts1[:, 0], kpts1[:, 1]

    return x0, y0, x1, y1    

def generate_query_kpt(img, mode, num_pts, h, w, pct_sift=0.9, nodata_value=-9999):
    """
    returns [N,2] coordinates of keypoints as a 2d array
    Args:
        img: np.array [H, W] PAN image
        mode: str mode for generating keypoints ['random', 'sift', 'mixed']
        num_pts: int number of keypoints to generate
        h: int height of image
        w: int width of image
        pct_sift: float percentage of sift keypoints
        nodata_value: int nodata value
    Returns:
        coord: np.array [N,2] [(x1,y1), (x2,y2),..,(xn,yn)] x->samp, y-> line
    """
    pct_rand = 1 - pct_sift
    if len(img.shape) == 3:
        _img = img[:,:,0]
    elif len(img.shape) == 2:
        _img = img
    else:
        TypeError("Wrong format for image (not gray scale or 3 channel)")
        exit(1)
    img_useable_mask = (_img != nodata_value).astype(np.uint8)
    # generate candidate query points
    if mode == 'random':
        kp1_x = np.random.rand(num_pts) * (w - 1)
        kp1_y = np.random.rand(num_pts) * (h - 1)
        coord = np.stack((kp1_x, kp1_y)).T

    elif mode == 'sift':        
        uint8_img = img_to_uInt8(img)
        sift = cv2.SIFT_create(nfeatures=num_pts)
        kp1 = sift.detect(uint8_img, mask=img_useable_mask)
        coord = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1])

    elif mode == 'mixed':
        "10% random points and 90% sift"
        kp1_x = np.random.rand(1 * int(pct_rand * num_pts)) * (w - 1)
        kp1_y = np.random.rand(1 * int(pct_rand * num_pts)) * (h - 1)
        kp1_rand = np.stack((kp1_x, kp1_y)).T

        sift = cv2.SIFT_create(nfeatures=int(pct_sift * num_pts))
        uint8_img = img_to_uInt8(img)
        kp1_sift = sift.detect(uint8_img, mask=img_useable_mask)
        kp1_sift = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1_sift])
        if len(kp1_sift) == 0:
            coord = kp1_rand
        else:
            coord = np.concatenate((kp1_rand, kp1_sift), 0) #[N, 2]

    else:
        raise Exception('unknown type of keypoints')
        exit()

    return coord