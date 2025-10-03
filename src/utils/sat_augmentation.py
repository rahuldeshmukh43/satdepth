import numpy as np
from typing import Tuple, Optional

from satdepth.src.utils.SatCV import warp_img_using_homography
import satdepth.src.utils.satdepth_utils as satdepth_utils

########################################################################
"""
Rotate Patch augmentation
"""
def get_rotation2d_mat(theta: float) -> np.array:
    """
    return 2D rotation matrix for angle theta in counter clockwise direction
    Args:
        theta (float) angle of rotation in radians
    Returns:
        R (np.array) 2x2 rotation matrix
    """
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R

def get_rotated_img_bbox(h:int, 
                        w:int,
                        theta:float,
                        xc:float,
                        yc:float) -> Tuple[int, int, int, int, float, float, np.array]:
    """
    Args:
        h (int) height of image
        w (int) width of image
        theta (float) angle of rotation in radians
        xc (float) x coordinate of center of rotation
        yc (float) y coordinate of center of rotation
    Returns:
        x_min_bbox (int) min x coordinate of bbox enclosing rotated patch on original image
        x_max_bbox (int) max x coordinate of bbox enclosing rotated patch on original image
        y_min_bbox (int) min y coordinate of bbox enclosing rotated patch on original image
        y_max_bbox (int) max y coordinate of bbox enclosing rotated patch on original image
        xc_new (float) x coordinate of center of patch in bbox frame
        yc_new (float) y coordinate of center of patch in bbox frame
        R (np.array) 2x2 rotation matrix
    """
    corners = np.array([[0., 0.],    # (x, y)
                        [w, 0.],
                        [0., h],
                        [w, h]]).T # [2, 4]

    R = get_rotation2d_mat(theta)
    H_Rot = np.zeros((3,3))
    H_Rot[:2,:2] = R
    H_Rot[-1,-1] = 1.0

    # shift origin to h/w, w/2
    corners[0, :] -= w / 2
    corners[1, :] -= h / 2
    corners_hc = np.vstack((corners, np.ones(4))) # [3, 4]
    rot_corners = H_Rot @ corners_hc
    rot_corners = rot_corners / rot_corners[-1,:]
    rot_xy = rot_corners[:-1,:] #[2, 4] (x,y)
    rot_xy[0, :] += xc # shift origin to xc, yc
    rot_xy[1, :] += yc

    # bbox coordinates
    x_min_bbox = np.min(rot_xy[0, :])
    x_max_bbox = np.max(rot_xy[0, :])
    y_min_bbox = np.min(rot_xy[1, :])
    y_max_bbox = np.max(rot_xy[1, :])

    x_min_bbox, x_max_bbox, y_min_bbox, y_max_bbox = list(map(lambda x: int(np.round(x)), [x_min_bbox, x_max_bbox, y_min_bbox, y_max_bbox]))

    # shift origin to bbox top left
    xc_new = xc - x_min_bbox
    yc_new = yc - y_min_bbox
    return x_min_bbox, x_max_bbox, y_min_bbox, y_max_bbox, xc_new, yc_new, R

def get_valid_patch(img_ds: np.array, 
                    lat_center: float,
                    lon_center: float,
                    ht_center: float,
                    theta: float,
                    patch_h: int,
                    patch_w: int) -> Optional[Tuple[int, int, int, int, float, float]]:
    """
    Args:
        img_ds (np.array) original image data sample
        lat_center (float) latitude of patch center
        lon_center (float) longitude of patch center
        ht_center (float) height of patch center
        theta (float) angle of rotation in radians
        patch_h (int) patch height
        patch_w (int) patch width

    Returns:
        Tuple containing:
        - x_min_bbox (int) min x coordinate of bbox enclosing rotated patch on original image
        - x_max_bbox (int) max x coordinate of bbox enclosing rotated patch on original image
        - y_min_bbox (int) min y coordinate of bbox enclosing rotated patch on original image
        - y_max_bbox (int) max y coordinate of bbox enclosing rotated patch on original image
        - xc_new (float) x coordinate of center of patch in bbox frame
        - yc_new (float) y coordinate of center of patch in bbox frame
    """
    # patch_center_pixel
    img_rpc = img_ds.ReadRPC()
    yc, xc = img_rpc.rpc(lat_center, lon_center, ht_center)
    xc = int(xc)
    yc = int(yc)

    x_min_bbox, x_max_bbox, y_min_bbox, y_max_bbox, xc_new, yc_new, _R  = get_rotated_img_bbox(patch_h, patch_w, theta, xc, yc)

    # check if patch is within valid bounds
    img_nrows, img_ncols = img_ds.ReadRasterSize()

    if np.any(np.array([x_min_bbox, y_min_bbox]) < 0 ) or np.any(np.array([x_max_bbox, y_max_bbox]) > np.array([img_ncols, img_nrows])):
        print("patch out of bounds")
        return None
    else:
        return x_min_bbox, x_max_bbox, y_min_bbox, y_max_bbox, xc_new, yc_new

from typing import Optional, Tuple

def get_crop_and_rotate_patch(img_ds: np.array, 
                              lat_center: float, 
                              lon_center: float, 
                              ht_center: float, 
                              theta: float, 
                              patch_h: int, 
                              patch_w: int, 
                              x_min_bbox: int, 
                              x_max_bbox: int, 
                              y_min_bbox: int, 
                              y_max_bbox: int, 
                              xc_new: float, 
                              yc_new: float, 
                              repeat: bool = True) -> Optional[Tuple[np.array, np.array, np.array, np.array, np.array]]:
    """
    Args:
        img_ds (np.array) original image data sample
        lat_center (float) latitude of patch center
        lon_center (float) longitude of patch center
        ht_center (float) height of patch center
        theta (float) angle of rotation in radians
        patch_h (int) patch height
        patch_w (int) patch width
        x_min_bbox (int) min x coordinate of bbox enclosing rotated patch on original image
        x_max_bbox (int) max x coordinate of bbox enclosing rotated patch on original image
        y_min_bbox (int) min y coordinate of bbox enclosing rotated patch on original image
        y_max_bbox (int) max y coordinate of bbox enclosing rotated patch on original image
        xc_new (float) x coordinate of center of patch in bbox frame
        yc_new (float) y coordinate of center of patch in bbox frame
        repeat (bool) whether to repeat the image channels or not

    Returns:
        Tuple containing:
        - img (np.array) cropped and rotated image
        - img_lat (np.array) cropped and rotated latitude image
        - img_lon (np.array) cropped and rotated longitude image
        - img_ht (np.array) cropped and rotated height image
        - affine_cam_rot_patch (np.array) affine camera transformation matrix
    """
    img = img_ds.ReadImg(x_min_bbox, y_min_bbox, x_max_bbox - x_min_bbox, y_max_bbox - y_min_bbox, repeat=repeat)
    if not satdepth_utils.ImageIsUseable(img):
        return None
    img_lat, img_lon, img_ht = img_ds.ReadLatLonHt(x_min_bbox, y_min_bbox, x_max_bbox - x_min_bbox, y_max_bbox - y_min_bbox)

    # chip rpc to bbox
    img_rpc = img_ds.ReadRPC()
    img_rpc.translate_linesamp(-1 * y_min_bbox, -1 * x_min_bbox)

    affine_cam_bbox = satdepth_utils.GetRPCAffine(img_rpc, lat_center, lon_center, ht_center)#, x_min_bbox, y_min_bbox) # [3,4] matrix

    # get rotated images (-theta)
    ## compute size of rotated image
    w, h = x_max_bbox - x_min_bbox, y_max_bbox - y_min_bbox
    x_min_bbox2, x_max_bbox2, y_min_bbox2, y_max_bbox2, xc, yc, R = get_rotated_img_bbox(h, w, -theta, xc_new, yc_new)
    H = np.vstack((np.hstack((R, np.array([[xc, yc]]).T )), np.array([[0, 0, 1]]))) @ np.array([[1., 0., -xc_new], [0., 1., -yc_new], [0., 0., 1.]])

    img, _ = warp_img_using_homography(img, H, same_size=False)
    img_lat, _= warp_img_using_homography(img_lat, H, same_size=False)
    img_lon, _ = warp_img_using_homography(img_lon, H, same_size=False)
    img_ht, _ = warp_img_using_homography(img_ht, H, same_size=False)

    ## crop images (h,w) about the new center
    x_lb = int(np.floor(xc  - patch_w/2))
    x_ub = x_lb + patch_w
    y_lb = int(np.floor(yc - patch_h/2))
    y_ub = y_lb + patch_h
    if repeat:
        img = img[y_lb:y_ub, x_lb:x_ub, :]
    else:
        img = img[y_lb:y_ub, x_lb:x_ub]
    img_lat = img_lat[y_lb:y_ub, x_lb:x_ub]
    img_lon = img_lon[y_lb:y_ub, x_lb:x_ub]
    img_ht = img_ht[y_lb:y_ub, x_lb:x_ub]

    # compute new camera
    H_crop = np.array([[1.,0., -x_lb],[0., 1., -y_lb],[0., 0., 1.]])
    affine_cam_rot_patch = H_crop @ H @  affine_cam_bbox
    return img, img_lat, img_lon, img_ht, affine_cam_rot_patch