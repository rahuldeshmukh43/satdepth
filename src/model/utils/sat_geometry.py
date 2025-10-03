import torch

from satdepth.src.model.utils.geospatial import compute_distance_lonlonh

@torch.no_grad()
def warp_kpts(kpts0, 
              lat0, 
              lon0, 
              ht0, 
              lat1, 
              lon1, 
              ht1, 
              affine_cam0, 
              affine_cam1,
              nodata_value=-9999, 
              distance_th=1.0):
    """
    Warp kpts0 from I0 to I1 with lat, lon, h
    Also check covisibility and lat, lon , ht consistency.
    world point  is consistent if distance < distance_th (hard-coded).

    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        lat0 (torch.Tensor): [N, H, W] lat for img0
        lon0 (torch.Tensor): [N, H, W] lon for img0
        ht0 (torch.Tensor): [N, H, W] ht for img0

        lat1 (torch.Tensor): [N, H, W] lat for img1
        lon1 (torch.Tensor): [N, H, W] lon for img1
        ht1 (torch.Tensor): [N, H, W] ht for img1

        affine_cam0 (torch.Tensor): [N, 3, 4] affine camera for img0
        affine_cam1 (torch.Tensor): [N, 3, 4] affine camera for img1
        nodata_value : -9999 default
        distance_th : threshold for true matching points (in meters)
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long()

    # Sample lat0, lon0, ht0, get calculable_mask on depth != 0
    kpts0_lat = torch.stack(
        [lat0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    kpts0_lon = torch.stack(
        [lon0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    kpts0_ht  = torch.stack(
        [ht0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    valid_mask0 = kpts0_lat != nodata_value

    # Project sampled lat0, lon0, ht0 to img1 using affine_cam1
    w_kpts0_h = torch.stack([kpts0_lon, kpts0_lat, kpts0_ht, torch.ones_like(kpts0_lat)], dim=-1) # (N, L, 4)
    kpts1 = (affine_cam1 @ w_kpts0_h.transpose(2, 1)).transpose(2,1) # (N, L, 3)
    # kpts1 = kpts1[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4) # (N, L, 2) for regular transformation we need to normalize hc
    kpts1 = kpts1[:, :, :2] # (N, L, 2) but for affine camera the third hc is always 1

    # covisible check
    h,w = lat1.shape[1:3]
    covisible_mask = (kpts1[:, :, 0] > 0) * (kpts1[:, :, 0] < w - 1) * \
                     (kpts1[:, :, 1] > 0) * (kpts1[:, :, 1] < h - 1)
    kpts1_long = kpts1.long()
    kpts1_long[~covisible_mask, :] = 0

    # Read true lat1, lon1, ht1 at projected points
    kpts1_lat = torch.stack(
        [lat1[i, kpts1_long[i, :, 1], kpts1_long[i, :, 0]] for i in range(kpts1.shape[0])], dim=0
    )  # (N, L)
    kpts1_lon = torch.stack(
        [lon1[i, kpts1_long[i, :, 1], kpts1_long[i, :, 0]] for i in range(kpts1.shape[0])], dim=0
    )  # (N, L)
    kpts1_ht  = torch.stack(
        [ht1[i, kpts1_long[i, :, 1], kpts1_long[i, :, 0]] for i in range(kpts1.shape[0])], dim=0
    )  # (N, L)
    valid_mask1 = kpts1_lat != nodata_value

    # compute consistency using ecef distance between two sets of kpts lat, lon, ht
    # consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    distance = compute_distance_lonlonh(kpts0_lat, kpts0_lon, kpts0_ht,
                                        kpts1_lat, kpts1_lon, kpts1_ht)
    consistent_mask = distance < distance_th # meters
    valid_mask = valid_mask0 * valid_mask1 * covisible_mask * consistent_mask

    return valid_mask, kpts1