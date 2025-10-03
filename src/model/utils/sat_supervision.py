import torch
from einops import repeat
from kornia.utils import create_meshgrid

from satdepth.src.model.utils.sat_geometry import warp_kpts

# Coarse and fine supervision of LoFTR adapted for satellite images

@torch.no_grad()
def coarse_supervision(data, config):
    """
    Update:
        data (dict):{
            "conf_matrix_gt": [N, h0*w0, h1*w1],
            "spv_b_ids": [M],
            "spv_i_ids": [M],
            "spv_j_ids": [M],
            "spv_w_pt0_i": [N, L, 2],
            "spv_pt1_i": [N, L, 2]
        }
    """
    # 1. misc
    device = data["image0"].device
    N, _, H0, W0 = data["image0"].shape
    _, _, H1, W1 = data["image1"].shape
    scale = config["resolution"][0]

    scale0 = scale * data["scale0"][:, None] if "scale0" in data else scale
    scale1 = scale * data["scale1"][:, None] if "scale0" in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
    
    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # get warped image coordinates
    # naming : w{warped}_pt{#image num}_{i: image, c: coarse, f:fine}
    _ , w_pt0_i = warp_kpts(grid_pt0_i, data["lat0"], data["lon0"], data["ht0"],
                            data["lat1"], data["lon1"], data["ht1"],
                            data["affine_cam0"], data["affine_cam1"], distance_th=config["gt_matches_3d_distance_thr"]) # (N, L, 2)
    _, w_pt1_i = warp_kpts(grid_pt0_i, data["lat1"], data["lon1"], data["ht1"],
                           data["lat0"], data["lon0"], data["ht0"],
                           data["affine_cam1"], data["affine_cam0"], distance_th=config["gt_matches_3d_distance_thr"])
    # scale image corrdinates to coarse
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. check if mutual nearest neighbor
    # converting 2d idx to 1d idx using raster order
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1 # (N , L)
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0
    
    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    # checking for mutual neighbors
    """
    for mutual neighbors, we find the index0 for every index1 for every warp grid pts.
    if the index0 are same as arange then they are mutual neighbors.
    we are not doing any lat,lon,ht checks here 
    """
    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1) # (N, L)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0) # i_ids: index on img0
    j_ids = nearest_index1[b_ids, i_ids] # j_ids: index on img1

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({"conf_matrix_gt": conf_matrix_gt})
    
    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        # logger.warning(f"No groundtruth coarse match found for: {data["pair_names"]}")
        # this won"t affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        "spv_b_ids": b_ids,
        "spv_i_ids": i_ids,
        "spv_j_ids": j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        "spv_w_pt0_i": w_pt0_i,  # warped location of gt kpt in img1
        "spv_pt1_i": grid_pt1_i  # location of gridded pt in img1 (center)
    })
    return

@torch.no_grad()
def fine_supervision(data, config):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop("spv_w_pt0_i"), data.pop("spv_pt1_i")
    w_pt0_i, pt1_i = data["spv_w_pt0_i"], data["spv_pt1_i"]
    scale = config["resolution"][1]
    radius = config["fine_window_size"] // 2

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data["b_ids"], data["i_ids"], data["j_ids"]

    # 3. compute gt
    scale = scale * data["scale1"][b_ids] if "scale0" in data else scale
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later
    expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius  # [M, 2]
    data.update({"expec_f_gt": expec_f_gt})

