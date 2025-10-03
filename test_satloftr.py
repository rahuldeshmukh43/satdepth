import os, sys, re
import configargparse
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import torch
from skimage import exposure

from satdepth.external.LoFTR.src.loftr import LoFTR
from satdepth.external.LoFTR.src.utils.misc import lower_config

from satdepth.src.config.satloftr_default import get_cfg_defaults
from satdepth.src.datasets.satdepth import SatDepthLoader, _read_pairs
import satdepth.src.model.utils.sat_supervision as sat_supervision
import satdepth.src.utils.sat_metrics as sat_metrics
import satdepth.src.utils.satdepth_utils as satdepth_utils
from satdepth.src.utils.useful_methods import setup_logger, get_basename, timeit, pkl_read, pkl_write

NUM_MATCH_LINES = 40 #20 for plotting 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt='f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} current value- {val:' + self.fmt + '} (running mean- {avg:' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def make_dsm_gridpts(dsm_file:str, 
                     dsm_shrink_buffer:int, 
                     patch_size:int):
    """
    Make patch_size windows on the dsm in the central area with a single side buffer of dsm_shrink_buffer
    -----------------------
    |    ____________     |
    |   | x | x | x |     |
    |   -------------     |
    |   | x | x | x |     |
    |   -------------     |
    |   | x | x | x |     |
    |   -------------     |
    |_____________________|

    Args:
        dsm_file: path to dsm file
        dsm_shrink_buffer: buffer to shrink the dsm from all sides
        patch_size: size of the patch

    Return: 
        list of center coordinates (lat, lon , ht) of the grided points
    """
    dsm_cam = satdepth_utils.ReadDSM(dsm_file)
    dsm = dsm_cam.ReadImg()
    nrows, ncols = dsm.shape

    num_cells_x = (ncols - 2 * dsm_shrink_buffer) // patch_size
    num_cells_y = (nrows - 2 * dsm_shrink_buffer) // patch_size

    # get center coordinates and corresponding lat, lon , ht
    grid_pts = []
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            center_x = (patch_size) * i + (patch_size // 2) + dsm_shrink_buffer
            center_y = (patch_size) * j + (patch_size // 2) + dsm_shrink_buffer
            if dsm_cam.nodata_mask[int(center_y), int(center_x)]:
                # no data point
                continue
            lon, lat = dsm_cam.backproject(center_y, center_x)
            ht = dsm[center_y, center_x]
            grid_pts.append((lat, lon, ht))

    return grid_pts

@timeit
def extract_matches(model, 
                    model_config, 
                    test_pair, 
                    args, 
                    epi_thrs:list, 
                    logger,  
                    device='cpu'):
    """
    extract matches for a single image pair, using all patches
    """
    img0_ds, img1_ds, intersection_angle = test_pair
    nMatchesKeep = args.nMatchesKeep

    # compute grid on dsm
    grid_pts = make_dsm_gridpts(img0_ds.dsm_path,
                                args.dsm_shrink_buffer,
                                args.img_patch_size)
    logger.info("Num grid patches: %d"%(len(grid_pts)))

    # init dataloader for this pair
    img_dataloader = SatDepthLoader(args, 
                                    "test", 
                                    img_pair=test_pair,
                                    grid_pts=grid_pts).load_data()

    @timeit
    def model_pass(model, batch):
        with torch.no_grad():
            model(batch)
        return

    pts0 = [[], []]
    pts1 = [[], []]
    scores = []
    p = [[] for _ in range(len(epi_thrs))]
    r = [[] for _ in range(len(epi_thrs))]
    n_correct = [[] for _ in range(len(epi_thrs))]
    n_gt_matches = [[] for _ in range(len(epi_thrs))]
    n_detected = [[] for _ in range(len(epi_thrs))]
    phi_errs = []
    theta_errs = []
    s_errs = []
    epi_errs = []
    for i_batch, data in enumerate(img_dataloader):
        if not isinstance(data, dict):
            print(
                "[Testing] recieved empty batch, continue to next batch (batch: %d)" % (i_batch))
            continue
        logger.info("batch : %d"%(i_batch))

        # send stuff to gpu
        data["image0"] = data["image0"].to(device)
        data["image1"] = data["image1"].to(device)

        # lat lon ht
        data["lat0"] = data["lat0"].to(device)
        data["lon0"] = data["lon0"].to(device)
        data["ht0"] = data["ht0"].to(device)

        data["lat1"] = data["lat1"].to(device)
        data["lon1"] = data["lon1"].to(device)
        data["ht1"] = data["ht1"].to(device)

        # affine camera
        data["affine_cam0"] = data["affine_cam0"].to(device)
        data["affine_cam1"] = data["affine_cam1"].to(device)

        # Fundamental matrix
        data["F_gt"] = data["F_gt"].to(device)

        # get matches for each patch
        model_pass(model, data)

        # do supervision, compute metrics, compute precision recall
        sat_supervision.coarse_supervision(data, model_config["loftr"])
        # sat_supervision.fine_supervision(data, model_config["loftr"])
        sat_metrics.compute_distance_errors(data)
        sat_metrics.compute_pose_errors(data)
        
        # shift matches using x_off, y_off
        for b_id in range(data["image0"].shape[0]):
            b_mask = data['m_bids'] == b_id
            if b_mask.shape[0] == 0: 
                print("No matches found for batch %d ... moving to next "%(b_id))
                continue

            x0_off = data["img0_extents"][b_id][0].numpy()
            y0_off = data["img0_extents"][b_id][1].numpy()
            x1_off = data["img1_extents"][b_id][0].numpy()
            y1_off = data["img1_extents"][b_id][1].numpy()

            mconf_coarse = data['mconf'][b_mask].cpu().numpy()
            sorted_score_idx = np.argsort(mconf_coarse)[::-1][:nMatchesKeep]
            kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()[sorted_score_idx,:]
            kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()[sorted_score_idx,:]
            mscore = mconf_coarse[sorted_score_idx]

            if kpts0.shape[0] == 0: continue
            pts0[0].extend(list(kpts0[:, 0] + x0_off))
            pts0[1].extend(list(kpts0[:, 1] + y0_off))
            pts1[0].extend(list(kpts1[:, 0] + x1_off))
            pts1[1].extend(list(kpts1[:, 1] + y1_off))
            scores.extend(list(mscore))

        for i_thr, thr in enumerate(epi_thrs):
            (_precision, 
            _recall,
            _count,
            _n_correct,
            _n_gt_matches,
            _n_detected, 
            _epi_errs) = sat_metrics.compute_scores(data,
                                                    epipolar_thr=thr,
                                                    nMatchesKeep=nMatchesKeep)
            p[i_thr].extend(_precision)
            r[i_thr].extend(_recall)
            n_correct[i_thr].extend(_n_correct)
            n_gt_matches[i_thr].extend(_n_gt_matches)
            n_detected[i_thr].extend(_n_detected)
            if i_thr == 0:
                epi_errs.extend(_epi_errs)
        
        #compute and get pose errors        
        _phi_errs, _theta_errs, _s_errs = data["phi_errs"], data["theta_errs"], data["s_errs"]
        phi_errs.extend(_phi_errs)
        theta_errs.extend(_theta_errs)
        s_errs.extend(_s_errs)

    # accumulate matches and match scores for the image pair
    pts0 = np.array(pts0).T #[N,2]
    pts1 = np.array(pts1).T
    matches = np.hstack((pts0, pts1)) #[N,4] # x0 y0 x1 y1
    match_scores = np.array(scores) #[N]
    
    return (matches,
            match_scores, 
            p, 
            r, 
            n_correct, 
            n_gt_matches, 
            n_detected, 
            phi_errs, 
            theta_errs, 
            s_errs, 
            epi_errs)

@timeit
def plot_matches(test_pair, 
                 matches, 
                 filename, 
                 plt_str=None, 
                 epi_errs=None, 
                 epi_thrs=1.0, 
                 plot_kp=False):
    WHITE_SEPARATION=20
    num_matches, _ = matches.shape
    if num_matches == 0:
        print("No matches found for %s ... moving to next "%(filename))
        return
    if epi_errs != None:
        if len(epi_errs) != num_matches:
            print("Length of epi errors (%d) is not the same as number of matches (%d)"%(len(epi_errs), num_matches))
            return
        mask = np.array(epi_errs) < epi_thrs
    else:
        raise ValueError("epi_errs is None")
    img0_ds, img1_ds, intersection_angle = test_pair
    filename = "%s.pdf" % (filename)

    img0 = img0_ds.ReadImg(repeat=False)
    img1 = img1_ds.ReadImg(repeat=False)

    img0 = exposure.equalize_hist(img0)
    img1 = exposure.equalize_hist(img1)

    h0,w0 = img0.shape
    h1, w1 = img1.shape
    img = np.ones((max(h0,h1), w0+w1+WHITE_SEPARATION))
    img[:h0,:w0] = img0
    img[:h1, w0+WHITE_SEPARATION:] = img1

    num_tp = mask.sum()
    num_fp = num_matches - num_tp
    precision = (num_tp/num_matches) * 100
    textstr = "P: %0.2f  N: %d"%(precision, num_matches)

    # cmap = matplotlib.cm.get_cmap("jet")
    # colors = cmap(np.arange(num_matches)/num_matches)
    cmap_tp = matplotlib.cm.get_cmap("cool")
    cmap_fp = matplotlib.cm.get_cmap("Wistia")
    colors_tp = cmap_tp(np.arange(num_tp)/num_tp)
    colors_fp = cmap_fp(np.arange(num_fp)/num_fp)
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    s = 5
    marker_lw = 0.5
    lw = 0.5
    alpha=0.7
    # plot the points
    # marker = 'o'
    # plt.scatter(matches[:,0], matches[:,1], color=colors,s=s, marker=marker, alpha=0.5)
    # plt.scatter(matches[:, 2]+ w0 + WHITE_SEPARATION, matches[:, 3] , color=colors, s=s, marker=marker, alpha=0.5)
    if plot_kp:
        plt.scatter(matches[mask,0], matches[mask,1], color=colors_tp,s=s, marker='o', alpha=alpha)
        plt.scatter(matches[~mask,0], matches[~mask,1], color=colors_fp,s=s, marker='x', alpha=alpha, linewidths=marker_lw)
        plt.scatter(matches[mask, 2]+ w0 + WHITE_SEPARATION, matches[mask, 3] , color=colors_tp, s=s, marker='o', alpha=alpha)
        plt.scatter(matches[~mask, 2]+ w0 + WHITE_SEPARATION, matches[~mask, 3] , color=colors_fp, s=s, marker='x', alpha=alpha, linewidths=marker_lw)
    
    # plot random true matches using line
    # idx_lines = np.random.permutation(np.arange(num_matches))[:NUM_MATCH_LINES]
    idx_true_matches = np.where(mask)[0]
    idx_lines = np.random.permutation(idx_true_matches)[:NUM_MATCH_LINES]
    for i in idx_lines:
        plt.plot( (matches[i,0], matches[i,2]+w0+WHITE_SEPARATION), (matches[i,1], matches[i,3]), color="#08FF08" , linewidth=lw)
    
    # idx_false_matches = np.where(~mask)[0]
    # idx_lines = np.random.permutation(idx_false_matches)[:NUM_MATCH_LINES]
    # for i in idx_lines:
    #     plt.plot( (matches[i,0], matches[i,2]+w0+WHITE_SEPARATION), (matches[i,1], matches[i,3]), color='r' , linewidth=lw)

    # put text box and put precision, Num matches
    # text box on top left corner
    props = dict(boxstyle='round', facecolor='red', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=14, fontweight='bold', color='yellow',
            verticalalignment='top', bbox=props)
    # put plt_str on bottom
    if plt_str:
        props = dict(boxstyle='round', facecolor='blue', alpha=0.5)
        plt.text(0.05, 0.05, plt_str, transform=plt.gca().transAxes,
                 fontsize=10, fontweight='bold', color='yellow',
                 verticalalignment='bottom', bbox=props)

    plt.savefig(filename,dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close("all")
    return

def run(args, 
        model_config, 
        epi_thrs,
        int_angle_bin_width:float, 
        num_int_angle_bins:int,
        rel_track_angle_bin_width:float, 
        num_rel_track_angle_bins:int, 
        out_folder:str):
    # tensorboard writer
    # tb_log_dir = os.path.join(args.outdir, 'tb_logger_testing')
    # print('tensorboard log files are stored in {}'.format(tb_log_dir))
    # tb_writer = SummaryWriter(tb_log_dir)

    # logger
    log_file = os.path.join(out_folder, args.log_name + '.log')
    logger = setup_logger(args.log_name, log_file)

    # define model
    model = LoFTR(config=model_config['loftr'])
    logger.info("Loading checkpoint from %s"%(args.ckpt_path))
    print("Loading checkpoint from %s"%(args.ckpt_path))

    state_dict = torch.load(args.ckpt_path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=True)

    device = "cuda:%d" % (args.device) if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # precision and recall meters
    precision_meter = AverageMeter('Average Precision (1pxl)', '0.6f')
    recall_meter = AverageMeter('Average Recall (1pxl)', '0.6f')

    # read pair list file
    test_pairs = _read_pairs(args.test_pairlist)
    Npairs = len(test_pairs)
    p_image = []
    r_image = []
    int_angle_p = [[] for _ in range(num_int_angle_bins)]
    int_angle_r = [[] for _ in range(num_int_angle_bins)]
    rel_track_angle_p = [[] for _ in range(num_rel_track_angle_bins)]
    rel_track_angle_r = [[] for _ in range(num_rel_track_angle_bins)]
    p_patches = [[] for _ in range(len(epi_thrs))]
    r_patches = [[] for _ in range(len(epi_thrs))]
    n_correct_patches = [[] for _ in range(len(epi_thrs))]
    n_gt_matches_patches = [[] for _ in range(len(epi_thrs))]
    n_detected_patches = [[] for _ in range(len(epi_thrs))]
    phi_errs_patches = []
    theta_errs_patches = []
    s_errs_patches = []
    int_angle_patches = []
    rel_track_angle_patches = []
    for i_pair,test_pair in enumerate(test_pairs):
        aoi_name = re.search("aoi_rect_piece_[0-9]*", test_pair[0].img_path)
        intersection_angle = test_pair[2]
        # relative_track_angle = satdepth_utils.compute_relative_track_angle(test_pair[0], test_pair[1])
        relative_track_angle = test_pair[3]

        int_bin_id = int(intersection_angle // int_angle_bin_width)
        rel_track_bin_id = int(relative_track_angle // rel_track_angle_bin_width)

        if aoi_name:
            aoi_name = aoi_name.group(0)

        img0_base = get_basename(test_pair[0].img_path)
        img1_base = get_basename(test_pair[1].img_path)

        filename = "%s_%s_and_%s_matches"%(aoi_name, img0_base, img1_base)
        filename = os.path.join(out_folder, filename)
        logger.info("---------------------------------------")
        logger.info("Processing testing img pair: %d/%d"%(i_pair+1, Npairs))
        logger.info(filename)

        # form the string for matching plot
        img0_sn = satdepth_utils.image_name_to_short_name(img0_base)
        img1_sn = satdepth_utils.image_name_to_short_name(img1_base)
        time_diff = satdepth_utils.time_difference(img0_sn, img1_sn)
        plot_str = r"(%s, %s) $\Delta t$ %d, $\alpha^{v}$ %0.2f, $\alpha^{t}$ %0.2f"%(img0_sn, 
                                                                                      img1_sn, 
                                                                                      time_diff, 
                                                                                      intersection_angle, 
                                                                                      relative_track_angle)

        if os.path.exists(filename+'.pkl'):
            # this pair was processed
            # read stuff and move to next
            logger.info("Reading Processed testing img pair: %d/%d" % (i_pair + 1, Npairs))
            logger.info("filename: %s" % (filename))
            out = pkl_read(filename +".pkl")
            matches = out["matches"]
            match_scores = out["match_scores"]
            this_image_precision = out["this_image_precision"]
            this_image_recall = out["this_image_recall"]
            p = out["p"]
            r = out["r"]
            _phi_errs = out["phi_errs"]
            _theta_errs = out["theta_errs"]
            _s_errs = out["s_errs"]
            n_correct = out["n_correct"] 
            n_gt_matches = out["n_gt_matches"] 
            n_detected = out["n_detected"] 
            this_image_n_correct = out["this_image_n_correct"] 
            this_image_n_gt_matches = out["this_image_n_gt_matches"] 
            this_image_n_detected = out["this_image_n_detected"] 
            if "epi_errs" in out:
                epi_errs = out["epi_errs"]
            else:
                epi_errs = None
            # store patch angles
            num_patches = len(p[0])
            _int_angle_patches = [intersection_angle]*num_patches
            _rel_track_angle_patches = [relative_track_angle]*num_patches

            # plot
            if args.do_match_plot and (not os.path.exists(filename+'.pdf')):
                plot_matches(test_pair, matches, filename, plt_str=plot_str, epi_errs=epi_errs)
        else:
            # process this pair
            logger.info("---------------------------------------")
            logger.info("Processing testing img pair: %d/%d"%(i_pair+1, Npairs))
            logger.info("filename: %s"%(filename))

            (matches,
            match_scores, 
            p,
            r,
            n_correct,
            n_gt_matches,
            n_detected,
            _phi_errs,
            _theta_errs,
            _s_errs, 
            epi_errs) = extract_matches(model, model_config, test_pair, args, epi_thrs, logger, device=device)

            num_patches = len(p[0])
            _int_angle_patches = [intersection_angle]*num_patches
            _rel_track_angle_patches = [relative_track_angle]*num_patches

            # summarize precision, recall for this image
            this_image_n_correct = np.array(n_correct).sum(axis=1)
            this_image_n_gt_matches = np.array(n_gt_matches).sum(axis=1)
            this_image_n_detected = np.array(n_detected).sum(axis=1) + 1e-11
            this_image_precision = this_image_n_correct/this_image_n_detected
            this_image_recall = this_image_n_correct/this_image_n_gt_matches

            # write matches
            pkl_write(filename + ".pkl", {"matches": matches,
                                        "match_scores": match_scores,
                                        "p": p,
                                        "r": r,
                                        "phi_errs": _phi_errs,
                                        "theta_errs": _theta_errs,
                                        "s_errs": _s_errs,
                                        "n_correct": n_correct,
                                        "n_gt_matches": n_gt_matches,
                                        "n_detected": n_detected,
                                        "this_image_precision": this_image_precision,
                                        "this_image_recall": this_image_recall,
                                        "this_image_n_correct": this_image_n_correct,
                                        "this_image_n_gt_matches": this_image_n_gt_matches,
                                        "this_image_n_detected": this_image_n_detected,
                                        "epi_errs":epi_errs,
                                        "int_angle_patches": _int_angle_patches,
                                        "rel_track_angle_patches": _rel_track_angle_patches
                                        })

            # plot
            if args.do_match_plot:
                plot_matches(test_pair, matches, filename, plt_str=plot_str,  epi_errs=epi_errs)

        if matches.shape[0]==0:
            print_txt = "No matches found for %s ... moving to next "%(filename)
            logger.info(print_txt)
            print(print_txt)
            continue

        phi_errs_patches.extend(_phi_errs)
        theta_errs_patches.extend(_theta_errs)
        s_errs_patches.extend(_s_errs)
        int_angle_patches.extend(_int_angle_patches)
        rel_track_angle_patches.extend(_rel_track_angle_patches)

        # store for patches
        for i in range(len(epi_thrs)):
            p_patches[i].extend(p[i])
            r_patches[i].extend(r[i])
            n_correct_patches[i].extend(n_correct[i])
            n_gt_matches_patches[i].extend(n_gt_matches[i])
            n_detected_patches[i].extend(n_detected[i])

        #store summarized precision recall
        p_image.append(this_image_precision)
        r_image.append(this_image_recall)

        precision_meter.update(this_image_precision.item(0))
        recall_meter.update(this_image_recall.item(0))

        # print running means
        print("Running means: %d/%d" % (i_pair + 1, Npairs))
        print(precision_meter)
        print(recall_meter)

        # put precision and recall in correct intersection angle bin
        int_angle_p[int_bin_id].append(this_image_precision)
        int_angle_r[int_bin_id].append(this_image_recall)
        rel_track_angle_p[rel_track_bin_id].append(this_image_precision)
        rel_track_angle_r[rel_track_bin_id].append(this_image_recall)

    return (p_image, 
            r_image, 
            int_angle_p, 
            int_angle_r, 
            rel_track_angle_p, 
            rel_track_angle_r, 
            p_patches, 
            r_patches, 
            n_correct_patches, 
            n_gt_matches_patches, 
            n_detected_patches, 
            phi_errs_patches, 
            theta_errs_patches, 
            s_errs_patches, 
            int_angle_patches, 
            rel_track_angle_patches)

if __name__ == "__main__":
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument("--config", is_config_file=True, help="config file path")

    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="specific checkpoint path to load the model from, "
                             "if not specified, automatically reload from most recent checkpoints")

    parser.add_argument("--logdir", type=str, default="./logs/", help="dir of tensorboard logs")
    parser.add_argument("--outdir", type=str, default="./out/", help="dir of output e.g., ckpts")
    parser.add_argument("--log_name", type=str, default="testing", help="base file name for logging")
    parser.add_argument("--testing_foldername", type=str, default="testing_set", help="folder name for testing set")

    # sat related
    parser.add_argument("--test_pairlist", type=str, help="path to pair list file for testing")

    # GENERAL OPTIONS: experiment name, mode
    parser.add_argument("--phase", type=str, default="test", help="train/val/test/benchmark/benchmark_patches")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id for single gpu training")
    parser.add_argument("--multi_gpu", action="store_true", help="flag for muti-gpu training")
    parser.add_argument("--force_cpu", action="store_true", help="flag for forcing cpu during testing")
    parser.add_argument("--num_gpu", type=int, default=1, help="number of gpus used for muli gpu training training")

    parser.add_argument("--img_patch_size", type=int, default=400, help="patch size extracted from original image")
    parser.add_argument("--train_img_size", type=int, default=400, help="size of patch used for training the network")
    parser.add_argument("--nodata_value", type=float, default=-9999, help="no data value for lat/lon/ht maps")
    parser.add_argument("--dsm_shrink_buffer", type=int, default=250, help="single side dsm shrinking buffer")
    parser.add_argument("--rot_aug", action="store_true", help="flag for rotation augmentation during training")
    parser.add_argument("--funda_method", type=str, default="cameras",
                        help="cameras/matches for calculating affine fundamental matrix")

    parser.add_argument("--num_pts", type=int, default=800, help="num of points to be extracted in each pair")
    parser.add_argument("--num_pts_retained", type=int, default=50, help="num of points to be retained for computing fundamental matrix")
    parser.add_argument("--kp_mode", type=str, default="mixed", help="sift/random/mixed")
    parser.add_argument("--pct_sift", type=float, default=0.9, help="percentage of sift points when mode is mixed")
    parser.add_argument("--kp_distance_thresh", type=float, default=0.25, help="3d distance threshold in meters to ascertain if a true match")

    # DATALOADER OPTIONS
    # parser.add_argument("--workers", type=int, help="number of data loading workers", default=4)

    # matcher options
    parser.add_argument("--match_thr", type=float,default=0.5, help="model coarse matching threshold")
    parser.add_argument("--nMatchesKeep", type=int, default=-1, help="number of matches to keep per data sample (ie per patch size). Top N score matches will be retained")

    parser.add_argument("--debug", action="store_true", help="flag for debugging code: only one image will be processed")

    parser.add_argument("--do_match_plot", action="store_true",
                        help="flag for doing match plot")

    args = parser.parse_args()

    # check args
    model_config = get_cfg_defaults()
    model_config.merge_from_file(args.main_cfg_path)
    model_config.merge_from_file(args.data_cfg_path)
    model_config = lower_config(model_config)
    model_config["loftr"]["match_coarse"]["thr"] = args.match_thr

    out_folder = os.path.join(args.outdir, args.testing_foldername)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # print args into the out_folder
    args_file = os.path.join(out_folder, "args.txt")
    with open(args_file, "w") as f:
        f.write(str(args))
    f.close()

    # run settings
    epi_thrs = np.linspace(1, 5, 5)
    int_angle_bin_width = 15
    int_angles = np.arange(0, 90, int_angle_bin_width)
    num_int_angle_bins = len(int_angles)
    rel_track_angle_bin_width = 60
    rel_track_angles = np.arange(0, 180, rel_track_angle_bin_width)
    num_rel_track_angle_bins = len(rel_track_angles)

    # run
    (p,
    r, 
    int_angle_p,
    int_angle_r, 
    rel_track_angle_p,
    rel_track_angle_r,
    p_patches,
    r_patches,
    n_correct_patches,
    n_gt_matches_patches,
    n_detected_patches,
    phi_errs_patches,
    theta_errs_patches,
    s_errs_patches,
    int_angle_patches,
    rel_track_angle_patches) = run(args, 
                            model_config,
                            epi_thrs,
                            int_angle_bin_width,
                            num_int_angle_bins,
                            rel_track_angle_bin_width,
                            num_rel_track_angle_bins,
                            out_folder)    

    summary_filename = os.path.join(out_folder, "summary_v2.pkl")
    summary_data = {
        "p": p,
        "r": r,
        "int_angle_p": int_angle_p,
        "int_angle_r": int_angle_r,
        "int_angles": int_angles,
        "rel_track_angle_p": rel_track_angle_p,
        "rel_track_angle_r": rel_track_angle_r,
        "rel_track_angles": rel_track_angles,
        "p_patches": p_patches,
        "r_patches": r_patches,
        "n_correct_patches": n_correct_patches,
        "n_gt_matches_patches": n_gt_matches_patches,
        "n_detected_patches": n_detected_patches,
        "phi_errs_patches": phi_errs_patches,
        "theta_errs_patches": theta_errs_patches,
        "s_errs_patches": s_errs_patches,
        "int_angle_patches": int_angle_patches,
        "rel_track_angle_patches": rel_track_angle_patches
    }

    pkl_write(summary_filename, summary_data)
    # print average precision and recall on entire testing dataset
    print("Testing dataset metrics (image wise):")
    print("For epi thrs (pxl): ", epi_thrs)
    print("Average Precision : ", np.mean(p, axis=0))
    print("Average Recall : ", np.mean(r, axis=0))

    # compute Pose AUC and av matching precision over all patches for the dataset
    # filter out nan
    phi_errs_patches =  np.array(phi_errs_patches)
    phi_errs_patches = phi_errs_patches[~np.isnan(phi_errs_patches)]
    theta_errs_patches =  np.array(theta_errs_patches)
    theta_errs_patches = theta_errs_patches[~np.isnan(theta_errs_patches)]
    # max_angular_error = [ max(_phi, _theta) for _phi, _theta in zip(phi_errs_patches, theta_errs_patches)]
    max_angular_error = np.maximum(phi_errs_patches, theta_errs_patches)
    pose_auc = sat_metrics.error_auc(max_angular_error, [5, 10, 20])
    print("Pose AUC")
    print(pose_auc)

    print("Patch wise performace: ")
    print("For epi thrs (pxl): ", epi_thrs)
    print("Average Precision : ", np.mean(p_patches, axis=1))
    print("Average Recall : ", np.mean(r_patches, axis=1))
    print("Average Num correctly detected matches : ", np.mean(n_correct_patches, axis=1))
    print("Average Num detected matches (nMatchesKeep %d) : "%(args.nMatchesKeep), np.mean(n_detected_patches, axis=1))