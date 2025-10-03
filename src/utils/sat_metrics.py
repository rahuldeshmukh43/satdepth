import numpy as np
import torch
from collections import OrderedDict
from loguru import logger

from kornia.geometry.conversions import convert_points_to_homogeneous

from satdepth.src.utils.SatCV import relative_pose_error, refine_affine_fundamental_matrix
from satdepth.src.utils.useful_methods import timeit

def _compute_conf_thresh(data):
    "this is threshold on epipolar distance / any distance metric to declare if a match is positive match"
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'satdepth':
        thr = 1.0 
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr

# @timeit
def compute_scores(data, epipolar_thr=1.0, nMatchesKeep=-1):
    p = []
    r = []
    count = 0
    n_correct = []
    n_gt_matches = []
    n_detected = []
    epi_errs = []
    for b_id in range(data["image0"].size(0)):
        (precision, 
         recall, 
         _n_correct, 
         _n_gt_matches, 
         _n_detected, 
         _epi_errs) = compute_precision_recall(data, 
                                               b_id, 
                                               epi_err_thr=epipolar_thr, 
                                               nMatchesKeep=nMatchesKeep)
        p.append(precision)
        r.append(recall)
        n_correct.append(_n_correct)
        n_gt_matches.append(_n_gt_matches)
        n_detected.append(_n_detected)
        epi_errs.extend(_epi_errs)
        count += 1
    return p, r, count, n_correct, n_gt_matches, n_detected, epi_errs

def compute_precision_recall(data, b_id, epi_err_thr=1.0, nMatchesKeep=-1):
    # conf_thr = _compute_conf_thresh(data) epi_err_thr
    b_mask = data['m_bids'] == b_id
    epi_errs = data['epi_errs'][b_mask].clone().detach().cpu().numpy()
    data['mconf'] = data['mconf'].clone().detach()

    if nMatchesKeep != -1:
        # filter top score matches for give b_id
        mconf_coarse = data['mconf'][b_mask].cpu().numpy()
        sorted_score_idx = np.argsort(mconf_coarse)[::-1][:nMatchesKeep]
        epi_errs = epi_errs[sorted_score_idx]

    correct_mask = epi_errs < epi_err_thr
    n_detected = len(correct_mask)
    precision = np.mean(correct_mask) if n_detected > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    recall = 0.0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.
    return precision, recall, n_correct, n_gt_matches, n_detected, epi_errs

def distance_errors(pts0, pts1, F_gt):
    """
    Function calculates two types of distance errors for N number of correspondences (x <-> x').
        (1) symmertic epipolar distnace: the distance of a point from its projected epipolar line, computed in each of the images from Hartley & Zisserman Eq 11.10 pg 287-288
        (2) sampsons distance: sampsons distance for the hyperplane x'Fx from Hartley & Zisserman Eq 11.9 pg 287 
    We use the ground truth F_gt to compute the distance errors. No RANSAC is used here.
    Args:
        pts0: torch.Tensor [N,2]
        pts1: torch.Tensor [N, 2]
        F_gt: torch.Tensor [3,3]
    Returns:
        d_symm: torch.tensor [N] symmetric epipolar distance
        d_samp: torch.tensor [N] sampsons distance
    """
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Fp0 = pts0 @ F_gt.T  #[N,3]
    p1Fp0 = torch.sum(pts1*Fp0, -1) #[N]
    Ftp1 = pts1 @ F_gt #[N,3]

    d_symm = p1Fp0**2 * (1.0/(Fp0[:,0]**2 + Fp0[:,1]**2) + 1.0/(Ftp1[:, 0]**2 + Ftp1[:, 1]**2) ) # [N]
    d_samp = p1Fp0**2 * (1.0 / (Fp0[:, 0] ** 2 + Fp0[:, 1] ** 2 + Ftp1[:, 0] ** 2 + Ftp1[:, 1] ** 2))  # [N]
    return d_symm, d_samp

def compute_distance_errors(data):
    """
    Update:
        data (dict):{"epi_errs": [M]}
    """
    F_gt = data["F_gt"]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']

    epi_errs = []
    sampson_errs = []
    for bs in range(F_gt.size(0)):
        mask = m_bids == bs
        epi_err, sampson_err = distance_errors(pts0[mask], pts1[mask], F_gt[bs])
        epi_errs.append(epi_err)
        sampson_errs.append(sampson_err)
    epi_errs = torch.cat(epi_errs, dim=0)
    sampson_errs = torch.cat(sampson_errs, dim=0)

    data.update({'epi_errs': epi_errs,
                 'sampson_errs': sampson_errs})

# @timeit
def compute_pose_errors(data, config=None, nMatchesKeep=-1):
    """
    Compute relative pose errors for each smaple in the batch
    """
    # get matches/ keypoints
    data.update({'phi_errs': [], 'theta_errs': [],'s_errs':[], 'inliers': []})
    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()

    # for each sample compute the relative error and write back to data
    BatchSize = len(data["dataset_name"])
    for bs in range(BatchSize):
        mask = m_bids == bs
        # make matches [N,4] for this batch
        mconf_coarse = data["mconf"][mask].cpu().numpy()
        sorted_score_idx = np.argsort(mconf_coarse)[::-1][:nMatchesKeep]
        matches = np.stack((pts0[mask][sorted_score_idx,0],
                            pts0[mask][sorted_score_idx,1],
                            pts1[mask][sorted_score_idx,0],
                            pts1[mask][sorted_score_idx,1])).T
        # print(matches.shape)
        #compute F
        ret = refine_affine_fundamental_matrix(matches, config=config) # pass ransac params here
        if ret is None:
            default = 9999
            data["phi_errs"].append(default)
            data["theta_errs"].append(default)
            data["s_errs"].append(default)
            # data["inliers"].append(np.array([]).astype(bool))
        else:
            F, inliers = ret
            F_gt = data["F_gt"][bs].cpu().numpy()
            phi_err, theta_err, s_err = relative_pose_error(F, F_gt)
            data["phi_errs"].append(phi_err)
            data["theta_errs"].append(theta_err)
            data["s_errs"].append(s_err)
            # print(F, F_gt)
            # print(phi_err, theta_err, s_err)
            # data["inliers"].append(inliers)
    return

def error_auc(errors, thresholds, auc_str='auc'):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    # if auc_str is 'auc':
    #     thresholds = [5, 10, 20]
    # elif auc_str is 'scale_auc':
    #     thresholds = [1, 2, 5]
    # else:
    #     NotImplementedError('%s is not a valid string'%(auc_str))

    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'{auc_str}@{t}': auc for t, auc in zip(thresholds, aucs)}

def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """ 
    Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([metrics['phi_errs'], metrics['theta_errs']]), axis=0)[unq_ids]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    #scale errors:
    scale_thresholds = [1, 2, 5]
    scale_errors = np.array(metrics["s_errs"])[unq_ids]
    scale_aucs = error_auc(scale_errors, scale_thresholds, auc_str='scale_auc')

    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids],
                         dist_thresholds, True)  # (prec@err_thr)

    return {**aucs, **scale_aucs, **precs}

def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs




