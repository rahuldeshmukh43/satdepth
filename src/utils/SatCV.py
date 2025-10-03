import numpy as np
import cv2

# Computing Fundamental Matrix using affine approx cameras for sampled patches
"""
We are given affine projective cameras (P1, P2) for two image patches. 
We also know the inlier matches using lat,lon,ht maps
We need to compute Fundamental matrix for the pair 

Affine camera Fundamental matrix is given by:
F = [[0 0 a]
     [0 0 b]
     [c d e]]  H&Z Eq14.1 Pg 345
"""

def relative_pose_error(F:np.array, 
                        F_gt:np.array):
    """
    Compute relative pose errors between F and F_gt using affine motion parameters
    Args:
        F: [3,3] affine fundamental matrix
        F_gt: [3,3] ground truth affine fundamental matrix
    Returns:
        phi_err: angle between axis of rotation
        theta_err: angle between cyclo rotation
        s_err: error in scale factor
    """
    # print(F, F_gt)
    phi, theta, s = affine_motion(F)
    phi_gt, theta_gt, s_gt = affine_motion(F_gt)

    # compute errors
    # angle between axis of rotation
    phi_err = np.rad2deg(np.linalg.norm(phi - phi_gt))
    # angle between cyclo rotation
    theta_err = np.rad2deg(np.linalg.norm(theta - theta_gt))
    # error in scale factor
    s_err = np.linalg.norm(s - s_gt)
    return phi_err, theta_err, s_err

def affine_motion(F : np.array):
    """
    Compute relative motion using affine fundamental matrix
    see H & Z eq 14.10 Pg 360
    Args:
        F (np.array) [3,3] affine fundamental matrix
    Returns:
        phi: axis of rotation
        theta: cyclo-rotation angle
        s: scale factor
    """
    eps = 1e-6
    a = F[0, 2]
    b = F[1, 2]
    c = F[2, 0]
    d = F[2, 1]
    e = F[2, 2]
    phi = np.arctan(b/(a + eps))
    theta = phi - np.arctan(d/(c + eps))
    s = np.sqrt((c**2 + d**2)/(a**2 + b**2 + eps))
    return phi, theta, s

def fundamental_matrix_cameras(P1 : np.array, P2 : np.array) -> np.array:
    """
    Cite: S2P
    Computes the fundamental matrix given the matrices of two cameras.

    Args:
        P1, P2: 2D arrays of size 3x4 containing the camera matrices

    Returns:
        the computed fundamental matrix, given by the formula 17.3 (p. 412) in
        Hartley & Zisserman book (2nd ed.).
        such that X2^T * F * X1 = 0 (this order is important)
    """
    X0 = P1[[1, 2], :]
    X1 = P1[[2, 0], :]
    X2 = P1[[0, 1], :]
    Y0 = P2[[1, 2], :]
    Y1 = P2[[2, 0], :]
    Y2 = P2[[0, 1], :]

    F = np.zeros((3, 3))
    F[0, 0] = np.linalg.det(np.vstack([X0, Y0]))
    F[0, 1] = np.linalg.det(np.vstack([X1, Y0]))
    F[0, 2] = np.linalg.det(np.vstack([X2, Y0]))
    F[1, 0] = np.linalg.det(np.vstack([X0, Y1]))
    F[1, 1] = np.linalg.det(np.vstack([X1, Y1]))
    F[1, 2] = np.linalg.det(np.vstack([X2, Y1]))
    F[2, 0] = np.linalg.det(np.vstack([X0, Y2]))
    F[2, 1] = np.linalg.det(np.vstack([X1, Y2]))
    F[2, 2] = np.linalg.det(np.vstack([X2, Y2]))

    F /= F[2,2]
    return F

def affine_fundamental_matrix(matches : np.array) -> np.array:
    """
    Estimates the affine fundamental matrix given a set of point correspondences
    between two images.

    Args:
        matches: 2D array of size Nx4 containing a list of pairs of matching
            points. Each line is of the form x1, y1, x2, y2, where (x1, y1) is
            the point in the first view while (x2, y2) is the matching point in
            the second view.

    Returns:
        the estimated affine fundamental matrix, given by the Gold Standard
        algorithm, as described in Hartley & Zisserman book (see chap. 14 pg 351 Algo 14.1).
    """

    # revert the order of points to fit H&Z convention (see algo 14.1)
    X = matches[:, [2, 3, 0, 1]]

    # compute the centroid
    N = len(X)
    XX = np.sum(X, axis=0) / N

    # compute the Nx4 matrix A
    A = X - np.tile(XX, (N, 1))

    # the solution is obtained as the singular vector corresponding to the
    # smallest singular value of matrix A. See Hartley and Zissermann for
    # details.
    # It is the last line of matrix V (because np.linalg.svd returns V^T)
    U, S, V = np.linalg.svd(A)
    N = V[-1, :]

    # extract values and build F
    F = np.zeros((3, 3))
    F[0, 2] = N[0] # a
    F[1, 2] = N[1] # b
    F[2, 0] = N[2] # c
    F[2, 1] = N[3] # d
    F[2, 2] = -np.dot(N, XX) # e
    F /= F[2,2] + 1e-8
    return F

def epipolar_distance(F : np.array, matches : np.array) -> np.array:
    """
    Computes scalar sum of residual of N matches. where residual is (X0.T F X1)
    Args:
        F: (np.array) [3,3]
        matches: (np.array) [N,4] [(x1,y1,x2,y2), ... ] matching coords
    Returns: 
        epipolar distance (X0.T F X1) -- scalar sum of residual for all N pts
    """
    X0 = np.vstack((matches[:,0],
                    matches[:,1],
                    np.ones_like(matches[:,0])))
    X1 = np.vstack((matches[:, 2],
                    matches[:, 3],
                    np.ones_like(matches[:, 2])))

    epipolar_line = np.matmul(F, X0) # [3,N]
    epipolar_line = epipolar_line / np.clip(np.linalg.norm(epipolar_line[:2, :], axis=0, keepdims=True), 1e-8, None)
    epi_distance = X1 * epipolar_line # [3,N]
    return np.abs(np.sum(epi_distance, axis=0) )

def refine_affine_fundamental_matrix(matches : np.array, config=None) -> np.array:
    """
    Refine Fundamental Matrix using RANSAC on matches
    Args:
        matches: (np.array) 2D array of size Nx4 containing a list of pairs of matching
            points. Each line is of the form x1, y1, x2, y2, where (x1, y1) is
            the point in the first view while (x2, y2) is the matching point in
            the second view.
        config: (dict) containing ransac_max_iters, ransac_pixel_thr, ransac_rand_sample_size
    Returns:
        F: [3,3] affine fundamental matrix 
        inliers: (np.array) [N,4] 
    """
    # ----------------#
    # RANSAC params
    if config == None:
        NTrials = 1000
        epi_thresh = 0.5
        Npts_trial = 10
    else:
        NTrials = config["ransac_max_iters"]
        epi_thresh = config["ransac_pixel_thr"]
        Npts_trial = config["ransac_rand_sample_size"]
    #----------------#

    Nmatches, _ = matches.shape
    # print("Num matches detected: ",Nmatches)
    if Nmatches < 2*Npts_trial:
        return None

    nInliers_max = -1
    idx_inliers_max = None
    for iTrial in range(NTrials):
        # get random subset
        subset_idx = np.random.permutation(Nmatches)[:Npts_trial]
        subset_matches = matches[subset_idx, :]

        # estimate F using subset
        F = affine_fundamental_matrix(subset_matches)

        # compute epi distance on rest and count size of inlier set
        epi_distances = epipolar_distance(F, matches)
        idx_inliers = np.where(epi_distances < epi_thresh)[0]
        nInliers = idx_inliers.size

        # print("RANSAC trial %d/%d- nInliers: "%(iTrial+1,NTrials),nInliers)

        # store best inlier set
        if nInliers > nInliers_max:
            nInliers_max = nInliers
            idx_inliers_max = idx_inliers

    # re-compute F using best inlier set
    F = affine_fundamental_matrix(matches[idx_inliers_max,:])
    # print('refined F: ', F)
    return F, matches[idx_inliers_max, :]

# affine camera rectification
def rectifying_similarities_from_affine_fundamental_matrix(F, debug=False):
    """
    Computes two similarities from an affine fundamental matrix.

    Args:
        F: 3x3 numpy array representing the input fundamental matrix
        debug (optional, default is False): boolean flag to activate verbose mode.
    Returns:
        S, S': two similarities such that, when used to resample the two images
            related by the fundamental matrix, the resampled images are
            stereo-rectified.
    """
    # check that the input matrix is an affine fundamental matrix
    assert(np.shape(F) == (3, 3))
    assert(np.linalg.matrix_rank(F) == 2)
    np.testing.assert_allclose(F[:2, :2], np.zeros((2, 2)))

    # notations
    a = F[0, 2]
    b = F[1, 2]
    c = F[2, 0]
    d = F[2, 1]
    e = F[2, 2]

    # rotations
    r = np.sqrt(c*c + d*d)
    s = np.sqrt(a*a + b*b)
    R1 = 1 / r * np.array([[d, -c], [c, d]])
    R2 = 1 / s * np.array([[-b, a], [-a, -b]])

    # zoom and translation
    z = np.sqrt(r / s)
    t = 0.5 * e / np.sqrt(r * s)

    # if debug:
    #     theta_1 = get_angle_from_cos_and_sin(d, c)
    #     print("reference image:")
    #     print("\trotation: %f deg" % np.rad2deg(theta_1))
    #     print("\tzoom: %f" % z)
    #     print("\tvertical translation: %f" % t)
    #     print()
    #     theta_2 = get_angle_from_cos_and_sin(-b, -a)
    #     print("secondary image:")
    #     print("\trotation: %f deg" % np.rad2deg(theta_2))
    #     print("\tzoom: %f" % (1.0 / z))
    #     print("\tvertical translation: %f" % -t)

    # output similarities
    S1 = np.zeros((3, 3))
    S1[0:2, 0:2] = z * R1
    S1[1, 2] = t
    S1[2, 2] = 1

    S2 = np.zeros((3, 3))
    S2[0:2, 0:2] = 1 / z * R2
    S2[1, 2] = -t
    S2[2, 2] = 1

    return S1, S2

def warp_img_using_homography(src_img, H, same_size=False):
    """
    Warp image using homography
    Args:
        src_img: (np.array) input image
        H: (np.array) [3,3] transformation homography
        same_size: (bool) true when we output should be of the same size as src_img
    Returns:
        dst: (np.array) warped image
        rewarp_homography: (np.array) [3,3] homography to re-warp the image back
    """
    h,w = src_img.shape[:2]
    corners = np.array([[0., 0., 1.],
                        [ w, 0., 1.],
                        [0.,  h, 1.],
                        [ w,  h, 1.]])
    # print(corners, H)
    new_corners = np.matmul(corners, H.T).T
    new_corners /= new_corners[-1, :]
    new_corners = new_corners.T
    new_x_min = np.min(new_corners[:, 0])
    new_x_max = np.max(new_corners[:, 0])
    new_y_min = np.min(new_corners[:, 1])
    new_y_max = np.max(new_corners[:, 1])
    out_w = np.round( new_x_max - new_x_min).astype(int)
    out_h = np.round(new_y_max - new_y_min).astype(int)
    # print(out_w, out_h)
    if same_size:
        rescaling_homography = np.array([[w/out_w, 0., -(w/out_w)*new_x_min],
                                         [0., h/out_h, -(h/out_h)*new_y_min],
                                         [0., 0., 1.]])
        rewarp_homography = np.matmul(rescaling_homography, H)
        dst = cv2.warpPerspective(src_img, rewarp_homography, (w, h))
    else:
        dst = cv2.warpPerspective(src_img, H, (out_w, out_h))
        rewarp_homography = H
    return dst, rewarp_homography

def transform_pts_using_homography(x : np.array, y: np.array, H : np.array):
    """
    Transform points using homography
    X_out = H * X_in
    Args:
        x: (np.array) [N,] x-coordinates
        y: (np.array) [N,] y-coordinates
        H: (np.array) [3,3] homography
    Returns:
        x_out: (np.array) [N,] x-coordinates
        y_out: (np.array) [N,] y-coordinates
    """
    xy_hc = np.matmul(H, np.vstack((x, y, np.ones_like(x))))
    xy_hc /= xy_hc[-1,:]
    return xy_hc[0, :], xy_hc[1, :]