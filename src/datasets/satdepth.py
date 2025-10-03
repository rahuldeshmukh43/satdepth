import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms

import satdepth.src.utils.satdepth_utils as satdepth_utils
import satdepth.src.utils.sat_augmentation as sat_augmentation
import satdepth.src.utils.SatCV as SatCV


MAX_RANDOM_TRIES = 10
SEED = 2024

def worker_init_fn(worker_id:int):
    # Set seed for each worker based on the worker_id
    np.random.seed(SEED + worker_id)
    torch.manual_seed(SEED + worker_id)
    return

class SatDepthLoader():
    def __init__(self, 
                 args, 
                 phase:str, 
                 rotation_augmentation:bool=False, 
                 img_pair=None, 
                 grid_pts=None, 
                 use_seed:bool=False):
        # super(SatDepthLoader, self).__init__()
        self.args = args
        assert phase in ["train", "val", "test"]
        self.phase = phase        

        if use_seed:
            _worker_init_fn = worker_init_fn
        else:
            _worker_init_fn = None

        print('\n######### Init Dataloader ##############')
        if rotation_augmentation:
            print('Using rotation augmentation for dataloader with phase: %s'%(phase))
        else:
            print('No rotation augmentation for dataloader with phase: %s' % (phase))
        print('#######################\n')

        self.dataset = SatDepthDataset(args, 
                                    phase,
                                    rotation_augmentation=rotation_augmentation,
                                    img_pair=img_pair,
                                    grid_pts=grid_pts)

        if not args.multi_gpu:
            self.data_loader = DataLoader( self.dataset,
                                           batch_size=args.batch_size,
                                           pin_memory=False,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           collate_fn=self.my_collate,
                                           worker_init_fn=_worker_init_fn)
        else:
            self.data_loader = DataLoader(self.dataset,
                                          batch_size=args.batch_size,
                                          pin_memory=False,
                                          shuffle=False,
                                          num_workers=args.num_workers,
                                          collate_fn=self.my_collate,
                                          sampler=DistributedSampler(self.dataset))


    def my_collate(self, batch):
        """
        Puts each data field into a tensor with outer dimension batch size
        Args:
            batch (list) list of items (can be nones too)
        Returns: returns collated batch removing Nones
        """
        #to filter out Nones when we could not extract any key points with corresponding matches
        batch = list(filter(lambda b: b is not None, batch))
        if len(batch) == 0:
            batch = [-1.,]
        return torch.utils.data.dataloader.default_collate(batch)

    def load_data(self):
        return self.data_loader

    def name(self):
        return "SatDepth"

    def __len__(self):
        return len(self.dataset)

class SatDepthDataset(Dataset):
    def __init__(self,
                args,
                phase: str,
                rotation_augmentation: bool = False,
                img_pair=None,
                grid_pts=None):
        """
        Args:
            args (argparse) : arguments
            phase (str) : "train/val/test" we use appropriate pairlist file for each phase
            rotation_augmentation (bool) : whether to use rotation augmentation
            img_pair (list | None) : list of two DataSample objects
            grid_pts (list | None) : list of lat, lon, ht points for which to extract patches
        """
        # super(SatDepthDataset, self).__init__()
        self.args = args
        self.phase = phase
        self.img_pair = img_pair
        self.grid_pts = grid_pts
        self.rotation_augmentation = rotation_augmentation

        self.kp_mode = args.kp_mode
        self.pct_sift = args.pct_sift
        self.num_pts = args.num_pts
        self.num_pts_retained = args.num_pts_retained
        self.kp_distance_thresh = args.kp_distance_thresh
        self.patch_size = args.img_patch_size
        self.dsm_shrink_buffer = args.dsm_shrink_buffer
        self.dataset_nodata_value = args.nodata_value
        # self.rectified = args.rectified
        self.funda_method = args.funda_method

        self.train_img_size = args.train_img_size

        if phase == "train":
            self.pairs_file = args.train_pairlist
        elif phase == "val":
            self.pairs_file = args.val_pairlist
        elif phase == "test":
            self.pairs_file = None
            try:
                self.img0_ds, self.img1_ds, self.intersection_angle,  self.relative_track_angle = self.img_pair
            except:
                self.img0_ds, self.img1_ds, self.intersection_angle = self.img_pair
            _ = self.img0_ds.ReadImg()
            _ = self.img1_ds.ReadImg()
            _ = self.img0_ds.ReadLatLonHt()
            _ = self.img1_ds.ReadLatLonHt()

        # read pairs info
        if phase != "test":
            self.pairs = _read_pairs(self.pairs_file)

        # dataset normalization stats for backbone
        # mean and std dev for imagenet -- resnet was trained on this
        # imgnet_mean = [0.485, 0.456, 0.406]
        # imgnet_std = [0.229, 0.224, 0.225]
        imgnet_mean = [0.0]
        imgnet_std = [1.0]
        # define transforms to be used
        if self.phase == "train":
            self.transform = transforms.Compose([transforms.Resize((self.train_img_size, self.train_img_size), antialias=True),
                                                transforms.Normalize(imgnet_mean, imgnet_std)])
        elif self.phase == "test" or self.phase == "val":
            self.transform = transforms.Compose([transforms.Normalize(imgnet_mean, imgnet_std)])
        else:
            self.transform = transforms.Compose([transforms.Resize((self.train_img_size, self.train_img_size), antialias=True)])

    def __getitem__(self, item):
        if self.phase == "train" or self.phase == "val":
            out = None
            for i in range(MAX_RANDOM_TRIES):
                out = self._get_item_random(item)
                if out is not None:
                    return out
            return None

        if self.phase == "test":
            return self._get_item_single_image_patches(item, 
                                                       self.img0_ds,
                                                       self.img1_ds,
                                                       self.intersection_angle)


    def _get_item_random(self, item):
        img0_ds, img1_ds, intersection_angle, relative_track_angle = self.pairs[item]
        # relative_track_angle = satdepth_utils.compute_relative_track_angle(img0_ds, img1_ds, method="ccw") #degree
        relative_track_angle = np.deg2rad(relative_track_angle) # radians

        # compute random angle of rotation
        if self.rotation_augmentation:
            theta = np.deg2rad(np.random.rand(1) * 360)[0] # radians
        else:
            theta = None

        # load dsm and draw random bbox on it
        img0_rpc = img0_ds.ReadRPC()
        img1_rpc = img1_ds.ReadRPC()

        if self.phase == "train" or self.phase=="val":
            lat0_center, lon0_center, ht0_center = satdepth_utils.get_random_pt_on_dsm(img0_ds, self.patch_size, self.dsm_shrink_buffer)
            lat1_center, lon1_center, ht1_center = lat0_center, lon0_center, ht0_center

            psize0 = satdepth_utils.get_valid_patch(img0_ds, img0_rpc,
                                                self.patch_size,
                                                lat0_center, lon0_center, ht0_center)
            if self.rotation_augmentation:
                psize1 = sat_augmentation.get_valid_patch(img1_ds,
                                                    lat1_center, lon1_center, ht1_center,
                                                    theta,
                                                    self.patch_size, self.patch_size)
            else:
                psize1 = satdepth_utils.get_valid_patch(img1_ds, img1_rpc,
                                                    self.patch_size,
                                                    lat1_center, lon1_center, ht1_center)

            if psize0 is None or psize1 is None:
                # cant draw patches on image, because the random patch was outside image extents
                print("cant draw random patch: %s and %s\n"%(img0_ds.img_path, img1_ds.img_path))
                return None

            # get patch extents for the two images
            x0_off, y0_off, x0_size, y0_size = psize0

            if self.rotation_augmentation:
                x1_min_bbox, x1_max_bbox, y1_min_bbox, y1_max_bbox, xc1_new, yc1_new = psize1
                x1_off = x1_min_bbox
                y1_off = y1_min_bbox
            else:
                x1_off, y1_off, x1_size, y1_size = psize1

            # chip the rpc to patch size
            img0_rpc.translate_linesamp(-1 * y0_off, -1 * x0_off)
            img1_rpc.translate_linesamp(-1 * y1_off, -1 * x1_off)

            # read the patches: img and lat, lon, ht maps, affine camera
            img0 = img0_ds.ReadImg(x0_off, y0_off, x0_size, y0_size, repeat=False)
            if not satdepth_utils.ImageIsUseable(img0):
                return None
            img0_lat, img0_lon, img0_ht = img0_ds.ReadLatLonHt(x0_off, y0_off, x0_size, y0_size)
            img0_rpc_affine_mat_3x4 = satdepth_utils.GetRPCAffine(img0_rpc, lat0_center, lon0_center, ht0_center)

            if not self.rotation_augmentation:
                img1 = img1_ds.ReadImg(x1_off, y1_off, x1_size, y1_size, repeat=False)
                if not satdepth_utils.ImageIsUseable(img1):
                    return None
                img1_lat, img1_lon, img1_ht = img1_ds.ReadLatLonHt(x1_off, y1_off, x1_size, y1_size)
                img1_rpc_affine_mat_3x4 = satdepth_utils.GetRPCAffine(img1_rpc, lat1_center, lon1_center, ht1_center) 
                relative_track_angle = np.abs(relative_track_angle) # \in [0, \pi]
            else:
                rot_aug_out = sat_augmentation.get_crop_and_rotate_patch(img1_ds, 
                                                    lat1_center, 
                                                    lon1_center, 
                                                    ht1_center, 
                                                    theta, 
                                                    self.patch_size, 
                                                    self.patch_size, 
                                                    x1_min_bbox, 
                                                    x1_max_bbox, 
                                                    y1_min_bbox, 
                                                    y1_max_bbox, 
                                                    xc1_new, 
                                                    yc1_new, 
                                                    repeat=False)
                if rot_aug_out is None: return None
                img1, img1_lat, img1_lon, img1_ht, img1_rpc_affine_mat_3x4 = rot_aug_out
                
                y1_size, x1_size = img1.shape[:2]
                # update relative track angle
                relative_track_angle = np.abs(relative_track_angle + theta - np.pi)/2 # \in [0, \pi]

            # compute the affine Fundamental matrix using the two cameras P1, P2
            if self.funda_method == "cameras":
                F_gt = SatCV.fundamental_matrix_cameras(img0_rpc_affine_mat_3x4, img1_rpc_affine_mat_3x4)
            elif self.funda_method == "matches":
                x0, y0, x1, y1 = satdepth_utils.get_matches(self.kp_mode, self.pct_sift, self.num_pts,
                                                        self.kp_distance_thresh, self.dataset_nodata_value,
                                                        img0, img0_rpc_affine_mat_3x4, img0_lat, img0_lon, img0_ht,
                                                        img1, img1_rpc_affine_mat_3x4, img1_lat, img1_lon, img1_ht,
                                                        x0_size, y0_size,
                                                        x1_size, y1_size)
                if x0.size <= self.num_pts_retained or x1.size <= self.num_pts_retained:
                    # print("cant draw points: %s and %s\n"
                    #       "sizes: img0: %d  img1: %d\n" % (img0_ds.img_path, img1_ds.img_path,x0.size, x1.size))
                    return None

                # correspondences can be different for each pair. Handle this!
                x0 = x0[:self.num_pts_retained].astype(np.float32)
                y0 = y0[:self.num_pts_retained].astype(np.float32)
                x1 = x1[:self.num_pts_retained].astype(np.float32)
                y1 = y1[:self.num_pts_retained].astype(np.float32)
                # compute the Fundamental matrix using matches
                matches = np.vstack((x0, y0, x1, y1)).T
                F_gt = SatCV.affine_fundamental_matrix(matches) # no ransac
                # F_gt = SatCV.refine_affine_fundamental_matrix(matches) # ransac refinement didnt improve since we have all true correspondences
            else:
                NotImplementedError("Wrong Fundamental matrix estimation method: %s"%(self.funda_method))
                exit(1)
            F_gt = F_gt.astype(np.float32)
    
        # transform images, coordinates to tensor and reshape image
        img0 = torch.from_numpy(img0).unsqueeze(0)
        img1 = torch.from_numpy(img1).unsqueeze(0)
    
        img0 = self.transform(img0)
        img1 = self.transform(img1)

        # make dictionary for data
        data = {
            "image0": img0,
            "image1": img1,
            "affine_cam0": img0_rpc_affine_mat_3x4,
            "lat0": img0_lat,
            "lon0": img0_lon,
            "ht0": img0_ht,
            # "img0_extents": [x0_off, y0_off, x0_size, y0_size],
            "affine_cam1": img1_rpc_affine_mat_3x4,
            "lat1": img1_lat,
            "lon1": img1_lon,
            "ht1": img1_ht,
            # "img1_extents": [x1_off, y1_off, x1_size, y1_size],
            "F_gt": F_gt,
            "intersection_angle": intersection_angle,
            "dataset_name": "SatDepth",
            "pair_names": (img0_ds.img_path, img1_ds.img_path),
            "rotation_aug_angle": np.rad2deg(theta) if theta != None else -9999,
            "relative_track_angle": np.rad2deg(relative_track_angle),
        }
        return data

    def _get_item_single_image_patches(self, item, img0_ds, img1_ds, intersection_angle):
        patch_center_lat, patch_center_lon, patch_center_ht = self.grid_pts[item]
        # img0_ds, img1_ds, intersection_angle = self.img_pair

        img0_rpc = img0_ds.ReadRPC()
        img1_rpc = img1_ds.ReadRPC()

        psizes = satdepth_utils.get_corresponding_patch_sizes(img0_ds, 
                                                              img0_rpc, 
                                                              img1_ds, 
                                                              img1_rpc, 
                                                              self.patch_size, 
                                                              patch_center_lat, 
                                                              patch_center_lon, 
                                                              patch_center_ht)
        if psizes != None:
            # get patch extents for the two images
            x0_off, y0_off, x0_size, y0_size, x1_off, y1_off, x1_size, y1_size, lat0_center, lon0_center, ht0_center = psizes
            # chip the rpc to patch size
            img0_rpc.translate_linesamp(-1 * y0_off, -1 * x0_off)
            img1_rpc.translate_linesamp(-1 * y1_off, -1 * x1_off)
            # read the patches: img and lat, lon, ht maps
            img0 = img0_ds.ReadImg(x0_off, y0_off, x0_size, y0_size, repeat=False)
            if not satdepth_utils.ImageIsUseable(img0):
                print("not useable patch on %s "% (img0_ds.img_path))
                return None
            img0_lat, img0_lon, img0_ht = img0_ds.ReadLatLonHt(x0_off, y0_off, x0_size, y0_size)

            img1 = img1_ds.ReadImg(x1_off, y1_off, x1_size, y1_size, repeat=False)
            if not satdepth_utils.ImageIsUseable(img1):
                print("not useable patch on %s" % (img1_ds.img_path))
                return None
            img1_lat, img1_lon, img1_ht = img1_ds.ReadLatLonHt(x1_off, y1_off, x1_size, y1_size)
            # compute the affine approximation of rpc camera for the two patches
            img0_rpc_affine_mat_3x4 = satdepth_utils.GetRPCAffine(img0_rpc, lat0_center, lon0_center, ht0_center)
            img1_rpc_affine_mat_3x4 = satdepth_utils.GetRPCAffine(img1_rpc, lat0_center, lon0_center, ht0_center)
            # compute the affine Fundamental matrix using the two cameras P1, P2
            # I compared F_cameras with F_matches, and F_matches gives a lower epipolar distance error
            if self.funda_method == "cameras":
                F_gt = SatCV.fundamental_matrix_cameras(img0_rpc_affine_mat_3x4, img1_rpc_affine_mat_3x4)
                F_gt = F_gt.astype(np.float32)
        else:
            # cant draw patches on image, because the random patch was outside image extents
            print("cant draw gridded patch: %s and %s\n" % (img0_ds.img_path, img1_ds.img_path))
            return None

        if self.funda_method == "matches":
            x0, y0, x1, y1 = satdepth_utils.get_matches(self.kp_mode, self.pct_sift, self.num_pts,
                                                    self.kp_distance_thresh, self.dataset_nodata_value,
                                                    img0, img0_rpc, img0_lat, img0_lon, img0_ht,
                                                    img1, img1_rpc, img1_lat, img1_lon, img1_ht,
                                                    x0_size, y0_size,
                                                    x1_size, y1_size)
            if x0.size <= self.num_pts_retained or x1.size <= self.num_pts_retained:
                # print("cant draw points: %s and %s\n"
                #       "sizes: img0: %d  img1: %d\n" % (img0_ds.img_path, img1_ds.img_path,x0.size, x1.size))
                return None

            # Gt correspondences can be different for each pair. Handle this!
            x0 = x0[:self.num_pts_retained].astype(np.float32)
            y0 = y0[:self.num_pts_retained].astype(np.float32)
            x1 = x1[:self.num_pts_retained].astype(np.float32)
            y1 = y1[:self.num_pts_retained].astype(np.float32)
            # compute the Fundamental matrix using matches
            matches = np.vstack((x0, y0, x1, y1)).T
            F_gt = SatCV.affine_fundamental_matrix(matches) # no ransac
            # F_gt = SatCV.refine_affine_fundamental_matrix(matches) # ransac refinement didnt improve since we have all true correspondences
            F_gt = F_gt.astype(np.float32)

        # transform images, coordinates to tensor and reshape image
        img0 = torch.from_numpy(img0).unsqueeze(0)
        img1 = torch.from_numpy(img1).unsqueeze(0)

        img0_lat  = torch.from_numpy(img0_lat)
        img0_lon = torch.from_numpy(img0_lon)
        img0_ht = torch.from_numpy(img0_ht)

        img1_lat = torch.from_numpy(img1_lat)
        img1_lon = torch.from_numpy(img1_lon)
        img1_ht = torch.from_numpy(img1_ht)

        img0 = self.transform(img0)
        img1 = self.transform(img1)

        img0_extents = torch.tensor([x0_off, y0_off, x0_size, y0_size])
        img1_extents = torch.tensor([x1_off, y1_off, x1_size, y1_size])

        # make dictionary for data
        data = {
            "image0": img0,
            "image1": img1,
            "affine_cam0": img0_rpc_affine_mat_3x4,
            "lat0": img0_lat,
            "lon0": img0_lon,
            "ht0": img0_ht,
            "img0_extents": img0_extents,
            "affine_cam1": img1_rpc_affine_mat_3x4,
            "lat1": img1_lat,
            "lon1": img1_lon,
            "ht1": img1_ht,
            "img1_extents": img1_extents,
            "F_gt": F_gt,
            "intersection_angle": intersection_angle,
            # "relative_track_angle": relative_track_angle,
            "dataset_name": "SatDepth",
            "pair_names": (img0_ds.img_path, img1_ds.img_path)
        }
        return data

    def __len__(self):
        if self.phase == "test":
            return len(self.grid_pts)
        return len(self.pairs)

def _read_pairs(pairs_file):
    "read pairs from csv file and make list of paired datasamples"
    pairs = []
    df = pd.read_csv(pairs_file)
    for i in range(len(df)):
        _df_idx, img0, img0_rpc, img0_lat, img0_lon, img0_ht, \
        img1, img1_rpc, img1_lat, img1_lon, img1_ht, \
        dsm_file, intersection_angle, relative_track_angle = df.iloc[i].to_list()

        img0_ds = satdepth_utils.DataSample(img0, img0_rpc, img0_lat, img0_lon, img0_ht, dsm_file)
        img1_ds = satdepth_utils.DataSample(img1, img1_rpc, img1_lat, img1_lon, img1_ht, dsm_file)
        pairs.append([img0_ds, img1_ds, intersection_angle, relative_track_angle])
    return pairs
    # return pairs[:20] # for debugging