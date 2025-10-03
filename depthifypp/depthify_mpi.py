#!/bin/python3
"""
Depthify a PAN image using corresponding DSM and corrected RPC
using MPI for executing depthify.py on multiple machines

The script first crops all images to AOI DSM extents and
then creates depth images for them

author: Rahul Deshmukh
date: 04 Dec 2022
"""
from osgeo import gdal
import os, sys, glob
import argparse
from multiprocessing import Pool

try:
    import mpi4py
    import mpi4py.MPI
except:
    pass

import numpy as np

from satdepth.utils.rpc import RPC
from utils import basename, check_if_exists, convert_to_epsg4326, get_vm_ip_address
from satdepth.utils.ortho_camera import Ortho_Camera
from satdepth.utils.useful_methods import setup_logger


class DepthifyArgs_Container:
    "class for storing args required for running depthify"
    def __init__(self,
                TopDir=None,
                aoi_base=None,
                dsm_file=None,
                rpc_file=None,  # path to RPB file
                img_file=None  # path to PAN file
                ):
        self.TopDir = TopDir
        self.aoi_base = aoi_base
        self.dsm_file = dsm_file
        self.rpc_file = rpc_file
        self.img_file = img_file
        self.img_base = basename(self.img_file)
        self.rpc_base = basename(self.rpc_file)
        assert self.img_base == self.rpc_base  # these names should be the same so that the cpp code can read the rpc
        return


def check_args(args):
    if not check_if_exists(args.WVTopDir): sys.exit(1)
    if not check_if_exists(args.outTopDir): os.makedirs(args.outTopDir)
    if (args.dem is not None) and (not check_if_exists(args.dem)):
        sys.exit(1)
    if args.dem_interp not in ["near", "bilinear", "cubic"]:
        raise ValueError("\n\nError: DEM interpolation should be one of near,"
                         " bilinear or cubic. "
                         "You have given %s \n" % args.dem_interp)
        sys.exit(1)
    if (args.dem_mask_file is not None) and (
            not check_if_exists(args.dem_mask_file)):
        sys.exit(1)
    return


def make_task_list(WVTopDir, aoi_name_prefix, aoi_list_file=None):
    if aoi_list_file == None:
        # find all aoi folders
        aoi_base_list = [f.name for f in os.scandir(WVTopDir) if
                             (f.is_dir() and aoi_name_prefix in f.name)]
    else:
        print("Reading aois to process from the file: %s"%(aoi_list_file))
        f = open(aoi_list_file,'r')
        aoi_base_list = f.readlines()
        f.close()
        aoi_base_list = [ line.replace('\n','') for line in aoi_base_list if aoi_name_prefix in line ]

    all_aoi_base_list = []
    dsm_list = []

    for aoi_base in aoi_base_list:
        dsm_file = os.path.join(WVTopDir, aoi_base, "dsm", "fusedDSM",
                                aoi_base + "_DSM-wgs84_unpadded.tif")
        if os.path.exists(dsm_file):
            dsm_list.append(dsm_file)
            all_aoi_base_list.append(aoi_base)

    # make list of task_args
    task_list_args = []
    for aoi_base, dsm_file in zip(all_aoi_base_list, dsm_list):
        aoi_corrected_rpc_folder = os.path.join(WVTopDir, aoi_base,
                                                "corrected_rpc",
                                                "bundle_adjustment",
                                                "pan", "RPB")
        if not os.path.exists(aoi_corrected_rpc_folder):
            aoi_corrected_rpc_folder = os.path.join(WVTopDir, aoi_base,
                                                    "corrected_rpc",
                                                    "bundle_adjustment_mitp",
                                                    "pan", "RPB")

        # find list of corrected RPCs
        for rpc_file in glob.glob(aoi_corrected_rpc_folder + "/*.RPB"):
            img_base = basename(rpc_file)
            img_file = os.path.join(WVTopDir, aoi_base, "chips", "PAN",
                                    img_base + ".tif")
            if os.path.exists(img_file):
                i_task = DepthifyArgs_Container(TopDir=WVTopDir,
                                                aoi_base=aoi_base,
                                                dsm_file=dsm_file,
                                                img_file=img_file,
                                                rpc_file=rpc_file)
                task_list_args.append(i_task)

    return task_list_args


def Crop_Image_to_DSM_extents(img_file: str, rpc_file: str, dsm_file: str,
                              dsm_nodata_value,
                              out_img_file: str, out_rpc_file: str, logger):
    "crop pan image to dsm extents using corrected rpc"
    rpc = RPC.from_file(rpc_file)
    dsm_cam = Ortho_Camera.from_file(dsm_file)
    dsm_img = dsm_cam.ReadImg()

    # get dsm no data mask
    dsm_nodata_mask = dsm_img == dsm_nodata_value
    # get rectangular border of valid dsm
    offset = 0
    dsm_r0, dsm_c0 = 0, 0
    dsm_r1, dsm_c1 = dsm_cam.nrows - 1, dsm_cam.ncols - 1
    inMask = np.any([dsm_nodata_mask[dsm_r0, dsm_c0],
                     dsm_nodata_mask[dsm_r1, dsm_c0],
                     dsm_nodata_mask[dsm_r0, dsm_c1],
                     dsm_nodata_mask[dsm_r1, dsm_c1]])
    while (inMask):
        dsm_r0 += 1;
        dsm_c0 += 1
        dsm_r1 -= 1;
        dsm_c1 -= 1
        offset += 1
        inMask = np.any([dsm_nodata_mask[dsm_r0, dsm_c0],
                         dsm_nodata_mask[dsm_r1, dsm_c0],
                         dsm_nodata_mask[dsm_r0, dsm_c1],
                         dsm_nodata_mask[dsm_r1, dsm_c1]])
    logger.info("\t Using  valid dsm with offset : %d\n" % offset)

    img_ds = gdal.Open(img_file)
    img = img_ds.ReadAsArray()
    img_nrows, img_ncols = img.shape

    # get four corners aoi dsm: lat, lon, ht values
    lon00, lat00 = dsm_cam.backproject(dsm_r0, dsm_c0)
    lon10, lat10 = dsm_cam.backproject(dsm_r1, dsm_c0)
    lon01, lat01 = dsm_cam.backproject(dsm_r0, dsm_c1)
    lon11, lat11 = dsm_cam.backproject(dsm_r1, dsm_c1)

    ht_r0c0 = dsm_img[dsm_r0, dsm_c0]
    ht_r1c0 = dsm_img[dsm_r1, dsm_c0]
    ht_r0c1 = dsm_img[dsm_r0, dsm_c1]
    ht_r1c1 = dsm_img[dsm_r1, dsm_c1]

    # project four corners to rpc image and get bbox
    r00, c00 = rpc.rpc(lat00, lon00, ht_r0c0)
    r10, c10 = rpc.rpc(lat10, lon10, ht_r1c0)
    r01, c01 = rpc.rpc(lat01, lon01, ht_r0c1)
    r11, c11 = rpc.rpc(lat11, lon11, ht_r1c1)

    r_min = int(np.floor(np.min([r00, r10, r01, r11])))
    r_max = int(np.ceil(np.max([r00, r10, r01, r11])))
    c_min = int(np.floor(np.min([c00, c10, c01, c11])))
    c_max = int(np.ceil(np.max([c00, c10, c01, c11])))

    r_min = np.clip(r_min, 0, img_nrows)
    r_max = np.clip(r_max, 0, img_nrows)
    assert r_min < r_max
    c_min = np.clip(c_min, 0, img_ncols)
    c_max = np.clip(c_max, 0, img_ncols)
    assert c_min < c_max

    # crop image to bbox
    img = img[r_min:r_max + 1, c_min:c_max + 1]
    out_nrows, out_ncols = img.shape

    # write new image
    write_raster(img_ds, out_img_file, img, out_ncols, out_nrows)

    # crops and write new rpc
    rpc.translate_linesamp(-1 * r_min, -1 * c_min)
    rpc.save_rpb(out_rpc_file)

    img_ds = None
    dsm_cam.ds = None
    return


def write_raster(input_ds, out_img_file, array_to_write, out_ncols, out_nrows):
    dtype_str = gdal.GetDataTypeName(
        input_ds.GetRasterBand(1).DataType)  # this is a string ex "UInt16"
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(out_img_file, out_ncols, out_nrows, 1,
                           gdal.GetDataTypeByName(dtype_str))
    out_ds.SetMetadata(input_ds.GetMetadata())
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array_to_write)
    out_ds = None
    return


def do_work_mpi_node( WVTopDir, outTopDir,
                     dem_file, dem_interp,
                     dem_mask_file,
                     aoi_name_prefix, aoi_list_file,
                     dsm_nodata_value,
                     num_threads_per_image,
                     batch_size,
                     USE_MPI, RANK, SIZE
                     ):
    if USE_MPI:
        # Get MPI params
        comm = mpi4py.MPI.COMM_WORLD
        RANK = comm.Get_rank()
        SIZE = comm.Get_size()

    ip = get_vm_ip_address()
    logging_file = os.path.join(outTopDir,
                                "depthify_mpi_commsize_%d_rank_%d_ip_%s.log" % (
                                SIZE, RANK, ip))
    logger = setup_logger("depthify_mpi", logging_file)

    logger.info(
        "Processing log for DEPTHIFY_MPI\n"
        "MPI Comm Size %d Rank %d\n"
        "on VM IP: %s\n"
        "------------------------------\n\n"
        % (SIZE, RANK, ip)
    )

    logger.info("Making a list of all work: all valid AOIS (ie with dsm) and all aligned images\n")
    all_task_list = make_task_list(WVTopDir, aoi_name_prefix, aoi_list_file)
    Num_tasks = len(all_task_list)
    Num_batches = int(np.ceil(Num_tasks / batch_size))
    logger.info("Total Number of Tasks: %d with batch size: %d" % ( Num_tasks, batch_size) )
    logger.info("Total Number of Batches: %d" % Num_batches)

    depthify_command_string = "python -u /mnt/cloudNAS2/rahul/GitReposCORE3D_newcopy/GitReposCORE3D/pyLEGO_rvl/lego_rvl/utils/depthImage/depthify.py "

    if (dem_file is not None) and (dem_mask_file is not None):
        depthify_command_string += "%s %s %s -rpc_file %s -dem %s -dem_interp %s -dem_mask_file %s -parallel -fast -num_threads_per_image %d"
    elif (dem_file is not None) and (dem_mask_file is None):
        depthify_command_string += "%s %s %s -rpc_file %s -dem %s -dem_interp %s " \
                                   "-parallel -fast -num_threads_per_image %d"
    elif (dem_file is None) and (dem_mask_file is not None):
        depthify_command_string += "%s %s %s -rpc_file %s " \
                                   "-dem_mask_file %s -parallel -fast -num_threads_per_image %d"
    elif (dem_file is None) and (dem_mask_file is None):
        depthify_command_string += "%s %s %s -rpc_file %s " \
                                   "-parallel -fast -num_threads_per_image %d"

    depthify_command_string += " 2>&1 >> %s" % (logging_file)


    num_node_batches = len(np.arange(RANK, Num_batches, SIZE))
    logger.info("\t This node will process a Total of %d batches ie max num of tasks:%d\n\n"
                %(num_node_batches, num_node_batches * batch_size))
    if USE_MPI: comm.barrier()

    # (3) sequentially carry out this node's work
    for batch_idx in np.arange(Num_batches):
        logger.info("Starting batch num: %d\n" % batch_idx)
        start_idx = RANK * batch_size + batch_idx * SIZE * batch_size
        end_idx = start_idx + batch_size
        # print("start_idx, end_idx: ", start_idx, end_idx)
        batch_task_args = all_task_list[ start_idx : end_idx]
        if len(batch_task_args) == 0:
            continue # no work for this batch on this node with rank=RANK
        # elif len(batch_task_args) == 1:
            #when only one job in this batch, then batch_task_args wone be a list, handling that
            # batch_task_args = [batch_task_args]

        cropped_images_outdir = [os.path.join(outTopDir, batch_task_args[i].aoi_base,
                                             "DSM_Cropped_Images") for i in range(len(batch_task_args))]
        depthify_outdir = [os.path.join(outTopDir, batch_task_args[i].aoi_base, "Depth") for i in range(len(batch_task_args))]
        for i in range(len(batch_task_args)):
            if not os.path.exists(cropped_images_outdir[i]):
                logger.info(" Creating aoi output directory for cropped images at %s" %(cropped_images_outdir[i]))
                try:
                    os.makedirs(cropped_images_outdir[i])
                except:
                    logger.warning("The aoi folder was already created %s"%(cropped_images_outdir[i]))
                    pass
            if not os.path.exists(depthify_outdir[i]):
                logger.info(
                    " Creating aoi output directory for depth images at %s" % (
                        depthify_outdir[i]))
                try:
                    os.makedirs(depthify_outdir[i])
                except:
                    logger.warning("The aoi folder was already created %s"%(cropped_images_outdir[i]))
                    pass

        logger.info("MPI BARRIER BEFORE CROPPING for synchronization for batch: %d\n" % batch_idx)
        # if USE_MPI: comm.barrier()
        #  crop images to DSM
        out_img_file = [os.path.join(cropped_images_outdir[i],
                                    batch_task_args[i].img_base + ".tif") for i in range(len(batch_task_args))]
        out_rpc_file = [os.path.join(cropped_images_outdir[i],
                                    batch_task_args[i].rpc_base + ".RPB") for i in range(len(batch_task_args))]

        for i in range(len(batch_task_args)):
            if not os.path.exists(out_img_file[i]):
                logger.info("Cropping Image to DSM\n"
                                "\t IMG FILE: %s\n"
                                "\t RPC FILE: %s\n"
                                "\t DSM FILE: %s\n"
                                "\t OUTPUT IMAGE FILE: %s\n"
                                "\t OUTPUT RPC FILE: %s\n"
                            %(batch_task_args[i].img_file, batch_task_args[i].rpc_file, batch_task_args[i].dsm_file,
                              out_img_file[i], out_rpc_file[i])
                            )
                Crop_Image_to_DSM_extents(batch_task_args[i].img_file,
                                          batch_task_args[i].rpc_file,
                                          batch_task_args[i].dsm_file, dsm_nodata_value,
                                          out_img_file[i], out_rpc_file[i],
                                          logger)

        logger.info("MPI BARRIER AFTER CROPPING for synchronization for batch: %d\n" % batch_idx)
        # comm.barrier()

        # depthify
        depthify_task_commands = []
        depth_img_file = []
        for i in range(len(batch_task_args)):
            if (dem_file is not None) and (dem_mask_file is not None):
                depthify_task_command = depthify_command_string % (
                    out_img_file[i],
                    batch_task_args[i].dsm_file,
                    depthify_outdir[i],
                    out_rpc_file[i],
                    dem_file, dem_interp,
                    dem_mask_file, num_threads_per_image)

            elif (dem_file is not None) and (dem_mask_file is None):
                depthify_task_command = depthify_command_string % (
                    out_img_file[i],
                    batch_task_args[i].dsm_file,
                    depthify_outdir[i],
                    out_rpc_file[i],
                    dem_file, dem_interp)

            elif (dem_file is None) and (dem_mask_file is not None):
                depthify_task_command = depthify_command_string % (
                    out_img_file[i],
                    batch_task_args[i].dsm_file,
                    depthify_outdir[i],
                    out_rpc_file[i],
                    dem_mask_file, num_threads_per_image)

            elif (dem_file is None) and (dem_mask_file is None):
                depthify_task_command = depthify_command_string % (
                    out_img_file[i],
                    batch_task_args[i].dsm_file,
                    depthify_outdir[i],
                    out_rpc_file[i],
                    num_threads_per_image)

            d_img_file =  os.path.join(depthify_outdir[i], batch_task_args[i].img_base+'_depth.tif')
            if not os.path.exists(d_img_file):
                depthify_task_commands.append(depthify_task_command)
                depth_img_file.append(d_img_file)

        for i in range(len(depth_img_file)):
            if not os.path.exists(depth_img_file[i]):
                logger.info("Depthifying the dsm cropped image for batch num: %d\n"
                            "\t Using the command\n"
                            "\t %s\n"
                            % (batch_idx, depthify_task_commands[i]))
        if len(depthify_task_commands) > 0:
            pool_workers = Pool(len(depthify_task_commands))
            pool_workers.map(os.system, depthify_task_commands)
            pool_workers.close()
            #os.system(depthify_task_command)

        logger.info("Finished batch num: %d\n"
                    "#--------------#\n"
                    %(batch_idx))
        #logger.info("MPI BARRIER AFTER DEPTHIFY for synchronization for task %d\n" % task_idx)
        #comm.barrier()
        # exit(1)
    return


def main():
    parser = argparse.ArgumentParser(prog="depthify_mpi",
                                     description="Depthify non-ortho images for"
                                                 " an AOI using MPI with "
                                                 "multiple virtual machines")
    parser.add_argument("WVTopDir", type=str,
                        help="Top directory path to WV AOIs"
                             " with PAN images, corrected RPCs and AOI DSMs")
    parser.add_argument("outTopDir", type=str,
                        help="Top directory for output, where we will write the depth data")
    parser.add_argument("-aoi_name_prefix", type=str,
                        help="prefix string for each aoi rect piece folder",
                        default="aoi_rect_piece_")
    parser.add_argument("-dsm_nodata_value", type=float,
                        help="no data value for DSMs to be read",
                        default=-9999)

    parser.add_argument("-dem", type=str,
                        help="path to low res DEM for ground height with full coverage")
    parser.add_argument("-dem_in_utm", help="flag to specify DEM is in UTM",
                        action="store_true")
    parser.add_argument("-dem_interp", type=str, default="bilinear",
                        help="interpolation method for dem")

    parser.add_argument("-dem_mask_file", help="uint8 bit mask file for "
                                               "dem indicating points to skip",
                        type=str,
                        default=None)
    parser.add_argument("-dem_mask_in_utm",
                        help="flag to specify DEM mask is in UTM",
                        action="store_true")

    parser.add_argument("-aoi_list", type=str, help="text file containing list of aois to process (only the base names)", default=None)

    # computation related args
    parser.add_argument("-num_threads_per_image", type=int,
                        help="number of parallel threads to be used "
                             "with fast flag it is the number of threads used "
                             "for computation for a block of data",
                        default=1)
    parser.add_argument("-batch_size", type=int, help="number of images to be processed in one VM parallely", default=1)

    parser.add_argument("-USE_MPI", action='store_true')
    parser.add_argument("-rank", type=int, default=0, help="rank of current vm \in [0 to size-1]")
    parser.add_argument("-size", type=int, default=1, help="world size ie number of vms working parallely")


    # Parse the commandline:
    try:
        args = parser.parse_args()
    except:
        print("\n")
        parser.print_help()
        sys.exit(1)

    check_args(args)

    # TODO: Assuming DSMs are in WGS, handle this later
    # low res dem
    if (args.dem is not None):
        if args.dem_in_utm:
            dem_file_epsg4326 = convert_to_epsg4326(args.dem)
        else:
            dem_file_epsg4326 = args.dem
    else:
        dem_file_epsg4326 = None

    # low res dem_mask
    if (args.dem_mask_file is not None):
        if args.dem_mask_in_utm:
            dem_mask_file_epsg4326 = convert_to_epsg4326(args.dem_mask_file)
        else:
            dem_mask_file_epsg4326 = args.dem_mask_file
    else:
        dem_mask_file_epsg4326 = None

    do_work_mpi_node(args.WVTopDir, args.outTopDir,
                     dem_file_epsg4326, args.dem_interp,
                     dem_mask_file_epsg4326,
                     args.aoi_name_prefix, args.aoi_list,
                     args.dsm_nodata_value,
                     args.num_threads_per_image,
                     args.batch_size,
                     args.USE_MPI, args.rank, args.size)
    return


if __name__ == "__main__":
    sys.exit(main())
