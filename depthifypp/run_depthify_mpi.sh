#!/bin/bash
# script for generating depth data
# We assume that the dsm has nodata value of -9999 (if not then write it!)
# Takes two args
#	$1: world size ie num vms: (default: 1)
#	$2: rank of this process (default: 0 this should be starting from 0)
# example usage: ./run_depthify_mpi_manually.sh <comm size> <rank>
# 
# You need to set a few variables before running the above command
#   WVTopDir -- path to WV type dir containing, chipped images, aligned rpcs, and upadded fused DSMs
#   outTopDir -- output directory for depth data
#   dem -- low-res dsm used for getting ground height for depthify (should be in wgs84)
#   dem_mask_file -- water mask from aster (should be in wgs84)
#   aoi_list -- list of aois to be depthified from <WVTopDir>

#* Set these
WVTopDir=""
outTopDir=""
dem=n*_w*_1arc_v3_wgs84.tif
dem_mask_file=ASTWBDV001_N*W*_att.tif
aoi_list=aoi_list.txt
# --------------------------------------------------------------------------------------------------------------

num_threads_per_image=$(nproc)
batch_size=1

wordl_size=$1
rank=$2

mkdir -p $outTopDir

cmd="time python depthify_mpi.py $WVTopDir $outTopDir -dem $dem -dem_mask_file $dem_mask_file -aoi_list $aoi_list -num_threads_per_image $num_threads_per_image -batch_size $batch_size -rank $rank -size $wordl_size 2>&1"

short_date=$(date +"%d_%B_%Y_%Hhrs_%Mmins_%Ssecs")
{
  echo 'Manual MPI-like Job Summary'
  echo -n 'start: '
  date
  echo ---------------------------------------
  echo Executing
  echo $cmd
  eval $cmd
  echo ---------------------------------------
  echo -n 'end: '
  date
  echo finished
} | tee $outTopDir/depthify_job_commsize_"$wordl_size"_rank_"$rank"_"$short_date".log
