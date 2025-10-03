<p align="center">
  <h1 align="center">Depthifypp</h1>
</p>

## Build depthifypp
To build depthifypp binary follow steps:
- Activate satdepth conda environment `conda activate satdepth`
- Go to `<path-to-satdepth-repo>/depthifypp/cpp` directory and run make `make all`
- Test the built binary using `./depthifypp`. If built correctly, it will throw an argument parsing error.

---
## Running depthifypp

Prior to running depthifypp we need to ensure that inputs are prepared correctly for our script. Please prepare the inputs following the guidelines given below.

<details>
  <summary>Folder Structure of Pipeline Output</summary>

  The depthifypp scripts are designed for our pipeline output folder structure which is as follows:

  ```
  .
  ├── aoi_rect_piece_0
  |   ├── chips
  |   │   └── PAN
  |   │       └── *.tif                                           # Satellite Image
  |   ├── corrected_rpc
  |   │   └── bundle_adjustment
  |   │       └── pan
  |   │           └── RPB
  |   |               └── *.RPB                                   # RPC Camera Model
  |   └── dsm
  |       └── fusedDSM
  |           └── aoi_rect_piece_0_DSM-wgs84_unpadded.tif         # DSM for the tile in wgs84
  |
  ├── aoi_rect_piece_1
  ...
  ```
</details>

<details>
  <summary>Contents of `aoi_list.txt` file</summary>

  This file contains a list of all tiles which need to be processed. The contents of the file are as follows:

  ```
  aoi_rect_piece_0
  aoi_rect_piece_0
  aoi_rect_piece_1
  aoi_rect_piece_2
  aoi_rect_piece_3
  aoi_rect_piece_4
  aoi_rect_piece_5
  ...
  ```
</details>

<details>
  <summary>Instructions for downloading DEM and Water Mask</summary>

  depthifypp needs a low-res DEM and water mask as inputs for creating SatDepth Maps. We use the SRTM DEM and ASTERWBDV001 datasets for low-res DEM and water mask respectively. To download this data follow the steps below:
  - [SRTM DEM] Go to [USGS Earth Explorer](https://earthexplorer.usgs.gov/) and select the `Digital Elevation\SRTM\SRTM 1 Arc-Second Global` dataset for your AOI and download DEM.
  - [ASTERWBDV001]  Go to [NASA EarthData Search](https://search.earthdata.nasa.gov/) and enter `C1575734433-LPDAAC_ECS` in the search bar for your area of interest then download the Water Mask.

  Please note that you may need to create an account with [USGS Earth Explorer](https://earthexplorer.usgs.gov/) and [NASA EarthData Search](https://search.earthdata.nasa.gov/) to download the data.

  Furthermore, please ensure that both the DEM and Water Mask are in wgs84 as our scripts expect it to be in this coordinate reference system. To convert any raster to wgs84 you can use the following command:

  ```shell
  gdalwarp -t_srs EPSG:4326 input.tif output_wgs84.tif
  ```

</details>

</br>

To generate SatDepth Maps use the following steps:
- Input the paths to files and folders in the script `run_depthify_mpi.sh`. You would need to setup paths for the varirables `WVTopDir` `outTopDir` `dem` `dem_mask_file` `aoi_list`.
- Run the script using the command `bash run_depthify_mpi.sh <world-size> <rank>`.

The script splits the tiles (listed in `aoi_list.txt`) equally among all workers. When running distributed job (ie multiple workers/rank), the `<world-size>` argument remains the same among all workers and `<rank>` goes from 0 to `<world-size> -1`. You will have to run the script manually on each worker, we do not provide functionality for sending the job to each worker.