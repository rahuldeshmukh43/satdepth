#include <omp.h> //openMP pragmas
#include <iostream> // for printing
#include <stdio.h>
#include <boost/program_options.hpp> // for nice argument parsing
#include <boost/stacktrace.hpp> //for call stack trace during exception
#include <boost/exception/all.hpp>

#include <cstdint> //uint16_t

//STL
#include <string>	// for string datatype
#include <vector>
//#include <cmath> //NaN

// gdal based includes
#include "gdal_priv.h" // for common GDAL functions
#include "gdal_alg.h"  // for GDALCreateRPCTransformer and GDALRPCTransform function
#include "cpl_conv.h"  // for CPLMalloc()

// custom utils
#include "depthify_worker_utils.hpp"
#include "depthify_worker_utils.cpp"

//#define DEBUG
#define LOG
#define TIMING
//#define OMP

namespace wu=worker_utils;

typedef boost::error_info<struct tag_stacktrace, boost::stacktrace::stacktrace> traced;

template <class E>
void throw_with_trace(const E& e) {
    throw boost::enable_error_info(e)
        << traced(boost::stacktrace::stacktrace());
}

struct Point3drc
{
	int dsm_row = -1;
	int dsm_col = -1;
	double lat = 0.;
	double lon = 0.;
	double ht = 0.;
	bool empty()
	{
		bool flag = (dsm_row == -1)? true: false;
		return flag;
	}
};
typedef struct Point3drc Point3drc;

int omp_thread_count() {
	/* for counting num threads in case of gcc */
    int n = 0;
    #pragma omp parallel
    {
		#pragma omp single
//		#ifdef OMP
    	n = omp_get_num_threads();
//		#else
//    	n= 1;
//		#endif
    }
    return n;
}

template<typename T, typename Tbounds>
bool InRange(T index, Tbounds lower_bound, Tbounds upper_bound )
{
	// checks for $$u\in [lower_bound, upper_bound)$$
	int index_int = (int) floor(index);
	lower_bound  = (int) floor(lower_bound);
	upper_bound  = (int) ceil(upper_bound);
	return (index_int >= lower_bound ) and (index_int < upper_bound);
}

template <typename T_img, typename T_out, typename T_latlon, typename T_idx>
bool CreateOutput(
	wu::RPCRaster<T_img, T_idx> *porpcRaster,
	int start_row, int start_col,
	int end_row, int end_col,
	std::vector<Point3drc> &flattened_grid_pts3d,
	int out_nrows, int out_ncols,
	std::vector<T_out>& outputArray_ht,
	std::vector<T_latlon>& outputArray_lat,
	std::vector<T_latlon>& outputArray_lon,
	T_out dst_nodata)
{
	//init output depth image
	#ifdef LOG
	std::cout << "compute max reduce z for the grid: use openMP" << std::endl;
	#endif


	#pragma omp parallel for
	for(uint32_t i=0; i < flattened_grid_pts3d.size(); i++)
	{
//		int tid = omp_get_thread_num();
		if (flattened_grid_pts3d[i].empty())
		{
//			#ifdef DEBUG
//			std::cout << "grid pt index i: "<< i <<" was empty" << std::endl;
//			#endif
			continue;
		}

		double i_lat = flattened_grid_pts3d[i].lat;
		double i_lon = flattened_grid_pts3d[i].lon;
		int success_flag;

		//project lat, lon, ht to image using rpc
		porpcRaster->Project(1, i_lat, i_lon, flattened_grid_pts3d[i].ht,
								  success_flag);

#ifdef DEBUG
		if(i<100)
		{
//		std::cout << "Projecting using rpc (tid: " << tid << ") " ;
		std::cout << "lat, lon, ht (" << flattened_grid_pts3d[i].lat;
		std::cout << ", " << flattened_grid_pts3d[i].lon;
		std::cout << ", " << flattened_grid_pts3d[i].ht << ") ";
		std::cout << " to r,c ("<< i_lat <<", "<< i_lon <<")"<< std::endl;
		}
#endif

		//check if projected pt lies within start/end row col
		if( (i_lat >= start_row ) and ( i_lat < end_row ) and
			(i_lon >= start_col ) and ( i_lon < end_col ))
		{
			//3d pt is inside the image -- put max ht
			//compute row, col in output array
			int out_row, out_col;
			out_row = (int) floor(i_lat - start_row);
			out_col = (int) floor(i_lon - start_col);

			// write max ht
			if(outputArray_ht[out_row*out_ncols + out_col] < flattened_grid_pts3d[i].ht)
			{
				#pragma omp atomic write
				outputArray_ht[out_row*out_ncols + out_col] = (T_out) flattened_grid_pts3d[i].ht;

				#pragma omp atomic write
				outputArray_lat[out_row*out_ncols + out_col] = (T_latlon) flattened_grid_pts3d[i].lat;

				#pragma omp atomic write
				outputArray_lon[out_row*out_ncols + out_col] = (T_latlon) flattened_grid_pts3d[i].lon;
			}
		}
		else
		{
			//3d pt not in the image -- continue
			continue;
		}
	}

	return 1;
}

template <typename T>
bool WriteGTiff(std::string outputRasterName,
		   int outputWidth,
		   int outputHeight,
		   int nbands,
		   GDALDataType outputDataType,
		   std::vector<T> &outBandsAsArray,
		   T dst_nodata)
{
	int writeflag;
	const char *pszFormat = "GTiff";
	GDALDriver *poDriver;
	char **papszMetadata;

	poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
	if(poDriver == NULL)
	{
		std::cout << "ERROR cannot create Driver object "<< std::endl;
		exit(1);
	}

	GDALDataset *poDstDS;
	char **ds_papszOptions = NULL;

	poDstDS = poDriver->Create(outputRasterName.c_str(), outputWidth,
			outputHeight, nbands, outputDataType, ds_papszOptions);
	GDALRasterBand *poDstBand;

	for (int bandIdx = 1; bandIdx <= nbands; bandIdx++)
	{
		poDstBand = poDstDS->GetRasterBand(bandIdx);
		poDstBand->SetNoDataValue(dst_nodata);
		writeflag = poDstBand->RasterIO( GF_Write, 0, 0, outputWidth, outputHeight, &outBandsAsArray[(bandIdx -1)*outputWidth*outputHeight],
				outputWidth, outputHeight, outputDataType, 0, 0);
	}

	GDALClose( (GDALDatasetH) poDstDS );

	return 1;
}

template<typename T_img=uint16_t, typename T_dsm=float, typename T_latlon=double, typename T_dem=int16_t, typename T_demMask=uint8_t, typename T_idx=double>
bool do_block_work(
		std::string outputRasterName,
		std::string inputRasterName, std::string rpcRaster_interp,
		int start_row, int end_row, // row col in rpc image space
		int start_col, int end_col,
		// lat lon in dsm space
		T_latlon UL_lat, T_latlon UL_lon,
		T_latlon UR_lat, T_latlon UR_lon,
		T_latlon LL_lat, T_latlon LL_lon,
		T_latlon LR_lat, T_latlon LR_lon,
		std::string dsmFileName, std::string dsm_interp,
		std::string demFileName, std::string dem_interp, int dem_offset,
		std::string demMaskFileName, std::string demMask_interp,
		int BUFFER_SIZE,
		double Z_GRID_SPACING,
		int numThreads)
{

	/*
	 * T_out is same as T_dsm usually float
	 */
//	int numThreads = omp_thread_count();

//	Default interpolations
//	std::string rpcRaster_interp = "bilinear";
//	std::string dsm_interp = "bilinear";
//	std::string dem_interp = "bilinear";
//	std::string demMask_interp = "near";

	double start, end;
	double total_start;

	#ifdef TIMING
	total_start = omp_get_wtime();
	#endif


	try
	{
		#ifdef TIMING
		start = omp_get_wtime();
		#endif

		GDALAllRegister();

		int dsm_nrows, dsm_ncols, out_nrows, out_ncols;
		int dem_nrows, dem_ncols, dem_mask_nrows, dem_mask_ncols;

		double dsm_ULrow, dsm_ULcol, dsm_LLrow, dsm_LLcol;
		double dsm_URrow, dsm_URcol, dsm_LRrow, dsm_LRcol;

		double dsm_Urow, dsm_Ucol, dsm_Lrow, dsm_Lcol;
		double dem_Urow, dem_Ucol, dem_Lrow, dem_Lcol;
		double demMask_Urow, demMask_Ucol, demMask_Lrow, demMask_Lcol;
		double ground_ht;

		wu::RPCRaster<T_img, T_idx> *porpcRaster;
		wu::OrthoRaster<T_dsm, T_idx> *podsmRaster;
		wu::OrthoRaster<T_dem, T_idx> *podemRaster;
		wu::OrthoRaster<T_demMask, T_idx> *podemMaskRaster;

		//compute output size
		out_nrows = end_row - start_row;
		out_ncols = end_col - start_col;

		//create array of RPC raster object
		porpcRaster = new wu::RPCRaster<T_img, T_idx>(inputRasterName, rpcRaster_interp);
		ground_ht = porpcRaster->po_rpc_info->dfHEIGHT_OFF - porpcRaster->po_rpc_info->dfHEIGHT_SCALE;
//		std::cout << "RPC ground ht : " << ground_ht << std::endl;

		//create array of ortho cam for dsm, dem, demMaskfile
		podsmRaster = new wu::OrthoRaster<T_dsm, T_idx>(dsmFileName, dsm_interp);
		dsm_ncols = podsmRaster->GetRasterXSize();
		dsm_nrows = podsmRaster->GetRasterYSize();

		//get no data value
		T_dsm dst_nodata = podsmRaster->GetNoDataValue();

		if(demFileName != "None")
		{
			podemRaster = new wu::OrthoRaster<T_dem, T_idx>(demFileName, dem_interp);
			dem_ncols = podemRaster->GetRasterXSize();
			dem_nrows = podemRaster->GetRasterYSize();
		}

		if(demMaskFileName != "None")
		{
			podemMaskRaster = new wu::OrthoRaster<T_demMask, T_idx>(demMaskFileName, demMask_interp);
			dem_mask_ncols = podemMaskRaster->GetRasterXSize();
			dem_mask_nrows = podemMaskRaster->GetRasterYSize();
		}
//		else
//		{
//			bool valid = true;
//		}

		#ifdef LOG
		std::cout << "find footprint on dsm (ie dsm pixels corresponding to start/end lat/lon)" << std::endl;
		#endif
		// project all four lat lon corners to dsm and get corresponding row, col
		podsmRaster->Project(UL_lat, UL_lon, &dsm_ULrow, &dsm_ULcol);
		podsmRaster->Project(UR_lat, UR_lon, &dsm_URrow, &dsm_URcol);
		podsmRaster->Project(LL_lat, LL_lon, &dsm_LLrow, &dsm_LLcol);
		podsmRaster->Project(LR_lat, LR_lon, &dsm_LRrow, &dsm_LRcol);

		// create min max bbox and then add buffer size
		dsm_Urow = std::min({dsm_ULrow, dsm_URrow, dsm_LLrow, dsm_LRrow});
		dsm_Lrow = std::max({dsm_ULrow, dsm_URrow, dsm_LLrow, dsm_LRrow});
		dsm_Ucol = std::min({dsm_ULcol, dsm_URcol, dsm_LLcol, dsm_LRcol});
		dsm_Lcol = std::max({dsm_ULcol, dsm_URcol, dsm_LLcol, dsm_LRcol});

		// add buffer size to bbox
		dsm_Urow  = floor( std::clamp( dsm_Urow - (double) BUFFER_SIZE, 0., dsm_nrows - 1. ) );
		dsm_Ucol  = floor( std::clamp( dsm_Ucol - (double) BUFFER_SIZE, 0., dsm_ncols - 1. ) );
		dsm_Lrow  = ceil( std::clamp( dsm_Lrow + (double) BUFFER_SIZE, 0., dsm_nrows - 1. ) );
		dsm_Lcol  = ceil( std::clamp( dsm_Lcol + (double) BUFFER_SIZE, 0., dsm_ncols - 1. ) );

		int dsm_bbox_nrows = (int) ceil(dsm_Lrow - dsm_Urow);
		int dsm_bbox_ncols = (int) ceil(dsm_Lcol - dsm_Ucol);
#ifdef DEBUG
		std::cout << "dsm bbox nrows: " << dsm_bbox_nrows << " ncols: "<< dsm_bbox_ncols << " ";
		std::cout << "Lrow: " << dsm_Urow << " Lrow " << dsm_Lrow << " ";
		std::cout << " Ucol: " << dsm_Ucol << " Lcol " << dsm_Lcol <<std::endl;
#endif
		// load partial array
		podsmRaster->ReadAsArray( (int) floor(dsm_Urow),
								  (int) floor(dsm_Ucol),
								  (int) ceil(dsm_Lrow - dsm_Urow) + 1,
								  (int) ceil(dsm_Lcol - dsm_Ucol) + 1 );

		double buffered_dsm_Ulat, buffered_dsm_Ulon, buffered_dsm_Llat, buffered_dsm_Llon;
		podsmRaster->BackProject(dsm_Urow, dsm_Ucol, &buffered_dsm_Ulat, &buffered_dsm_Ulon );
		podsmRaster->BackProject(dsm_Lrow, dsm_Lcol, &buffered_dsm_Llat, &buffered_dsm_Llon );
		if(demFileName != "None")
		{ 	// compute dem footprint
			podemRaster->Project(buffered_dsm_Ulat, buffered_dsm_Ulon, &dem_Urow, &dem_Ucol);
			podemRaster->Project(buffered_dsm_Llat, buffered_dsm_Llon, &dem_Lrow, &dem_Lcol);

			// load partial array
			podemRaster->ReadAsArray( (int) floor(dem_Urow),
									  (int) floor(dem_Ucol),
									  (int) ceil(dem_Lrow - dem_Urow) + 1,
									  (int) ceil(dem_Lcol - dem_Ucol) + 1 );
#ifdef DEBUG
			std::cout << "dem : ";
			std::cout << "Lrow: " << dem_Urow << " Lrow " << dem_Lrow << " ";
			std::cout << " Ucol: " << dem_Ucol << " Lcol " << dem_Lcol <<std::endl;
#endif
		}

		if(demMaskFileName != "None")
		{
			podemMaskRaster->Project(buffered_dsm_Ulat, buffered_dsm_Ulon, &demMask_Urow, &demMask_Ucol);
			podemMaskRaster->Project(buffered_dsm_Llat, buffered_dsm_Llon, &demMask_Lrow, &demMask_Lcol);

			// load partial array
			podemMaskRaster->ReadAsArray( (int) floor(demMask_Urow),
										  (int) floor(demMask_Ucol),
										  (int) ceil(demMask_Lrow - demMask_Urow) + 1,
										  (int) ceil(demMask_Lcol - demMask_Ucol) + 1);
#ifdef DEBUG
			std::cout << "dem Mask : ";
			std::cout << "Lrow: " << demMask_Urow << " Lrow " << demMask_Lrow << " ";
			std::cout << " Ucol: " << demMask_Ucol << " Lcol " << demMask_Lcol <<std::endl;
#endif

		}



		#ifdef TIMING
		end = omp_get_wtime();
		printf("Time for initial work : %lf\n", end - start);
		start = omp_get_wtime();
		#endif

		//compute min and max ht for each row,col in dsm
		std::vector <double> minHt(dsm_bbox_nrows*dsm_bbox_ncols, ground_ht);
//		if(demFileName == "None")
//		{
//			std::fill( minHt.begin(), minHt.end(), (double) dst_nodata);
//		}
//		else
//		{
//			std::fill( minHt.begin(), minHt.end(), ground_ht);
//		}

		std::vector <double> maxHt(dsm_bbox_nrows*dsm_bbox_ncols, (double) dst_nodata);
		std::vector <int> numHtpts(dsm_bbox_nrows*dsm_bbox_ncols, 0);

		#ifdef LOG
		std::cout << "populate min , max ht and numhtpts" << std::endl;
		#endif
		//TODO: change below to a function and then instead of single pixel read from disk, load entire footprint into memory and then assign (do tiling for further optimization). this is a bottleneck at the moment as disk i/o is slow
		#pragma omp parallel for
		for(int i=0; i < dsm_bbox_nrows*dsm_bbox_ncols; i++)
		{
//			int tid = omp_get_thread_num();
			//compute 2d row col index for dsm
			int r = (i / dsm_bbox_ncols) + (int) dsm_Urow;
			int c = (i % dsm_bbox_ncols) + (int) dsm_Ucol;

//			#ifdef DEBUG
//			std::cout << "dsm access r,c "<< r << ", "<< c << std::endl;
//			#endif

			//compute corresponding lat lon
			double i_lat, i_lon;
			podsmRaster->BackProject(r, c, &i_lat, &i_lon);

			if(demMaskFileName != "None")
			{
				double dem_mask_r, dem_mask_c;
				podemMaskRaster->Project(i_lat, i_lon, &dem_mask_r, &dem_mask_c);
				T_demMask demMaskValue;
//#ifdef DEBUG
//				std::cout << "mask access r,c "<< dem_mask_r << ", "<< dem_mask_c << " int: " << (int) floor(dem_mask_r) << ", " << (int) floor(dem_mask_c) << std::endl;
//#endif

//				if (not (InRange(dem_mask_r, 0, dem_mask_nrows) and InRange(dem_mask_c, 0, dem_mask_ncols))) // when reading from disk
				if (not (InRange(dem_mask_r, demMask_Urow, demMask_Lrow) and InRange(dem_mask_c, demMask_Ucol, demMask_Lcol)))// when readiing from partila array in memory
				{
					//numHtpts[i] = 0; // numpts is already set as zero so go to next
					continue;
				}//else get dem mask raster value

//#ifdef DEBUG
//				std::cout << "Inside mask" << std::endl;
//#endif
//				std::cout << "reading partial array" << std::endl;
				podemMaskRaster->ReadArray(dem_mask_r, dem_mask_c,  demMaskValue);
				if ( demMaskValue != 0) //if not land
				{
					//numHtpts[i] = 0; // numpts is already set as zero so go to next
					continue;
				}

			} // else find min max ht using dsm and dem

			//get max ht value using dsm
//#ifdef DEBUG
//			std::cout << "Reading max ht from dsm" << std::endl;
//#endif
			T_dem i_min_ht, i_min_ht_temp;
			T_dsm i_max_ht;
//			std::cout << "reading partial array" << std::endl;
			podsmRaster->ReadArray(r, c, i_max_ht);
			maxHt[i] = (double) i_max_ht;

			//handle no data value
			if (maxHt[i] == (double) dst_nodata)
			{
				continue;
				//numHtPts is zero
			}

			double dem_r, dem_c;
			if(demFileName != "None")
			{
				//compute dem r and c
//				double dem_r, dem_c;
				podemRaster->Project(i_lat, i_lon, &dem_r, &dem_c);

//				if ( (InRange(dem_r, 0, dem_nrows) and InRange(dem_c, 0, dem_ncols))) //when reading from disk
				if ( (InRange(dem_r, dem_Urow, dem_Lrow) and InRange(dem_c, dem_Ucol, dem_Lcol))) // when reading from partial array in memory
				{// get dem mask raster value
					//read min value from dem
//					std::cout << "reading partial array" << std::endl;
					podemRaster->ReadArray(dem_r, dem_c, i_min_ht);
					podemRaster->ReadRasterValue(dem_r, dem_c, i_min_ht_temp); // reading from disk
					//handle no data value
					if((double) i_min_ht != (double) dst_nodata)
					{
						//if not no data then set value
						minHt[i] = (double) i_min_ht - (double) dem_offset;
					}
//					else
//					{
//						minHt[i] = ground_ht; //ground_ht is double
//					}
				}
//				else
//				{
//					minHt[i] = ground_ht;
//				}
			}

			numHtpts[i] = (int) ceil((maxHt[i] - minHt[i])/Z_GRID_SPACING);
			if (numHtpts[i] < 0){numHtpts[i]=0;}//do nothing this is noise

#ifdef DEBUG
			if (numHtpts[i] < 0)
			{
			std::cout << "i : " << i << " " ;
//			std::cout << "demMaskValue: " << demMaskValue <<" ";
			std::cout << "dsm r,c: " << r << ", " << c << " ";
			if(demFileName != "None") std::cout << "dem r,c: " << dem_r << ", " << dem_c;
			std::cout << " maxHt[i] " << maxHt[i];
//			std::cout << " i_max_ht "<< i_max_ht;
			std::cout << " minHt[i] " << minHt[i];
			//for debug
			std::cout << " i_min_ht: " << i_min_ht << " i_min_ht_temp: " << i_min_ht_temp;
//			std::cout << "dem r,c " << dem_r << ", " << dem_c;
			std::cout << " partial array rowoff, coloff, rowsize, colsize: " << podemRaster->poPartialArray->rowOff << " " << podemRaster->poPartialArray->colOff << " " << podemRaster->poPartialArray->rowSize << " " << podemRaster->poPartialArray->colSize;
//			std::cout <<  " i_min_ht " << i_min_ht;
			std::cout << " numHtpts[i] " << numHtpts[i] << std::endl;
			}
#endif
		}


		//compute total 3d points
		int numTotalPts = 0;
		#pragma omp parallel for reduction( + : numTotalPts)
		for(int i=0; i < numHtpts.size(); i++  )
		{
			if(numHtpts[i] >= 0)
			{
			numTotalPts += numHtpts[i];
			}
		}
		std::cout << "Total number of 3D Points that need to be projected : " << numTotalPts << std::endl << std::endl;

		#ifdef TIMING
		end = omp_get_wtime();
		printf("Time for populating heights: %lf\n", end - start);
		start = omp_get_wtime();
		#endif

		std::cout << "Creating 3d grid" << std::endl;
		// create the 3d grid as a vector of vector
		// This is done on single cpu but we are creating only an empty vector so its not costly
		std::vector  <std::vector <Point3drc> > grid_pts3d;
		std::vector <Point3drc> *i_ht_pts3d;
		for(int i=0; i<dsm_bbox_nrows*dsm_bbox_ncols; i++)
		{
			if(numHtpts[i] >= 0)
			{
				i_ht_pts3d = new std::vector<Point3drc>(numHtpts[i]);
				grid_pts3d.push_back(*i_ht_pts3d);
			}
		}


		#ifdef TIMING
		end = omp_get_wtime();
		printf("Time for serial creation of 3d points container: %lf\n", end - start);
		start = omp_get_wtime();
		#endif

		#ifdef LOG
		std::cout << "Populating 3d grid" << std::endl;
		#endif

		#pragma omp parallel for
		for(int i=0; i<dsm_bbox_nrows*dsm_bbox_ncols; i++)
		{
//			int tid = omp_get_thread_num();

			if(numHtpts[i] <= 0)
			{
				//dont do anything for empty container
				continue;
			}
			//compute 2d row col index
			int r = i / dsm_bbox_ncols + (int) dsm_Urow;
			int c = i % dsm_bbox_ncols + (int) dsm_Ucol;

			double i_lat, i_lon;
			podsmRaster->BackProject(r, c, &i_lat, &i_lon);

			for(int j=0; j<numHtpts[i]; j++)
			{
				grid_pts3d[i][j].dsm_row = r;
				grid_pts3d[i][j].dsm_col = c;
				grid_pts3d[i][j].lat = i_lat;
				grid_pts3d[i][j].lon = i_lon;
				grid_pts3d[i][j].ht = minHt[i] + j*Z_GRID_SPACING;
			}
			//change last one to maxht
			grid_pts3d[i][numHtpts[i]-1].ht = maxHt[i];
		}


		#ifdef TIMING
		end = omp_get_wtime();
		printf("Time for populating 3d points: %lf\n", end - start);
		start = omp_get_wtime();
		#endif

		#ifdef LOG
		std::cout << "Flattening 3d grid"<< std::endl;
		#endif
		// flatten the nested vector - sequential
		// std::vector <Point3drc> flattened_grid_pts3d;
		// for (auto const &v: grid_pts3d) // time=9.5815 for debug test case 2000*2000
		// {
		// 	flattened_grid_pts3d.insert(flattened_grid_pts3d.end(), v.begin(), v.end());
		// }

		
		// parallelized flattening
		std::vector <Point3drc> flattened_grid_pts3d;
		size_t grid_size = grid_pts3d.size();
		#pragma omp parallel
		{
			std::vector<Point3drc> private_flattened_grid_pts3d;

			#pragma omp for schedule(static)
			for (int i = 0; i < grid_size; i++)
			{
				for (int j = 0; j < grid_pts3d[i].size(); j++)
				{
					private_flattened_grid_pts3d.push_back(grid_pts3d[i][j]);
				}
			}

			#pragma omp critical
			{
				flattened_grid_pts3d.insert(flattened_grid_pts3d.end(),
											private_flattened_grid_pts3d.begin(),
											private_flattened_grid_pts3d.end());
			}
		}
		

		#ifdef TIMING
		end = omp_get_wtime();
		printf("Time for serial flattening of 3d points: %lf\n", end - start);
		start = omp_get_wtime();
		#endif

		#ifdef LOG
		std::cout << "writing the depth image"<< std::endl;
		#endif

		bool writeSuccess=false, createSuccess=false;
		GDALDataType outputDataType;
		outputDataType = podsmRaster->GetDataType();

		#ifdef LOG
		std::cout << "creating empty depth image"<< std::endl;
		#endif

		std::vector<T_dsm> outputArray_ht(out_nrows*out_ncols, (T_dsm) dst_nodata);
		std::vector<T_latlon> outputArray_lat(out_nrows*out_ncols, (T_latlon) dst_nodata);
		std::vector<T_latlon> outputArray_lon(out_nrows*out_ncols, (T_latlon) dst_nodata);


		#ifdef LOG
		std::cout << "writing max depth values"<< std::endl;
		#endif
		createSuccess = CreateOutput(porpcRaster, start_row, start_col,
									 end_row, end_col, flattened_grid_pts3d,
									 out_nrows, out_ncols,
									 outputArray_ht, outputArray_lat, outputArray_lon,
									 dst_nodata);

		#ifdef TIMING
		end = omp_get_wtime();
		printf("Time for computing max depth values: %lf\n", end - start);
		start = omp_get_wtime();
		#endif

		//merge the three into a single vector: band order lat, lon, ht
//		outputArray_lat.insert(outputArray_lat.end(), outputArray_lon.begin(), outputArray_lon.end());
//		outputArray_lat.insert(outputArray_lat.end(), outputArray_ht.begin(), outputArray_ht.end());

		#ifdef TIMING
		end = omp_get_wtime();
		printf("Time for serial combination of lat, lon, depth values: %lf\n", end - start);
		start = omp_get_wtime();
		#endif

		#ifdef LOG
		std::cout << "writing the lat, lon, depth image as GTiff"<< std::endl;
		#endif

		writeSuccess = WriteGTiff(outputRasterName + "_depth.tif", out_ncols, out_nrows,
								  1, outputDataType, outputArray_ht, (T_dsm) dst_nodata);

		writeSuccess = WriteGTiff(outputRasterName + "_lat.tif", out_ncols, out_nrows,
								  1, GDT_Float64, outputArray_lat, (T_latlon) dst_nodata);

		writeSuccess = WriteGTiff(outputRasterName + "_lon.tif", out_ncols, out_nrows,
									  1, GDT_Float64, outputArray_lon, (T_latlon) dst_nodata);


		#ifdef TIMING
		end = omp_get_wtime();
		printf("Time for writing GTiff: %lf\n", end - start);
		printf("Total Time for depthipp: %lf\n", end - total_start);
//		start = omp_get_wtime();
		#endif


		if(!writeSuccess || !createSuccess)
		{
			std::cout << "ERROR cannot write output file" << std::endl;
			exit(1);
		}

		//clean-up:
		//delete rasters
		delete podemRaster, porpcRaster, podsmRaster, podemMaskRaster;

		return 0;
	}
	catch(std::exception &err)
	{
		std::cout << "Unhandled Exception " << err.what() << ", for file: "<< outputRasterName << " Exiting " << std::endl;
	    const boost::stacktrace::stacktrace* st = boost::get_error_info<traced>(err);
	    if (st) {
	        std::cout << *st << '\n';
	    }
		return 1;
	}

}


int main(int argc, char **argv)
{
	//set precision of std::cout
	std::cout.precision(17);

	//declare input args
	std::string outputRasterName;
	std::string inputRasterName; std::string rpcRaster_interp;
	int start_row; // uy
	int start_col; // ux
	int end_row;   // ly
	int end_col;   // lx
	double UL_lat; double UL_lon;
	double UR_lat; double UR_lon;
	double LL_lat; double LL_lon;
	double LR_lat; double LR_lon;
	std::string dsmFileName; std::string dsm_interp;
	std::string demFileName; std::string dem_interp; int dem_offset;
	std::string demMaskFileName; std::string demMask_interp;
//	double dst_nodata;
	int BUFFER_SIZE;
	double Z_GRID_SPACING;
	int numThreads;

	// parse args from command line
	try
	{	//create parser using boost
		namespace po = boost::program_options;
		po::options_description desc("Allowed options");
		desc.add_options()
				("help", "Help for depthifypp")
				("output", po::value<std::string>(&outputRasterName)->required(), "Output raster base name")
				("input", po::value<std::string>(&inputRasterName)->required(), "Input raster file name with rpc at same location or written in image")
				("input_interp", po::value<std::string>(&rpcRaster_interp)->default_value("bilinear"), "Input raster interp type (bilinear or near)")
				("start_row", po::value<int>(&start_row)->required(), "starting row index for the block")
				("end_row", po::value<int>(&end_row)->required(), "ending row index for the block")
				("start_col", po::value<int>(&start_col)->required(), "starting col index for the block")
				("end_col", po::value<int>(&end_col)->required(), "ending col index for the block")

				("UL_lat", po::value<double>(&UL_lat)->required(), "Upper Left lat for the block")
				("UL_lon", po::value<double>(&UL_lon)->required(), "Upper Left lon for the block")
				("UR_lat", po::value<double>(&UR_lat)->required(), "Upper Right lat for the block")
				("UR_lon", po::value<double>(&UR_lon)->required(), "Upper Right lon for the block")
				("LL_lat", po::value<double>(&LL_lat)->required(), "Upper Left lat for the block")
				("LL_lon", po::value<double>(&LL_lon)->required(), "Upper Left lon for the block")
				("LR_lat", po::value<double>(&LR_lat)->required(), "Lower Right lat for the block")
				("LR_lon", po::value<double>(&LR_lon)->required(), "Lower Right lon for the block")

				("dsm_file", po::value<std::string>(&dsmFileName)->required(), "high-res DSM file name to be used as height source")
				("dsm_interp", po::value<std::string>(&dsm_interp)->default_value("bilinear"), "high-res DSM interpolation type (bilinear or near)")
				("BUFFER_SIZE", po::value<int>(&BUFFER_SIZE)->required(), "Buffer size for increasing footprint on dsm")
				("Z_GRID_SPACING", po::value<double>(&Z_GRID_SPACING)->required(), "spacing for Z grid")
				("dem_file", po::value<std::string>(&demFileName)->default_value("None"), "low-res DEM file name to be used as rough height source")
				("dem_interp", po::value<std::string>(&dem_interp)->default_value("bilinear"), "low-res DEM interpolation type (bilinear or near)")
				("dem_offset", po::value<int>(&dem_offset)->default_value(50), "offset to be substracted for dem ht")
				("dem_mask_file", po::value<std::string>(&demMaskFileName)->default_value("None"), "optional DEM mask file name")
				("dem_mask_interp", po::value<std::string>(&demMask_interp)->default_value("near"), "dem mask interpolation type (bilinear or near)")
				("num_threads", po::value<int>(&numThreads)->default_value(-1), "optional number of openMP threads");

		try{//parse the command line
			po::command_line_parser parser{argc, argv};
			parser.options(desc)
				   .style(
					po::command_line_style::unix_style |
					po::command_line_style::allow_long_disguise
					);
			po::parsed_options parsed_options = parser.run();

			po::variables_map vm;
			po::store(parsed_options, vm);
			po::notify(vm);
		}
		catch(po::error& err) // parsing error
		{
			std::cerr << "ERROR: " << err.what() << std::endl << std::endl;
			std::cerr << desc << std::endl;
			return 1;
		}
	}
	catch(std::exception& err) // std exception
	{
		std::cerr << "Unhandled Exception " << err.what() << ", Exiting " << std::endl;
		return 1;
	}

	//run worker function
	if(numThreads == -1)
	{ 	//user did not set this
		numThreads = omp_thread_count();
	}

	std::cout << "using numThreads : "<<numThreads << std::endl;

	bool workerFailed = do_block_work
			(outputRasterName,
			 inputRasterName, rpcRaster_interp,
			 start_row, end_row,
			 start_col, end_col,
			 UL_lat, UL_lon,
			 UR_lat, UR_lon,
			 LL_lat, LL_lon,
			 LR_lat, LR_lon,
			 dsmFileName, dsm_interp,
			 demFileName, dem_interp, dem_offset,
			 demMaskFileName, demMask_interp,
			 BUFFER_SIZE,
			 Z_GRID_SPACING,
			 numThreads);

	if(workerFailed)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}
