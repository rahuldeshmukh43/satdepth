#ifndef DEPTHIFY_WORKER_UTILS_HPP
#define DEPTHIFY_WORKER_UTILS_HPP

#include <iostream>

#include <string>	// for string datatype
#include <vector>

// gdal based includes
#include "gdal_priv.h" // for common GDAL functions
#include "gdal_alg.h" // for GDALCreateRPCTransformer and GDALRPCTransform function
#include "cpl_conv.h" // for CPLMalloc()


namespace worker_utils
{
	template<typename Timg>
	struct PartialArray
	{
		int rowOff;
		int colOff;
		int rowSize;
		int colSize;
		Timg *poArray; // pointer to 1D partial array
		void at(int row, int col, Timg &val);
	};


	template<typename Timg, typename Tidx>
	class Raster
	{
		int xsize; //ncols
		int ysize; //nrows
		Timg noDataVal;

	public:

		std::string raster_file; //path to file
		std::string interp_type; // bilinear or nearest;
		GDALDataset* poRasterDataset= NULL; //pointer to raster dataset
		GDALRasterBand* poRasterBand= NULL;
		GDALDataType dtype;
		struct PartialArray<Timg> *poPartialArray = NULL;


		//attributes
		int nbands;

		//constructors
		Raster();
		Raster(std::string raster_file, std::string interp_type);


		//accessors
		std::string GetRasterFile();
		std::string GetInterpType();
		int GetRasterXSize();
		int GetRasterYSize();
		Timg GetNoDataValue();
		GDALDataType GetDataType();

		//mutators
		void SetRasterFile(std::string raster_file);

		//methods
		void Open(int band=1);
		void Open(std::string raster_file, int band=1);

		//read raster from disk
		int ReadRasterValue(Tidx row, Tidx col, Timg &val);
		int ReadRasterValue_int(int row, int col, Timg &val);

		//read by loading partial array into memory
		void ReadAsArray(); //read whole array
		void ReadAsArray(int rowOff, int colOff, int rowSize, int colSize); //load partial array into memory for quick access
		int ReadArray_int(int row, int col, Timg &val);
		int ReadArray(Tidx row, Tidx col, Timg &val);

		//destructors
		virtual ~Raster();
	};

	template <typename Timg, typename Tidx>
	class OrthoRaster: public Raster<Timg, Tidx>
	{

		std::vector<double> geoTransform; //length 6 vector

	public:
		//attributes


		//constructor
		OrthoRaster();
		OrthoRaster(std::string raster_file, std::string interp_type);

		//methods
		void Project(double lat, double lon, Tidx* row, Tidx* col);
		void BackProject(int row, int col, double* lat, double* lon);

		//destructors
		virtual ~OrthoRaster();

	};

	template <typename Timg, typename Tidx>
	class RPCRaster: public Raster<Timg, Tidx>
	{
		void *rpc_transformer;

	public:
		//atributes
		GDALRPCInfo *po_rpc_info;

		//constructor
		RPCRaster();
		RPCRaster(std::string raster_file, std::string interp_type);

		//methods
		//project - inplace writes row->lat, col->lon
		void Project(int nPointCount, double &lat, double &lon, double &ht, int &success_flag);

		//destructors
		virtual ~RPCRaster();

	};

}

#endif
