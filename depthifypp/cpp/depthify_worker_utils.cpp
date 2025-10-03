#include "depthify_worker_utils.hpp"

using namespace worker_utils;


//---------------Partial Array ------------------
template<typename Timg>
void PartialArray<Timg>::at(int row, int col, Timg &val){
	//read value at given row col
	int idx1d = row*this->colSize + col;
	val = this->poArray[idx1d];
}

//--------------- Raster ------------------------
template <typename Timg, typename Tidx>
Raster<Timg, Tidx>::Raster(){}

template <typename Timg, typename Tidx>
Raster<Timg, Tidx>::Raster(std::string raster_file, std::string interp_type): raster_file(raster_file), interp_type(interp_type)
{
	this->Open(raster_file);
}

template <typename Timg, typename Tidx>
std::string Raster<Timg, Tidx>::GetRasterFile()
{
	return this->raster_file;
}

template <typename Timg, typename Tidx>
std::string Raster<Timg, Tidx>::GetInterpType()
{
	return this->interp_type;
}

template <typename Timg, typename Tidx>
void Raster<Timg, Tidx>::SetRasterFile(std::string raster_file)
{
	this->raster_file = raster_file;
}

template <typename Timg, typename Tidx>
void Raster<Timg, Tidx>::Open(int band)
{
	this->poRasterDataset = (GDALDataset *) GDALOpen(this->raster_file.c_str(), GA_ReadOnly );
	if(this->poRasterDataset == NULL )
	{
		std::cout<< "ERROR IN OPENING RASTER DATASET" << std::endl;
		abort();
	}
	this->nbands = this->poRasterDataset->GetRasterCount();

	this->poRasterBand = this->poRasterDataset->GetRasterBand(band);
	if(this->poRasterDataset == NULL )
	{
		std::cout<< "ERROR IN OPENING RASTER BAND" << std::endl;
		abort();
	}

	dtype = this->poRasterBand->GetRasterDataType();
	xsize =  this->poRasterDataset->GetRasterXSize();
	ysize =  this->poRasterDataset->GetRasterYSize();
	noDataVal = this->poRasterBand->GetNoDataValue();
}

template <typename Timg, typename Tidx>
void Raster<Timg, Tidx>::Open(std::string raster_file, int band)
{
	this->SetRasterFile(raster_file);
	this->Open(band);
}

template <typename Timg, typename Tidx>
int Raster<Timg, Tidx>::GetRasterXSize()
{
	// number of cols
	return xsize;
}

template <typename Timg, typename Tidx>
int Raster<Timg, Tidx>::GetRasterYSize()
{
	//number of rows
	return ysize;
}

template <typename Timg, typename Tidx>
Timg Raster<Timg, Tidx>::GetNoDataValue()
{
	return this->noDataVal;
}

template <typename Timg, typename Tidx>
GDALDataType Raster<Timg, Tidx>::GetDataType()
{
	return this->dtype;
}

template <typename Timg, typename Tidx>
int Raster<Timg, Tidx>::ReadRasterValue_int(int row, int col, Timg &val)
{
	// read value at given pixel location
	int readFailed;
	readFailed = this->poRasterBand->RasterIO(GF_Read, col, row, 1, 1, &val, 1, 1, this->dtype, 0, 0);
//	std::cout << "disk arrayRow, col: "<< row << " " << col <<" val: "<< val<< std::endl;

	return readFailed;
}

template <typename Timg, typename Tidx>
int Raster<Timg, Tidx>::ReadRasterValue(Tidx row, Tidx col, Timg &val)
{	// Can only do bilinear and nearest interp
	// read value at given pixel location
	int readFailed;

	if (this->interp_type == "near")
	{
		return this->ReadRasterValue_int((int) floor(row), (int) floor(col), val);
	}

	// compute upper and lower int pixels
	double row0, col0, row1, col1;
	row0 = floor(row);
	col0 = floor(col);
	row1 = row0 + 1;
	col1 = col0 + 1;

	double drow = row - row0; //decimal values for row col
	double dcol = col - col0;

	//check for border pixels
	if (row0 < 0) row0 = row1;
	if (col0 < 0) col0 = col1;
	if (row1 >= ysize) row1 = row0;
	if (col1 >= xsize) col1 = col0;

	// get value at the int pixels
	Timg imgr0c0, imgr1c0, imgr0c1, imgr1c1;
	readFailed = this->ReadRasterValue_int( (int) row0, (int) col0, imgr0c0);
	readFailed = this->ReadRasterValue_int( (int) row1, (int) col0, imgr1c0);
	readFailed = this->ReadRasterValue_int( (int) row0, (int) col1, imgr0c1);
	readFailed = this->ReadRasterValue_int( (int) row1, (int) col1, imgr1c1);

	//handle nodata
	if (imgr0c0 == this->noDataVal) imgr0c0 = 0.;
	if (imgr1c0 == this->noDataVal) imgr1c0 = 0.;
	if (imgr0c1 == this->noDataVal) imgr0c1 = 0.;
	if (imgr1c1 == this->noDataVal) imgr1c1 = 0.;

	// interpolate the value
	val = (Timg) (
				( (double) (imgr0c0) ) * ( 1. - drow ) * ( 1. - dcol ) +
				( (double) (imgr1c0) ) * ( drow ) * ( 1. - dcol ) +
				( (double) (imgr0c1) ) * ( 1. - drow ) * ( dcol ) +
				( (double) (imgr1c1) ) * ( drow ) * ( dcol )
			);

	return readFailed;
}

template <typename Timg, typename Tidx>
void Raster<Timg, Tidx>::ReadAsArray(int _rowOff, int _colOff, int _rowSize, int _colSize)
{
	int rowOff = _rowOff;
	int colOff = _colOff;
	int rowSize = _rowSize;
	int colSize = _colSize;

	// allocate partial Array struct
	poPartialArray = new PartialArray<Timg>;

	//do sanity check: check if projected window is inside the image?
	if(_rowOff<0) rowOff = 0;
	if(_colOff<0) colOff = 0;
	if(_rowOff + _rowSize > this->ysize) rowSize = this->ysize - rowOff;
	if(_colOff + _colSize > this->xsize) colSize = this->xsize - colOff;

	//populate partialArray struct fields
	poPartialArray->rowOff = rowOff;
	poPartialArray->colOff = colOff;
	poPartialArray->rowSize = rowSize;
	poPartialArray->colSize = colSize;

	//allocate and read partial array into the struct
	poPartialArray->poArray = new Timg[rowSize*colSize];

	int readFailed;
	readFailed = this->poRasterBand->RasterIO(GF_Read, colOff, rowOff, colSize, rowSize, 			poPartialArray->poArray, colSize, rowSize, this->dtype, 0, 0);
	return;
}

template <typename Timg, typename Tidx>
void Raster<Timg, Tidx>::ReadAsArray()
{
	//read whole array
	int rowOff = 0;
	int colOff = 0;
	int rowSize = this->ysize;
	int colSize = this->xsize;

	this->ReadAsArray(rowOff, colOff, rowSize, colSize);

	return;
}

template <typename Timg, typename Tidx>
int Raster<Timg, Tidx>::ReadArray_int(int row, int col, Timg &val)
{
	//read from partial array without any interpolation
	// compute array row and col
	int arrayRow = row - poPartialArray->rowOff;
	int arrayCol = col - poPartialArray->colOff;
	poPartialArray->at(arrayRow, arrayCol, val);
//	std::cout << "RAM row, col: " << row << " " << col << " arrayRow, col: "<< arrayRow << " " << arrayCol <<" val: "<< val<< std::endl;
//	std::cout << " partial array rowoff, coloff, rowsize, colsize: " << poPartialArray->rowOff << " " << poPartialArray->colOff << " " << poPartialArray->rowSize << " " << poPartialArray->colSize <<std::endl;
	return 0;
}

template <typename Timg, typename Tidx>
int Raster<Timg, Tidx>::ReadArray(Tidx row, Tidx col, Timg &val)
{ 	// Can only do bilinear and nearest interp
	// read value at given pixel location
	int readFailed;

	if (this->interp_type == "near")
	{
		return this->ReadArray_int((int) floor(row), (int) floor(col), val);
	}

	// compute upper and lower int pixels
	double row0, col0, row1, col1;
	row0 = floor(row);
	col0 = floor(col);
	row1 = row0 + 1;
	col1 = col0 + 1;

	double drow = row - row0; //decimal values for row col
	double dcol = col - col0;

	//check for border pixels
	if (row0 < poPartialArray->rowOff ) row0 = row1;
	if (col0 < poPartialArray->colOff ) col0 = col1;
	if (row1 >= poPartialArray->rowOff + poPartialArray->rowSize) row1 = row0;
	if (col1 >= poPartialArray->colOff + poPartialArray->colSize) col1 = col0;

	// get value at the int pixels
	Timg imgr0c0, imgr1c0, imgr0c1, imgr1c1;
	readFailed = this->ReadArray_int( (int) row0, (int) col0, imgr0c0);
	readFailed = this->ReadArray_int( (int) row1, (int) col0, imgr1c0);
	readFailed = this->ReadArray_int( (int) row0, (int) col1, imgr0c1);
	readFailed = this->ReadArray_int( (int) row1, (int) col1, imgr1c1);

	//handle nodata
	if (imgr0c0 == this->noDataVal) imgr0c0 = 0.;
	if (imgr1c0 == this->noDataVal) imgr1c0 = 0.;
	if (imgr0c1 == this->noDataVal) imgr0c1 = 0.;
	if (imgr1c1 == this->noDataVal) imgr1c1 = 0.;

	// interpolate the value
	val = (Timg) (
				( (double) (imgr0c0) ) * ( 1. - drow ) * ( 1. - dcol ) +
				( (double) (imgr1c0) ) * ( drow ) * ( 1. - dcol ) +
				( (double) (imgr0c1) ) * ( 1. - drow ) * ( dcol ) +
				( (double) (imgr1c1) ) * ( drow ) * ( dcol )
			);

	return readFailed;
}


template <typename Timg, typename Tidx>
Raster<Timg, Tidx>::~Raster()
{
//	std::cout << "Trying to delete Raster obj" <<std::endl;
//	std::cout << "dataset pointer: " << this->poRasterDataset << std::endl;
	if (this->poRasterDataset != NULL)
	{
//		std::cout << "dataset pointer: " << this->poRasterDataset << std::endl;
		GDALClose(this->poRasterDataset);
		this->poRasterDataset= NULL;
		delete this->poRasterDataset;
		this->poRasterBand = NULL;
		delete this->poRasterBand;
		// delete partial array
		if (this->poPartialArray != NULL)
		{
		delete this->poPartialArray->poArray;
		delete this->poPartialArray; // pointer to struct
		}

	}
}

//--------------- OrthoRaster ------------------------
template <typename Timg, typename Tidx>
OrthoRaster<Timg, Tidx>::OrthoRaster(): Raster<Timg, Tidx>(), geoTransform(6,0){}

template <typename Timg, typename Tidx>
OrthoRaster<Timg, Tidx>::OrthoRaster(std::string raster_file, std::string interp_type): Raster<Timg, Tidx>(raster_file, interp_type), geoTransform(6,0)
{
	this->poRasterDataset->GetGeoTransform( &( (this->geoTransform)[0] ) );
}

template <typename Timg, typename Tidx>
void OrthoRaster<Timg, Tidx>::Project(double lat, double lon, Tidx* porow, Tidx* pocol)
{
	//lon is X, lat is Y
	*porow = (lat - this->geoTransform[3])/this->geoTransform[5]; //X
	*pocol = (lon - this->geoTransform[0])/this->geoTransform[1]; //Y
}

template <typename Timg, typename Tidx>
void OrthoRaster<Timg, Tidx>::BackProject(int row, int col, double* lat, double* lon)
{
	*lon = this->geoTransform[0] + this->geoTransform[1]*col + this->geoTransform[2]*row; //X
	*lat = this->geoTransform[3] + this->geoTransform[5]*row + this->geoTransform[4]*col; //Y
}

template <typename Timg, typename Tidx>
OrthoRaster<Timg, Tidx>::~OrthoRaster()
{
//	delete geoTransform;
//	std::cout << "Trying to delete OrthoRaster obj" <<std::endl;
}

//--------------- RPCRaster ------------------------
template <typename Timg, typename Tidx>
RPCRaster<Timg, Tidx>::RPCRaster(): Raster<Timg, Tidx>(){}

template <typename Timg, typename Tidx>
RPCRaster<Timg, Tidx>::RPCRaster(std::string raster_file, std::string interp_type): Raster<Timg, Tidx>(raster_file, interp_type)
{
	char *papszOptions = NULL;
	this->po_rpc_info = new GDALRPCInfo;

	if( !GDALExtractRPCInfo(this->poRasterDataset->GetMetadata( "RPC" ), this->po_rpc_info) )
	{
		std::cout << " ERROR: No rpc metadata found" << std::endl;
		abort();
	}

	// Create rpc transformer. Note gdal's convention is opposite to normal rpc. So by default
	// it maps pixel line sample to lon, lat
	/*GDALCreateRPCTransformer( GDALRPCInfo *psRPCInfo, int bReversed,
	 * 							 double dfPixErrThreshold, char **papszOptions )*/
	this->rpc_transformer = GDALCreateRPCTransformer(this->po_rpc_info, 0, 0, &papszOptions);

}

template <typename Timg, typename Tidx>
void RPCRaster<Timg, Tidx>::Project(int nPointCount, double &lat, double &lon, double &ht, int &success_flag)
{
	//GDALRPCTransform : takes in a array of lat, lon, ht and writes the projection (line,samp) into the passed array lat, lon
	// all gdal fucntions take args as X,Y ie lon, lat
	GDALRPCTransform(this->rpc_transformer, 1, nPointCount, &lon, &lat, &ht, &success_flag);
}

template <typename Timg, typename Tidx>
RPCRaster<Timg, Tidx>::~RPCRaster()
{
//	std::cout << "Trying to delete RPCRaster obj" <<std::endl;
	delete po_rpc_info;
	GDALDestroyRPCTransformer(rpc_transformer);
	rpc_transformer = NULL;
//	delete rpc_transformer;
}


