#ifndef __GlobalTypes__
#define __GlobalTypes__

#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <Eigen/Core>
#include <Eigen/StdVector>

#include <DBoW3/DBoW3.h>
#include "sophus/se3.hpp"
#include "sophus/sim3.hpp"

#define SCALE_IDEPTH 1.0f // scales internal value to idepth.
#define SCALE_XI_ROT 1.0f
#define SCALE_XI_TRANS 0.5f
#define SCALE_F 50.0f
#define SCALE_C 50.0f
#define SCALE_W 1.0f
#define SCALE_A 10.0f
#define SCALE_B 1000.0f

#define SCALE_IDEPTH_INVERSE (1.0f / SCALE_IDEPTH)
#define SCALE_XI_ROT_INVERSE (1.0f / SCALE_XI_ROT)
#define SCALE_XI_TRANS_INVERSE (1.0f / SCALE_XI_TRANS)
#define SCALE_F_INVERSE (1.0f / SCALE_F)
#define SCALE_C_INVERSE (1.0f / SCALE_C)
#define SCALE_W_INVERSE (1.0f / SCALE_W)
#define SCALE_A_INVERSE (1.0f / SCALE_A)
#define SCALE_B_INVERSE (1.0f / SCALE_B)

#define SOLVER_SVD (int)1
#define SOLVER_ORTHOGONALIZE_SYSTEM (int)2
#define SOLVER_ORTHOGONALIZE_POINTMARG (int)4
#define SOLVER_ORTHOGONALIZE_FULL (int)8
#define SOLVER_SVD_CUT7 (int)16
#define SOLVER_REMOVE_POSEPRIOR (int)32
#define SOLVER_USE_GN (int)64
#define SOLVER_FIX_LAMBDA (int)128
#define SOLVER_ORTHOGONALIZE_X (int)256
#define SOLVER_MOMENTUM (int)512
#define SOLVER_STEPMOMENTUM (int)1024
#define SOLVER_ORTHOGONALIZE_X_LATER (int)2048

#if ThreadCount != 0              //Number of thread is set from cmake, cant use for some built in std::thread to detect number of cores!
#define NUM_THREADS (ThreadCount) // const int NUM_THREADS = boost::thread::hardware_concurrency();
#else
#define NUM_THREADS (6)
#endif

namespace FSLAM
{

enum Dataset
{
    Tum_mono = 0,
    Euroc,
    Kitti,
    Emptyd
};
enum Sensor
{
    Monocular = 0,
    Stereo,
    RGBD,
    Emptys
};
enum PhotoUnDistMode
{
    HaveCalib = 0,
    OnlineCalib,
    NoCalib,
    Emptyp
};

enum ResState {IN=0, OOB, OUTLIER};

struct ImageData
{
public:
    cv::Mat cvImgL;
    cv::Mat cvImgR;
    double timestamp;
    float ExposureL;
    float ExposureR;

    std::vector<float> fImgL;
    std::vector<float> fImgR;

    inline void deepCopy(ImageData &NewData) //deep copy was never tested!!
    {
        if (!cvImgL.empty())
        {
            cvImgL.copyTo(NewData.cvImgL);
            memcpy(&NewData.fImgL[0], &fImgL[0], cvImgL.cols * cvImgL.rows * sizeof(float));
        }

        if (!cvImgR.empty())
        {
            cvImgR.copyTo(NewData.cvImgR);
            memcpy(&NewData.fImgL[0], &fImgL[0], cvImgR.rows * cvImgR.cols * sizeof(float));
        }
        NewData.timestamp = timestamp;
        NewData.ExposureL = ExposureL;
        NewData.ExposureR = ExposureR;
    }
    ImageData(int width, int height)
    {
        int size = width * height;
        fImgL.resize(size);
        fImgR.resize(size);
    }
    ~ImageData() {}
};

typedef DBoW3::Vocabulary ORBVocabulary;

typedef Sophus::SE3d SE3;
typedef Sophus::SO3d SO3;
typedef Sophus::Sim3d Sim3;

const int CPARS = 4;             // number of camera calibration parameters
const int MAX_RES_PER_POINT = 8; // number of residuals in each point, see dso's paper for the pattern

// double matricies
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;
typedef Eigen::Matrix<double, CPARS, CPARS> MatCC;
typedef Eigen::Matrix<double, CPARS, 10> MatC10;
typedef Eigen::Matrix<double, 10, 10> Mat1010;
typedef Eigen::Matrix<double, 13, 13> Mat1313;
typedef Eigen::Matrix<double, 8, 10> Mat810;
typedef Eigen::Matrix<double, 8, 3> Mat83;
typedef Eigen::Matrix<double, 6, 6> Mat66;
typedef Eigen::Matrix<double, 5, 3> Mat53;
typedef Eigen::Matrix<double, 4, 3> Mat43;
typedef Eigen::Matrix<double, 4, 2> Mat42;
typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, 2, 2> Mat22;
typedef Eigen::Matrix<double, 8, CPARS> Mat8C;
typedef Eigen::Matrix<double, CPARS, 8> MatC8;
typedef Eigen::Matrix<double, 8, 8> Mat88;
typedef Eigen::Matrix<double, 7, 7> Mat77;
typedef Eigen::Matrix<double, 4, 9> Mat49;
typedef Eigen::Matrix<double, 8, 9> Mat89;
typedef Eigen::Matrix<double, 9, 4> Mat94;
typedef Eigen::Matrix<double, 9, 8> Mat98;
typedef Eigen::Matrix<double, 8, 1> Mat81;
typedef Eigen::Matrix<double, 1, 8> Mat18;
typedef Eigen::Matrix<double, 9, 1> Mat91;
typedef Eigen::Matrix<double, 1, 9> Mat19;
typedef Eigen::Matrix<double, 8, 4> Mat84;
typedef Eigen::Matrix<double, 4, 8> Mat48;
typedef Eigen::Matrix<double, 4, 4> Mat44;
typedef Eigen::Matrix<double, 14, 14> Mat1414;
typedef Eigen::Matrix<double, 8 + CPARS + 1, 8 + CPARS + 1> MatPCPC;

// float matricies
typedef Eigen::Matrix<float, 3, 3> Mat33f;
typedef Eigen::Matrix<float, 10, 3> Mat103f;
typedef Eigen::Matrix<float, 2, 2> Mat22f;
typedef Eigen::Matrix<float, 3, 1> Vec3f;
typedef Eigen::Matrix<float, 2, 1> Vec2f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;
typedef Eigen::Matrix<float, 8, CPARS> Mat8Cf;
typedef Eigen::Matrix<float, CPARS, 8> MatC8f;
typedef Eigen::Matrix<float, 1, 8> Mat18f;
typedef Eigen::Matrix<float, 6, 6> Mat66f;
typedef Eigen::Matrix<float, 8, 8> Mat88f;
typedef Eigen::Matrix<float, 8, 4> Mat84f;
typedef Eigen::Matrix<float, 6, 6> Mat66f;
typedef Eigen::Matrix<float, 4, 4> Mat44f;
typedef Eigen::Matrix<float, 12, 12> Mat1212f;
typedef Eigen::Matrix<float, 13, 13> Mat1313f;
typedef Eigen::Matrix<float, 10, 10> Mat1010f;
typedef Eigen::Matrix<float, 9, 9> Mat99f;
typedef Eigen::Matrix<float, 4, 2> Mat42f;
typedef Eigen::Matrix<float, 6, 2> Mat62f;
typedef Eigen::Matrix<float, 1, 2> Mat12f;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatXXf;
typedef Eigen::Matrix<float, 8 + CPARS + 1, 8 + CPARS + 1> MatPCPCf;
typedef Eigen::Matrix<float, 14, 14> Mat1414f;

// double vectors
typedef Eigen::Matrix<double, CPARS, 1> VecC;
typedef Eigen::Matrix<double, 14, 1> Vec14;
typedef Eigen::Matrix<double, 13, 1> Vec13;
typedef Eigen::Matrix<double, 10, 1> Vec10;
typedef Eigen::Matrix<double, 9, 1> Vec9;
typedef Eigen::Matrix<double, 8, 1> Vec8;
typedef Eigen::Matrix<double, 7, 1> Vec7;
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 5, 1> Vec5;
typedef Eigen::Matrix<double, 4, 1> Vec4;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;
typedef Eigen::Matrix<double, 8 + CPARS + 1, 1> VecPC;

// float vectors
typedef Eigen::Matrix<float, CPARS, 1> VecCf;
typedef Eigen::Matrix<float, MAX_RES_PER_POINT, 1> VecNRf;
typedef Eigen::Matrix<float, 12, 1> Vec12f;
typedef Eigen::Matrix<float, 8, 1> Vec8f;
typedef Eigen::Matrix<float, 10, 1> Vec10f;
typedef Eigen::Matrix<float, 4, 1> Vec4f;
typedef Eigen::Matrix<float, 12, 1> Vec12f;
typedef Eigen::Matrix<float, 13, 1> Vec13f;
typedef Eigen::Matrix<float, 9, 1> Vec9f;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VecXf;
typedef Eigen::Matrix<float, 8 + CPARS + 1, 1> VecPCf;
typedef Eigen::Matrix<float, 14, 1> Vec14f;

// unsigned char vectors
typedef Eigen::Matrix<unsigned char, 3, 1> Vec3b;

// Vector of Eigen vectors
typedef std::vector<Vec2, Eigen::aligned_allocator<Vec2>> VecVec2;
typedef std::vector<Vec3, Eigen::aligned_allocator<Vec3>> VecVec3;
typedef std::vector<Vec2f, Eigen::aligned_allocator<Vec2f>> VecVec2f;
typedef std::vector<Vec3f, Eigen::aligned_allocator<Vec3f>> VecVec3f;

static std::vector<std::vector<std::vector<int>>> staticPattern {

        {{0,0}, 	  {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},	// .
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-1},	  {-1,0},	   {0,0},	    {1,0},	     {0,1}, 	  {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},	// +
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{-1,-1},	  {1,1},	   {0,0},	    {-1,1},	     {1,-1}, 	  {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},	// x
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{-1,-1},	  {-1,0},	   {-1,1},		{-1,0},		 {0,0},		  {0,1},	   {1,-1},		{1,0},		 {1,1},       {-100,-100},	// full-tight
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {-100,-100},	// full-spread-9
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {-2,-2},   // full-spread-13
		 {-2,2},      {2,-2},      {2,2},       {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{-2,-2},     {-2,-1}, {-2,-0}, {-2,1}, {-2,2}, {-1,-2}, {-1,-1}, {-1,-0}, {-1,1}, {-1,2}, 										// full-25
		 {-0,-2},     {-0,-1}, {-0,-0}, {-0,1}, {-0,2}, {+1,-2}, {+1,-1}, {+1,-0}, {+1,1}, {+1,2},
		 {+2,-2}, 	  {+2,-1}, {+2,-0}, {+2,1}, {+2,2}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {-2,-2},   // full-spread-21
		 {-2,2},      {2,-2},      {2,2},       {-3,-1},     {-3,1},      {3,-1}, 	   {3,1},       {1,-3},      {-1,-3},     {1,3},
		 {-1,3},      {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{0,2},		 {-100,-100}, {-100,-100},	// 8 for SSE efficiency
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{-4,-4},     {-4,-2}, {-4,-0}, {-4,2}, {-4,4}, {-2,-4}, {-2,-2}, {-2,-0}, {-2,2}, {-2,4}, 										// full-45-SPREAD
		 {-0,-4},     {-0,-2}, {-0,-0}, {-0,2}, {-0,4}, {+2,-4}, {+2,-2}, {+2,-0}, {+2,2}, {+2,4},
		 {+4,-4}, 	  {+4,-2}, {+4,-0}, {+4,2}, {+4,4}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200},
		 {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}}
         
         };

static int staticPatternNum[10] = {
		1,
		5,
		5,
		9,
		9,
		13,
		25,
		21,
		8,
		25
};

static int staticPatternPadding[10] = {
		1,
		1,
		1,
		1,
		2,
		2,
		2,
		3,
		2,
		4
};

struct AffLight
{
	AffLight(double a_, double b_) : a(a_), b(b_) {};
	AffLight() : a(0), b(0) {};

	// Affine Parameters:
	double a,b;	// I_frame = exp(a)*I_global + b. // I_global = exp(-a)*(I_frame - b).

	static Vec2 fromToVecExposure(float exposureF, float exposureT, AffLight g2F, AffLight g2T)
	{
		if(exposureF==0 || exposureT==0)
		{
			exposureT = exposureF = 1;
			//printf("got exposure value of 0! please choose the correct model.\n");
			//assert(setting_brightnessTransferFunc < 2);
		}

		double a = exp(g2T.a-g2F.a) * exposureT / exposureF;
		double b = g2T.b - a*g2F.b;
		return Vec2(a,b);
	}

	Vec2 vec()
	{
		return Vec2(a,b);
	}
};

// EIGEN_ALWAYS_INLINE Eigen::Vector3f getInterpolatedElement33BiLin(const Eigen::Vector3f* const mat, const float x, const float y, const int width)
EIGEN_ALWAYS_INLINE Eigen::Vector3f getInterpolatedElement33BiLin(const std::vector<Vec3f>& mat, const float x, const float y, const int width)

{
	int ix = (int)x;
	int iy = (int)y;
	const Eigen::Vector3f* bp = &mat[ix+iy*width];

	float tl = (*(bp))[0];
	float tr = (*(bp+1))[0];
	float bl = (*(bp+width))[0];
	float br = (*(bp+width+1))[0];

	float dx = x - ix;
	float dy = y - iy;
	float topInt = dx * tr + (1-dx) * tl;
	float botInt = dx * br + (1-dx) * bl;
	float leftInt = dy * bl + (1-dy) * tl;
	float rightInt = dy * br + (1-dy) * tr;

	return Eigen::Vector3f(
			dx * rightInt + (1-dx) * leftInt,
			rightInt-leftInt,
			botInt-topInt);
}

EIGEN_ALWAYS_INLINE Eigen::Vector3f getInterpolatedElement33(const std::vector<Vec3f>& mat, const float x, const float y, const int width)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const Eigen::Vector3f* bp = &mat [ix+iy*width];


	return dxdy * *(const Eigen::Vector3f*)(bp+1+width)
	        + (dy-dxdy) * *(const Eigen::Vector3f*)(bp+width)
	        + (dx-dxdy) * *(const Eigen::Vector3f*)(bp+1)
			+ (1-dx-dy+dxdy) * *(const Eigen::Vector3f*)(bp);
}

EIGEN_ALWAYS_INLINE float getInterpolatedElement31(const std::vector<Vec3f>& mat, const float x, const float y, const int width)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const Eigen::Vector3f* bp = &mat [ix+iy*width];


	return dxdy * (*(const Eigen::Vector3f*)(bp+1+width))[0]
	        + (dy-dxdy) * (*(const Eigen::Vector3f*)(bp+width))[0]
	        + (dx-dxdy) * (*(const Eigen::Vector3f*)(bp+1))[0]
			+ (1-dx-dy+dxdy) * (*(const Eigen::Vector3f*)(bp))[0];
}

inline Vec3b makeRainbow3B(float id)
{
	if(!(id > 0))
		return Vec3b(255,255,255);

	int icP = id;
	float ifP = id-icP;
	icP = icP%3;
	if(icP == 0) return Vec3b(255*(1-ifP), 255*ifP,     0);
	if(icP == 1) return Vec3b(0,           255*(1-ifP), 255*ifP);
	if(icP == 2) return Vec3b(255*ifP,     0,           255*(1-ifP));
	return Vec3b(255,255,255);
}

inline void setPixel9(cv::Mat& Img, const int &u, const int &v, Vec3b& val)
{
	Img.at<cv::Vec3b>(u+1,v-1) = cv::Vec3b(val[0], val[1], val[2]);
	Img.at<cv::Vec3b>(u+1,v) = cv::Vec3b(val[0], val[1], val[2]);
	Img.at<cv::Vec3b>(u+1,v+1) = cv::Vec3b(val[0], val[1], val[2]);
	Img.at<cv::Vec3b>(u,v-1) = cv::Vec3b(val[0], val[1], val[2]);
	Img.at<cv::Vec3b>(u,v) = cv::Vec3b(val[0], val[1], val[2]);
	Img.at<cv::Vec3b>(u,v+1) = cv::Vec3b(val[0], val[1], val[2]);
	Img.at<cv::Vec3b>(u-1,v-1) = cv::Vec3b(val[0], val[1], val[2]);
	Img.at<cv::Vec3b>(u-1,v) = cv::Vec3b(val[0], val[1], val[2]);
	Img.at<cv::Vec3b>(u-1,v+1) = cv::Vec3b(val[0], val[1], val[2]);

}

} // namespace FSLAM
#endif
