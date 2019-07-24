#ifndef __UNDISTORTER__
#define __UNDISTORTER__
#include <string>
#include <Eigen/Core>
#include "GlobalTypes.h"

namespace FSLAM
{

class GeometricUndistorter
{

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    GeometricUndistorter(std::string GeomCalib_)
    {
        LoadGeometricCalibration(GeomCalib_);
    }

    ~GeometricUndistorter() { if(remapX != 0) delete[] remapX; if(remapY != 0) delete[] remapY;}

    enum CamModel {RadTan = 0, Pinhole, Atan, KannalaBrandt, EquiDistant} Cameramodel;
    void LoadGeometricCalibration(std::string GeomCalib);
    void makeOptimalK_crop();
    void distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n);
    void undistort(cv::Mat &Input_, cv::Mat &Output);

    cv::Mat M1l, M2l, M1r, M2r; //rectification and remapping matrices for stereo rectification
    float ic[10];
    std::string StereoState;
    int w, h, wOrg, hOrg;
    float baseline;

    
protected:
    
    Mat33 K;
    bool passthrough;
    float* remapX;
	float* remapY;

};

}

#endif