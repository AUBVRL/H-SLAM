#ifndef __UNDISTORTER__
#define __UNDISTORTER__
#include <string>
#include <Eigen/Core>
#include "GlobalTypes.h"

namespace FSLAM
{

class Undistorter
{

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Undistorter(std::string GeomCalib_, std::string Gamma_, std::string Vignette_)
    {
        LoadGeometricCalibration(GeomCalib_);
    }
    ~Undistorter()
    {
	    if(remapX != 0) delete[] remapX;
	    if(remapY != 0) delete[] remapY;
    }
    enum CamModel {RadTan = 0, Pinhole, Atan, KannalaBrandt, EquiDistant, Empty};
    CamModel Cameramodel;
    void LoadGeometricCalibration(std::string GeomCalib);
    void makeOptimalK_crop();
    void distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n);
    cv::Mat M1l, M2l, M1r, M2r; //rectification and remapping matrices for stereo rectification
    float ic[10];

    // void LoadPhotometricCalibration(std::string Gamma, std::string Vignette);
    
protected:

    int w, h, wOrg, hOrg;
    float baseline;
    Mat33 K;
    bool passthrough;
    float* remapX;
	float* remapY;

};

}

#endif