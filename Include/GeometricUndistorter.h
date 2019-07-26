#ifndef __UNDISTORTER__
#define __UNDISTORTER__
#include <string>
#include <Eigen/Core>
#include <Settings.h>

namespace FSLAM
{

class GeometricUndistorter
{

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    GeometricUndistorter(std::string GeomCalib_)
    {
        LoadGeometricCalibration(GeomCalib_);
        in_data = new float[WidthOri * HeightOri];
        in_data2 = new float[WidthOri * HeightOri];
    }

    ~GeometricUndistorter() 
    { 
        if(remapX != 0) delete[] remapX; if(remapY != 0) delete[] remapY;
        if(in_data != 0) delete[] in_data; if(in_data2 != 0) delete[] in_data2;
    }

    enum CamModel {RadTan = 0, Pinhole, Atan, KannalaBrandt, EquiDistant} Cameramodel;
    void LoadGeometricCalibration(std::string GeomCalib);
    void makeOptimalK_crop();
    void distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n);
    void undistort(std::shared_ptr<ImageData> ImgData, float* In_L, float* In_R); //used when performing photometric calib first
    void undistort(std::shared_ptr<ImageData>ImgData,  bool NoPhoCalib = false); //used when performing geometric calib first


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

    float * in_data;
    float * in_data2;

};

}

#endif