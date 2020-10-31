#ifndef __UNDISTORTER__
#define __UNDISTORTER__
#include <string>
#include <opencv2/core.hpp>
#include <Settings.h>

namespace SLAM
{

class geomUndistorter
{

public:

    geomUndistorter(std::string GeomCalib_);
    ~geomUndistorter();
    void LoadGeometricCalibration(std::string GeomCalib);
    void makeOptimalK_crop();
    void distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n);
    void undistort(cv::Mat &Img);

    enum CamModel {RadTan = 0, Pinhole, Atan, KannalaBrandt, EquiDistant} Cameramodel;
    int w, h, wOrg, hOrg; //w, h is the largest resolution SLAM will operate at. wOrg, hOrg is the input res.
    double ic[10];
    Mat33 K;


protected:
    
    bool passthrough;
    cv::Mat remapX_;
    cv::Mat remapY_;

    float* remapX;
	float* remapY;

};

}

#endif