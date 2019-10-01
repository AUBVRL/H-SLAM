#ifndef __UNDISTORTER__
#define __UNDISTORTER__
#include <string>
#include <Settings.h>

namespace FSLAM
{

class GeometricUndistorter
{

public:

    GeometricUndistorter(std::string GeomCalib_);
    ~GeometricUndistorter();
    void LoadGeometricCalibration(std::string GeomCalib);
    void makeOptimalK_crop();
    void distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n);
    void undistort(cv::Mat &Img, bool isRight);

    enum CamModel {RadTan = 0, Pinhole, Atan, KannalaBrandt, EquiDistant} Cameramodel;
    cv::Mat M1l, M2l, M1r, M2r; //rectification and remapping matrices for stereo rectification
    std::string StereoState;
    int w, h, wOrg, hOrg; //w, h is the largest resolution SLAM will operate at. wOrg, hOrg is the input res.
    float baseline;
    float ic[10];
    Mat33f K;


protected:
    
    bool passthrough;
    cv::Mat remapX_;
    cv::Mat remapY_;

    float* remapX;
	float* remapY;

};

}

#endif