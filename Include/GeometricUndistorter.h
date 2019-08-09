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
    void undistort(cv::Mat &ImgL, cv::Mat &ImgR); //used when performing photometric calib first

    enum CamModel {RadTan = 0, Pinhole, Atan, KannalaBrandt, EquiDistant} Cameramodel;
    cv::Mat M1l, M2l, M1r, M2r; //rectification and remapping matrices for stereo rectification
    std::string StereoState;
    int w, h, wOrg, hOrg;
    float baseline;
    float ic[10];

protected:
    
    Mat33 K;
    bool passthrough;
    cv::Mat remapX_;
    cv::Mat remapY_;

    float* remapX;
	float* remapY;

};

}

#endif