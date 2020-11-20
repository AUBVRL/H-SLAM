#ifndef __photoUndistorter__H__
#define __photoUndistorter__H__
#pragma once

#include <mutex>

namespace SLAM{

class photoUndistorter
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    photoUndistorter(std::string gamma_path, std::string vignetteImage, bool onPhotoCalib); 
    ~photoUndistorter() 
    {
	    if(invignetteMapInv) delete[] invignetteMapInv;
    }

    void undistort(cv::Mat &Image, cv::Mat &Output, float factor = 1.0f);

    void ResetGamma();
    void ResetVignette();
    void Reset();
    void UpdateGamma(float *_BInv);

    EIGEN_STRONG_INLINE float getBGradOnly(float color)
    {
        // std::unique_lock<std::mutex> lock (mlock);
        int c = color + 0.5f;
        if (c < 5) c = 5;
        if (c > 250) c = 250;
        return B[c + 1] - B[c];
    }

    EIGEN_STRONG_INLINE uchar getB(uchar color)
    {
        return B[color]; //cv::saturate_cast<uchar>()
    }

    EIGEN_STRONG_INLINE float getBInvGradOnly(float color)
    {
        // std::unique_lock<std::mutex> lock (mlock);
        int c = color + 0.5f;
        if (c < 5) c = 5;
        if (c > 250) c = 250;
        return Binv[c + 1] - Binv[c];
    }
    int GDepth;
    bool GammaValid;
    bool VignetteValid;

private:
    float inG[256]; //Read from preCalibrated information. Otherwise initialize to identity.
    float *invignetteMapInv; //Read from preCalibrated information. Otherwise initialize to identity.
    
    cv::Mat incvVignette; //Read from preCalibrated information.

    float B[256];
    float Binv[256];
    float* vignetteMapInv;
    
    std::mutex mlock;

};
}
#endif
