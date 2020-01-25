#include "GlobalTypes.h"
#include "boost/thread/mutex.hpp"

namespace FSLAM
{
class PhotometricUndistorter
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    PhotometricUndistorter(std::string gamma_path, std::string vignetteImage); 
    ~PhotometricUndistorter() 
    {
	    if(invignetteMapInv) delete[] invignetteMapInv;
    }


    void undistort(cv::Mat &Image, bool isRightRGBD = false, float factor = 1.0f);

    void ResetGamma();
    void ResetVignette();
    void Reset();
    void UpdateGamma(float *_BInv);

    EIGEN_STRONG_INLINE float getBGradOnly(float color)
    {
        boost::unique_lock<boost::mutex> lock (mlock);
        int c = color + 0.5f;
        if (c < 5) c = 5;
        if (c > 250) c = 250;
        return B[c + 1] - B[c];
    }

    EIGEN_STRONG_INLINE float getBInvGradOnly(float color)
    {
        boost::unique_lock<boost::mutex> lock (mlock);
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
    
    boost::mutex mlock;


};
} // namespace FSLAM