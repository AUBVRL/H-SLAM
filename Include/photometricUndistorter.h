#include "GlobalTypes.h"

namespace FSLAM
{
class PhotometricUndistorter
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    PhotometricUndistorter(std::string gamma_path, std::string vignetteImage, int w_, int h_);
    ~PhotometricUndistorter() 
    {
	    if(vignetteMapInv) delete[] vignetteMapInv;
    }


    void undistort(cv::Mat &Image, float* fImage, float factor = 1.0f);

    float *getG()
    {
        if (!GammaValid)
            return 0;
        else
            return G;
    };

private:
    float G[256 * 256];
    int GDepth;
    float *vignetteMapInv;
    int w, h;
    bool GammaValid;
    bool VignetteValid;
    cv::Mat cvVignette;
};
} // namespace FSLAM