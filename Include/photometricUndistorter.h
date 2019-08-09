#include "GlobalTypes.h"

namespace FSLAM
{
class ORBDetector;
class PhotometricUndistorter
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    PhotometricUndistorter(std::string gamma_path, std::string vignetteImage); 
    ~PhotometricUndistorter() 
    {
	    if(vignetteMapInv) delete[] vignetteMapInv;
    }


    void undistort(cv::Mat &Image, bool isRightRGBD = false, float factor = 1.0f);


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
    bool GammaValid;
    bool VignetteValid;
    cv::Mat cvVignette;

};
} // namespace FSLAM