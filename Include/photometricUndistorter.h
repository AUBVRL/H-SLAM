#include "GlobalTypes.h"

namespace FSLAM
{
class ORBDetector;
class PhotometricUndistorter
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    PhotometricUndistorter(std::string gamma_path, std::string vignetteImage, bool isRight = false); //, int w_, int h_);
    ~PhotometricUndistorter() 
    {
	    if(vignetteMapInv) delete[] vignetteMapInv;
    }


    void undistort(cv::Mat &Image, float* fImage, bool isRightRGBD = false, float factor = 1.0f);
    void undistort(float* fImg, cv::Mat& Img, int w, int h, bool isRightRGBD = false, float factor = 1.0f);


    float *getG()
    {
        if (!GammaValid)
            return 0;
        else
            return G;
    };

private:
    std::shared_ptr<ORBDetector> Detector;
    float G[256 * 256];
    int GDepth;
    float *vignetteMapInv;
    bool GammaValid;
    bool VignetteValid;
    cv::Mat cvVignette;

    std::string name;
};
} // namespace FSLAM