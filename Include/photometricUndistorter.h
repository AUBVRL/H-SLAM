#include "GlobalTypes.h"

namespace FSLAM
{
class PhotometricUndistorter
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    PhotometricUndistorter(std::string gamma_path, std::string vignetteImage, int w_, int h_);
    ~PhotometricUndistorter();


    void undistort(std::shared_ptr<ImageData> Img, float factor = 1);

    float *getG()
    {
        if (!valid)
            return 0;
        else
            return G;
    };

private:
    float G[256 * 256];
    int GDepth;
    float *vignetteMap;
    float *vignetteMapInv;
    int w, h;
    bool valid;
};
} // namespace FSLAM