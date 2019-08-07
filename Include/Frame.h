#ifndef __FRAME__
#define __FRAME__

#include "Settings.h"

namespace FSLAM
{


class Frame
{
public:
    Frame();
    ~Frame();
    
    std::vector<cv::KeyPoint> mvKeys;
    int nFeatures;
    cv::Mat Descriptors;
    const int patchsize = 2;

private:

};


} // namespace FSLAM









#endif