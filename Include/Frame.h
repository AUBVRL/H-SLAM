#ifndef __FRAME__
#define __FRAME__

#include "Settings.h"

namespace FSLAM
{

class ORBDetector;
class ImageData;

class Frame
{
public:

    Frame(std::shared_ptr<ImageData>&Img, std::shared_ptr<ORBDetector>_Detector);
    ~Frame();
    
    std::shared_ptr<ORBDetector> Detector;

    std::vector<float*> vfImgL;
    std::vector<float*> vfImgR;

    std::vector<cv::KeyPoint> mvKeysL;
    std::vector<cv::KeyPoint> mvKeysR;

    cv::Mat DescriptorsL;
    cv::Mat DescriptorsR;

    int nFeaturesL;
    int nFeaturesR;




};







} // namespace FSLAM









#endif