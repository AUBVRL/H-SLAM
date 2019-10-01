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
	// EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Frame(std::shared_ptr<ImageData>Img, std::shared_ptr<ORBDetector>_Detector);
    ~Frame();
    void CreatePyrs(std::shared_ptr<ImageData> Img);

    std::shared_ptr<ORBDetector> Detector;

    // std::vector<float*> vfImgL;
    // std::vector<float*> vfImgR;

    // Vec6f* Image; //[0]I, [1]Ix, [2]Iy, [3]Ixy, [4]Ixx, [5]Iyy

    std::vector<cv::KeyPoint> mvKeysL;
    std::vector<cv::KeyPoint> mvKeysR;

    cv::Mat DescriptorsL;
    cv::Mat DescriptorsR;

    int nFeaturesL;
    int nFeaturesR;






};







} // namespace FSLAM









#endif