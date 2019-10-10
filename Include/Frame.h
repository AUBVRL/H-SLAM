#ifndef __FRAME__
#define __FRAME__

#include "Settings.h"
namespace FSLAM
{

class ORBDetector;
class ImageData;
template<typename Type> class IndexThreadReduce;

class Frame
{
public:
	// EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Frame(std::shared_ptr<ImageData>Img, std::shared_ptr<ORBDetector>_Detector, std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft,
            std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolRight);
    ~Frame();
    void CreatePyrs(cv::Mat& Img, std::vector<cv::Mat>& Pyr);
    void RightThread(cv::Mat& Img, std::vector<cv::Mat>& Pyr, std::vector<cv::KeyPoint>& mvKeysR, cv::Mat& DescriptorsR, int& nFeaturesR, std::shared_ptr<IndexThreadReduce<Vec10>>& FrontEndThreadPoolRight);

    std::shared_ptr<ORBDetector> Detector;
    
    
    std::vector<cv::Mat> LeftPyr;
    std::vector<cv::Mat> RightPyr;

    // Vec6f* Image; //[0]I, [1]Ix, [2]Iy, [3]Ixy, [4]Ixx, [5]Iyy

    std::vector<cv::KeyPoint> mvKeysL;
    std::vector<cv::KeyPoint> mvKeysR;

    cv::Mat DescriptorsL;
    cv::Mat DescriptorsR;

    int nFeaturesL;
    int nFeaturesR;

    //Pyramid params
    int EDGE_THRESHOLD; 



};







} // namespace FSLAM









#endif