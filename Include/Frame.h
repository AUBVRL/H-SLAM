#ifndef __FRAME__
#define __FRAME__

#include "Settings.h"
namespace FSLAM
{

class ORBDetector;
class ImageData;
class CalibData;
template<typename Type> class IndexThreadReduce;

class Frame
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Frame(std::shared_ptr<ImageData>Img, std::shared_ptr<ORBDetector>_Detector, std::shared_ptr<CalibData>_Calib, std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft);
     //std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolRight)
    ~Frame();

    void CreateIndPyrs(cv::Mat& Img, std::vector<cv::Mat>& Pyr);
    // void RightThread(cv::Mat& Img, std::vector<cv::Mat>& Pyr, std::vector<cv::KeyPoint>& mvKeysR, cv::Mat& DescriptorsR, int& nFeaturesR, std::shared_ptr<IndexThreadReduce<Vec10>>& FrontEndThreadPoolRight);
    
    void CreateDirPyrs(std::vector<float>& Img, std::vector<std::vector<Vec3f>> &DirPyr);

    std::shared_ptr<ORBDetector> Detector;
    
    
    std::vector<cv::Mat> LeftIndPyr; //temporary CV_8U pyramids to extract features
    std::vector<std::vector<Vec3f>> LeftDirPyr; //float representation of image pyramid with computation of dIx and dIy
    std::vector<std::vector<Vec3f>> RightDirPyr;


    std::vector<cv::KeyPoint> mvKeysL;
    // std::vector<cv::KeyPoint> mvKeysR;

    cv::Mat DescriptorsL;
    // cv::Mat DescriptorsR;

    int nFeaturesL;
    // int nFeaturesR;

    //Pyramid params
    int EDGE_THRESHOLD; 

    cv::Mat ImgR; //For display purposes only!

    std::shared_ptr<CalibData> Calib;



};







} // namespace FSLAM









#endif