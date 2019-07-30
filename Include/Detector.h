#ifndef __DETECTOR__
#define __DETECTOR__
#include <vector>
#include <opencv2/core/types.hpp>

#include <iostream>

namespace FSLAM
{

class ORBDetector
{
public:
    ORBDetector()
    {
        HALF_PATCH_SIZE = 15;
        PATCH_SIZE = 31;
        EDGE_THRESHOLD = 19;
        umax = InitUmax();
        pattern = InitPattern();
        
    }

    ~ORBDetector(){}
    void ExtractFeatures(cv::Mat &Image, std::vector<cv::KeyPoint> &mvKeys, cv::Mat &Descriptors, int &nOrb, std::string name = "");


private:

    std::vector<int> InitUmax();
    std::vector<cv::Point> InitPattern();
    void computeOrbDescriptor(const cv::Mat &Orig, const cv::Mat &img, std::vector<cv::KeyPoint> &Keys, cv::Mat &Descriptors_);
    int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);
    float IC_Angle(const cv::Mat &image, cv::Point2f pt, const std::vector<int> &u_max);
    std::vector<cv::KeyPoint> Ssc(std::vector<cv::KeyPoint> keyPoints, int numRetPoints,float tolerance, int cols, int rows);


    std::vector<int> umax;
    int HALF_PATCH_SIZE;
    int PATCH_SIZE;
    int EDGE_THRESHOLD;
    std::vector<cv::Point> pattern;




};


}













#endif
