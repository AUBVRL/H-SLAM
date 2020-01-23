#ifndef __DETECTOR__
#define __DETECTOR__
// #include <vector>
#include <opencv2/core/types.hpp>
#include "GlobalTypes.h"

namespace FSLAM
{
template<typename Type> class IndexThreadReduce;

class ORBDetector
{
public:
    ORBDetector();
    ~ORBDetector();
    int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);
    void ExtractFeatures(cv::Mat &Image, std::vector<std::vector<float>>& GradPyr, std::vector<cv::KeyPoint> &mvKeys, cv::Mat &Descriptors, int &nOrb, int NumFeatures, std::shared_ptr<IndexThreadReduce<Vec10>>thPool);

private:

    std::vector<int> InitUmax();
    std::vector<cv::Point> InitPattern();
    void computeOrbDescriptor(const cv::Mat &Orig, const cv::Mat &img, std::vector<cv::KeyPoint> &Keys, cv::Mat &Descriptors_, int min, int max);
    float IC_Angle(const cv::Mat &image, cv::Point2f pt, const std::vector<int> &u_max);
    std::vector<cv::KeyPoint> Ssc(std::vector<cv::KeyPoint> keyPoints, int numRetPoints, int minDist, float tolerance, int cols, int rows);
    void DetectFeatures(std::vector<std::vector<float>>& Gradients, std::vector<cv::KeyPoint>& Kpts);


    std::vector<int> umax;
    int HALF_PATCH_SIZE;
    int PATCH_SIZE;
    int EDGE_THRESHOLD;
    std::vector<cv::Point> pattern;




};


}













#endif