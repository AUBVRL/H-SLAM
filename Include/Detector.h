#ifndef __FeatureDetector__
#define __FeatureDetector__

#include <opencv2/core/types.hpp>
#include "globalTypes.h"


namespace SLAM
{

template <typename Type> class IndexThreadReduce;
class PixelSelector;
class CalibData;
class FeatureDetector
{
public:
    FeatureDetector(std::shared_ptr<CalibData> _Calib);
    ~FeatureDetector();
    // void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);
    void ExtractFeatures(cv::Mat &Image, FeatureType FeatType, std::vector<Vec3f*>&DirPyr, int id, std::vector<float*>& GradPyr, std::vector<cv::KeyPoint> &mvKeys, cv::Mat &Descriptors, int &NumFeatures, int NumFeaturesToExtract, shared_ptr<IndexThreadReduce<Vec10>> ThreadPool);

private:

    std::vector<int> InitUmax();
    std::vector<cv::Point> InitPattern();
    void computeOrbDescriptor(const cv::Mat &Orig, const cv::Mat &img, std::vector<cv::KeyPoint> &Keys, cv::Mat &Descriptors_, int min, int max);
    float IC_Angle(const cv::Mat &image, cv::Point2f pt, const std::vector<int> &u_max);
    std::vector<cv::KeyPoint> Ssc(std::vector<cv::KeyPoint> keyPoints, int numRetPoints, int minDist, float tolerance, int cols, int rows);
    float ShiTomasiScore(vector<Vec3f*>&DirPyr, cv::Size imgSize, const float &u, const float &v, int halfbox = 4);

    std::vector<int> umax;
    int HALF_PATCH_SIZE;
    int PATCH_SIZE;
    int EDGE_THRESHOLD;
    std::vector<cv::Point> pattern;
    std::shared_ptr<PixelSelector> PixSelector;
    float* selectionMap;

    
};


} // namespace SLAM

#endif