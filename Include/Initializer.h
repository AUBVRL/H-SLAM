#ifndef __INITIALIZER__
#define __INITIALIZER__

#include "Settings.h"

namespace FSLAM
{
class Frame;
class CalibData;
class GUI;

class Initializer
{
private:

    void FindHomography(std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
    void FindFundamental(std::vector<bool> &vbInliers, float &score, cv::Mat &F21);
    cv::Mat ComputeH21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);
    cv::Mat ComputeF21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);
    float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, std::vector<bool> &vbMatchesInliers, float sigma);
    float CheckFundamental(const cv::Mat &F21, std::vector<bool> &vbMatchesInliers, float sigma);
    bool ReconstructF(std::vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
    bool ReconstructH(std::vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
    void Triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
    void Normalize(const std::vector<cv::Point2f> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
    int CheckRT(const cv::Mat &R, const cv::Mat &t, const std::vector<cv::Point2f> &vKeys1, const std::vector<cv::Point2f> &vKeys2,
                std::vector<bool> &vbInliers,
                const cv::Mat &K, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, float &parallax);
    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
    int FindMatches(int windowSize = 10, int maxL1Error = 7);
    bool FindTransformation(cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);
    float ComputeSceneMedianDepth(const int q, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);
    float ComputeMeanOpticalFlow(std::vector<cv::Point2f> &Prev, std::vector<cv::Point2f> &New);

    std::shared_ptr<CalibData> Calib;
    std::shared_ptr<GUI> displayhandler;

    float mSigma, mSigma2; // Standard Deviation and Variance
    int mMaxIterations;    // Ransac max iterations
    std::shared_ptr<cv::RNG> randomGen;

    std::vector<std::vector<size_t>> mvSets; // Ransac sets
    
    cv::Mat TransitImage;
    std::vector<cv::Point2f> FirstFramePts;
    std::vector<cv::Point2f> mvbPrevMatched; //p0 updated
    std::vector<cv::Point2f> MatchedPts;
    std::vector<cv::Point2f> MatchedPtsBkp; //used to detect stationary camera
    std::vector<bool> MatchedStatus;
    std::vector<cv::Scalar> ColorVec; 
    std::vector<cv::Point3f> mvIniP3D;

    //structures containing extracted features
    std::shared_ptr<Frame> FirstFrame;
    std::shared_ptr<Frame> SecondFrame;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    bool Initialize(std::shared_ptr<Frame> _Frame);
    Initializer(std::shared_ptr<CalibData> _Calib, std::shared_ptr<GUI>_DisplayHandler);
    ~Initializer(){};

};

} // namespace FSLAM

#endif