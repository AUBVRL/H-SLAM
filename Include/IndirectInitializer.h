#ifndef __IndirectInitializer_H__
#define __IndirectInitializer_H__

#include "Settings.h"
#include <opencv2/core/types.hpp>

namespace FSLAM
{
class Frame;
class CalibData;
class ORBDetector;

class IndirectInitializer
{
    typedef std::pair<int, int> Match;

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
    void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
    void Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
    int CheckRT(const cv::Mat &R, const cv::Mat &t, const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2,
                const std::vector<Match> &vMatches12, std::vector<bool> &vbInliers,
                const cv::Mat &K, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, float &parallax);
    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
    int FindMatches(std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize=10, int TH_LOW = 50, float mfNNratio = 0.9, bool CheckOrientation = true);




    std::shared_ptr<ORBDetector> Detector;

    //Camera calibration information
    std::shared_ptr<CalibData> Calib;

    // Current Matches from Reference to Current
    std::vector<Match> mvMatches12;
    std::vector<bool> mvbMatched1;

    std::vector<int> mvIniMatches;
    std::vector<cv::Point3f> mvIniP3D;



    // Standard Deviation and Variance
    float mSigma, mSigma2;
    // Ransac max iterations
    int mMaxIterations;
    // Ransac sets
    std::vector<std::vector<size_t> > mvSets;   
    std::vector<cv::Point2f> mvbPrevMatched;

    static const int HISTO_LENGTH = 30;

public:
    IndirectInitializer(std::shared_ptr<CalibData> _Calib, std::shared_ptr<ORBDetector> _Detector);
    ~IndirectInitializer();
    bool Initialize(std::shared_ptr<Frame> _Frame);
    
    //structures containing extracted features
    std::shared_ptr<Frame> FirstFrame;
    std::shared_ptr<Frame> SecondFrame;
};

} // namespace FSLAM

#endif