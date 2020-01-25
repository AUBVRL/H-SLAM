#include <opencv2/core/mat.hpp>
#include <memory>
#include <map>
#include <queue>
#include <Eigen/Core>

namespace FSLAM
{
//     class Frame;
//     class LandMark;
//     class FeatureDetector;

// class OnlineCalibrator
// {
// public:
//     OnlineCalibrator();
//     ~OnlineCalibrator() {}
//     std::shared_ptr<FeatureDetector> Detector;
//     void ProcessFrame(cv::Mat Image);


// private:
//     std::queue<std::shared_ptr<Frame>> Frames;
//     int NumFrames;
//     std::map<std::weak_ptr<Frame>,std::pair<std::weak_ptr<Frame>,int>,std::owner_less<std::weak_ptr<Frame>>> Connectivity;
// };



// class Frame
// {
// public:
// 	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
//     Frame(cv::Mat Image, std::shared_ptr<FeatureDetector> Detector_);
//     ~Frame() {}
//     void InterpolateFeaturePatches(int keyInd);

//     cv::Mat image;
//     cv::Mat image_corrected;
//     cv::Mat gradient_image;
//     std::vector<std::shared_ptr<LandMark>> features;
//     double exp_time;
//     double gt_exp_time;

    
//     cv::Mat Descriptors;
//     std::vector<std::vector<double>> mvKeysPatches; // unrotated interpolated patch
//     int nOrb;
//     std::vector<cv::KeyPoint> mvKeys;
//     std::shared_ptr<FeatureDetector> Detector;
//     const int patchsize = 2; 

// };

// class LandMark
// {
// public:
//     LandMark(cv::KeyPoint Kp_): Kp(Kp_){}
//     ~LandMark() {}

//     // std::weak_ptr<Frame> Src;
//     cv::KeyPoint Kp;
//     std::vector<double> output_values;
//     std::vector<double> radiance_estimate;
//     std::vector<double> gradient_value;
    
// };

} // namespace FSLAM