#include <opencv2/core/mat.hpp>
#include <memory>
#include <map>
#include <queue>

namespace FSLAM
{
    class Frame;
    class Feature;
    class ORBDetector;

class OnlineCalibrator
{
public:
    OnlineCalibrator();
    ~OnlineCalibrator() {}
    std::shared_ptr<ORBDetector> Detector;
    void AddFrame(cv::Mat Image);


private:
    std::queue<std::shared_ptr<Frame>> Frames;
    int NumFrames;

};



class Frame
{
public:
    Frame(cv::Mat Image, std::shared_ptr<ORBDetector>& Detector_);
    ~Frame() {}
    void InterpolateFeaturePatch(int keyInd);

    cv::Mat image;
    cv::Mat image_corrected;
    cv::Mat gradient_image;
    std::vector<std::shared_ptr<Feature>> features;
    double exp_time;
    double gt_exp_time;
    
    cv::Mat Descriptors;
    std::vector<std::vector<double>> mvKeysPatches; // unrotated interpolated patch
    int nOrb;
    std::vector<cv::KeyPoint> mvKeys;
    std::shared_ptr<ORBDetector> Detector;
    const int patchsize = 2; 

};

class Feature
{
public:
    Feature(cv::KeyPoint Kp_);
    ~Feature() {}

    // std::weak_ptr<Frame> Src;
    cv::KeyPoint Kp;
    std::vector<double> output_values;
    std::vector<double> radiance_estimate;
    std::vector<double> gradient_value;
    // std::map<std::shared_ptr<Frame>, cv::Point2f, std::owner_less<std::shared_ptr<Frame>>> Connectivity;

    // Feature* m_prev_feature;
    // Feature* m_next_feature;
    
};

} // namespace FSLAM