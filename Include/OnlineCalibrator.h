#include <opencv2/core/mat.hpp>
#include <memory>
#include <map>

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
    std::vector<std::shared_ptr<Frame>> Frames;

};



class Frame
{
public:
    Frame(cv::Mat Image, std::shared_ptr<ORBDetector>& Detector_);
    ~Frame() {}
    cv::Mat image;
    cv::Mat image_corrected;
    cv::Mat gradient_image;
    std::vector<std::shared_ptr<Feature>> features;
    double exp_time;
    double gt_exp_time;
    
    cv::Mat Descriptors;
    int nOrb;
    std::vector<cv::KeyPoint> mvKeys;
    std::shared_ptr<ORBDetector> Detector;

};

class Feature
{
public:
    Feature(cv::KeyPoint Kp_);
    ~Feature() {}
    void InterpolateFeaturePatch(cv::Mat Image);

    std::shared_ptr<Frame> Src;
    cv::KeyPoint Kp;
    std::vector<double> output_values;
    std::vector<double> radiance_estimate;
    std::vector<double> gradient_value;
    // std::map<std::shared_ptr<Frame>, cv::Point2f, std::owner_less<std::shared_ptr<Frame>>> Connectivity;
    const int patchsize = 2; 

    // Feature* m_prev_feature;
    // Feature* m_next_feature;
    
};

} // namespace FSLAM