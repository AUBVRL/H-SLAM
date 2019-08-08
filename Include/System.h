#include <memory>


namespace FSLAM
{
class ImageData;
class OnlineCalibrator;
class ORBDetector;
class Frame;

class System
{

public:
    System();
    ~System();
    void ProcessNewFrame(std::shared_ptr<ImageData> &DataIn);

private:
    std::shared_ptr<OnlineCalibrator> OnlinePhCalibL;
    std::shared_ptr<OnlineCalibrator> OnlinePhCalibR;

    std::shared_ptr<ORBDetector> Detector;
    std::shared_ptr<Frame> CurrentFrame;

    /* data */
};

} // namespace FSLAM