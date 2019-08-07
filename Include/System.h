#include <memory>


namespace FSLAM
{
class ImageData;
class OnlineCalibrator;
class ORBDetector;

class System
{

public:
    System();
    ~System();
    void ProcessNewFrame(std::shared_ptr<ImageData> &Frame);

private:
    std::shared_ptr<OnlineCalibrator> OnlinePhCalibL;
    std::shared_ptr<OnlineCalibrator> OnlinePhCalibR;

    std::shared_ptr<ORBDetector> Detector;


    /* data */
};

} // namespace FSLAM