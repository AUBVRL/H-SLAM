#include <memory>


namespace FSLAM
{
class ImageData;
class OnlineCalibrator;

class System
{

public:
    System();
    ~System();
    void ProcessNewFrame(ImageData& Frame);

private:
    std::shared_ptr<OnlineCalibrator> OnlinePhCalibL;
    std::shared_ptr<OnlineCalibrator> OnlinePhCalibR;

    /* data */
};

} // namespace FSLAM