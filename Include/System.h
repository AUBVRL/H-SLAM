#include <memory>

namespace FSLAM
{
class ImageData;
class OnlineCalibrator;
class ORBDetector;
class Frame;
class CalibData;
class GeometricUndistorter;
class PhotometricUndistorter;

class System
{

public:

    System(std::shared_ptr<GeometricUndistorter> GeomUndist, std::shared_ptr<PhotometricUndistorter> PhoUndistL, std::shared_ptr<PhotometricUndistorter> PhoUndistR);
    ~System();
    void ProcessNewFrame(std::shared_ptr<ImageData> DataIn);

private:
    std::shared_ptr<PhotometricUndistorter> PhotoL; //The input photometric undistorter 
    std::shared_ptr<PhotometricUndistorter> PhotoR; //The input photometric undistorter 
    std::shared_ptr<CalibData> Calib; //Calibration data that is used for projection and optimization

    //geometric calib data used for undistorting the images. Only used to initialize the calib ptr.
    //stored here without being used within the system.
    std::shared_ptr<GeometricUndistorter> GeomCalibIn; 
    
    
    // std::shared_ptr<OnlineCalibrator> OnlinePhCalibL;
    // std::shared_ptr<OnlineCalibrator> OnlinePhCalibR;

    std::shared_ptr<ORBDetector> Detector;
    std::shared_ptr<Frame> CurrentFrame;

    /* data */
};

} // namespace FSLAM