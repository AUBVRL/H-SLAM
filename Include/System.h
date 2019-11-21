#include <memory>
#include "Settings.h"
#include "IndexThreadReduce.h"

namespace FSLAM
{
class ImageData;
class OnlineCalibrator;
class ORBDetector;
class Frame;
class CalibData;
class GeometricUndistorter;
class PhotometricUndistorter;
class GUI;

class System
{

public:

    System(std::shared_ptr<GeometricUndistorter> GeomUndist, std::shared_ptr<PhotometricUndistorter> 
            PhoUndistL, std::shared_ptr<PhotometricUndistorter> PhoUndistR, std::shared_ptr<GUI> _DisplayHandler);
    ~System();
    void ProcessNewFrame(std::shared_ptr<ImageData> DataIn);
	
    std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft;
    // std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolRight;

    std::shared_ptr<IndexThreadReduce<Vec10>> BackEndThreadPool;


private:
    void DrawImages();
    std::shared_ptr<CalibData> Calib; //Calibration data that is used for projection and optimization
    std::shared_ptr<GUI> DisplayHandler;


    
    
    // std::shared_ptr<OnlineCalibrator> OnlinePhCalibL;
    // std::shared_ptr<OnlineCalibrator> OnlinePhCalibR;

    std::shared_ptr<ORBDetector> Detector;
    std::shared_ptr<Frame> CurrentFrame;





    //stored here without being used within the system.
    std::shared_ptr<PhotometricUndistorter> PhoUndistL; //The input photometric undistorter 
    std::shared_ptr<PhotometricUndistorter> PhoUndistR; //The input photometric undistorter 
    std::shared_ptr<GeometricUndistorter> GeomUndist;     //geometric calib data used for undistorting the images. Only used to initialize the calib ptr.
                                                


    /* data */
};

} // namespace FSLAM