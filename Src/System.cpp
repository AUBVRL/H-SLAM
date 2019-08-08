#include "System.h"
#include "OnlineCalibrator.h"
#include "Detector.h"
#include "Settings.h"
#include "Frame.h"

namespace FSLAM
{

System::System()
{
    Detector = std::make_shared<ORBDetector>();
    //Setup Online Photometric Calibrators
    // OnlinePhCalibL = OnlinePhCalibL = NULL;
    // if (PhoUndistMode == OnlineCalib)
    // {
    //     OnlinePhCalProcessibL = std::make_shared<OnlineCalibrator>();
    //     if (Sensortype == Stereo)
    //         OnlinePhCalibR = std::make_shared<OnlineCalibrator>();
    // }
}

System::~System()
{
}

void System::ProcessNewFrame(std::shared_ptr<ImageData> &DataIn)
{
    CurrentFrame = std::make_shared<Frame>(DataIn, Detector);
    //only called if online photometric calibration is required (keep this here and not in the photometric undistorter to have access to slam data)
    // if(OnlinePhCalibL) 
    //     OnlinePhCalibL->ProcessFrame(Frame.cvImgL);
    // if(OnlinePhCalibR)
    //     OnlinePhCalibL->ProcessFrame(Frame.cvImgR);




}
} // namespace FSLAM
