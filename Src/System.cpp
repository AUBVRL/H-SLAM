#include "System.h"
#include "OnlineCalibrator.h"
#include "Settings.h"

namespace FSLAM
{

System::System()
{
    //Setup Online Photometric Calibrators
    // OnlinePhCalibL = OnlinePhCalibL = NULL;
    // if (PhoUndistMode == OnlineCalib)
    // {
    //     OnlinePhCalibL = std::make_shared<OnlineCalibrator>();
    //     if (Sensortype == Stereo)
    //         OnlinePhCalibR = std::make_shared<OnlineCalibrator>();
    // }
}

System::~System()
{
}

void System::ProcessNewFrame(ImageData &Frame)
{

    //only called if online photometric calibration is required (keep this here and not in the photometric undistorter to have access to slam data)
    // if(OnlinePhCalibL) 
    //     OnlinePhCalibL->ProcessFrame(Frame.cvImgL);
    // if(OnlinePhCalibR)
    //     OnlinePhCalibL->ProcessFrame(Frame.cvImgR);




}
} // namespace FSLAM
