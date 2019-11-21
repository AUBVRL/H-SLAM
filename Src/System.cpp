#include "System.h"
// #include "OnlineCalibrator.h"
#include "Detector.h"
#include "Settings.h"
#include "Frame.h"
#include "CalibData.h"
#include "GeometricUndistorter.h"
#include "photometricUndistorter.h"
#include "Display.h"

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

namespace FSLAM
{


System::System(std::shared_ptr<GeometricUndistorter> _GeomUndist, std::shared_ptr<PhotometricUndistorter> _PhoUndistL, 
            std::shared_ptr<PhotometricUndistorter> _PhoUndistR,    std::shared_ptr<GUI> _DisplayHandler): DisplayHandler(_DisplayHandler)
{
    FrontEndThreadPoolLeft = std::shared_ptr<IndexThreadReduce<Vec10>>(new IndexThreadReduce<Vec10>);
    // FrontEndThreadPoolRight = std::shared_ptr<IndexThreadReduce<Vec10>>(new IndexThreadReduce<Vec10>);
    BackEndThreadPool = std::shared_ptr<IndexThreadReduce<Vec10>>(new IndexThreadReduce<Vec10>);

    Detector = std::make_shared<ORBDetector>();
    GeomUndist = _GeomUndist;
    PhoUndistR = _PhoUndistR;
    PhoUndistL = _PhoUndistL;

    Calib = std::shared_ptr<CalibData>(new CalibData(GeomUndist->w, GeomUndist->h, GeomUndist->K, GeomUndist->baseline, PhoUndistL, PhoUndistR, DirPyrLevels, DirPyrScaleFactor));
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

void System::ProcessNewFrame(std::shared_ptr<ImageData> DataIn)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    CurrentFrame = std::make_shared<Frame>(DataIn, Detector, Calib ,FrontEndThreadPoolLeft); //FrontEndThreadPoolRight
    std::cout << "time: " << (float)(((std::chrono::duration<double>)(std::chrono::high_resolution_clock::now() - start)).count() * 1e3) << std::endl;
    //only called if online photometric calibration is required (keep this here and not in the photometric undistorter to have access to slam data)
    // if(OnlinePhCalibL) 
    //     OnlinePhCalibL->ProcessFrame(Frame.cvImgL);
    // if(OnlinePhCalibR)
    //     OnlinePhCalibL->ProcessFrame(Frame.cvImgR);



    DrawImages();
}

void System::DrawImages()
{
    cv::Mat Dest;
    if (Sensortype == Stereo || Sensortype == RGBD)
        cv::hconcat(CurrentFrame->LeftIndPyr[0], CurrentFrame->ImgR, Dest);
    else
        Dest = CurrentFrame->LeftIndPyr[0];

    cv::cvtColor(Dest, Dest, CV_GRAY2BGR);

    if (DrawDetected)
    {

        for (size_t i = 0; i < CurrentFrame->mvKeysL.size(); ++i)
            cv::circle(Dest, CurrentFrame->mvKeysL[i].pt, 3, cv::Scalar(255.0, 0.0, 0.0), -1, cv::LineTypes::LINE_8, 0);
        // if (Sensortype == Stereo)
        // {
        //     cv::Point2f Shift(CurrentFrame->LeftIndPyr[0].size().width, 0.0f);
        //     for (size_t i = 0; i < CurrentFrame->mvKeysR.size(); ++i)
        //         cv::circle(Dest, CurrentFrame->mvKeysR[i].pt + Shift, 3, cv::Scalar(255.0, 0.0, 0.0), -1, cv::LineTypes::LINE_8, 0);
        // }
    }
    DisplayHandler->UploadFrameImage(Dest.data, Dest.size().width, Dest.size().height);
}

} // namespace FSLAM
