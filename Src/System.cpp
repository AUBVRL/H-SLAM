#include "System.h"
// #include "OnlineCalibrator.h"
#include "Detector.h"
#include "Settings.h"
#include "Frame.h"
#include "CalibData.h"
#include "GeometricUndistorter.h"
#include "photometricUndistorter.h"
#include "Display.h"
#include "Map.h"
#include "ImmaturePoint.h"
#include "IndirectInitializer.h"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

namespace FSLAM
{


System::System(std::shared_ptr<GeometricUndistorter> _GeomUndist, std::shared_ptr<PhotometricUndistorter> _PhoUndistL, 
            std::shared_ptr<PhotometricUndistorter> _PhoUndistR,    std::shared_ptr<GUI> _DisplayHandler): DisplayHandler(_DisplayHandler), Initialized(false),
            NeedNewKFAfter(-1)
{
    GeomUndist = _GeomUndist;
    PhoUndistR = _PhoUndistR;
    PhoUndistL = PhoUndistL;
    FrontEndThreadPoolLeft = std::shared_ptr<IndexThreadReduce<Vec10>>(new IndexThreadReduce<Vec10>);
    BackEndThreadPool = std::shared_ptr<IndexThreadReduce<Vec10>>(new IndexThreadReduce<Vec10>);
    Detector = std::make_shared<ORBDetector>();
    Calib = std::shared_ptr<CalibData>(new CalibData(GeomUndist->w, GeomUndist->h, GeomUndist->K, GeomUndist->baseline, PhoUndistL, PhoUndistR,
                                        DirPyrLevels, DirPyrScaleFactor, IndPyrLevels, IndPyrScaleFactor));
    SlamMap = std::shared_ptr<Map>(new Map());

    // selectionMap = new float[wG[0]*hG[0]];
	// coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
	// coarseTracker = new CoarseTracker(wG[0], hG[0]);
	// coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
	// coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	// pixelSelector = new PixelSelector(wG[0], hG[0]);
    // lastCoarseRMSE.setConstant(100);
	// currentMinActDist=2;

    // ef = new EnergyFunctional();
	// ef->red = &this->treadReduce;
	// isLost=false;
	// initFailed=false;

    RunMapping = true;
    // lastRefStopID=0;

	tMappingThread = boost::thread(&System::MappingThread, this);

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
    BlockUntilMappingIsFinished();
    if(PhoUndistL)
        PhoUndistL->Reset(); 
    if(PhoUndistR)
        PhoUndistR->Reset();
}

void System::ProcessNewFrame(std::shared_ptr<ImageData> DataIn)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    
    std::shared_ptr<Frame> CurrentFrame = std::shared_ptr<Frame>(new Frame(DataIn, Detector, Calib, FrontEndThreadPoolLeft, !Initialized)); // FrontEndThreadPoolRight

    if(!Initialized)
    {   //Initialize..
        if(!Initializer)
            Initializer = std::shared_ptr<IndirectInitializer>(new IndirectInitializer(Calib,Detector,DisplayHandler));
        Initialized = Initializer->Initialize(CurrentFrame);
        
    }
    else
    {
        
        bool NeedNewKf = false;
        switch (Sensortype)
        {
        case Monocular: 
            //TrackMonocular()
            break;
        case Stereo:
            CurrentFrame->ImmaturePointsLeftRight.resize(CurrentFrame->mvKeys.size());
            GetStereoDepth(CurrentFrame);
            // FrontEndThreadPoolLeft->reduce(boost::bind(&Frame::ComputeStereoDepth, CurrentFrame, CurrentFrame, _1, _2), 0, CurrentFrame->mvKeys.size(), std::ceil(CurrentFrame->mvKeys.size() / NUM_THREADS));
            //TrackStereo()
            break;
        case RGBD: 
            //TrackRGBD()
            break;
        }

        if (SequentialOperation)
        {
            if (NeedNewKf)
                AddKeyframe(CurrentFrame);
            else
                ProcessNonKeyframe(CurrentFrame);
        }
        else //Parallel Computation
        {
            boost::unique_lock<boost::mutex> lock(MapThreadMutex);
            UnmappedTrackedFrames.push_back(CurrentFrame);
            if (NeedNewKf)
                NeedNewKFAfter = CurrentFrame->id;
            TrackedFrameSignal.notify_all();
            while (!Initialized)
                MappedFrameSignal.wait(lock);
            lock.unlock();
        }
    }
    
    std::cout << "time: " << (float)(((std::chrono::duration<double>)(std::chrono::high_resolution_clock::now() - start)).count() * 1e3) << std::endl;
    //only called if online photometric calibration is required (keep this here and not in the photometric undistorter to have access to slam data)
    // if(OnlinePhCalibL) 
    //     OnlinePhCalibL->ProcessFrame(Frame.cvImgL);
    // if(OnlinePhCalibR)
    //     OnlinePhCalibL->ProcessFrame(Frame.cvImgR);
    DrawImages(CurrentFrame);

}

void System::DrawImages(std::shared_ptr<Frame> CurrentFrame)
{
    if(!DisplayHandler)
        return;

    if(!DisplayHandler->Show2D->Get())
        return;

    if(DisplayHandler->ShowImages->Get())
    {
        cv::Mat Dest;
        if (Sensortype == Stereo || Sensortype == RGBD)
            cv::hconcat(CurrentFrame->LeftIndPyr[0], CurrentFrame->ImgR, Dest);
        else
            Dest = CurrentFrame->LeftIndPyr[0];

       cv::cvtColor(Dest, Dest, CV_GRAY2BGR);

        if (DrawDetected)
        {

            for (size_t i = 0; i < CurrentFrame->mvKeys.size(); ++i)
                cv::circle(Dest, CurrentFrame->mvKeys[i].pt, 3, cv::Scalar(255.0, 0.0, 0.0), -1, cv::LineTypes::LINE_8, 0);
            // if (Sensortype == Stereo)
            // {
            //     cv::Point2f Shift(CurrentFrame->LeftIndPyr[0].size().width, 0.0f);
            //     for (size_t i = 0; i < CurrentFrame->mvKeysR.size(); ++i)
            //         cv::circle(Dest, CurrentFrame->mvKeysR[i].pt + Shift, 3, cv::Scalar(255.0, 0.0, 0.0), -1, cv::LineTypes::LINE_8, 0);
            // }
        }
        DisplayHandler->UploadFrameImage(Dest.data, Dest.size().width, Dest.size().height);
        
        // DisplayHandler->UploadDepthKeyFrameImage(Dest.data, Dest.size().width, Dest.size().height);

    }
}

void System::ProcessNonKeyframe(std::shared_ptr<Frame> Frame)
{
    return;
}

//MOVE THIS INTO FRAME.CC
void System::GetStereoDepth(std::shared_ptr<Frame> _In)
{
    if(Sensortype != Stereo)
        return;

    int NumImmature = _In->mvKeys.size();
    std::vector<std::shared_ptr<ImmaturePoint>> ImmatureStereoPoints;
    ImmatureStereoPoints.resize(NumImmature);
    FrontEndThreadPoolLeft->reduce(boost::bind(&Frame::ComputeStereoDepth, _In, _In, ImmatureStereoPoints, _1,_2), 0, NumImmature, std::ceil(NumImmature/NUM_THREADS));
    return;
}


} // namespace FSLAM
