#include "System.h"
// #include "OnlineCalibrator.h"
#include "Detector.h"
#include "Settings.h"
#include "Frame.h"
#include "ImmaturePoint.h"
#include "MapPoint.h"
#include "CalibData.h"
#include "GeometricUndistorter.h"
#include "photometricUndistorter.h"
#include "Display.h"
#include "Map.h"
#include "Initializer.h"


#include "CoarseTracker.h"
#include "EnergyFunctional.h"

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
    Detector = std::make_shared<FeatureDetector>();
    Calib = std::shared_ptr<CalibData>(new CalibData(GeomUndist->w, GeomUndist->h, GeomUndist->K, GeomUndist->baseline, PhoUndistL, PhoUndistR,
                                        DirPyrLevels, IndPyrLevels, IndPyrScaleFactor));
    SlamMap = std::shared_ptr<Map>(new Map());


    //----------------begin dso------------------
    selectionMap = new float[Calib->Width* Calib->Height];
	coarseDistanceMap = new CoarseDistanceMap(Calib->Width, Calib->Height);
	coarseTracker = new CoarseTracker(Calib->Width, Calib->Height);
	coarseTracker_forNewKF = new CoarseTracker(Calib->Width, Calib->Height);
	// coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	// pixelSelector = new PixelSelector(wG[0], hG[0]);
    lastCoarseRMSE.setConstant(100);
	currentMinActDist=2;

    ef = new EnergyFunctional();
	ef->red = &this->treadReduce;


    //----------------end dso------------------


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

    //----------------begin dso------------------
    delete[] selectionMap;

    // for (FrameShell *s : allFrameHistory)
    //     delete s;
    // for (Frame *fh : unmappedTrackedFrames)
    //     delete fh;

    delete coarseDistanceMap;
    delete coarseTracker;
    delete coarseTracker_forNewKF;
    // delete coarseInitializer;
    // delete pixelSelector;
    delete ef;
    
    //----------------end dso------------------
    
    
}

void System::ProcessNewFrame(std::shared_ptr<ImageData> DataIn)
{
    std::shared_ptr<Frame> CurrentFrame = std::shared_ptr<Frame>(new Frame(DataIn, Detector, Calib, FrontEndThreadPoolLeft, !Initialized)); // FrontEndThreadPoolRight

    if(!Initialized)
    {   //Initialize..
        if(!cInitializer)
            cInitializer = std::shared_ptr<Initializer>(new Initializer(Calib, FrontEndThreadPoolLeft ,DisplayHandler));
        if(cInitializer->Initialize(CurrentFrame))
            InitFromInitializer(cInitializer);
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
    

    //only called if online photometric calibration is required (keep this here and not in the photometric undistorter to have access to slam data)
    // if(OnlinePhCalibL) 
    //     OnlinePhCalibL->ProcessFrame(Frame.cvImgL);
    // if(OnlinePhCalibR)
    //     OnlinePhCalibL->ProcessFrame(Frame.cvImgR);
    DrawImages(DataIn, CurrentFrame);


}

void System::DrawImages(std::shared_ptr<ImageData> DataIn, std::shared_ptr<Frame> CurrentFrame)
{
    if(!DisplayHandler)
        return;

    if(!DisplayHandler->Show2D->Get())
        return;

    if(DisplayHandler->ShowImages->Get())
    {
        cv::Mat Dest;
        if (Sensortype == Stereo || Sensortype == RGBD)
            cv::hconcat(DataIn->cvImgL, DataIn->cvImgR, Dest);
        else
            Dest = DataIn->cvImgL;

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

void System::InitFromInitializer(std::shared_ptr<Initializer> _cInit)
{
    std::shared_ptr<Frame> firstFrame = _cInit->FirstFrame;
    std::shared_ptr<Frame> secondFrame = _cInit->SecondFrame;
    firstFrame->idx = 0;
    // frameHessians.push_back(_cInit->FirstFrame);
    firstFrame->id = 0;

    // allKeyFramesHistory.push_back(firstFrame->shell);
    // ef->insertFrame(firstFrame, &Hcalib);
    // setPrecalcValues();

    
    // firstFrame->pointHessians.reserve(FirstFrame->nFeatures);
    // firstFrame->pointHessiansMarginalized.reserve(FirstFrame->nFeatures);
    // firstFrame->pointHessiansOut.reserve(FirstFrame->nFeatures);

    for (int i = 0; i < firstFrame->nFeatures; ++i)
    {
        if(_cInit->videpth[i] <= 0.0f)
            continue;

        // Pnt *point = coarseInitializer->points[0] + i;
        // ImmaturePoint *pt = new ImmaturePoint(firstFrame->mvKeys[i].pt.x + 0.5f, firstFrame->mvKeys[i].pt.y + 0.5f, firstFrame, 1, &Hcalib);

        // if (!std::isfinite(pt->energyTH))
        // {
        //     delete pt;
        //     continue;
        // }

        // pt->idepth_max = pt->idepth_min = 1;
        // PointHessian *ph = new PointHessian(pt, &Hcalib);
        // delete pt;
        // if (!std::isfinite(ph->energyTH))
        // {
        //     delete ph;
        //     continue;
        // }

        // ph->setIdepthScaled(videpth[i]);
        // ph->setIdepthZero(videpth[i]);
        // ph->hasDepthPrior = true;
        // ph->setPointStatus(PointHessian::ACTIVE);

        // firstFrame->pointHessians.push_back(ph);
        // ef->insertPoint(ph);
    }

    SE3 firstToNew = _cInit->Pose;
    // firstToNew.translation() /= rescaleFactor;

    // really no lock required, as we are initializing.
    {
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        firstFrame->camToWorld = SE3();
        firstFrame->aff_g2l_internal = AffLight(0, 0);
        firstFrame->setEvalPT_scaled(firstFrame->camToWorld.inverse(), firstFrame->aff_g2l_internal);
        firstFrame->trackingRef.reset();
        firstFrame->camToTrackingRef = SE3();

        secondFrame->camToWorld = firstToNew.inverse();
        secondFrame->aff_g2l_internal = AffLight(0, 0);
        secondFrame->setEvalPT_scaled(secondFrame->camToWorld.inverse(), secondFrame->aff_g2l_internal);
        secondFrame->trackingRef = firstFrame;
        secondFrame->camToTrackingRef =  secondFrame->camToWorld;
    }

    // Initialized = true;
    // printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

} // namespace FSLAM
