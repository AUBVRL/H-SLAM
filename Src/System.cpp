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
    // selectionMap = new float[Calib->Width* Calib->Height];
	coarseDistanceMap = new CoarseDistanceMap(Calib->Width, Calib->Height);
	coarseTracker = new CoarseTracker(Calib->Width, Calib->Height);
	coarseTracker_forNewKF = new CoarseTracker(Calib->Width, Calib->Height);
	// coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	// pixelSelector = new PixelSelector(wG[0], hG[0]);
    lastCoarseRMSE.setConstant(100);
	currentMinActDist=2;

    ef = std::shared_ptr<EnergyFunctional> (new EnergyFunctional());
	ef->red = &this->treadReduce;


    //----------------end dso------------------


	isLost=false;
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
    // delete[] selectionMap;

    // for (FrameShell *s : allFrameHistory)
    //     delete s;
    // for (Frame *fh : unmappedTrackedFrames)
    //     delete fh;

    delete coarseDistanceMap;
    delete coarseTracker;
    delete coarseTracker_forNewKF;
    // delete coarseInitializer;
    // delete pixelSelector;
    
    //----------------end dso------------------
    
    
}

void System::ProcessNewFrame(std::shared_ptr<ImageData> DataIn)
{
    std::shared_ptr<Frame> CurrentFrame = std::shared_ptr<Frame>(new Frame(DataIn, Detector, Calib, FrontEndThreadPoolLeft, !Initialized)); // FrontEndThreadPoolRight
    allFrameHistory.push_back(CurrentFrame);

    boost::unique_lock<boost::mutex> lock(trackMutex);

    bool NeedNewKf = false;

    if(!Initialized)
    {   //Initialize..
        if(!cInitializer)
            cInitializer = std::shared_ptr<Initializer>(new Initializer(Calib, FrontEndThreadPoolLeft ,DisplayHandler));

        if(cInitializer->Initialize(CurrentFrame))
        {
            InitFromInitializer(cInitializer);
            cInitializer.reset();
            NeedNewKf = true;
        }
    }
    else
    {
        if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
        {
            boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
            CoarseTracker *tmp = coarseTracker;
            coarseTracker = coarseTracker_forNewKF;
            coarseTracker_forNewKF = tmp;
        }
        
        Vec4 tres;
        
        switch (Sensortype)
        {
        case Monocular:
            tres = trackNewCoarse(CurrentFrame);
            break;
        case Stereo:
            //TrackStereo()
            break;
        case RGBD:
            //TrackRGBD()
            break;
        }

        if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
        {
            printf("Initial Tracking failed: LOST!\n");
            isLost = true;
            return;
        }

        if (setting_keyframesPerSecond > 0)
        {
            NeedNewKf = allFrameHistory.size() == 1 ||
                           (CurrentFrame->TimeStamp - allKeyFramesHistory.back()->TimeStamp) > 0.95f / setting_keyframesPerSecond;
        }
        else
        {
            Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, CurrentFrame->ab_exposure,
                                                       coarseTracker->lastRef_aff_g2l, CurrentFrame->aff_g2l_internal);

            // BRIGHTNESS CHECK
            NeedNewKf = allFrameHistory.size() == 1 ||
                           setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double)tres[1]) / (Calib->Width + Calib->Height) +
                                   setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double)tres[2]) / (Calib->Width + Calib->Height) +
                                   setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (Calib->Width + Calib->Height) +
                                   setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float)refToFh[0])) >
                               1 ||
                           2 * coarseTracker->firstCoarseRMSE < tres[0];
        }
    }

    DrawImages(DataIn, CurrentFrame);
    if (SequentialOperation && Initialized)
    {
        if (NeedNewKf)
            AddKeyframe(CurrentFrame);
        else
            ProcessNonKeyframe(CurrentFrame);
    }
    else if(Initialized) //Parallel Computation
    {
        boost::unique_lock<boost::mutex> lock(MapThreadMutex);
        UnmappedTrackedFrames.push_back(CurrentFrame);
        if (NeedNewKf)
            NeedNewKFAfter = CurrentFrame->trackingRef->id;
        
        TrackedFrameSignal.notify_all();        
        while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
		{
			MappedFrameSignal.wait(lock);
		}

        // int count = 0;
        // for (auto it : allKeyFramesHistory)
        //     for (auto it2 : it->pointHessians)
        //         if (it2 && (it2->status == MapPoint::ACTIVE || it2->status == MapPoint::MARGINALIZED))
        //             count++;
        // std::cout << count << std::endl;

        lock.unlock();
    }

    //only called if online photometric calibration is required (keep this here and not in the photometric undistorter to have access to slam data)
    // if(OnlinePhCalibL) 
    //     OnlinePhCalibL->ProcessFrame(Frame.cvImgL);
    // if(OnlinePhCalibR)
    //     OnlinePhCalibL->ProcessFrame(Frame.cvImgR);
    

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

void System::ProcessNonKeyframe(std::shared_ptr<Frame> fh)
{
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->trackingRef != nullptr);
		fh->camToWorld = fh->trackingRef->camToWorld * fh->camToTrackingRef;
		fh->setEvalPT_scaled(fh->camToWorld.inverse(),fh->aff_g2l_internal);
	}

	traceNewCoarse(fh);
    fh->ReduceToEssential(false);
	fh.reset();
}

void System::InitFromInitializer(std::shared_ptr<Initializer> _cInit)
{
    std::shared_ptr<Frame> firstFrame = _cInit->FirstFrame;
    std::shared_ptr<Frame> secondFrame = _cInit->SecondFrame;
    firstFrame->idx = 0;
    frameHessians.push_back(_cInit->FirstFrame);
    firstFrame->id = 0;

    allKeyFramesHistory.push_back(firstFrame);
    ef->insertFrame(firstFrame, Calib);
    setPrecalcValues();

    
    firstFrame->pointHessians.resize(firstFrame->nFeatures, nullptr);

    for (int i = 0; i < firstFrame->nFeatures; ++i)
    {
        if(_cInit->videpth[i] <= 0.0f)
            continue;

        std::shared_ptr<ImmaturePoint> pt = std::shared_ptr<ImmaturePoint>(new ImmaturePoint(firstFrame->mvKeys[i].pt.x + 0.5f, firstFrame->mvKeys[i].pt.y + 0.5f, firstFrame, 1, Calib));

        if (!std::isfinite(pt->energyTH))
            continue;

        pt->idepth_max = pt->idepth_min = _cInit->videpth[i];
        std::shared_ptr<MapPoint> ph = std::shared_ptr<MapPoint>(new MapPoint(pt,Calib));
    
        if (!std::isfinite(ph->energyTH))
            continue;
        

        ph->setIdepthScaled(_cInit->videpth[i]);
        ph->setIdepthZero(_cInit->videpth[i]);
        ph->hasDepthPrior = true;
        ph->setPointStatus(MapPoint::ACTIVE);

        firstFrame->pointHessians[i] = ph;
        ef->insertPoint(ph);
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

    Initialized = true;
    printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size() - (int)std::count(firstFrame->pointHessians.begin(), firstFrame->pointHessians.end(), nullptr) );
}

void System::setPrecalcValues()
{
	for(auto fh : frameHessians)
	{
		fh->targetPrecalc.resize(frameHessians.size());
		for(unsigned int i=0, iend = frameHessians.size(); i<iend; ++i)
			fh->targetPrecalc[i].set(fh, frameHessians[i], Calib);
	}

	ef->setDeltaF(Calib);
}

Vec4 System::trackNewCoarse(std::shared_ptr<Frame> fh)
{

	assert(allFrameHistory.size() > 0);
	// set pose initialization.

    // for(IOWrap::Output3DWrapper* ow : outputWrapper)
    //     ow->pushLiveFrame(fh);



	std::shared_ptr<Frame> lastF = coarseTracker->lastRef;

	AffLight aff_last_2_l = AffLight(0,0);

	std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
	if(allFrameHistory.size() == 2)
		for(unsigned int i=0;i<lastF_2_fh_tries.size();i++) lastF_2_fh_tries.push_back(SE3());
	else
	{
		std::shared_ptr<Frame> slast = allFrameHistory[allFrameHistory.size()-2];
		std::shared_ptr<Frame> sprelast = allFrameHistory[allFrameHistory.size()-3];
		SE3 slast_2_sprelast;
		SE3 lastF_2_slast;
		{	// lock on global pose consistency!
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
			lastF_2_slast = slast->camToWorld.inverse() * lastF->camToWorld;
			aff_last_2_l = slast->aff_g2l_internal;
		}
		SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.


		// get last delta-movement.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	// assume constant motion.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
		lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast); // assume half motion.
		lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
		lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.


		// just try a TON of different initializations (all rotations). In the end,
		// if they don't work they will only be tried on the coarsest level, which is super fast anyway.
		// also, if tracking rails here we loose, so we really, really want to avoid that.
		for(float rotDelta=0.02; rotDelta < 0.05; rotDelta++)
		{
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		}

		if(!slast->poseValid || !sprelast->poseValid || !lastF->poseValid)
		{
			lastF_2_fh_tries.clear();
			lastF_2_fh_tries.push_back(SE3());
		}
	}


	Vec3 flowVecs = Vec3(100,100,100);
	SE3 lastF_2_fh = SE3();
	AffLight aff_g2l = AffLight(0,0);


	// as long as maxResForImmediateAccept is not reached, I'll continue through the options.
	// I'll keep track of the so-far best achieved residual for each level in achievedRes.
	// If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.


	Vec5 achievedRes = Vec5::Constant(NAN);
	bool haveOneGood = false;
	int tryIterations=0;
	for(unsigned int i=0;i<lastF_2_fh_tries.size();i++)
	{
		AffLight aff_g2l_this = aff_last_2_l;
		SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
		bool trackingIsGood = coarseTracker->trackNewestCoarse(
				fh, lastF_2_fh_this, aff_g2l_this,
				DirPyrLevels-1,
				achievedRes);	// in each level has to be at least as good as the last try.
		tryIterations++;

		// if(i != 0)
		// {
		// 	printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
		// 			i,
		// 			i, DirPyrLevels-1,
		// 			aff_g2l_this.a,aff_g2l_this.b,
		// 			achievedRes[0],
		// 			achievedRes[1],
		// 			achievedRes[2],
		// 			achievedRes[3],
		// 			achievedRes[4],
		// 			coarseTracker->lastResiduals[0],
		// 			coarseTracker->lastResiduals[1],
		// 			coarseTracker->lastResiduals[2],
		// 			coarseTracker->lastResiduals[3],
		// 			coarseTracker->lastResiduals[4]);
		// }


		// do we have a new winner?
		if(trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >=  achievedRes[0]))
		{
			//printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
			flowVecs = coarseTracker->lastFlowIndicators;
			aff_g2l = aff_g2l_this;
			lastF_2_fh = lastF_2_fh_this;
			haveOneGood = true;
		}

		// take over achieved res (always).
		if(haveOneGood)
		{
			for(int i=0;i<5;i++)
			{
				if(!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])	// take over if achievedRes is either bigger or NAN.
					achievedRes[i] = coarseTracker->lastResiduals[i];
			}
		}


        if(haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
            break;

	}

	if(!haveOneGood)
	{
        printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
		flowVecs = Vec3(0,0,0);
		aff_g2l = aff_last_2_l;
		lastF_2_fh = lastF_2_fh_tries[0];
	}

	lastCoarseRMSE = achievedRes;

	// no lock required, as fh is not used anywhere yet.
	fh->camToTrackingRef = lastF_2_fh.inverse();
	fh->trackingRef = lastF;
	fh->aff_g2l_internal = aff_g2l;
	fh->camToWorld = fh->trackingRef->camToWorld * fh->camToTrackingRef;
    fh->poseValid = true;

	if(coarseTracker->firstCoarseRMSE < 0)
		coarseTracker->firstCoarseRMSE = achievedRes[0];

    // if(!setting_debugout_runquiet)
    //     printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);



	// if(setting_logStuff)
	// {
	// 	(*coarseTrackingLog) << std::setprecision(16)
	// 					<< fh->shell->id << " "
	// 					<< fh->shell->timestamp << " "
	// 					<< fh->ab_exposure << " "
	// 					<< fh->shell->camToWorld.log().transpose() << " "
	// 					<< aff_g2l.a << " "
	// 					<< aff_g2l.b << " "
	// 					<< achievedRes[0] << " "
	// 					<< tryIterations << "\n";
	// }


	return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}



} // namespace FSLAM
