#include "System.h"
#include "geomUndistorter.h"
#include "photoUndistorter.h"
#include "Detector.h"

#include "CalibData.h"
#include "MapPoint.h"
#include "Frame.h"
#include "ImmaturePoint.h"
#include "Initializer.h"
#include "EnergyFunctional.h"
#include "OptimizationClasses.h"
#include "Display.h"
#include "CoarseTracker.h"
#include <opencv2/imgproc.hpp>
#include "IndexThreadReduce.h"
// #include "DBoW3/Vocabulary.h"

#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <numeric>
#include <chrono>

namespace SLAM
{

    System::System(shared_ptr<geomUndistorter> geomUndist, shared_ptr<photoUndistorter> phoUndist, 
    shared_ptr<GUI> _DisplayHandler): gUndist(geomUndist), pUndist(phoUndist), isInitialized(false), 
    NeedNewKFAfter(-1), currentMinActDist(2.0f), initFailed(false), displayHandler(_DisplayHandler)
         //
    {
        geomCalib = shared_ptr<CalibData>(new CalibData(gUndist->w, gUndist->h, gUndist->K, pyramidSize));
        frontEndThreadPool = shared_ptr<IndexThreadReduce<Vec10>>(new IndexThreadReduce<Vec10>());
        backEndThreadPool = shared_ptr<IndexThreadReduce<Vec10>>(new IndexThreadReduce<Vec10>());

        detector = shared_ptr<FeatureDetector>(new FeatureDetector(geomCalib));

        ef = shared_ptr<EnergyFunctional> (new EnergyFunctional());
	    ef->red = backEndThreadPool;

        mappingThread = thread(&System::MappingThread, this);

        coarseDistanceMap = shared_ptr<CoarseDistanceMap>(new CoarseDistanceMap(geomCalib));
        coarseTracker = shared_ptr<CoarseTracker> (new CoarseTracker(geomCalib));
	    coarseTracker_forNewKF = shared_ptr<CoarseTracker> (new CoarseTracker(geomCalib));
        lastCoarseRMSE.setConstant(100);
        isLost = false;
    }

    System::~System()
    {
        BlockUntilMappingIsFinished();
        coarseDistanceMap.reset();
        coarseTracker_forNewKF.reset(); 
        coarseTracker.reset();
        UnmappedTrackedFrames.clear(); UnmappedTrackedFrames.resize(0);
        frameHessians.clear(); frameHessians.resize(0);
        allKeyFramesHistory.clear(); allKeyFramesHistory.resize(0); 
	    allFrameHistory.clear(); allFrameHistory.resize(0);
        ef.reset();
        backEndThreadPool.reset();
        frontEndThreadPool.reset();
        geomCalib.reset();
        detector.reset();
        pUndist.reset();
        gUndist.reset();
        initializer.reset();
        displayHandler.reset();
    }

    void System::ProcessFrame(shared_ptr<ImageData> dataIn)
    {
        TrackTime.startTime();
        bool needNewKf = false;
        shared_ptr<FrameShell> CurrentFrame = shared_ptr<FrameShell>(new FrameShell(dataIn, geomCalib, pUndist, false)); // FrontEndThreadPoolRight    
        CurrentFrame->frame->Extract(detector, CurrentFrame->id, true, frontEndThreadPool);

        allFrameHistory.push_back(CurrentFrame);

        if(!isInitialized)
        {
            if(!initializer)
                initializer = shared_ptr<Initializer>(new Initializer(geomCalib, frontEndThreadPool));
            if(initializer->Initialize(CurrentFrame))
            {
                cout<<"Initialized!!"<<endl;
                InitFromInitializer(initializer);
                initializer.reset();
                needNewKf = true;
            }
            else
            {
                if (CurrentFrame != initializer->FirstFrame)
                    CurrentFrame->frame.reset();
                
            }
        }
        else
        {
            // Perform TRACKING!

            // UPDATE TRACKER ,,, BETTER WAY OF DOING THIS WITHOUT HOLDING 2 tracker copies!!
            if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
            {
                unique_lock<mutex> crlock(coarseTrackerSwapMutex);
                shared_ptr<CoarseTracker> tmp = coarseTracker;
                coarseTracker = coarseTracker_forNewKF;
                coarseTracker_forNewKF = tmp;

                // coarseTracker.swap(coarseTracker_forNewKF);
            }

            // Perform tracking
            Vec4 tres = trackNewCoarse(CurrentFrame);

            // Check if tracking failed
            if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
            {
                cout<<"Initial Tracking failed: LOST!\n"<<endl;
                CurrentFrame->frame->efFrame.reset();
                CurrentFrame->frame.reset();
                isLost = true;
                return;
            }

            // Check if we need new keyframe
             Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, CurrentFrame->ab_exposure,
                                                       coarseTracker->lastRef_aff_g2l, CurrentFrame->aff_g2l);

            // BRIGHTNESS CHECK
            needNewKf = allFrameHistory.size() == 1 ||
                           setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double)tres[1]) / (geomCalib->Width + geomCalib->Height) +
                                   setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double)tres[2]) / (geomCalib->Width + geomCalib->Height) +
                                   setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (geomCalib->Width + geomCalib->Height) +
                                   setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 1 
                                   || 2 * coarseTracker->firstCoarseRMSE < tres[0];
        
        }
        
        if(displayHandler)
            displayHandler->UploadCurrentFrame(dataIn, CurrentFrame->frame->mvKeys, CurrentFrame->camToWorld, CurrentFrame->poseValid);
        
        if(isInitialized)
            toMapping(CurrentFrame, needNewKf);

        
        // CurrentFrame->frame->Extract(CurrentFrame->id, false);
    
    
        // cv::Mat Output;
        // CurrentFrame->frame->Image.convertTo(Output, CV_8UC3);
        // cv::drawKeypoints(Output, CurrentFrame->frame->mvKeys, Output, cv::Scalar(0,255,0));
        // cv::namedWindow("test", cv::WINDOW_KEEPRATIO);
        // cv::imshow("test", Output);
        // cv::waitKey(1);
        TrackTime.endTime(true);
        return;
    }

    void System::InitFromInitializer(shared_ptr<Initializer> _cInit)
    {
        shared_ptr<FrameShell> firstFrame = _cInit->FirstFrame;
        shared_ptr<FrameShell> secondFrame = _cInit->SecondFrame;
        firstFrame->frame->idx = 0;
        frameHessians.push_back(_cInit->FirstFrame);
        
        firstFrame->KfId = 0;
        FrameShell::GlobalKfId = 1;
        allKeyFramesHistory.push_back(firstFrame);
        ef->insertFrame(firstFrame, geomCalib);
        setPrecalcValues();

        // firstFrame->frame->pointHessians.resize(firstFrame->frame->nFeatures, nullptr);
        firstFrame->frame->pointHessians.reserve(firstFrame->frame->nFeatures);

        for (int i = 0; i < firstFrame->frame->nFeatures; ++i)
        {
            if (_cInit->videpth[i] <= 0.0f)
                continue;

            shared_ptr<ImmaturePoint> pt = shared_ptr<ImmaturePoint>(
                new ImmaturePoint(firstFrame->frame->mvKeys[i].pt.x, firstFrame->frame->mvKeys[i].pt.y, firstFrame, 1, geomCalib)); //offset +0.5f

            if (!std::isfinite(pt->energyTH))
                continue;

            pt->idepth_max = pt->idepth_min = _cInit->videpth[i];
            shared_ptr<MapPoint> ph = shared_ptr<MapPoint>(new MapPoint(pt, geomCalib));

            if (!std::isfinite(ph->energyTH))
                continue;

            ph->setIdepth(_cInit->videpth[i]);
            ph->efPoint->setIdepthZero(_cInit->videpth[i]);
            ph->hasDepthPrior = true;
            ph->setPointStatus(ACTIVE);

            firstFrame->frame->pointHessians.push_back(ph);
            ef->insertPoint(ph);

        }

        SE3 firstToNew = _cInit->Pose;

        firstFrame->camToWorld = SE3();
        firstFrame->aff_g2l = AffLight(0, 0);
        firstFrame->frame->efFrame->setEvalPT_scaled(firstFrame->camToWorld.inverse(), firstFrame->aff_g2l);
        firstFrame->trackingRef.reset();
        firstFrame->camToTrackingRef = SE3();

        firstFrame->frame->NeedRefresh = true;
        if (displayHandler)
            displayHandler->UploadKeyFrame(firstFrame);

        secondFrame->camToWorld = firstToNew.inverse();
        secondFrame->aff_g2l = AffLight(0, 0);
        secondFrame->frame->efFrame->setEvalPT_scaled(secondFrame->camToWorld.inverse(), secondFrame->aff_g2l);
        secondFrame->trackingRef = firstFrame;
        secondFrame->camToTrackingRef = secondFrame->camToWorld;

        isInitialized = true;
        firstFrame->poseValid = true;
        secondFrame->poseValid = true;
        printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->frame->pointHessians.size() -
                                                          (int)count(firstFrame->frame->pointHessians.begin(), 
                                                          firstFrame->frame->pointHessians.end(), nullptr));
        return;
    }

    void System::setPrecalcValues()
    {
        for (auto &fh : frameHessians)
        {
            fh->frame->targetPrecalc.resize(frameHessians.size());
            for (unsigned int i = 0, iend = frameHessians.size(); i < iend; ++i)
                fh->frame->targetPrecalc[i].set(fh, frameHessians[i], geomCalib);
        }

        ef->setDeltaF(geomCalib);
        return;
    }

    void System::toMapping(shared_ptr<FrameShell> currentFrame, bool needKeyframe)
    {
        if (isLost)
            return;
        if (SequentialOperation)
        {
            if (needKeyframe)
            {
                AddKeyframe(currentFrame);
            }
            else
            {   
                ProcessNonKeyframe(currentFrame);
            }
        }
        else
        {
		    unique_lock<mutex> lock(trackMapSyncMutex);
            UnmappedTrackedFrames.push_back(currentFrame);
            if (needKeyframe)
                NeedNewKFAfter = currentFrame->trackingRef->id;

            TrackedFrameSignal.notify_all();
            while (coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1)
                MappedFrameSignal.wait(lock);
        }

        return;
    }

    void System::AddKeyframe(shared_ptr<FrameShell> currentFrame)
    {
        {
            unique_lock<mutex> crlock(shellPoseMutex);
            assert(currentFrame->trackingRef != nullptr);
            currentFrame->camToWorld = currentFrame->trackingRef->camToWorld * currentFrame->camToTrackingRef;
            currentFrame->frame->efFrame->setEvalPT_scaled(currentFrame->camToWorld.inverse(), currentFrame->aff_g2l);
            currentFrame->KfId = FrameShell::GlobalKfId;
            FrameShell::GlobalKfId++;
            currentFrame->isKeyFrame = true;
        }
        if (displayHandler)
            displayHandler->UploadKeyFrame(currentFrame);

        traceNewCoarse(currentFrame);

        unique_lock<mutex> lock(mapMutex);

        // =========================== Flag Frames to be Marginalized. =========================
        flagFramesForMarginalization();

        // =========================== add New Frame to Hessian Struct. =========================
        currentFrame->frame->idx = frameHessians.size();
        frameHessians.push_back(currentFrame);
        // fh->frameID = allKeyFramesHistory.size();
        allKeyFramesHistory.push_back(currentFrame);
        ef->insertFrame(currentFrame, geomCalib);

        setPrecalcValues();
        // =========================== add new residuals for old points =========================
        int numFwdResAdde = 0;
        for (auto &fh1 : frameHessians) // go through all active frames
        {
            if (fh1 == currentFrame)
                continue;
            for (auto &ph : fh1->frame->pointHessians)
            {
                // if(!ph || ph->getPointStatus() != ACTIVE)
                //     continue;
                shared_ptr<PointFrameResidual> r = shared_ptr<PointFrameResidual>(new PointFrameResidual(ph, fh1, currentFrame));
                r->setState(ResState::IN);
                ph->residuals.push_back(r);
                ef->insertResidual(r, ph->residuals.size());
                ph->lastResiduals[1] = ph->lastResiduals[0];
                ph->lastResiduals[0] = pair<shared_ptr<PointFrameResidual>, ResState>(r, ResState::IN);
                numFwdResAdde += 1;
            }
        }
        // =========================== Activate Points (& flag for marginalization). =========================
        activatePoints();
        ef->makeIDX();

        // =========================== OPTIMIZE ALL =========================

        currentFrame->frame->frameEnergyTH = frameHessians.back()->frame->frameEnergyTH;
        float rmse = optimize(setting_maxOptIterations);

        // =========================== Figure Out if INITIALIZATION FAILED =========================
        if (allKeyFramesHistory.size() <= 4)
        {
            if (allKeyFramesHistory.size() == 2 && rmse > 20 * benchmark_initializerSlackFactor)
            {
                printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
                initFailed = true;
            }
            if (allKeyFramesHistory.size() == 3 && rmse > 13 * benchmark_initializerSlackFactor)
            {
                printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
                initFailed = true;
            }
            if (allKeyFramesHistory.size() == 4 && rmse > 9 * benchmark_initializerSlackFactor)
            {
                printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
                initFailed = true;
            }
        }


        // =========================== REMOVE OUTLIER =========================
        removeOutliers();

        {
            unique_lock<mutex> crlock(coarseTrackerSwapMutex);
            // coarseTracker_forNewKF->makeK(geomCalib);
            coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);

           
            if(displayHandler) //Draw upcoming active keyframe depth map
            {
                vector<pair<float, int>> dmap = coarseTracker_forNewKF->GetKFDepthMap();
                displayHandler->UploadActiveKF(coarseTracker_forNewKF->lastRef->frame->Image, dmap);
            }
        }

        // =========================== (Activate-)Marginalize Points =========================
        flagPointsForRemoval();
        ef->dropPointsF();
        getNullspaces(
            ef->lastNullspaces_pose,
            ef->lastNullspaces_scale,
            ef->lastNullspaces_affA,
            ef->lastNullspaces_affB);
        ef->marginalizePointsF();

        // =========================== add new Immature points & new residuals =========================
        makeNewTraces(currentFrame);

        // for(IOWrap::Output3DWrapper* ow : outputWrapper)
        // {
        //     ow->publishGraph(ef->connectivityMap);
        //     ow->publishKeyframes(frameHessians, false, &Hcalib);
        // }

        // =========================== Let the display know that these KFs need updating =========================
        if (displayHandler)
        {
            // boost::unique_lock<boost::mutex> lock(DisplayHandler->KeyframesMutex);
            for (auto &it : frameHessians)
                it->frame->NeedRefresh = true;
        }

        // =========================== Marginalize Frames =========================

        for (unsigned int i = 0; i < frameHessians.size(); i++)
        {
            if (frameHessians[i]->frame->FlaggedForMarginalization)
            {
                marginalizeFrame(frameHessians[i]);
                i = 0;
            }
        }

        return;
    }

    void System::ProcessNonKeyframe(shared_ptr<FrameShell> currentFrame)
    {
        {
            unique_lock<mutex> crlock(shellPoseMutex);
            assert(currentFrame->trackingRef != nullptr);
            currentFrame->camToWorld = currentFrame->trackingRef->camToWorld * currentFrame->camToTrackingRef;
            currentFrame->frame->efFrame->setEvalPT_scaled(currentFrame->camToWorld.inverse(), currentFrame->aff_g2l);
        }

        traceNewCoarse(currentFrame);

        if (currentFrame->frame->efFrame)
            currentFrame->frame->efFrame.reset();
        if (currentFrame->frame)
            currentFrame->frame.reset();
        currentFrame.reset();

        return;
    }

    void System::traceNewCoarse(shared_ptr<FrameShell> currentFrame)
    {
        unique_lock<mutex> lock(mapMutex);
        int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

        Mat33f K = Mat33f::Identity();
        K(0, 0) = geomCalib->fxl();
        K(1, 1) = geomCalib->fyl();
        K(0, 2) = geomCalib->cxl();
        K(1, 2) = geomCalib->cyl();
        for (auto &host : frameHessians)
        {

            SE3 hostToNew = currentFrame->frame->efFrame->PRE_worldToCam * host->frame->efFrame->PRE_camToWorld;
            Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
            Vec3f Kt = K * hostToNew.translation().cast<float>();

            Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, currentFrame->ab_exposure, host->frame->efFrame->aff_g2l(), currentFrame->frame->efFrame->aff_g2l()).cast<float>();

            for (auto &ph : host->frame->ImmaturePoints)
            {
                if (!ph)
                    continue;
                ph->traceOn(currentFrame, KRKi, Kt, aff, geomCalib, false);
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD)
                    trace_good++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION)
                    trace_badcondition++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB)
                    trace_oob++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
                    trace_out++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED)
                    trace_skip++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED)
                    trace_uninitialized++;
                trace_total++;
            }
        }       
        return;
    }


    void System::marginalizeFrame(shared_ptr<FrameShell> frame)
    {
        // marginalize or remove all this frames points.

        // assert((int)frame->pointHessians.size()==0);

        ef->marginalizeFrame(frame);

        // drop all observations of existing points in that frame.

        for (auto &fh : frameHessians)
        {
            if (fh == frame)
                continue;

            for (auto &ph : fh->frame->pointHessians)
            {
                // size_t n = ph->residuals.size();
                for (unsigned int i = 0; i < ph->residuals.size(); i++)
                {
                    shared_ptr<PointFrameResidual> r = ph->residuals[i];

                    if (r->target == frame)
                    {
                        if (ph->lastResiduals[0].first == r)
                            ph->lastResiduals[0].first.reset();
                        else if (ph->lastResiduals[1].first == r)
                            ph->lastResiduals[1].first.reset();

                        // if(r->host->KfId < r->target->KfId)
                        // 	statistics_numForceDroppedResFwd++;
                        // else
                        // 	statistics_numForceDroppedResBwd++;

                        ef->dropResidual(r); //this should remove the only holding copy of the pointframeresidual

                        // deleteOut<PointFrameResidual>(ph->residuals, i);
                        break;
                    }
                }
            }
        }

        {
            // std::vector<Frame*> v;
            // v.push_back(frame);
            // for(IOWrap::Output3DWrapper* ow : outputWrapper)
            //     ow->publishKeyframes(v, true, &Hcalib);
        }

        frame->MarginalizedAt = frameHessians.back()->id;
        frame->frame->MovedByOpt = frame->frame->efFrame->w2c_leftEps().norm();
        frame->frame->ReduceToEssential();

        if (displayHandler)
        {
            // boost::unique_lock<boost::mutex> lock(DisplayHandler->KeyframesMutex);
            frame->frame->NeedRefresh = true;
        }

        deleteOutOrder<FrameShell>(frameHessians, frame);
        for (unsigned int i = 0; i < frameHessians.size(); i++)
            frameHessians[i]->frame->idx = i;

        setPrecalcValues();
        ef->setAdjointsF(geomCalib);
    }


    void System::BlockUntilMappingIsFinished()
    {
        unique_lock<mutex> lock(trackMapSyncMutex);
        RunMapping = false;
        TrackedFrameSignal.notify_all();
        lock.unlock();

        mappingThread.join();
    }


    void System::MappingThread()
    {
        unique_lock<mutex> lock(trackMapSyncMutex);
        bool NeedToCatchUp = false;
        RunMapping = true;
        while (RunMapping)
        {
            while (UnmappedTrackedFrames.size() == 0)
            {
                TrackedFrameSignal.wait(lock);
                if (!RunMapping)
                    return;
            }

            shared_ptr<FrameShell> frame = UnmappedTrackedFrames.front();
            UnmappedTrackedFrames.pop_front();

            // guaranteed to make a KF for the very first two tracked frames.
            if (allKeyFramesHistory.size() <= 2)
            {
                lock.unlock();
                AddKeyframe(frame);
                lock.lock();
                MappedFrameSignal.notify_all();
                continue;
            }

            if (UnmappedTrackedFrames.size() > 3)
                NeedToCatchUp = true;

            if (UnmappedTrackedFrames.size() > 0) // if there are other frames to track, do that first.
            {
                lock.unlock();
                ProcessNonKeyframe(frame);
                lock.lock();

                if (NeedToCatchUp && UnmappedTrackedFrames.size() > 0)
                {
                    shared_ptr<FrameShell> frame = UnmappedTrackedFrames.front();
                    UnmappedTrackedFrames.pop_front();
                    {
                        unique_lock<mutex> crlock(shellPoseMutex);
                        assert(frame->trackingRef);
                        frame->camToWorld = frame->trackingRef->camToWorld * frame->camToTrackingRef;
                        frame->frame->efFrame->setEvalPT_scaled(frame->camToWorld.inverse(), frame->aff_g2l);
                    }

                    frame->frame->efFrame.reset();
                    frame->frame.reset();
                    frame.reset();
                }
            }
            else
            {
                if (setting_realTimeMaxKF || NeedNewKFAfter >= frameHessians.back()->id)
                {
                    lock.unlock();
                    AddKeyframe(frame);
                    NeedToCatchUp = false;
                    lock.lock();
                }
                else
                {
                    lock.unlock();
                    ProcessNonKeyframe(frame);
                    lock.lock();
                }
            }
            MappedFrameSignal.notify_all();
        }
        printf("MAPPING FINISHED!\n");
    }


    Vec4 System::trackNewCoarse(shared_ptr<FrameShell> &fh)
    {

        assert(allFrameHistory.size() > 0);
        // set pose initialization.

        // for(IOWrap::Output3DWrapper* ow : outputWrapper)
        //     ow->pushLiveFrame(fh);

        shared_ptr<FrameShell> lastF = coarseTracker->lastRef;

        AffLight aff_last_2_l = AffLight(0, 0);

        vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
        if (allFrameHistory.size() == 2)
            for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
                lastF_2_fh_tries.push_back(SE3());
        else
        {
            shared_ptr<FrameShell> slast = allFrameHistory[allFrameHistory.size() - 2];
            shared_ptr<FrameShell> sprelast = allFrameHistory[allFrameHistory.size() - 3];
            SE3 slast_2_sprelast;
            SE3 lastF_2_slast;
            { // lock on global pose consistency!
                unique_lock<mutex> crlock(shellPoseMutex);
                slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
                lastF_2_slast = slast->camToWorld.inverse() * lastF->camToWorld;
                aff_last_2_l = slast->aff_g2l;
            }
            SE3 fh_2_slast = slast_2_sprelast; // assumed to be the same as fh_2_slast.

            // get last delta-movement.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);                        // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast); // assume double motion (frame skipped)
            lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast);  // assume half motion.
            lastF_2_fh_tries.push_back(lastF_2_slast);                                               // assume zero motion.
            lastF_2_fh_tries.push_back(SE3());                                                       // assume zero motion FROM KF.

            // just try a TON of different initializations (all rotations). In the end,
            // if they don't work they will only be tried on the coarsest level, which is super fast anyway.
            // also, if tracking rails here we loose, so we really, really want to avoid that.
            for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta++)
            {
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0, 0)));                  // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0, 0)));                  // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0, 0)));                  // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0, 0)));                 // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0, 0)));                 // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0, 0)));                 // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0), Vec3(0, 0, 0)));           // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta), Vec3(0, 0, 0)));           // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta), Vec3(0, 0, 0)));           // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0), Vec3(0, 0, 0)));          // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));          // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));         // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta), Vec3(0, 0, 0)));         // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));         // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));   // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));   // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));   // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));    // assume constant motion.
            }

            if (!slast->poseValid || !sprelast->poseValid || !lastF->poseValid)
            {
                lastF_2_fh_tries.clear();
                lastF_2_fh_tries.push_back(SE3());
            }
        }

        Vec3 flowVecs = Vec3(100, 100, 100);
        SE3 lastF_2_fh = SE3();
        AffLight aff_g2l = AffLight(0, 0);

        // as long as maxResForImmediateAccept is not reached, I'll continue through the options.
        // I'll keep track of the so-far best achieved residual for each level in achievedRes.
        // If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.

        Vec5 achievedRes = Vec5::Constant(NAN);
        bool haveOneGood = false;
        int tryIterations = 0;
        for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
        {
            AffLight aff_g2l_this = aff_last_2_l;
            SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
            bool trackingIsGood = coarseTracker->trackNewestCoarse(
                fh, lastF_2_fh_this, aff_g2l_this,
                pyramidSize - 1,
                achievedRes); // in each level has to be at least as good as the last try.
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
            if (trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >= achievedRes[0]))
            {
                //printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
                flowVecs = coarseTracker->lastFlowIndicators;
                aff_g2l = aff_g2l_this;
                lastF_2_fh = lastF_2_fh_this;
                haveOneGood = true;
            }

            // take over achieved res (always).
            if (haveOneGood)
            {
                for (int i = 0; i < 5; i++)
                {
                    if (!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i]) // take over if achievedRes is either bigger or NAN.
                        achievedRes[i] = coarseTracker->lastResiduals[i];
                }
            }

            if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)
                break;
        }

        if (!haveOneGood)
        {
            printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
            flowVecs = Vec3(0, 0, 0);
            aff_g2l = aff_last_2_l;
            lastF_2_fh = lastF_2_fh_tries[0];
        }

        lastCoarseRMSE = achievedRes;

        // no lock required, as fh is not used anywhere yet.
        fh->camToTrackingRef = lastF_2_fh.inverse();
        fh->trackingRef = lastF;
        fh->aff_g2l = aff_g2l;
        fh->camToWorld = fh->trackingRef->camToWorld * fh->camToTrackingRef;
        if (haveOneGood)
            fh->poseValid = true;

        if (coarseTracker->firstCoarseRMSE < 0)
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

} // SLAM