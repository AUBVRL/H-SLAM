#include "System.h"
#include "Map.h"
#include "Frame.h"
#include "MapPoint.h"
#include "EnergyFunctional.h"
#include "ImmaturePoint.h"
#include "CalibData.h"
#include "CoarseTracker.h"
namespace FSLAM
{
void System::AddKeyframe(std::shared_ptr<Frame> fh)
{
    {
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        assert(fh->trackingRef != nullptr);
        fh->camToWorld = fh->trackingRef->camToWorld * fh->camToTrackingRef;
        fh->setEvalPT_scaled(fh->camToWorld.inverse(), fh->aff_g2l_internal);
    }

    traceNewCoarse(fh);
    boost::unique_lock<boost::mutex> lock(mapMutex);

    // =========================== Flag Frames to be Marginalized. =========================
    flagFramesForMarginalization(fh);

    // =========================== add New Frame to Hessian Struct. =========================
    fh->idx = frameHessians.size();
    frameHessians.push_back(fh);
    // fh->frameID = allKeyFramesHistory.size();
    allKeyFramesHistory.push_back(fh);
    ef->insertFrame(fh, Calib);

    setPrecalcValues();
    // =========================== add new residuals for old points =========================
    int numFwdResAdde = 0;
    for (auto fh1 : frameHessians) // go through all active frames
    {
        if (fh1 == fh)
            continue;
        for (auto ph : fh1->pointHessians)
        {
            if(!ph)
                continue;
            std::shared_ptr<PointFrameResidual> r = std::shared_ptr<PointFrameResidual>(new PointFrameResidual(ph, fh1, fh));
            r->setState(ResState::IN);
            ph->residuals.push_back(r);
            ef->insertResidual(r);
            ph->lastResiduals[1] = ph->lastResiduals[0];
            ph->lastResiduals[0] = std::pair<std::shared_ptr<PointFrameResidual>, ResState>(r, ResState::IN);
            numFwdResAdde += 1;
        }
    }
    // =========================== Activate Points (& flag for marginalization). =========================
    activatePointsMT();
    ef->makeIDX();

    // =========================== OPTIMIZE ALL =========================

    fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
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

    if (isLost)
        return;

    // =========================== REMOVE OUTLIER =========================
    removeOutliers();

    {
    	boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
    	coarseTracker_forNewKF->makeK(Calib);
    	coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);

    //     coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
    //     coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
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
    makeNewTraces(fh);

    // for(IOWrap::Output3DWrapper* ow : outputWrapper)
    // {
    //     ow->publishGraph(ef->connectivityMap);
    //     ow->publishKeyframes(frameHessians, false, &Hcalib);
    // }

    // =========================== Marginalize Frames =========================

    for (unsigned int i = 0; i < frameHessians.size(); i++)
        if (frameHessians[i]->FlaggedForMarginalization)
        {
            marginalizeFrame(frameHessians[i]);
            i = 0;
        }

    std::cout << "adding keyframes!!" << std::endl;
    return;
}

void System::MappingThread()
{
    boost::unique_lock<boost::mutex> lock(MapThreadMutex);

    while (RunMapping)
    {
        while (UnmappedTrackedFrames.size() == 0)
        {
            TrackedFrameSignal.wait(lock);
            if (!RunMapping)
                return;
        }

        std::shared_ptr<Frame> frame = UnmappedTrackedFrames.front();
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
                std::shared_ptr<Frame> frame = UnmappedTrackedFrames.front();
                UnmappedTrackedFrames.pop_front();
                {
                    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
                    assert(frame->trackingRef);
                    frame->camToWorld = frame->trackingRef->camToWorld * frame->camToTrackingRef;
                    frame->setEvalPT_scaled(frame->camToWorld.inverse(), frame->aff_g2l_internal);
                }
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

void System::BlockUntilMappingIsFinished()
{
    boost::unique_lock<boost::mutex> lock(MapThreadMutex);
    RunMapping = false;
    TrackedFrameSignal.notify_all();
    lock.unlock();

    tMappingThread.join();
}

void System::makeNewTraces(std::shared_ptr<Frame> newFrame)
{
    // pixelSelector->allowFast = true;
    //int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
    // int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,setting_desiredImmatureDensity);

    // newFrame->pointHessians.reserve(numPointsTotal*1.2f);
    newFrame->pointHessians.resize(newFrame->nFeatures);
    newFrame->ImmaturePoints.resize(newFrame->nFeatures);

    //fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
    newFrame->pointHessiansMarginalized.reserve(newFrame->nFeatures);
    newFrame->pointHessiansOut.reserve(newFrame->nFeatures);

    for (int i = 0; i < newFrame->nFeatures; ++i)
    {
        // if (newFrame->pointHessians[i])
        //     continue;
        std::shared_ptr<ImmaturePoint> impt = std::shared_ptr<ImmaturePoint>(new ImmaturePoint(newFrame->mvKeys[i].pt.x, newFrame->mvKeys[i].pt.y, newFrame, 1, Calib));
        if (!std::isfinite(impt->energyTH))
            continue;
        else
            newFrame->ImmaturePoints[i] = impt;
    }
}

void System::flagPointsForRemoval()
{
    assert(EFIndicesValid);

    std::vector<std::shared_ptr<Frame>> fhsToMargPoints;

    for (int i = 0; i < (int)frameHessians.size(); i++)
        if (frameHessians[i]->FlaggedForMarginalization)
            fhsToMargPoints.push_back(frameHessians[i]);

    int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

    for (auto host : frameHessians) // go through all active frames
    {
        for (unsigned int i = 0; i < host->pointHessians.size(); i++)
        {
            auto ph = host->pointHessians[i];
            if (!ph)
                continue;

            if (ph->idepth_scaled < 0 || ph->residuals.size() == 0)
            {
                host->pointHessiansOut.push_back(ph);
                ph->status = MapPoint::OUTLIER;
                host->pointHessians[i] = 0;
                flag_nores++;
            }
            else if (ph->isOOB(fhsToMargPoints) || host->FlaggedForMarginalization)
            {
                flag_oob++;
                if (ph->isInlierNew())
                {
                    flag_in++;
                    int ngoodRes = 0;
                    for (auto r : ph->residuals)
                    {
                        r->resetOOB();
                        r->linearize(Calib);
                        r->isLinearized = false;
                        r->applyRes(true);
                        if (r->isActive())
                        {
                            r->fixLinearizationF(ef);
                            ngoodRes++;
                        }
                    }
                    if (ph->idepth_hessian > setting_minIdepthH_marg)
                    {
                        flag_inin++;
                        ph->status = MapPoint::MARGINALIZED;
                        host->pointHessiansMarginalized.push_back(ph);
                    }
                    else
                    {
                        ph->status = MapPoint::OUTLIER;
                        host->pointHessiansOut.push_back(ph);
                    }
                }
                else
                {
                    host->pointHessiansOut.push_back(ph);
                    ph->status = MapPoint::OUTLIER;
                }
                host->pointHessians[i].reset();
            }
        }
        // for(int i=0;i<(int)host->pointHessians.size();i++)
        // {
        // 	if(host->pointHessians[i]==0)
        // 	{
        // 		host->pointHessians[i] = host->pointHessians.back();
        // 		host->pointHessians.pop_back();
        // 		i--;
        // 	}
        // }
    }
}

void System::activatePointsMT()
{

	if(ef->nPoints < setting_desiredPointDensity*0.66)
		currentMinActDist -= 0.8;
	if(ef->nPoints < setting_desiredPointDensity*0.8)
		currentMinActDist -= 0.5;
	else if(ef->nPoints < setting_desiredPointDensity*0.9)
		currentMinActDist -= 0.2;
	else if(ef->nPoints < setting_desiredPointDensity)
		currentMinActDist -= 0.1;

	if(ef->nPoints > setting_desiredPointDensity*1.5)
		currentMinActDist += 0.8;
	if(ef->nPoints > setting_desiredPointDensity*1.3)
		currentMinActDist += 0.5;
	if(ef->nPoints > setting_desiredPointDensity*1.15)
		currentMinActDist += 0.2;
	if(ef->nPoints > setting_desiredPointDensity)
		currentMinActDist += 0.1;

	if(currentMinActDist < 0) currentMinActDist = 0;
	if(currentMinActDist > 4) currentMinActDist = 4;

    // if(!setting_debugout_runquiet)
    //     printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
    //             currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);

	std::shared_ptr<Frame> newestHs = frameHessians.back();

	// // make dist map.
	coarseDistanceMap->makeK(Calib);
	coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);


	std::vector<std::shared_ptr<ImmaturePoint>> toOptimize; toOptimize.reserve(20000);


	for(auto host : frameHessians)		// go through all active frames
	{
		if(host == newestHs) continue;

		SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
        // Mat33f KRKi = (Calib->pyrK[0] * fhToNew.rotationMatrix().cast<float>() * Calib->pyrKi[0] );
        // Vec3f Kt = (Calib->pyrK[0] * fhToNew.translation().cast<float>());

		Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
		Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());


		for(unsigned int i=0;i<host->ImmaturePoints.size();i+=1)
		{
			std::shared_ptr<ImmaturePoint> ph = host->ImmaturePoints[i];
            if(!ph)
                continue;
			ph->idxInImmaturePoints = i;

			// delete points that have never been traced successfully, or that are outlier on the last trace.
			if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
			{
				ph.reset();
				host->ImmaturePoints[i].reset();
				continue;
			}

			// can activate only if this is true.
			bool canActivate = (ph->lastTraceStatus == IPS_GOOD || ph->lastTraceStatus == IPS_SKIPPED || ph->lastTraceStatus == IPS_BADCONDITION || ph->lastTraceStatus == IPS_OOB )
							&& ph->lastTracePixelInterval < 8 && ph->quality > setting_minTraceQuality && (ph->idepth_max+ph->idepth_min) > 0;


			// if I cannot activate the point, skip it. Maybe also delete it.
			if(!canActivate)
			{
				// if point will be out afterwards, delete it instead.
				if(ph->host.lock()->FlaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
				{
					ph.reset();
					host->ImmaturePoints[i].reset();
				}
				continue;
			}


			// see if we need to activate point due to distance map.
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;

			if((u > 0 && v > 0 && u < Calib->wpyr[1] && v < Calib->hpyr[1]))
			{

				float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+Calib->wpyr[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

				if(dist>=currentMinActDist* ph->my_type)
				{
					coarseDistanceMap->addIntoDistFinal(u,v);
					toOptimize.push_back(ph);
				}
			}
			else
			{
				ph.reset();
				host->ImmaturePoints[i].reset();
			}
		}
	}


//	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
//			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

	std::vector<std::shared_ptr<MapPoint>> optimized; optimized.resize(toOptimize.size());

	if(multiThreading)
		treadReduce.reduce(boost::bind(&System::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);
	else
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);


	for(unsigned k=0;k<toOptimize.size();k++)
	{
		std::shared_ptr<MapPoint> newpoint = optimized[k];
		std::shared_ptr<ImmaturePoint> ph = toOptimize[k];

		if(newpoint != nullptr)
		{
			newpoint->host.lock()->ImmaturePoints[ph->idxInImmaturePoints].reset();
			newpoint->host.lock()->pointHessians[ph->idxInImmaturePoints] = newpoint;
			ef->insertPoint(newpoint);
			for(auto r : newpoint->residuals)
				ef->insertResidual(r);
			ph.reset();
		}
		else if(!newpoint || ph->lastTraceStatus==IPS_OOB)
		{
            ph->host.lock()->ImmaturePoints[ph->idxInImmaturePoints].reset();
			ph.reset();
			
		}
		else
		{
			assert(newpoint = nullptr);
		}
	}


	// for(auto host : frameHessians)
	// {
	// 	for(int i=0;i<(int)host->ImmaturePoints.size();i++)
	// 	{
	// 		if(host->ImmaturePoints[i]==0)
	// 		{
	// 			host->ImmaturePoints[i] = host->ImmaturePoints.back();
	// 			host->ImmaturePoints.pop_back();
	// 			i--;
	// 		}
	// 	}
	// }
}

void System::activatePointsMT_Reductor( std::vector<std::shared_ptr<MapPoint>>* optimized, std::vector<std::shared_ptr<ImmaturePoint>>* toOptimize, int min, int max, Vec10* stats, int tid)
{
	std::vector<std::shared_ptr<ImmaturePointTemporaryResidual>> tr;
  
	for(int k=min;k<max;k++)
		(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
	
	
}

void System::traceNewCoarse(std::shared_ptr<Frame> fh)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

	Mat33f K = Mat33f::Identity();
	K(0,0) = Calib->fxl();
	K(1,1) = Calib->fyl();
	K(0,2) = Calib->cxl();
	K(1,2) = Calib->cyl();

	for(auto host : frameHessians)		// go through all active frames
	{

		SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
		Vec3f Kt = K * hostToNew.translation().cast<float>();

		Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

		for(auto ph : host->ImmaturePoints)
		{
            if(!ph)
                continue;
			ph->traceOn(fh, KRKi, Kt, aff, false );

			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
			trace_total++;
		}
	}
//	printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
//			trace_total,
//			trace_good, 100*trace_good/(float)trace_total,
//			trace_skip, 100*trace_skip/(float)trace_total,
//			trace_badcondition, 100*trace_badcondition/(float)trace_total,
//			trace_oob, 100*trace_oob/(float)trace_total,
//			trace_out, 100*trace_out/(float)trace_total,
//			trace_uninitialized, 100*trace_uninitialized/(float)trace_total);
}


} // namespace FSLAM