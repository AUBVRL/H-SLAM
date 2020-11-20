#include "System.h"
#include "CalibData.h"
#include "Frame.h"
#include "MapPoint.h"
#include "EnergyFunctional.h"
#include "IndexThreadReduce.h"
#include "CoarseTracker.h"

#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
namespace SLAM
{

    void System::flagFramesForMarginalization()
    {
        if (setting_minFrameAge > setting_maxFrames)
        {
            for (int i = setting_maxFrames; i < (int)frameHessians.size(); i++)
            {
                auto &fh = frameHessians[i - setting_maxFrames];
                fh->frame->FlaggedForMarginalization = true;
            }
            return;
        }

        int flagged = 0;
        // marginalize all frames that have not enough points.
        for (auto &fh : frameHessians)
        {
            int in = fh->frame->pointHessians.size() + fh->frame->ImmaturePoints.size();
            int out = fh->frame->pointHessiansMarginalized.size() + fh->frame->pointHessiansOut.size();

            Vec2 refToFh = AffLight::fromToVecExposure(frameHessians.back()->ab_exposure, fh->ab_exposure, frameHessians.back()->frame->efFrame->aff_g2l(), fh->frame->efFrame->aff_g2l());

            if ((in < setting_minPointsRemaining * (in + out) || fabs(logf((float)refToFh[0])) > setting_maxLogAffFacInWindow) && ((int)frameHessians.size()) - flagged > setting_minFrames)
            {
                fh->frame->FlaggedForMarginalization = true;
                flagged++;
            }
        }

        // marginalize one.
        if ((int)frameHessians.size() - flagged >= setting_maxFrames)
        {
            double smallestScore = 1;
            shared_ptr<FrameShell> toMarginalize;
            auto latest = frameHessians.back();

            for (auto &fh : frameHessians)
            {
                if (fh->KfId > latest->KfId - setting_minFrameAge || fh->KfId == 0)
                    continue;
                //if(fh==frameHessians.front() == 0) continue;

                double distScore = 0;
                for (FrameFramePrecalc &ffh : fh->frame->targetPrecalc)
                {
                    if (ffh.target->KfId > latest->KfId - setting_minFrameAge + 1 || ffh.target == ffh.host)
                        continue;
                    distScore += 1 / (1e-5 + ffh.distanceLL);
                }
                distScore *= -sqrtf(fh->frame->targetPrecalc.back().distanceLL);

                if (distScore < smallestScore)
                {
                    smallestScore = distScore;
                    toMarginalize = fh;
                }
            }

            toMarginalize->frame->FlaggedForMarginalization = true;
            flagged++;
        }
    }

    void System::activatePoints()
    {
        if (ef->nPoints < setting_desiredPointDensity * 0.66)
            currentMinActDist -= 0.8;
        if (ef->nPoints < setting_desiredPointDensity * 0.8)
            currentMinActDist -= 0.5;
        else if (ef->nPoints < setting_desiredPointDensity * 0.9)
            currentMinActDist -= 0.2;
        else if (ef->nPoints < setting_desiredPointDensity)
            currentMinActDist -= 0.1;

        if (ef->nPoints > setting_desiredPointDensity * 1.5)
            currentMinActDist += 0.8;
        if (ef->nPoints > setting_desiredPointDensity * 1.3)
            currentMinActDist += 0.5;
        if (ef->nPoints > setting_desiredPointDensity * 1.15)
            currentMinActDist += 0.2;
        if (ef->nPoints > setting_desiredPointDensity)
            currentMinActDist += 0.1;

        if (currentMinActDist < 0)
            currentMinActDist = 0;
        if (currentMinActDist > 4)
            currentMinActDist = 4;
        // if(!setting_debugout_runquiet)
        //     printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
        //             currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);

        shared_ptr<FrameShell> newestHs = frameHessians.back();
        // // make dist map.
        // coarseDistanceMap->makeK(geomCalib);
        coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

        vector<shared_ptr<ImmaturePoint>> toOptimize;
        toOptimize.reserve(20000);

        for (auto &host : frameHessians) // go through all active frames
        {
            if (host == newestHs)
                continue;

            SE3 fhToNew = newestHs->frame->efFrame->PRE_worldToCam * host->frame->efFrame->PRE_camToWorld;

            Mat33f KRKi = (geomCalib->pyrK[1] * fhToNew.rotationMatrix() * geomCalib->pyrKi[0]).cast<float>();
            Vec3f Kt = (geomCalib->pyrK[1] * fhToNew.translation()).cast<float>();

            for (unsigned int i = 0; i < host->frame->ImmaturePoints.size(); i++)
            {
                shared_ptr<ImmaturePoint> &ph = host->frame->ImmaturePoints[i];
                ph->idxInImmaturePoints = i;

                // delete points that have never been traced successfully, or that are outlier on the last trace.
                if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
                {
                    ph.reset();
                    // host->frame->ImmaturePoints[i]= nullptr;
                    continue;
                }

                // can activate only if this is true.
                bool canActivate = (ph->lastTraceStatus == IPS_GOOD || ph->lastTraceStatus == IPS_SKIPPED 
                                    || ph->lastTraceStatus == IPS_BADCONDITION || ph->lastTraceStatus == IPS_OOB) 
                                    && ph->lastTracePixelInterval < 8 && ph->quality > setting_minTraceQuality 
                                    && (ph->idepth_max + ph->idepth_min) > 0;

                // if I cannot activate the point, skip it. Maybe also delete it.
                if (!canActivate)
                {
                    // if point will be out afterwards, delete it instead.
                    if (ph->host->frame->FlaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
                    {
                        ph.reset();
                        // host->frame->ImmaturePoints[i]= nullptr;
                    }
                    continue;
                }

                // see if we need to activate point due to distance map.
                Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * (0.5f * (ph->idepth_max + ph->idepth_min));
                int u = ptp[0] / ptp[2] + 0.5f;
                int v = ptp[1] / ptp[2] + 0.5f;

                if ((u > 0 && v > 0 && u < geomCalib->wpyr[1] && v < geomCalib->hpyr[1]))
                {

                    float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u + geomCalib->wpyr[1] * v] + (ptp[0] - floorf((float)(ptp[0])));

                    if (dist >= currentMinActDist * ph->my_type)
                    {
                        coarseDistanceMap->addIntoDistFinal(u, v);
                        toOptimize.push_back(ph);
                    }
                }
                else
                {
                    ph.reset();
                    host->frame->ImmaturePoints[i] = nullptr;
                }
            }
        }

        //	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
        //			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

        vector<shared_ptr<MapPoint>> optimized;
        optimized.resize(toOptimize.size());

        
        backEndThreadPool->reduce(bind(&System::activateImmaturePts, this, &optimized, &toOptimize,
                                 placeholders::_1, placeholders::_2, placeholders::_3, placeholders::_4), 0, toOptimize.size(), 0);

        for (unsigned k = 0; k < toOptimize.size(); k++)
        {
            shared_ptr<MapPoint> newpoint = optimized[k];
            shared_ptr<ImmaturePoint> ph = toOptimize[k];

            if (newpoint != nullptr) //&& newpoint != (PointHessian*)((long)(-1))
            {
                newpoint->host->frame->ImmaturePoints[ph->idxInImmaturePoints].reset();
                newpoint->host->frame->pointHessians.push_back(newpoint);
                ef->insertPoint(newpoint);
                for (int i = 0; i < newpoint->residuals.size(); i++)
                    ef->insertResidual(newpoint->residuals[i], i);

                ph.reset();
            }
            else if (!newpoint || ph->lastTraceStatus == IPS_OOB)
            {
                ph->host->frame->ImmaturePoints[ph->idxInImmaturePoints].reset();
                ph.reset();
            }
            else
            {
                assert(newpoint == nullptr);
            }
        }

        for (auto &host : frameHessians)
        {
            for (int i = 0; i < (int)host->frame->ImmaturePoints.size(); i++)
            {
                if (host->frame->ImmaturePoints[i] == 0)
                {
                    host->frame->ImmaturePoints[i] = host->frame->ImmaturePoints.back();
                    host->frame->ImmaturePoints.pop_back();
                    i--;
                }
            }
        }
    }

    void System::activateImmaturePts(vector<shared_ptr<MapPoint>> *optimized, vector<shared_ptr<ImmaturePoint>> *toOptimize, 
                                     int min, int max, Vec10 *stats, int tid)
    {
        vector<shared_ptr<ImmaturePointTemporaryResidual>> tr;
        tr.reserve(frameHessians.size());
        for (int i = 0, iend = frameHessians.size(); i < iend; ++i)
            tr.push_back(make_shared<ImmaturePointTemporaryResidual>());

        for (int k = min; k < max; ++k)
            (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr);
    }


    void System::removeOutliers()
    {
        int numPointsDropped = 0;
        for (auto &fh : frameHessians)
        {
            for (unsigned int i = 0; i < fh->frame->pointHessians.size(); i++)
            {
                std::shared_ptr<MapPoint> &ph = fh->frame->pointHessians[i];
                if (!ph)
                    continue;
                // if(ph->status!= MapPoint::ACTIVE)
                // 	continue;
                if (ph->residuals.size() == 0)
                {
                    fh->frame->pointHessiansOut.push_back(ph);
                    ph->efPoint->stateFlag = energyStatus::toDrop;
                    fh->frame->pointHessians[i] = fh->frame->pointHessians.back();
                    fh->frame->pointHessians.pop_back();
                    i--;
                    // ph->setPointStatus(OUTLIER); //debug this
                    numPointsDropped++;
                }
            }

            for (auto &ph : fh->frame->pointHessiansOut)
            {
                if (!ph || !ph->efPoint)
                    continue;
                if (ph->efPoint->stateFlag == energyStatus::toDrop)
                    ef->removePoint(ph);
            }
        }

        EFIndicesValid = false;
        ef->makeIDX();
        // ef->dropPointsF();
    }


    void System::makeNewTraces(std::shared_ptr<FrameShell> newFrame)
    {
        // pixelSelector->allowFast = true;
        //int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
        // int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,setting_desiredImmatureDensity);
        if(newFrame->frame->mvKeys.size() == 0)
            newFrame->frame->Extract(detector, newFrame->id, false, backEndThreadPool);
        newFrame->frame->pointHessians.reserve(newFrame->frame->nFeatures);
        newFrame->frame->pointHessiansMarginalized.reserve(newFrame->frame->nFeatures);
        newFrame->frame->pointHessiansOut.reserve(newFrame->frame->nFeatures);

        for (int i = 0; i < newFrame->frame->nFeatures; ++i)
        {
            // if (newFrame->pointHessians[i])
            //     continue;
            std::shared_ptr<ImmaturePoint> impt = std::shared_ptr<ImmaturePoint>(new ImmaturePoint(newFrame->frame->mvKeys[i].pt.x, 
                                                                                 newFrame->frame->mvKeys[i].pt.y, newFrame, 1, geomCalib));
            if (!std::isfinite(impt->energyTH))
                continue;
            else
            {
                newFrame->frame->ImmaturePoints.push_back(impt);
            }
        }
        // cv::Mat Output;
        // newFrame->frame->Image.convertTo(Output, CV_8UC3);
        // cv::drawKeypoints(Output, newFrame->frame->mvKeys, Output, cv::Scalar(0, 255, 0));
        // cv::namedWindow("test", cv::WINDOW_KEEPRATIO);
        // cv::imshow("test", Output);
        // cv::waitKey(1);
    }

    void System::flagPointsForRemoval()
    {
        assert(EFIndicesValid);

        std::vector<std::shared_ptr<FrameShell>> fhsToMargPoints;

        for (int i = 0; i < (int)frameHessians.size(); i++)
            if (frameHessians[i]->frame->FlaggedForMarginalization)
                fhsToMargPoints.push_back(frameHessians[i]);

        int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

        for (auto &host : frameHessians) // go through all active frames
        {
            for (unsigned int i = 0; i < host->frame->pointHessians.size(); i++)
            {
                auto &ph = host->frame->pointHessians[i];
                if (!ph)
                    continue;

                if (ph->idepth < 0 || ph->residuals.size() == 0)
                {
                    ph->efPoint->stateFlag = energyStatus::toDrop;
                    ph->setPointStatus(OUTLIER);
                    host->frame->pointHessiansOut.push_back(ph);
                    host->frame->pointHessians[i] = nullptr;
                    flag_nores++;
                }
                else if (ph->isOOB(fhsToMargPoints) || host->frame->FlaggedForMarginalization)
                {
                    flag_oob++;
                    if (ph->isInlierNew())
                    {
                        flag_in++;
                        int ngoodRes = 0;
                        for (auto &r : ph->residuals)
                        {
                            r->resetOOB();
                            r->linearize(geomCalib);
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
                            ph->efPoint->stateFlag = energyStatus::toMarg;
                            ph->setPointStatus(MARGINALIZED);
                            host->frame->pointHessiansMarginalized.push_back(ph);
                        }
                        else
                        {
                            ph->efPoint->stateFlag = energyStatus::toDrop;
                            ph->setPointStatus(OUTLIER);
                            host->frame->pointHessiansOut.push_back(ph);
                        }
                    }
                    else
                    {
                        ph->efPoint->stateFlag = energyStatus::toDrop;
                        ph->setPointStatus(OUTLIER);
                        host->frame->pointHessiansOut.push_back(ph);
                    }
                    host->frame->pointHessians[i].reset();
                }
            }
            for (int i = 0; i < (int)host->frame->pointHessians.size(); i++)
            {
                if (!host->frame->pointHessians[i])
                {
                    host->frame->pointHessians[i] = host->frame->pointHessians.back();
                    host->frame->pointHessians.pop_back();
                    i--;
                }
            }
        }
    }

} // namespace SLAM