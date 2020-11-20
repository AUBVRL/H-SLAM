#include "System.h"
#include "CalibData.h"
#include "Frame.h"
#include "MapPoint.h"
#include "EnergyFunctional.h"
#include "IndexThreadReduce.h"
#include "OptimizationClasses.h"

namespace SLAM
{
    Vec3 System::linearizeAll(bool fixLinearization)
    {
        double lastEnergyP = 0;
        double lastEnergyR = 0;
        double num = 0;

        vector<shared_ptr<PointFrameResidual>> toRemove[NUM_THREADS];
        for (int i = 0; i < NUM_THREADS; i++)
            toRemove[i].clear();


        backEndThreadPool->reduce(bind(&System::linearizeAll_Reductor, this, fixLinearization, toRemove, placeholders::_1, 
                                      placeholders::_2, placeholders::_3, placeholders::_4), 0, activeResiduals.size(), 0);
        lastEnergyP = backEndThreadPool->stats[0];


        setNewFrameEnergyTH();

        if (fixLinearization)
        {
            for (auto &r : activeResiduals)
            {
                auto &ph = r->point;
                if (ph->lastResiduals[0].first == r)
                    ph->lastResiduals[0].second = r->state_state;
                else if (ph->lastResiduals[1].first == r)
                    ph->lastResiduals[1].second = r->state_state;
            }

            for (int i = 0; i < NUM_THREADS; i++)
            {
                for (auto r : toRemove[i])
                {
                    auto &ph = r->point;

                    if (ph->lastResiduals[0].first == r)
                        ph->lastResiduals[0].first.reset();
                    else if (ph->lastResiduals[1].first == r)
                        ph->lastResiduals[1].first.reset();

                    for (unsigned int k = 0; k < ph->residuals.size(); k++)
                        if (ph->residuals[k] == r)
                        {
                            ef->dropResidual(r);
                            // ph->residuals[k].reset();
                            // ph->residuals[k] = ph->residuals.back();
                            // ph->residuals.pop_back();
                            break;
                        }
                }
            }
            //printf("FINAL LINEARIZATION: removed %d / %d residuals!\n", nResRemoved, (int)activeResiduals.size());
        }

        return Vec3(lastEnergyP, lastEnergyR, num);
    }


    void System::linearizeAll_Reductor(bool fixLinearization, vector<shared_ptr<PointFrameResidual>> *toRemove, int min, 
                                       int max, Vec10 *stats, int tid)
    {
        for (int k = min; k < max; k++)
        {
            auto &r = activeResiduals[k];
            (*stats)[0] += r->linearize(geomCalib);

            if (fixLinearization)
            {
                r->applyRes(true);

                if (r->isActive())
                {
                    if (r->isNew)
                    {
                        auto p = r->point;
                        Vec3f ptp_inf = r->host->frame->targetPrecalc[r->target->frame->idx].PRE_KRKiTll * Vec3f(p->u, p->v, 1); // projected point assuming infinite depth.
                        Vec3f ptp = ptp_inf + r->host->frame->targetPrecalc[r->target->frame->idx].PRE_KtTll * p->idepth;        // projected point with real depth.
                        float relBS = 0.01 * ((ptp_inf.head<2>() / ptp_inf[2]) - (ptp.head<2>() / ptp[2])).norm();               // 0.01 = one pixel.

                        if (relBS > p->maxRelBaseline)
                            p->maxRelBaseline = relBS;

                        p->numGoodResiduals++;
                    }
                }
                else
                {
                    toRemove[tid].push_back(activeResiduals[k]);
                }
            }
        }
    }

    void System::setNewFrameEnergyTH()
    {
        // collect all residuals and make decision on TH.
        allResVec.clear();
        allResVec.reserve(activeResiduals.size() * 2);
        shared_ptr<FrameShell> newFrame = frameHessians.back();

        for (auto &r : activeResiduals)
            if (r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame)
            {
                allResVec.push_back(r->state_NewEnergyWithOutlier);
            }

        if (allResVec.size() == 0)
        {
            newFrame->frame->frameEnergyTH = 12 * 12 * patternNum;
            return; // should never happen, but lets make sure.
        }

        int nthIdx = setting_frameEnergyTHN * allResVec.size();

        assert(nthIdx < (int)allResVec.size());
        assert(setting_frameEnergyTHN < 1);

        std::nth_element(allResVec.begin(), allResVec.begin() + nthIdx, allResVec.end());
        float nthElement = sqrtf(allResVec[nthIdx]);

        newFrame->frame->frameEnergyTH = nthElement * setting_frameEnergyTHFacMedian;
        newFrame->frame->frameEnergyTH = 26.0f * setting_frameEnergyTHConstWeight + newFrame->frame->frameEnergyTH * (1 - setting_frameEnergyTHConstWeight);
        newFrame->frame->frameEnergyTH = newFrame->frame->frameEnergyTH * newFrame->frame->frameEnergyTH;
        newFrame->frame->frameEnergyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;

        //
        //	int good=0,bad=0;
        //	for(float f : allResVec) if(f<newFrame->frameEnergyTH) good++; else bad++;
        //	printf("EnergyTH: mean %f, median %f, result %f (in %d, out %d)! \n",
        //			meanElement, nthElement, sqrtf(newFrame->frameEnergyTH),
        //			good, bad);
    }


    void System::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10 *stats, int tid)
    {
        for (int k = min; k < max; k++)
            activeResiduals[k]->applyRes(true);
        return;
    }


    // sets linearization point.
    void System::backupState(bool backupLastStep)
    {
        if (setting_solverMode & SOLVER_MOMENTUM)
        {
            if (backupLastStep)
            {
                geomCalib->step_backup = geomCalib->step;
                geomCalib->value_backup = geomCalib->value;
                for (auto &fh : frameHessians)
                {
                    fh->frame->efFrame->step_backup = fh->frame->efFrame->step;
                    fh->frame->efFrame->state_backup = fh->frame->efFrame->get_state();
                    for (auto &ph : fh->frame->pointHessians)
                    {
                        ph->efPoint->idepth_backup = ph->idepth;
                        ph->efPoint->step_backup = ph->efPoint->step;
                    }
                }
            }
            else
            {
                geomCalib->step_backup.setZero();
                geomCalib->value_backup = geomCalib->value;
                for (auto &fh : frameHessians)
                {
                    fh->frame->efFrame->step_backup.setZero();
                    fh->frame->efFrame->state_backup = fh->frame->efFrame->get_state();
                    for (auto &ph : fh->frame->pointHessians)
                    {
                        ph->efPoint->idepth_backup = ph->idepth;
                        ph->efPoint->step_backup = 0;
                    }
                }
            }
        }
        else
        {
            geomCalib->value_backup = geomCalib->value;
            for (auto &fh : frameHessians)
            {
                fh->frame->efFrame->state_backup = fh->frame->efFrame->get_state();
                for (auto &ph : fh->frame->pointHessians)
                    ph->efPoint->idepth_backup = ph->idepth;
            }
        }
    }


    vector<VecX> System::getNullspaces(vector<VecX> &nullspaces_pose, vector<VecX> &nullspaces_scale, vector<VecX> &nullspaces_affA,
                                       vector<VecX> &nullspaces_affB)
    {
        nullspaces_pose.clear();
        nullspaces_scale.clear();
        nullspaces_affA.clear();
        nullspaces_affB.clear();

        int n = CPARS + frameHessians.size() * 8;
        vector<VecX> nullspaces_x0_pre;
        for (int i = 0; i < 6; i++)
        {
            VecX nullspace_x0(n);
            nullspace_x0.setZero();
            for (auto fh : frameHessians)
            {
                nullspace_x0.segment<6>(CPARS + fh->frame->idx * 8) = fh->frame->efFrame->nullspaces_pose.col(i);
                nullspace_x0.segment<3>(CPARS + fh->frame->idx * 8) *= SCALE_XI_TRANS_INVERSE;
                nullspace_x0.segment<3>(CPARS + fh->frame->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
            }
            nullspaces_x0_pre.push_back(nullspace_x0);
            nullspaces_pose.push_back(nullspace_x0);
        }
        for (int i = 0; i < 2; i++)
        {
            VecX nullspace_x0(n);
            nullspace_x0.setZero();
            for (auto &fh : frameHessians)
            {
                nullspace_x0.segment<2>(CPARS + fh->frame->idx * 8 + 6) = fh->frame->efFrame->nullspaces_affine.col(i).head<2>();
                nullspace_x0[CPARS + fh->frame->idx * 8 + 6] *= SCALE_A_INVERSE;
                nullspace_x0[CPARS + fh->frame->idx * 8 + 7] *= SCALE_B_INVERSE;
            }
            nullspaces_x0_pre.push_back(nullspace_x0);
            if (i == 0)
                nullspaces_affA.push_back(nullspace_x0);
            if (i == 1)
                nullspaces_affB.push_back(nullspace_x0);
        }

        VecX nullspace_x0(n);
        nullspace_x0.setZero();
        for (auto &fh : frameHessians)
        {
            nullspace_x0.segment<6>(CPARS + fh->frame->idx * 8) = fh->frame->efFrame->nullspaces_scale;
            nullspace_x0.segment<3>(CPARS + fh->frame->idx * 8) *= SCALE_XI_TRANS_INVERSE;
            nullspace_x0.segment<3>(CPARS + fh->frame->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
        }
        nullspaces_x0_pre.push_back(nullspace_x0);
        nullspaces_scale.push_back(nullspace_x0);

        return nullspaces_x0_pre;
    }


    // applies step to linearization point.
    bool System::doStepFromBackup(float stepfacC, float stepfacT, float stepfacR, float stepfacA, float stepfacD)
    {
        //	float meanStepC=0,meanStepP=0,meanStepD=0;
        //	meanStepC += Hcalib.step.norm();

        Vec10 pstepfac;
        pstepfac.segment<3>(0).setConstant(stepfacT);
        pstepfac.segment<3>(3).setConstant(stepfacR);
        pstepfac.segment<4>(6).setConstant(stepfacA);

        float sumA = 0, sumB = 0, sumT = 0, sumR = 0, sumID = 0, numID = 0;

        float sumNID = 0;

        if (setting_solverMode & SOLVER_MOMENTUM)
        {
            geomCalib->setValue(geomCalib->value_backup + geomCalib->step);
            for (auto &fh : frameHessians)
            {
                Vec10 step = fh->frame->efFrame->step;
                step.head<6>() += 0.5f * (fh->frame->efFrame->step_backup.head<6>());

                fh->frame->efFrame->setState(fh->frame->efFrame->state_backup + step);
                sumA += step[6] * step[6];
                sumB += step[7] * step[7];
                sumT += step.segment<3>(0).squaredNorm();
                sumR += step.segment<3>(3).squaredNorm();

                for (auto &ph : fh->frame->pointHessians)
                {
                    float step = ph->efPoint->step + 0.5f * (ph->efPoint->step_backup);
                    ph->setIdepth(ph->efPoint->idepth_backup + step);
                    sumID += step * step;
                    sumNID += fabsf(ph->efPoint->idepth_backup);
                    numID++;

                    ph->efPoint->setIdepthZero(ph->efPoint->idepth_backup + step);
                }
            }
        }
        else
        {
            geomCalib->setValue(geomCalib->value_backup + stepfacC * geomCalib->step);
            for (auto &fh : frameHessians)
            {
                fh->frame->efFrame->setState(fh->frame->efFrame->state_backup + pstepfac.cwiseProduct(fh->frame->efFrame->step));
                sumA += fh->frame->efFrame->step[6] * fh->frame->efFrame->step[6];
                sumB += fh->frame->efFrame->step[7] * fh->frame->efFrame->step[7];
                sumT += fh->frame->efFrame->step.segment<3>(0).squaredNorm();
                sumR += fh->frame->efFrame->step.segment<3>(3).squaredNorm();

                for (auto &ph : fh->frame->pointHessians)
                {
                    ph->setIdepth(ph->efPoint->idepth_backup + stepfacD * ph->efPoint->step);
                    sumID += ph->efPoint->step * ph->efPoint->step;
                    sumNID += fabsf(ph->efPoint->idepth_backup);
                    numID++;

                    ph->efPoint->setIdepthZero(ph->efPoint->idepth_backup + stepfacD * ph->efPoint->step);
                }
            }
        }

        sumA /= frameHessians.size();
        sumB /= frameHessians.size();
        sumR /= frameHessians.size();
        sumT /= frameHessians.size();
        sumID /= numID;
        sumNID /= numID;

        // if(!setting_debugout_runquiet)
        //     printf("STEPS: A %.1f; B %.1f; R %.1f; T %.1f. \t",
        //             sqrtf(sumA) / (0.0005*setting_thOptIterations),
        //             sqrtf(sumB) / (0.00005*setting_thOptIterations),
        //             sqrtf(sumR) / (0.00005*setting_thOptIterations),
        //             sqrtf(sumT)*sumNID / (0.00005*setting_thOptIterations));

        EFDeltaValid = false;
        setPrecalcValues();

        return sqrtf(sumA) < 0.0005 * setting_thOptIterations &&
               sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
               sqrtf(sumR) < 0.00005 * setting_thOptIterations &&
               sqrtf(sumT) * sumNID < 0.00005 * setting_thOptIterations;
        //
        //	printf("mean steps: %f %f %f!\n",
        //			meanStepC, meanStepP, meanStepD);
    }


    // sets linearization point.
    void System::loadSateBackup()
    {
        geomCalib->setValue(geomCalib->value_backup);
        for (auto &fh : frameHessians)
        {
            fh->frame->efFrame->setState(fh->frame->efFrame->state_backup);
            for (auto &ph : fh->frame->pointHessians)
            {
                ph->setIdepth(ph->efPoint->idepth_backup);
                ph->efPoint->setIdepthZero(ph->efPoint->idepth_backup);
            }
        }

        EFDeltaValid = false;
        setPrecalcValues();
    }


    shared_ptr<MapPoint> System::optimizeImmaturePoint(shared_ptr<ImmaturePoint> &point, int minObs,
                                                       vector<shared_ptr<ImmaturePointTemporaryResidual>> &residuals)
    {
        int nres = 0;

        for (auto &fh : frameHessians)
        {
            if (fh != point->host)
            {
                residuals[nres]->state_NewEnergy = residuals[nres]->state_energy = 0;
                residuals[nres]->state_NewState = ResState::OUT;
                residuals[nres]->state_state = ResState::IN;
                residuals[nres]->target = fh;
                nres++;
            }
        }
        assert(nres == ((int)frameHessians.size()) - 1);

        bool print = false; //rand()%50==0;

        float lastEnergy = 0;
        float lastHdd = 0;
        float lastbd = 0;
        float currentIdepth = (point->idepth_max + point->idepth_min) * 0.5f;

        for (int i = 0; i < nres; i++)
        {
            lastEnergy += point->linearizeResidual(1000, residuals[i], lastHdd, lastbd, currentIdepth, geomCalib);
            residuals[i]->state_state = residuals[i]->state_NewState;
            residuals[i]->state_energy = residuals[i]->state_NewEnergy;
        }

        if (!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
        {
            if (print)
                printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
                       nres, lastHdd, lastEnergy);
            return 0;
        }

        if (print)
            printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n",
                   nres, lastHdd, lastEnergy, currentIdepth);

        float lambda = 0.1;
        for (int iteration = 0; iteration < setting_GNItsOnPointActivation; iteration++)
        {
            float H = lastHdd;
            H *= 1 + lambda;
            float step = (1.0 / H) * lastbd;
            float newIdepth = currentIdepth - step;

            float newHdd = 0;
            float newbd = 0;
            float newEnergy = 0;
            for (int i = 0; i < nres; i++)
                newEnergy += point->linearizeResidual(1, residuals[i], newHdd, newbd, newIdepth, geomCalib);

            if (!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)
            {
                if (print)
                    printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
                           nres,
                           newHdd,
                           lastEnergy);
                return 0;
            }

            if (print)
                printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
                       (true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
                       iteration,
                       log10(lambda),
                       "",
                       lastEnergy, newEnergy, newIdepth);

            if (newEnergy < lastEnergy)
            {
                currentIdepth = newIdepth;
                lastHdd = newHdd;
                lastbd = newbd;
                lastEnergy = newEnergy;
                for (int i = 0; i < nres; i++)
                {
                    residuals[i]->state_state = residuals[i]->state_NewState;
                    residuals[i]->state_energy = residuals[i]->state_NewEnergy;
                }

                lambda *= 0.5;
            }
            else
            {
                lambda *= 5;
            }

            if (fabsf(step) < 0.0001 * currentIdepth)
                break;
        }

        if (!std::isfinite(currentIdepth))
        {
            printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
            return nullptr;
        }

        int numGoodRes = 0;
        for (int i = 0; i < nres; i++)
            if (residuals[i]->state_state == ResState::IN)
                numGoodRes++;

        if (numGoodRes < minObs)
        {
            if (print)
                printf("OptPoint: OUTLIER!\n");
            return nullptr;
        }

        shared_ptr<MapPoint> p = shared_ptr<MapPoint>(new MapPoint(point, geomCalib));

        if (!std::isfinite(p->energyTH))
        {
            return nullptr;
        }

        p->lastResiduals[0].first = nullptr;
        p->lastResiduals[0].second = ResState::OOB;
        p->lastResiduals[1].first = nullptr;
        p->lastResiduals[1].second = ResState::OOB;
        p->efPoint->setIdepthZero(currentIdepth);
        p->setIdepth(currentIdepth);
        p->setPointStatus(ACTIVE);

        for (int i = 0; i < nres; i++)
            if (residuals[i]->state_state == ResState::IN)
            {
                shared_ptr<PointFrameResidual> r = shared_ptr<PointFrameResidual>(new PointFrameResidual(p, p->host, residuals[i]->target));
                r->state_NewEnergy = r->state_energy = 0;
                r->state_NewState = ResState::OUT;
                r->setState(ResState::IN);
                p->residuals.push_back(r);

                if (r->target == frameHessians.back())
                {
                    p->lastResiduals[0].first = r;
                    p->lastResiduals[0].second = ResState::IN;
                }
                else if (r->target == (frameHessians.size() < 2 ? nullptr : frameHessians[frameHessians.size() - 2]))
                {
                    p->lastResiduals[1].first = r;
                    p->lastResiduals[1].second = ResState::IN;
                }
            }

        if (print)
            printf("point activated!\n");

        // statistics_numActivatedPoints++;
        return p;
    }


    float System::optimize(int mnumOptIts)
    {
        if (frameHessians.size() < 2)
            return 0;
        if (frameHessians.size() < 3)
            mnumOptIts = 20;
        if (frameHessians.size() < 4)
            mnumOptIts = 15;

        // get statistics and active residuals.

        activeResiduals.clear();
        int numPoints = 0;
        int numLRes = 0;
        for (auto &fh : frameHessians)
            for (auto &ph : fh->frame->pointHessians)
            {
                // if(!ph || !ph->efPoint) //ph->getPointStatus() != ACTIVE)
                // 	continue;

                for (auto &r : ph->residuals)
                {
                    if (!r->isLinearized)
                    {
                        activeResiduals.push_back((r));
                        r->resetOOB();
                    }
                    else
                        numLRes++;
                }
                numPoints++;
            }

        // if(!setting_debugout_runquiet)
        //     printf("OPTIMIZE %d pts, %d active res, %d lin res!\n",ef->nPoints,(int)activeResiduals.size(), numLRes);

        Vec3 lastEnergy = linearizeAll(false);
        double lastEnergyL = setting_forceAceptStep? 0.0: ef->calcLEnergyF_MT();// calcLEnergy();
        double lastEnergyM = setting_forceAceptStep? 0.0: ef->calcMEnergyF(); //calcMEnergy();

        backEndThreadPool->reduce(bind(&System::applyRes_Reductor, this, true, placeholders::_1, placeholders::_2, 
                               placeholders::_3, placeholders::_4), 0, activeResiduals.size(), 0);

        // if(!setting_debugout_runquiet)
        // {
        //     printf("Initial Error       \t");
        //     printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
        // }

        // debugPlotTracking();

        double lambda = 1e-1;
        float stepsize = 1;
        VecX previousX = VecX::Constant(CPARS + 8 * frameHessians.size(), NAN);
        for (int iteration = 0; iteration < mnumOptIts; iteration++)
        {
            // solve!
            backupState(iteration != 0);
            //solveSystemNew(0);
            ef->lastNullspaces_forLogging = getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);

	        ef->solveSystemF(iteration, lambda, geomCalib);

            double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm());
            previousX = ef->lastX;

            if (std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM))
            {
                float newStepsize = exp(incDirChange * 1.4);
                if (incDirChange < 0 && stepsize > 1)
                    stepsize = 1;

                stepsize = sqrtf(sqrtf(newStepsize * stepsize * stepsize * stepsize));
                if (stepsize > 2)
                    stepsize = 2;
                if (stepsize < 0.25)
                    stepsize = 0.25;
            }

            bool canbreak = doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);

            // eval new energy!
            Vec3 newEnergy = linearizeAll(false);
            double newEnergyL = setting_forceAceptStep? 0.0: ef->calcLEnergyF_MT();
            double newEnergyM = setting_forceAceptStep? 0.0: ef->calcMEnergyF(); //calcMEnergy();

            // if(!setting_debugout_runquiet)
            // {
            //     printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",
            // 		(newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
            // 				lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) ? "ACCEPT" : "REJECT",
            // 		iteration,
            // 		log10(lambda),
            // 		incDirChange,
            // 		stepsize);
            //     printOptRes(newEnergy, newEnergyL, newEnergyM , 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
            // }

            if (setting_forceAceptStep || (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
                                           lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM))
            {

        
                backEndThreadPool->reduce(bind(&System::applyRes_Reductor, this, true, placeholders::_1, placeholders::_2, 
                                   placeholders::_3, placeholders::_4), 0, activeResiduals.size(), 0);

                lastEnergy = newEnergy;
                lastEnergyL = newEnergyL;
                lastEnergyM = newEnergyM;

                lambda *= 0.25;
            }
            else
            {
                loadSateBackup();
                lastEnergy = linearizeAll(false);
                lastEnergyL = setting_forceAceptStep? 0.0: ef->calcLEnergyF_MT();// calcLEnergy();
                lastEnergyM = setting_forceAceptStep? 0.0: ef->calcMEnergyF(); //calcMEnergy();
                lambda *= 1e2;
            }

            if (canbreak && iteration >= setting_minOptIterations)
                break;
        }

        Vec10 newStateZero = Vec10::Zero();
        newStateZero.segment<2>(6) = frameHessians.back()->frame->efFrame->get_state().segment<2>(6);

        frameHessians.back()->frame->efFrame->setEvalPT(frameHessians.back()->frame->efFrame->PRE_worldToCam,
                                                        newStateZero);
        EFDeltaValid = false;
        EFAdjointsValid = false;
        ef->setAdjointsF(geomCalib);
        setPrecalcValues();

        lastEnergy = linearizeAll(true);

        if (!std::isfinite((double)lastEnergy[0]) || !std::isfinite((double)lastEnergy[1]) || !std::isfinite((double)lastEnergy[2]))
        {
            printf("KF Tracking failed: LOST!\n");
            isLost = true;
        }

        // statistics_lastFineTrackRMSE = sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA)));

        // if(calibLog != 0)
        // {
        // 	(*calibLog) << Calib->value_scaled.transpose() <<
        // 			" " << frameHessians.back()->get_state_scaled().transpose() <<
        // 			" " << sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA))) <<
        // 			" " << ef->resInM << "\n";
        // 	calibLog->flush();
        // }

        {
            unique_lock<mutex> crlock(shellPoseMutex);
            for (auto &fh : frameHessians)
            {
                fh->camToWorld = fh->frame->efFrame->PRE_camToWorld;
                fh->aff_g2l = fh->frame->efFrame->aff_g2l();
            }
        }

        // debugPlotTracking();

        return sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));
    }

} // namespace SLAM