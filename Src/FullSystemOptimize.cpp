

#include "System.h"
 
// #include "stdio.h"
#include "GlobalTypes.h"
#include "CalibData.h"
#include "Frame.h"
#include "MapPoint.h"
#include "DirectProjection.h"

#include "EnergyFunctional.h"


namespace FSLAM
{


void System::linearizeAll_Reductor(bool fixLinearization, vector<shared_ptr<PointFrameResidual>>* toRemove, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
	{
		auto &r = activeResiduals[k];
		(*stats)[0] += r->linearize(Calib);

		if(fixLinearization)
		{
			r->applyRes(true);

			if(r->isActive())
			{
				if(r->isNew)
				{
					auto p = r->point;
					Vec3f ptp_inf = r->host->frame->targetPrecalc[r->target->frame->idx].PRE_KRKiTll * Vec3f(p->u, p->v, 1);	// projected point assuming infinite depth.
					Vec3f ptp = ptp_inf + r->host->frame->targetPrecalc[r->target->frame->idx].PRE_KtTll* p->idepth;	// projected point with real depth.
					float relBS = 0.01*((ptp_inf.head<2>() / ptp_inf[2])-(ptp.head<2>() / ptp[2])).norm();	// 0.01 = one pixel.


					if(relBS > p->maxRelBaseline)
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


void System::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
		activeResiduals[k]->applyRes(true);
}
void System::setNewFrameEnergyTH()
{
	// collect all residuals and make decision on TH.
	allResVec.clear();
	allResVec.reserve(activeResiduals.size()*2);
	shared_ptr<FrameShell> newFrame = frameHessians.back();

	for(auto &r : activeResiduals)
		if(r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame)
		{
			allResVec.push_back(r->state_NewEnergyWithOutlier);
		}

	if(allResVec.size()==0)
	{
		newFrame->frame->frameEnergyTH = 12*12*patternNum;
		return;		// should never happen, but lets make sure.
	}


	int nthIdx = setting_frameEnergyTHN*allResVec.size();

	assert(nthIdx < (int)allResVec.size());
	assert(setting_frameEnergyTHN < 1);

	std::nth_element(allResVec.begin(), allResVec.begin()+nthIdx, allResVec.end());
	float nthElement = sqrtf(allResVec[nthIdx]);

    newFrame->frame->frameEnergyTH = nthElement*setting_frameEnergyTHFacMedian;
	newFrame->frame->frameEnergyTH = 26.0f*setting_frameEnergyTHConstWeight + newFrame->frame->frameEnergyTH*(1-setting_frameEnergyTHConstWeight);
	newFrame->frame->frameEnergyTH = newFrame->frame->frameEnergyTH*newFrame->frame->frameEnergyTH;
	newFrame->frame->frameEnergyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;



//
//	int good=0,bad=0;
//	for(float f : allResVec) if(f<newFrame->frameEnergyTH) good++; else bad++;
//	printf("EnergyTH: mean %f, median %f, result %f (in %d, out %d)! \n",
//			meanElement, nthElement, sqrtf(newFrame->frameEnergyTH),
//			good, bad);
}
Vec3 System::linearizeAll(bool fixLinearization)
{
	double lastEnergyP = 0;
	double lastEnergyR = 0;
	double num = 0;


	vector<shared_ptr<PointFrameResidual>> toRemove[NUM_THREADS];
	for(int i=0;i<NUM_THREADS;i++) toRemove[i].clear();

	if(multiThreading)
	{
		treadReduce.reduce(boost::bind(&System::linearizeAll_Reductor, this, fixLinearization, toRemove, _1, _2, _3, _4), 0, activeResiduals.size(), 0);
		lastEnergyP = treadReduce.stats[0];
	}
	else
	{
		Vec10 stats;
		linearizeAll_Reductor(fixLinearization, toRemove, 0,activeResiduals.size(),&stats,0);
		lastEnergyP = stats[0];
	}

	setNewFrameEnergyTH();

	if(fixLinearization)
	{
		for(auto &r : activeResiduals)
		{
			auto &ph = r->point;
			if(ph->lastResiduals[0].first == r)
				ph->lastResiduals[0].second = r->state_state;
			else if(ph->lastResiduals[1].first == r)
				ph->lastResiduals[1].second = r->state_state;
		}

		for(int i=0;i<NUM_THREADS;i++)
		{
			for(auto r : toRemove[i])
			{
				auto &ph = r->point;

				if(ph->lastResiduals[0].first == r)
					ph->lastResiduals[0].first.reset();
				else if (ph->lastResiduals[1].first == r)
					ph->lastResiduals[1].first.reset();

				for(unsigned int k=0; k<ph->residuals.size();k++)
					if(ph->residuals[k] == r)
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




// applies step to linearization point.
bool System::doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD)
{
//	float meanStepC=0,meanStepP=0,meanStepD=0;
//	meanStepC += Hcalib.step.norm();

	Vec10 pstepfac;
	pstepfac.segment<3>(0).setConstant(stepfacT);
	pstepfac.segment<3>(3).setConstant(stepfacR);
	pstepfac.segment<4>(6).setConstant(stepfacA);


	float sumA=0, sumB=0, sumT=0, sumR=0, sumID=0, numID=0;

	float sumNID=0;

	if(setting_solverMode & SOLVER_MOMENTUM)
	{
		Calib->setValue(Calib->value_backup + Calib->step);
		for(auto &fh : frameHessians)
		{
			Vec10 step = fh->frame->efFrame->step;
			step.head<6>() += 0.5f*(fh->frame->efFrame->step_backup.head<6>());

			fh->frame->efFrame->setState(fh->frame->efFrame->state_backup + step);
			sumA += step[6]*step[6];
			sumB += step[7]*step[7];
			sumT += step.segment<3>(0).squaredNorm();
			sumR += step.segment<3>(3).squaredNorm();

			for(auto &ph : fh->frame->pointHessians)
			{
				float step = ph->efPoint->step+0.5f*(ph->efPoint->step_backup);
				ph->setIdepth(ph->efPoint->idepth_backup + step);
				sumID += step*step;
				sumNID += fabsf(ph->efPoint->idepth_backup);
				numID++;

                ph->efPoint->setIdepthZero(ph->efPoint->idepth_backup + step);
			}
		}
	}
	else
	{
		Calib->setValue(Calib->value_backup + stepfacC *Calib->step);
		for(auto &fh : frameHessians)
		{
			fh->frame->efFrame->setState(fh->frame->efFrame->state_backup + pstepfac.cwiseProduct(fh->frame->efFrame->step));
			sumA += fh->frame->efFrame->step[6]*fh->frame->efFrame->step[6];
			sumB += fh->frame->efFrame->step[7]*fh->frame->efFrame->step[7];
			sumT += fh->frame->efFrame->step.segment<3>(0).squaredNorm();
			sumR += fh->frame->efFrame->step.segment<3>(3).squaredNorm();

			for(auto &ph : fh->frame->pointHessians)
			{
				ph->setIdepth(ph->efPoint->idepth_backup + stepfacD*ph->efPoint->step);
				sumID += ph->efPoint->step*ph->efPoint->step;
				sumNID += fabsf(ph->efPoint->idepth_backup);
				numID++;

                ph->efPoint->setIdepthZero(ph->efPoint->idepth_backup + stepfacD*ph->efPoint->step);
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


	EFDeltaValid=false;
	setPrecalcValues();



	return sqrtf(sumA) < 0.0005*setting_thOptIterations &&
			sqrtf(sumB) < 0.00005*setting_thOptIterations &&
			sqrtf(sumR) < 0.00005*setting_thOptIterations &&
			sqrtf(sumT)*sumNID < 0.00005*setting_thOptIterations;
//
//	printf("mean steps: %f %f %f!\n",
//			meanStepC, meanStepP, meanStepD);
}



// sets linearization point.
void System::backupState(bool backupLastStep)
{
	if(setting_solverMode & SOLVER_MOMENTUM)
	{
		if(backupLastStep)
		{
			Calib->step_backup = Calib->step;
			Calib->value_backup = Calib->value;
			for(auto &fh : frameHessians)
			{
				fh->frame->efFrame->step_backup = fh->frame->efFrame->step;
				fh->frame->efFrame->state_backup = fh->frame->efFrame->get_state();
				for(auto &ph : fh->frame->pointHessians)
				{
					ph->efPoint->idepth_backup = ph->idepth;
					ph->efPoint->step_backup = ph->efPoint->step;
				}
			}
		}
		else
		{
			Calib->step_backup.setZero();
			Calib->value_backup = Calib->value;
			for(auto &fh : frameHessians)
			{
				fh->frame->efFrame->step_backup.setZero();
				fh->frame->efFrame->state_backup = fh->frame->efFrame->get_state();
				for(auto &ph : fh->frame->pointHessians)
				{
					ph->efPoint->idepth_backup = ph->idepth;
					ph->efPoint->step_backup=0;
				}
			}
		}
	}
	else
	{
		Calib->value_backup = Calib->value;
		for(auto &fh : frameHessians)
		{
			fh->frame->efFrame->state_backup = fh->frame->efFrame->get_state();
			for(auto &ph : fh->frame->pointHessians)
				ph->efPoint->idepth_backup = ph->idepth;
		}
	}
}

// sets linearization point.
void System::loadSateBackup()
{
	Calib->setValue(Calib->value_backup);
	for(auto &fh : frameHessians)
	{
		fh->frame->efFrame->setState(fh->frame->efFrame->state_backup);
		for(auto &ph : fh->frame->pointHessians)
		{
			ph->setIdepth(ph->efPoint->idepth_backup);
            ph->efPoint->setIdepthZero(ph->efPoint->idepth_backup);
		}

	}


	EFDeltaValid=false;
	setPrecalcValues();
}


double System::calcMEnergy()
{
	if(setting_forceAceptStep) return 0;
	// calculate (x-x0)^T * [2b + H * (x-x0)] for everything saved in L.
	//ef->makeIDX();
	//ef->setDeltaF(&Hcalib);
	return ef->calcMEnergyF();

}


void System::printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b)
{
	printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n",
			res[0],
			sqrtf((float)(res[0] / (patternNum*ef->resInA))),
			ef->resInA,
			ef->resInM,
			a,
			b
	);

}


float System::optimize(int mnumOptIts)
{

	if(frameHessians.size() < 2) return 0;
	if(frameHessians.size() < 3) mnumOptIts = 20;
	if(frameHessians.size() < 4) mnumOptIts = 15;

	// get statistics and active residuals.

	activeResiduals.clear();
	int numPoints = 0;
	int numLRes = 0;
	for(auto &fh : frameHessians)
		for(auto &ph : fh->frame->pointHessians)
		{
			// if(!ph || !ph->efPoint) //ph->getPointStatus() != ACTIVE)
			// 	continue;
			
			for(auto &r : ph->residuals)
			{
				if(!r->isLinearized)
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
	double lastEnergyL = calcLEnergy();
	double lastEnergyM = calcMEnergy();


	if(multiThreading)
		treadReduce.reduce(boost::bind(&System::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
	else
		applyRes_Reductor(true,0,activeResiduals.size(),0,0);


    // if(!setting_debugout_runquiet)
    // {
    //     printf("Initial Error       \t");
    //     printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
    // }

	// debugPlotTracking();


	double lambda = 1e-1;
	float stepsize=1;
	VecX previousX = VecX::Constant(CPARS+ 8*frameHessians.size(), NAN);
	for(int iteration=0;iteration<mnumOptIts;iteration++)
	{
		// solve!
		backupState(iteration!=0);
		//solveSystemNew(0);
		solveSystem(iteration, lambda);
		double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm());
		previousX = ef->lastX;


		if(std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM))
		{
			float newStepsize = exp(incDirChange*1.4);
			if(incDirChange<0 && stepsize>1) stepsize=1;

			stepsize = sqrtf(sqrtf(newStepsize*stepsize*stepsize*stepsize));
			if(stepsize > 2) stepsize=2;
			if(stepsize <0.25) stepsize=0.25;
		}

		bool canbreak = doStepFromBackup(stepsize,stepsize,stepsize,stepsize,stepsize);







		// eval new energy!
		Vec3 newEnergy = linearizeAll(false);
		double newEnergyL = calcLEnergy();
		double newEnergyM = calcMEnergy();




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

		if(setting_forceAceptStep || (newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
				lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM))
		{

			if(multiThreading)
				treadReduce.reduce(boost::bind(&System::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
			else
				applyRes_Reductor(true,0,activeResiduals.size(),0,0);

			lastEnergy = newEnergy;
			lastEnergyL = newEnergyL;
			lastEnergyM = newEnergyM;

			lambda *= 0.25;
		}
		else
		{
			loadSateBackup();
			lastEnergy = linearizeAll(false);
			lastEnergyL = calcLEnergy();
			lastEnergyM = calcMEnergy();
			lambda *= 1e2;
		}


		if(canbreak && iteration >= setting_minOptIterations) break;
	}



	Vec10 newStateZero = Vec10::Zero();
	newStateZero.segment<2>(6) = frameHessians.back()->frame->efFrame->get_state().segment<2>(6);

	frameHessians.back()->frame->efFrame->setEvalPT(frameHessians.back()->frame->efFrame->PRE_worldToCam,
			newStateZero);
	EFDeltaValid=false;
	EFAdjointsValid=false;
	ef->setAdjointsF(Calib);
	setPrecalcValues();


	lastEnergy = linearizeAll(true);


	if(!std::isfinite((double)lastEnergy[0]) || !std::isfinite((double)lastEnergy[1]) || !std::isfinite((double)lastEnergy[2]))
    {
        printf("KF Tracking failed: LOST!\n");
		isLost=true;
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
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		for(auto &fh : frameHessians)
		{
			fh->camToWorld = fh->frame->efFrame->PRE_camToWorld;
			fh->aff_g2l = fh->frame->efFrame->aff_g2l();
		}
	}




	// debugPlotTracking();

	return sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA)));

}





void System::solveSystem(int iteration, double lambda)
{
	ef->lastNullspaces_forLogging = getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);

	ef->solveSystemF(iteration, lambda, Calib);
}



double System::calcLEnergy()
{
	if(setting_forceAceptStep) return 0;

	double Ef = ef->calcLEnergyF_MT();
	return Ef;

}


void System::removeOutliers()
{
	int numPointsDropped=0;
	for(auto &fh : frameHessians)
	{
		for(unsigned int i=0; i < fh->frame->pointHessians.size(); i++)
		{
			std::shared_ptr<MapPoint>& ph = fh->frame->pointHessians[i];
			if(!ph )
				continue;
			// if(ph->status!= MapPoint::ACTIVE)
			// 	continue;
			if(ph->residuals.size() == 0)
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

		for (auto& ph: fh->frame->pointHessiansOut)
		{
			if(!ph || !ph->efPoint)
				continue;
			if(ph->efPoint->stateFlag == energyStatus::toDrop)
				ef->removePoint(ph);		
		}
	}

	EFIndicesValid = false;
	ef->makeIDX();
	// ef->dropPointsF();
}




std::vector<VecX> System::getNullspaces(
		std::vector<VecX> &nullspaces_pose,
		std::vector<VecX> &nullspaces_scale,
		std::vector<VecX> &nullspaces_affA,
		std::vector<VecX> &nullspaces_affB)
{
	nullspaces_pose.clear();
	nullspaces_scale.clear();
	nullspaces_affA.clear();
	nullspaces_affB.clear();


	int n=CPARS+frameHessians.size()*8;
	std::vector<VecX> nullspaces_x0_pre;
	for(int i=0;i<6;i++)
	{
		VecX nullspace_x0(n);
		nullspace_x0.setZero();
		for(auto fh : frameHessians)
		{
			nullspace_x0.segment<6>(CPARS+fh->frame->idx*8) = fh->frame->efFrame->nullspaces_pose.col(i);
			nullspace_x0.segment<3>(CPARS+fh->frame->idx*8) *= SCALE_XI_TRANS_INVERSE;
			nullspace_x0.segment<3>(CPARS+fh->frame->idx*8+3) *= SCALE_XI_ROT_INVERSE;
		}
		nullspaces_x0_pre.push_back(nullspace_x0);
		nullspaces_pose.push_back(nullspace_x0);
	}
	for(int i=0;i<2;i++)
	{
		VecX nullspace_x0(n);
		nullspace_x0.setZero();
		for(auto &fh : frameHessians)
		{
			nullspace_x0.segment<2>(CPARS+fh->frame->idx*8+6) = fh->frame->efFrame->nullspaces_affine.col(i).head<2>();
			nullspace_x0[CPARS+fh->frame->idx*8+6] *= SCALE_A_INVERSE;
			nullspace_x0[CPARS+fh->frame->idx*8+7] *= SCALE_B_INVERSE;
		}
		nullspaces_x0_pre.push_back(nullspace_x0);
		if(i==0) nullspaces_affA.push_back(nullspace_x0);
		if(i==1) nullspaces_affB.push_back(nullspace_x0);
	}

	VecX nullspace_x0(n);
	nullspace_x0.setZero();
	for(auto &fh : frameHessians)
	{
		nullspace_x0.segment<6>(CPARS+fh->frame->idx*8) = fh->frame->efFrame->nullspaces_scale;
		nullspace_x0.segment<3>(CPARS+fh->frame->idx*8) *= SCALE_XI_TRANS_INVERSE;
		nullspace_x0.segment<3>(CPARS+fh->frame->idx*8+3) *= SCALE_XI_ROT_INVERSE;
	}
	nullspaces_x0_pre.push_back(nullspace_x0);
	nullspaces_scale.push_back(nullspace_x0);

	return nullspaces_x0_pre;
}

}
