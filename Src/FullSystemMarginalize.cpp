#include "System.h"
 
// #include "stdio.h"
#include "GlobalTypes.h"
#include "CalibData.h"
#include "DirectProjection.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Display.h"
#include "ImmaturePoint.h"
#include "EnergyFunctional.h"

#include "CoarseTracker.h"

namespace FSLAM
{

void System::flagFramesForMarginalization()
{
	if(setting_minFrameAge > setting_maxFrames)
	{
		for(int i=setting_maxFrames;i<(int)frameHessians.size();i++)
		{
			auto &fh = frameHessians[i-setting_maxFrames];
			fh->frame->FlaggedForMarginalization = true;
		}
		return;
	}

	int flagged = 0;
	// marginalize all frames that have not enough points.
	for(auto &fh : frameHessians)
	{
		int in = 0;
		int out = 0;
		for (auto &it : fh->frame->pointHessians)
		{
			if(!it)
				continue;
			PtStatus status = it->getPointStatus();
			if( status == ACTIVE)
				in++;
			else if (status == OUTLIER || status == MARGINALIZED)
				out++;
		}
		for (auto &it2 : fh->frame->ImmaturePoints)
		{
			if(!it2)
				continue;
			if(!std::isfinite(it2->idepth_max) || it2->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER || it2->lastTraceStatus == ImmaturePointStatus::IPS_OOB)
				continue;
			in++;
		}

		Vec2 refToFh=AffLight::fromToVecExposure(frameHessians.back()->ab_exposure, fh->ab_exposure, frameHessians.back()->frame->efFrame->aff_g2l(), fh->frame->efFrame->aff_g2l());


		if( (in < setting_minPointsRemaining *(in+out) || fabs(logf((float)refToFh[0])) > setting_maxLogAffFacInWindow) && ((int)frameHessians.size())-flagged > setting_minFrames)
		{
			fh->frame->FlaggedForMarginalization = true;
			flagged++;
		}
	}

	// marginalize one.
	if((int)frameHessians.size()-flagged >= setting_maxFrames)
	{
		double smallestScore = 1;
		std::shared_ptr<FrameShell> toMarginalize;
		auto latest = frameHessians.back();


		for(auto &fh : frameHessians)
		{
			if(fh->KfId > latest->KfId-setting_minFrameAge || fh->KfId == 0) continue;
			//if(fh==frameHessians.front() == 0) continue;

			double distScore = 0;
			for(FrameFramePrecalc &ffh : fh->frame->targetPrecalc)
			{
				if(ffh.target->KfId > latest->KfId-setting_minFrameAge+1 || ffh.target == ffh.host) continue;
				distScore += 1/(1e-5+ffh.distanceLL);

			}
			distScore *= -sqrtf(fh->frame->targetPrecalc.back().distanceLL);


			if(distScore < smallestScore)
			{
				smallestScore = distScore;
				toMarginalize = fh;
			}
		}

//		printf("MARGINALIZE frame %d, as it is the closest (score %.2f)!\n",
//				toMarginalize->frameID, smallestScore);
		toMarginalize->frame->FlaggedForMarginalization = true;
		flagged++;
	}

//	printf("FRAMES LEFT: ");
//	for(FrameHessian* fh : frameHessians)
//		printf("%d ", fh->frameID);
//	printf("\n");
}




void System::marginalizeFrame(std::shared_ptr<FrameShell> frame)
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
			if (ph)
			{
				if (ph->getPointStatus() == ACTIVE)
				{
					size_t n = ph->residuals.size();
					for (unsigned int i = 0; i < n; i++)
					{
						std::shared_ptr<PointFrameResidual> r = ph->residuals[i];
						
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

							ef->dropResidual(ph, r); //this should remove the only holding copy of the pointframeresidual
							i--;
							n--;
							// deleteOut<PointFrameResidual>(ph->residuals, i);
							break;
						}
					}
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

	if (DisplayHandler)
	{
		boost::unique_lock<boost::mutex> lock(DisplayHandler->KeyframesMutex); 
		frame->frame->NeedRefresh = true;
	}

	deleteOutOrder<FrameShell>(frameHessians, frame);
	for(unsigned int i=0;i<frameHessians.size();i++)
		frameHessians[i]->frame->idx = i;

	setPrecalcValues();
	ef->setAdjointsF(Calib);
}




}
