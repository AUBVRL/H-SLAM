#include "System.h"
 
// #include "stdio.h"
#include "GlobalTypes.h"
#include "CalibData.h"
#include "DirectProjection.h"
#include "Frame.h"
#include "MapPoint.h"

// #include <Eigen/LU>
// #include <algorithm>
// #include "IOWrapper/ImageDisplay.h"
// #include "util/globalCalib.h"

// #include <Eigen/SVD>
// #include <Eigen/Eigenvalues>
// #include "FullSystem/ResidualProjections.h"
// #include "FullSystem/ImmaturePoint.h"

#include "EnergyFunctional.h"
#include "EnergyFunctionalStructs.h"

// #include "IOWrapper/Output3DWrapper.h"

#include "CoarseTracker.h"

namespace FSLAM
{

void System::flagFramesForMarginalization(std::shared_ptr<Frame> newFH)
{
	if(setting_minFrameAge > setting_maxFrames)
	{
		for(int i=setting_maxFrames;i<(int)frameHessians.size();i++)
		{
			auto fh = frameHessians[i-setting_maxFrames];
			fh->FlaggedForMarginalization = true;
		}
		return;
	}


	int flagged = 0;
	// marginalize all frames that have not enough points.
	for(int i=0;i<(int)frameHessians.size();i++)
	{
		auto fh = frameHessians[i];
		int in = fh->pointHessians.size() + fh->ImmaturePoints.size();
		int out = fh->pointHessiansMarginalized.size() + fh->pointHessiansOut.size();


		Vec2 refToFh=AffLight::fromToVecExposure(frameHessians.back()->ab_exposure, fh->ab_exposure,
				frameHessians.back()->aff_g2l(), fh->aff_g2l());


		if( (in < setting_minPointsRemaining *(in+out) || fabs(logf((float)refToFh[0])) > setting_maxLogAffFacInWindow)
				&& ((int)frameHessians.size())-flagged > setting_minFrames)
		{
//			printf("MARGINALIZE frame %d, as only %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
//					fh->frameID, in, in+out,
//					(int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
//					(int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
//					visInLast, outInLast,
//					fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
			fh->FlaggedForMarginalization = true;
			flagged++;
		}
		else
		{
//			printf("May Keep frame %d, as %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
//					fh->frameID, in, in+out,
//					(int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
//					(int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
//					visInLast, outInLast,
//					fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
		}
	}

	// marginalize one.
	if((int)frameHessians.size()-flagged >= setting_maxFrames)
	{
		double smallestScore = 1;
		std::shared_ptr<Frame> toMarginalize;
		auto latest = frameHessians.back();


		for(auto fh : frameHessians)
		{
			if(fh->id > latest->id-setting_minFrameAge || fh->id == 0) continue;
			//if(fh==frameHessians.front() == 0) continue;

			double distScore = 0;
			for(FrameFramePrecalc &ffh : fh->targetPrecalc)
			{
				if(ffh.target.lock()->id > latest->id-setting_minFrameAge+1 || ffh.target.lock() == ffh.host.lock()) continue;
				distScore += 1/(1e-5+ffh.distanceLL);

			}
			distScore *= -sqrtf(fh->targetPrecalc.back().distanceLL);


			if(distScore < smallestScore)
			{
				smallestScore = distScore;
				toMarginalize = fh;
			}
		}

//		printf("MARGINALIZE frame %d, as it is the closest (score %.2f)!\n",
//				toMarginalize->frameID, smallestScore);
		toMarginalize->FlaggedForMarginalization = true;
		flagged++;
	}

//	printf("FRAMES LEFT: ");
//	for(FrameHessian* fh : frameHessians)
//		printf("%d ", fh->frameID);
//	printf("\n");
}




void System::marginalizeFrame(std::shared_ptr<Frame> frame)
{
	// marginalize or remove all this frames points.

	assert((int)frame->pointHessians.size()==0);


	ef->marginalizeFrame(frame);

	// drop all observations of existing points in that frame.

	for (auto fh : frameHessians)
	{
		if (fh == frame)
			continue;

		for (auto ph : fh->pointHessians)
		{
			if (ph)
			{
				if (ph->status == MapPoint::ACTIVE)
				{
					size_t n = ph->residuals.size();
					for (unsigned int i = 0; i < n; i++)
					{
						std::shared_ptr<PointFrameResidual> r = ph->residuals[i];
						
						if (r->target.lock() == frame)
						{
							if (ph->lastResiduals[0].first == r)
								ph->lastResiduals[0].first.reset();
							else if (ph->lastResiduals[1].first == r)
								ph->lastResiduals[1].first.reset();

							// if(r->host->frameID < r->target->frameID)
							// 	statistics_numForceDroppedResFwd++;
							// else
							// 	statistics_numForceDroppedResBwd++;

							ef->dropResidual(r);
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
	frame->MovedByOpt = frame->w2c_leftEps().norm();

	deleteOutOrder<Frame>(frameHessians, frame);
	for(unsigned int i=0;i<frameHessians.size();i++)
		frameHessians[i]->idx = i;




	setPrecalcValues();
	ef->setAdjointsF(Calib);
}




}
