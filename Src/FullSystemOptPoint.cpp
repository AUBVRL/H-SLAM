#include "System.h"
 
// #include "stdio.h"
// #include "GlobalTypes.h"
// #include <Eigen/LU>
// #include <algorithm>
// #include "IOWrapper/ImageDisplay.h"
// #include "util/globalCalib.h"

// #include <Eigen/SVD>
// #include <Eigen/Eigenvalues>
#include "ImmaturePoint.h"
#include "MapPoint.h"
#include "OptimizationClasses.h"
#include "Frame.h"
// #include "math.h"

namespace FSLAM
{



std::shared_ptr<MapPoint> System::optimizeImmaturePoint(std::shared_ptr<ImmaturePoint>& point, int minObs, std::vector<std::shared_ptr<ImmaturePointTemporaryResidual>>& residuals)
{
	int nres = 0;
	
	for(auto &fh : frameHessians)
	{
		if(fh != point->host)
		{
			residuals[nres]->state_NewEnergy = residuals[nres]->state_energy = 0;
			residuals[nres]->state_NewState = ResState::OUT;
			residuals[nres]->state_state = ResState::IN;
			residuals[nres]->target = fh;
			nres++;
		}
	}
	assert(nres == ((int)frameHessians.size())-1);

	bool print = false;//rand()%50==0;

	float lastEnergy = 0;
	float lastHdd=0;
	float lastbd=0;
	float currentIdepth=(point->idepth_max+point->idepth_min)*0.5f;

	for(int i=0;i<nres;i++)
	{
		lastEnergy += point->linearizeResidual( 1000, residuals[i],lastHdd, lastbd, currentIdepth, Calib);
		residuals[i]->state_state = residuals[i]->state_NewState;
		residuals[i]->state_energy = residuals[i]->state_NewEnergy;
	}

	if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
	{
		if(print)
			printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
				nres, lastHdd, lastEnergy);
		return 0;
	}

	if(print) printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n" ,
			nres, lastHdd,lastEnergy,currentIdepth);

	float lambda = 0.1;
	for(int iteration=0;iteration<setting_GNItsOnPointActivation;iteration++)
	{
		float H = lastHdd;
		H *= 1+lambda;
		float step = (1.0/H) * lastbd;
		float newIdepth = currentIdepth - step;

		float newHdd=0; float newbd=0; float newEnergy=0;
		for(int i=0;i<nres;i++)
			newEnergy += point->linearizeResidual(1, residuals[i],newHdd, newbd, newIdepth, Calib);

		if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)
		{
			if(print) printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
					nres,
					newHdd,
					lastEnergy);
			return 0;
		}

		if(print) printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
				(true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
				iteration,
				log10(lambda),
				"",
				lastEnergy, newEnergy, newIdepth);

		if(newEnergy < lastEnergy)
		{
			currentIdepth = newIdepth;
			lastHdd = newHdd;
			lastbd = newbd;
			lastEnergy = newEnergy;
			for(int i=0;i<nres;i++)
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

		if(fabsf(step) < 0.0001*currentIdepth)
			break;
	}

	if(!std::isfinite(currentIdepth))
	{
		printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
		return nullptr;
	}


	int numGoodRes=0;
	for(int i=0;i<nres;i++)
		if(residuals[i]->state_state == ResState::IN) numGoodRes++;

	if(numGoodRes < minObs)
	{
		if(print) printf("OptPoint: OUTLIER!\n");
		return nullptr;
	}


	std::shared_ptr<MapPoint> p = std::shared_ptr<MapPoint>(new MapPoint(point, Calib));

	if(!std::isfinite(p->energyTH))
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

	for(int i=0;i<nres;i++)
		if(residuals[i]->state_state == ResState::IN)
		{
			std::shared_ptr<PointFrameResidual> r = std::shared_ptr<PointFrameResidual>(new PointFrameResidual(p->host, residuals[i]->target));
			r->state_NewEnergy = r->state_energy = 0;
			r->state_NewState = ResState::OUT;
			r->setState(ResState::IN);
			p->residuals.push_back(r);

			if(r->target == frameHessians.back())
			{
				p->lastResiduals[0].first = r;
				p->lastResiduals[0].second = ResState::IN;
			}
			else if(r->target == (frameHessians.size()<2 ? nullptr : frameHessians[frameHessians.size()-2]))
			{
				p->lastResiduals[1].first = r;
				p->lastResiduals[1].second = ResState::IN;
			}
		}

	if(print) printf("point activated!\n");

	// statistics_numActivatedPoints++;
	return p;
}



}
