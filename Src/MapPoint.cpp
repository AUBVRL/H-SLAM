#include "MapPoint.h"
#include "CalibData.h"
#include "Frame.h"
#include "ImmaturePoint.h"

namespace FSLAM
{

MapPoint::MapPoint(const ImmaturePoint* const rawPoint, std::shared_ptr<CalibData> Hcalib)
{
	instanceCounter++;
    Calib = Hcalib;
	host = rawPoint->host;
	hasDepthPrior=false;

	idepth_hessian=0;
	maxRelBaseline=0;
	numGoodResiduals=0;

	// set static values & initialization.
	u = rawPoint->u;
	v = rawPoint->v;
	assert(std::isfinite(rawPoint->idepth_max));
	//idepth_init = rawPoint->idepth_GT;

	my_type = rawPoint->my_type;

	setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min)*0.5);
	setPointStatus(INACTIVE);

	int n = patternNum;
	memcpy(color, rawPoint->color, sizeof(float)*n);
	memcpy(weights, rawPoint->weights, sizeof(float)*n);
	energyTH = rawPoint->energyTH;

	// efPoint=0;

}

void MapPoint::release()
{
	for (unsigned int i = 0; i < residuals.size(); i++)
		delete residuals[i];
	residuals.clear();
}

bool MapPoint::isOOB(const std::vector<Frame*> &toMarg) const
{

	int visInToMarg = 0;
	for (auto r : residuals)
	{

		if (r->state_state != ResState::IN)
			continue;
		for (auto k : toMarg)
			if (r->target == k)
				visInToMarg++;
	}
	if ((int)residuals.size() >= setting_minGoodActiveResForMarg &&
		numGoodResiduals > setting_minGoodResForMarg + 10 &&
		(int)residuals.size() - visInToMarg < setting_minGoodActiveResForMarg)
		return true;

	if (lastResiduals[0].second == ResState::OOB)
		return true;
	if (residuals.size() < 2)
		return false;
	if (lastResiduals[0].second == ResState::OUTLIER && lastResiduals[1].second == ResState::OUTLIER)
		return true;
	return false;
}

bool MapPoint::isInlierNew()
{
	return (int)residuals.size() >= setting_minGoodActiveResForMarg && numGoodResiduals >= setting_minGoodResForMarg;
}

}