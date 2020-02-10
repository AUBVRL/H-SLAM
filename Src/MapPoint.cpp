#include "MapPoint.h"
#include "CalibData.h"
#include "Frame.h"
#include "ImmaturePoint.h"

namespace FSLAM
{

MapPoint::MapPoint(std::shared_ptr<ImmaturePoint> rawPoint, std::shared_ptr<CalibData> Hcalib)
{
	// instanceCounter++;
    Calib = Hcalib;
	host = rawPoint->host.lock();
	hasDepthPrior=false;

	// set static values & initialization.
	u = rawPoint->u;
	v = rawPoint->v;

	idepth_hessian=0;
	maxRelBaseline=0;
	numGoodResiduals=0;
	WasMarginalized = false;
    // std::weak_ptr<Frame> host;

	assert(std::isfinite(rawPoint->idepth_max));

	my_type = rawPoint->my_type;

	setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min)*0.5);
	setPointStatus(INACTIVE);

	int n = patternNum;
	memcpy(color, rawPoint->color, sizeof(float)*n);
	memcpy(weights, rawPoint->weights, sizeof(float)*n);
	energyTH = rawPoint->energyTH;

	// efPoint=0;

}

bool MapPoint::isOOB(const std::vector<std::shared_ptr<Frame>> &toMarg) const
{
	int visInToMarg = 0;
	for (auto r : residuals)
	{
		if (r->state_state != ResState::IN)
			continue;
		for (auto k : toMarg)
			if (r->target.lock() == k)
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