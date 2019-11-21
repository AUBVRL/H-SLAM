#pragma once

 
#include "GlobalTypes.h"

// #include "FullSystem/HessianBlocks.h"
namespace FSLAM
{

class Frame;
class CalibData;

struct ImmaturePointTemporaryResidual
{
public:
	enum ResState {IN=0, OOB, OUTLIER};

	ResState state_state;
	double state_energy;
	ResState state_NewState;
	double state_NewEnergy;
	std::weak_ptr<Frame> target;
};


enum ImmaturePointStatus {
	IPS_GOOD=0,					// traced well and good
	IPS_OOB,					// OOB: end tracking & marginalize!
	IPS_OUTLIER,				// energy too high: if happens again: outlier!
	IPS_SKIPPED,				// traced well and good (but not actually traced).
	IPS_BADCONDITION,			// not traced because of bad condition.
	IPS_UNINITIALIZED};			// not even traced once.


class ImmaturePoint
{
public:
	// EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// static values
	float color[MAX_RES_PER_POINT];
	float weights[MAX_RES_PER_POINT];

	Mat22f gradH;
	Vec2f gradH_ev;
	Mat22f gradH_eig;
	float energyTH;
	float u,v;
	std::weak_ptr<Frame> hostFrame;
	int idxInImmaturePoints;

	float quality;

	float my_type;

	float idepth_min;
	float idepth_max;
	ImmaturePoint(int u_, int v_, std::weak_ptr<Frame> host_, float type, std::weak_ptr<CalibData> HCalib);
	~ImmaturePoint();

	ImmaturePointStatus traceOn(std::weak_ptr<Frame> frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f &hostToFrame_affine, std::weak_ptr<CalibData> HCalib, bool debugPrint=false);

	ImmaturePointStatus lastTraceStatus;
	Vec2f lastTraceUV;
	float lastTracePixelInterval;

	float idepth_GT;

	double linearizeResidual( std::weak_ptr<CalibData> HCalib, const float outlierTHSlack, ImmaturePointTemporaryResidual* tmpRes, float &Hdd, float &bd, float idepth);
	float getdPixdd(std::weak_ptr<CalibData> HCalib, ImmaturePointTemporaryResidual* tmpRes, float idepth);

	float calcResidual(
			std::weak_ptr<CalibData> HCalib, const float outlierTHSlack, ImmaturePointTemporaryResidual* tmpRes, float idepth);
private:

};

}

