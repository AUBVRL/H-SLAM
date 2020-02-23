#ifndef __IMMATUREPOINT_H__
#define __IMMATUREPOINT_H__

 
#include "GlobalTypes.h"
 
namespace FSLAM
{

class FrameShell;
class CalibData;

struct ImmaturePointTemporaryResidual
{
public:
	ResState state_state;
	double state_energy;
	ResState state_NewState;
	double state_NewEnergy;
	shared_ptr<FrameShell> target;
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
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	float color[MAX_RES_PER_POINT];
	float weights[MAX_RES_PER_POINT];

	Mat22f gradH;
	Vec2f gradH_ev;
	Mat22f gradH_eig;
	float energyTH;
	float u,v;
	shared_ptr<FrameShell> host;
	int idxInImmaturePoints;

	float quality;
	float my_type;
	float idepth_min;
	float idepth_max;
	ImmaturePoint(float u_, float v_, shared_ptr<FrameShell> host_, float type, shared_ptr<CalibData> Calib);
	~ImmaturePoint(){};

	ImmaturePointStatus lastTraceStatus;
	Vec2f lastTraceUV;
	float lastTracePixelInterval;

	ImmaturePointStatus traceOn(shared_ptr<FrameShell> frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f &hostToFrame_affine,
								shared_ptr<CalibData>& Calib, bool debugPrint = false);
	double linearizeResidual( const float outlierTHSlack, shared_ptr<ImmaturePointTemporaryResidual> tmpRes, float &Hdd, float &bd, float idepth, shared_ptr<CalibData> &Calib);
	float calcResidual(const float outlierTHSlack, shared_ptr<ImmaturePointTemporaryResidual> tmpRes, float idepth, shared_ptr<CalibData> &Calib);

};

}

#endif