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
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// static values
	float color[MAX_RES_PER_POINT];
	float weights[MAX_RES_PER_POINT];

	Mat22f gradH;
	Vec2f gradH_ev;
	Mat22f gradH_eig;
	float energyTH;
	float u,v;
	float u_stereo, v_stereo;
	int index;
	std::weak_ptr<Frame> hostFrame;
	int idxInImmaturePoints;

	float quality;

	float my_type;
	float idepth_min;
	float idepth_max;
	float idepth_min_stereo;
	float idepth_max_stereo;
	float idepth_stereo;
	ImmaturePoint(int u_, int v_, int index_, std::shared_ptr<Frame> host_, float type, std::shared_ptr<CalibData> Calib);
	~ImmaturePoint();

	ImmaturePointStatus traceOn(Vec3f* frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f &hostToFrame_affine,
								std::shared_ptr<CalibData> Calib, bool debugPrint = false);

	ImmaturePointStatus traceStereo(Vec3f* frame, std::shared_ptr<CalibData> Calib);

	ImmaturePointStatus lastTraceStatus;
	Vec2f lastTraceUV;
	float lastTracePixelInterval;

	float idepth_GT;

	double linearizeResidual(std::shared_ptr<CalibData> Calib, const float outlierTHSlack, ImmaturePointTemporaryResidual *tmpRes, float &Hdd, float &bd,
							 float idepth);

private:
	EIGEN_STRONG_INLINE float derive_idepth(const Vec3f &t, const float &u, const float &v, const int &dx, const int &dy, const float &dxInterp,
											const float &dyInterp, const float &drescale)
	{
		return (dxInterp * drescale * (t[0] - t[2] * u) + dyInterp * drescale * (t[1] - t[2] * v)) * SCALE_IDEPTH;
	}

	EIGEN_STRONG_INLINE bool projectPoint( const float &u_pt, const float &v_pt, const float &idepth, const int &dx, const int &dy,
											std::shared_ptr<CalibData> const &Calib, const Mat33f &R, const Vec3f &t, float &drescale, 
											float &u, float &v, float &Ku, float &Kv, Vec3f &KliP, float &new_idepth);
};

}

