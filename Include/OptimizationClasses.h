#ifndef __OptimizationClasses__
#define __OptimizationClasses__

#include "GlobalTypes.h"

namespace FSLAM
{

class Frame;
class CalibData;
class MapPoint;
class EnergyFunctional;
// class EFResidual;

struct FrameFramePrecalc
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    // static values
    // static int instanceCounter;
    std::weak_ptr<Frame> host;   // defines row
    std::weak_ptr<Frame> target; // defines column

    // precalc values
    Mat33f PRE_RTll;
    Mat33f PRE_KRKiTll;
    Mat33f PRE_RKiTll;
    Mat33f PRE_RTll_0;

    Vec2f PRE_aff_mode;
    float PRE_b0_mode;

    Vec3f PRE_tTll;
    Vec3f PRE_KtTll;
    Vec3f PRE_tTll_0;

    float distanceLL;

    inline ~FrameFramePrecalc() {}
    inline FrameFramePrecalc() { }
    void set(std::shared_ptr<Frame> host, std::shared_ptr<Frame> target, std::shared_ptr<CalibData> HCalib);

};

struct RawResidualJacobian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// ================== new structure: save independently =============.
	VecNRf resF;

	// the two rows of d[x,y]/d[xi].
	Vec6f Jpdxi[2];			// 2x6

	// the two rows of d[x,y]/d[C].
	VecCf Jpdc[2];			// 2x4

	// the two rows of d[x,y]/d[idepth].
	Vec2f Jpdd;				// 2x1

	// the two columns of d[r]/d[x,y].
	VecNRf JIdx[2];			// 9x2

	// = the two columns of d[r] / d[ab]
	VecNRf JabF[2];			// 9x2


	// = JIdx^T * JIdx (inner product). Only as a shorthand.
	Mat22f JIdx2;				// 2x2
	// = Jab^T * JIdx (inner product). Only as a shorthand.
	Mat22f JabJIdx;			// 2x2
	// = Jab^T * Jab (inner product). Only as a shorthand.
	Mat22f Jab2;			// 2x2

};

class PointFrameResidual
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// EFResidual* efResidual;
	// static int instanceCounter;
	ResState state_state;
	double state_energy;
	ResState state_NewState;
	double state_NewEnergy;
	double state_NewEnergyWithOutlier;


	void setState(ResState s) {state_state = s;}

	std::weak_ptr<MapPoint> point;
	std::weak_ptr<Frame> host;
	std::weak_ptr<Frame> target;
	std::shared_ptr<RawResidualJacobian> J;

	bool isNew;


	Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
	Vec3f centerProjectedTo;

	~PointFrameResidual();
	PointFrameResidual()
	{
		J = std::shared_ptr<RawResidualJacobian>(new RawResidualJacobian);
	}
	PointFrameResidual(std::shared_ptr<MapPoint> point_, std::shared_ptr<Frame> host_, std::shared_ptr<Frame> target_): point(point_), host(host_), target(target_)
	{
		J = std::shared_ptr<RawResidualJacobian>(new RawResidualJacobian);
		resetOOB();
		isNew = true;

	}
	double linearize(std::shared_ptr<CalibData> HCalib);


	void resetOOB()
	{
		state_NewEnergy = state_energy = 0;
		state_NewState = ResState::OUTLIER;

		setState(ResState::IN);
	};
	void applyRes( bool copyJacobians);

	
    inline bool isActive() const { return isActiveAndIsGoodNEW; }
    void fixLinearizationF(std::shared_ptr<EnergyFunctional> ef); 	// fix the jacobians
	int hostIDX = 0;
	int targetIDX = 0;
	VecNRf res_toZeroF;
	Vec8f JpJdF = Vec8f::Zero();
	bool isLinearized = false; // if linearization is fixed.
	bool isActiveAndIsGoodNEW = false;

	void takeData()
	{
		Vec2f JI_JI_Jd = J->JIdx2 * J->Jpdd;
		for (int i = 0; i < 6; i++)
			JpJdF[i] = J->Jpdxi[0][i] * JI_JI_Jd[0] + J->Jpdxi[1][i] * JI_JI_Jd[1];
		JpJdF.segment<2>(6) = J->JabJIdx * J->Jpdd;
	}

};

} // namespace FSLAM

#endif