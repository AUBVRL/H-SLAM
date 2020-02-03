#ifndef __ENERGYFUNCTIONALSTRUCTS_H__
#define __ENERGYFUNCTIONALSTRUCTS_H__
 
#include "GlobalTypes.h"
// #include "vector"
// #include <math.h>
#include "OptimizationClasses.h"

namespace FSLAM
{

class PointFrameResidual;
class CalibData;
class Frame;
class MapPoint;

class EFResidual;
class EFPoint;
class EFFrame;
class EnergyFunctional;






class EFResidual
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	inline EFResidual(PointFrameResidual* org, EFPoint* point_, EFFrame* host_, EFFrame* target_) :
		data(org), point(point_), host(host_), target(target_)
	{
		isLinearized=false;
		isActiveAndIsGoodNEW=false;
		J = new RawResidualJacobian();
		assert(((long)this)%16==0);
		assert(((long)J)%16==0);
	}
	inline ~EFResidual()
	{
		delete J;
	}


	void takeDataF();


	void fixLinearizationF(EnergyFunctional* ef);


	// structural pointers
	PointFrameResidual* data;
	int hostIDX, targetIDX;
	EFPoint* point;
	EFFrame* host;
	EFFrame* target;
	int idxInAll;

	RawResidualJacobian* J;

	VecNRf res_toZeroF;
	Vec8f JpJdF;


	// status.
	bool isLinearized;

	// if residual is not OOB & not OUTLIER & should be used during accumulations
	bool isActiveAndIsGoodNEW;
	inline const bool &isActive() const {return isActiveAndIsGoodNEW;}
};


enum EFPointStatus {PS_GOOD=0, PS_MARGINALIZE, PS_DROP};

class EFPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EFPoint(MapPoint* d, EFFrame* host_) : data(d),host(host_)
	{
		takeData();
		stateFlag=EFPointStatus::PS_GOOD;
	}
	void takeData();

	MapPoint* data;



	float priorF;
	float deltaF;


	// constant info (never changes in-between).
	int idxInPoints;
	EFFrame* host;

	// contains all residuals.
	std::vector<EFResidual*> residualsAll;

	float bdSumF;
	float HdiF;
	float Hdd_accLF;
	VecCf Hcd_accLF;
	float bd_accLF;
	float Hdd_accAF;
	VecCf Hcd_accAF;
	float bd_accAF;


	EFPointStatus stateFlag;
};



class EFFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EFFrame(Frame* d) : data(d)
	{
		takeData();
	}
	void takeData();


	Vec8 prior;				// prior hessian (diagonal)
	Vec8 delta_prior;		// = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
	Vec8 delta;				// state - state_zero.



	std::vector<EFPoint*> points;
	Frame* data;
	int idx;	// idx in frames.

	int frameID;
};

}

#endif

