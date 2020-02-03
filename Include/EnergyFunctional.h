#ifndef __ENERGYFUNCTIONAL_H__
#define __ENERGYFUNCTIONAL_H__

 
#include "GlobalTypes.h"
// #include "vector"
// #include <math.h>
// #include "map"


namespace FSLAM
{

template <typename Type> class IndexThreadReduce;

class PointFrameResidual;
class CalibData;
class Frame;
class MapPoint;


class EFResidual;
class EFPoint;
class EFFrame;
class EnergyFunctional;
class AccumulatedTopHessian;
class AccumulatedTopHessianSSE;
class AccumulatedSCHessian;
class AccumulatedSCHessianSSE;


extern bool EFAdjointsValid;
extern bool EFIndicesValid;
extern bool EFDeltaValid;



class EnergyFunctional {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	friend class EFFrame;
	friend class EFPoint;
	friend class EFResidual;
	friend class AccumulatedTopHessian;
	friend class AccumulatedTopHessianSSE;
	friend class AccumulatedSCHessian;
	friend class AccumulatedSCHessianSSE;

	EnergyFunctional();
	~EnergyFunctional();


	EFResidual* insertResidual(PointFrameResidual* r);
	EFFrame* insertFrame(Frame* fh, std::shared_ptr<CalibData> Hcalib);
	EFPoint* insertPoint(MapPoint* ph);

	void dropResidual(EFResidual* r);
	void marginalizeFrame(EFFrame* fh);
	void removePoint(EFPoint* ph);



	void marginalizePointsF();
	void dropPointsF();
	void solveSystemF(int iteration, double lambda, std::shared_ptr<CalibData> HCalib);
	double calcMEnergyF();
	double calcLEnergyF_MT();


	void makeIDX();

	void setDeltaF(std::shared_ptr<CalibData> HCalib);

	void setAdjointsF(std::shared_ptr<CalibData> Hcalib);

	std::vector<EFFrame*> frames;
	int nPoints, nFrames, nResiduals;

	MatXX HM;
	VecX bM;

	int resInA, resInL, resInM;
	MatXX lastHS;
	VecX lastbS;
	VecX lastX;
	std::vector<VecX> lastNullspaces_forLogging;
	std::vector<VecX> lastNullspaces_pose;
	std::vector<VecX> lastNullspaces_scale;
	std::vector<VecX> lastNullspaces_affA;
	std::vector<VecX> lastNullspaces_affB;

	IndexThreadReduce<Vec10>* red;


	std::map<uint64_t,
	  Eigen::Vector2i,
	  std::less<uint64_t>,
	  Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>
	  > connectivityMap;

private:

	VecX getStitchedDeltaF() const;

	void resubstituteF_MT(VecX x, std::shared_ptr<CalibData> HCalib, bool MT);
    void resubstituteFPt(const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid);

	void accumulateAF_MT(MatXX &H, VecX &b, bool MT);
	void accumulateLF_MT(MatXX &H, VecX &b, bool MT);
	void accumulateSCF_MT(MatXX &H, VecX &b, bool MT);

	void calcLEnergyPt(int min, int max, Vec10* stats, int tid);

	void orthogonalize(VecX* b, MatXX* H);
	Mat18f* adHTdeltaF;

	Mat88* adHost;
	Mat88* adTarget;

	Mat88f* adHostF;
	Mat88f* adTargetF;


	VecC cPrior;
	VecCf cDeltaF;
	VecCf cPriorF;

	AccumulatedTopHessianSSE* accSSE_top_L;
	AccumulatedTopHessianSSE* accSSE_top_A;


	AccumulatedSCHessianSSE* accSSE_bot;

	std::vector<EFPoint*> allPoints;
	std::vector<EFPoint*> allPointsToMarg;

	float currentLambda;
};
}

#endif

