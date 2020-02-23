#ifndef __ENERGYFUNCTIONAL_H__
#define __ENERGYFUNCTIONAL_H__

 
#include "GlobalTypes.h"

namespace FSLAM
{

template <typename Type> class IndexThreadReduce;

class PointFrameResidual;
class CalibData;
class FrameShell;
class MapPoint;

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

	friend class AccumulatedTopHessian;
	friend class AccumulatedSCHessian;

	EnergyFunctional();
	~EnergyFunctional();


	void insertResidual(shared_ptr<MapPoint>& ph, shared_ptr<PointFrameResidual>& r);
	void insertFrame(shared_ptr<FrameShell>& frame, shared_ptr<CalibData>& Hcalib);
	void insertPoint(shared_ptr<MapPoint>& ph);

	void dropResidual(shared_ptr<MapPoint>& ph, shared_ptr<PointFrameResidual> r);
	void marginalizeFrame(shared_ptr<FrameShell>& fh);
	void removePoint(shared_ptr<MapPoint>& ph);



	void marginalizePointsF();
	void dropPointsF();
	void solveSystemF(int iteration, double lambda, shared_ptr<CalibData>& HCalib);
	double calcMEnergyF();
	double calcLEnergyF_MT();


	void makeIDX();

	void setDeltaF(shared_ptr<CalibData>& HCalib);

	void setAdjointsF(shared_ptr<CalibData>& Hcalib);

	vector<shared_ptr<FrameShell>> frames;
	int nPoints, nFrames, nResiduals;

	MatXX HM;
	VecX bM;

	int resInA, resInL, resInM;
	MatXX lastHS;
	VecX lastbS;
	VecX lastX;
	vector<VecX> lastNullspaces_forLogging;
	vector<VecX> lastNullspaces_pose;
	vector<VecX> lastNullspaces_scale;
	vector<VecX> lastNullspaces_affA;
	vector<VecX> lastNullspaces_affB;

	IndexThreadReduce<Vec10>* red;


	std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> connectivityMap;


	VecX getStitchedDeltaF() const;

	void resubstituteF_MT(VecX x, shared_ptr<CalibData>& HCalib, bool MT);
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

	vector<shared_ptr<MapPoint>> allPoints;
	vector<shared_ptr<MapPoint>> allPointsToMarg;

	float currentLambda;
};
}

#endif

