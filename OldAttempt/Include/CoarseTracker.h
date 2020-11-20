#ifndef __COARSETRACKER_H__
#define __COARSETRACKER_H__

 
#include "Settings.h"
#include "MatrixAccumulators.h"

namespace SLAM
{
struct CalibData;
struct FrameShell;
struct PointFrameResidual;

class CoarseTracker {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CoarseTracker(shared_ptr<CalibData> _geomCalib);
	~CoarseTracker();

	bool trackNewestCoarse(
			shared_ptr<FrameShell>& newFrameHessian,
			SE3 &lastToNew_out, AffLight &aff_g2l_out,
			int coarsestLvl, Vec5 minResForAbort);

	void setCoarseTrackingRef( vector<shared_ptr<FrameShell>>& frameHessians);

	// void makeK(shared_ptr<CalibData> HCalib);

	bool debugPrint, debugPlot;
	static const int PYR_LEVELS = 10; //compiler hack since we create at most 10 pyramid levels from the input class
	shared_ptr<CalibData> geomCalib;
	// Mat33f K[PYR_LEVELS];
	// Mat33f Ki[PYR_LEVELS];
	// float fx[PYR_LEVELS];
	// float fy[PYR_LEVELS];
	// float fxi[PYR_LEVELS];
	// float fyi[PYR_LEVELS];
	// float cx[PYR_LEVELS];
	// float cy[PYR_LEVELS];
	// float cxi[PYR_LEVELS];
	// float cyi[PYR_LEVELS];
	// int w[PYR_LEVELS];
	// int h[PYR_LEVELS];

    // void debugPlotIDepthMap(float* minID, float* maxID, std::vector<IOWrap::Output3DWrapper*> &wraps);
    // void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps);
	std::vector<pair<float, int>> GetKFDepthMap();

	shared_ptr<FrameShell> lastRef;
	AffLight lastRef_aff_g2l;
	shared_ptr<FrameShell> newFrame;
	int refFrameID;

	// act as pure ouptut
	Vec5 lastResiduals;
	Vec3 lastFlowIndicators;
	double firstCoarseRMSE;
private:


	void makeCoarseDepthL0(vector<shared_ptr<FrameShell>>& frameHessians);
	float* idepth[PYR_LEVELS];
	float* weightSums[PYR_LEVELS];
	float* weightSums_bak[PYR_LEVELS];


	Vec6 calcResAndGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
	Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
	void calcGSSSE(float fx, float fy, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);
	void calcGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);

	// pc buffers
	float* pc_u[PYR_LEVELS];
	float* pc_v[PYR_LEVELS];
	float* pc_idepth[PYR_LEVELS];
	float* pc_color[PYR_LEVELS];
	int pc_n[PYR_LEVELS];

	// warped buffers
	float* buf_warped_idepth;
	float* buf_warped_u;
	float* buf_warped_v;
	float* buf_warped_dx;
	float* buf_warped_dy;
	float* buf_warped_residual;
	float* buf_warped_weight;
	float* buf_warped_refColor;
	int buf_warped_n;


    vector<float*> ptrToDelete;


	Accumulator9 acc;
};


class CoarseDistanceMap {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CoarseDistanceMap(shared_ptr<CalibData> _geomCalib);
	~CoarseDistanceMap();

	void makeDistanceMap( vector<shared_ptr<FrameShell>>& frameHessians, shared_ptr<FrameShell>& frame);


	float* fwdWarpedIDDistFinal;
	static const int PYR_LEVELS = 10; //compiler hack since we create at most 10 pyramid levels from the input class
	shared_ptr<CalibData> geomCalib;

	void addIntoDistFinal(int u, int v);


private:

	PointFrameResidual** coarseProjectionGrid;
	int* coarseProjectionGridNum;
	Eigen::Vector2i* bfsList1;
	Eigen::Vector2i* bfsList2;

	void growDistBFS(int bfsNum);
};

}

#endif