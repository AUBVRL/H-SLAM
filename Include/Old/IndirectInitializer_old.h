#ifndef __IndirectInitializer_H__
#define __IndirectInitializer_H__

#include "Settings.h"
#include "MatrixAccumulators.h"
#include <opencv2/core/types.hpp>

namespace FSLAM
{
class Frame;
class CalibData;
class ORBDetector;
class GUI;

struct InitializerTempResidual
{
public:
	ResState state_state;
	double state_energy;
	ResState state_NewState;
	double state_NewEnergy;
    std::weak_ptr<Frame> target;

};

struct Pnt
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// index in jacobian. never changes (actually, there is no reason why).
	float u,v;

	// idepth / isgood / energy during optimization.
	float idepth;
	bool isGood;
	Vec2f energy;		// (UenergyPhotometric, energyRegularizer)
	bool isGood_new;
	float idepth_new;
	Vec2f energy_new;

	float iR;
	float iRSumNum;

	float lastHessian;
	float lastHessian_new;

	// max stepsize for idepth (corresponding to max. movement in pixel-space).
	float maxstep;

	// idx (x+y*w) of closest point one pyramid level above.
	int parent;
	float parentDist;

	// idx (x+y*w) of up to 10 nearest points in pixel space.
	int neighbours[10];
	float neighboursDist[10];

	float my_type;
	float outlierTH;


    float color[MAX_RES_PER_POINT];
	float weights[MAX_RES_PER_POINT];
    InitializerTempResidual Residual;
};



class IndirectInitializer
{
    typedef std::pair<int, int> Match;

private:
    void FindHomography(std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
    void FindFundamental(std::vector<bool> &vbInliers, float &score, cv::Mat &F21);
    cv::Mat ComputeH21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);
    cv::Mat ComputeF21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);
    float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, std::vector<bool> &vbMatchesInliers, float sigma);
    float CheckFundamental(const cv::Mat &F21, std::vector<bool> &vbMatchesInliers, float sigma);
    bool ReconstructF(std::vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
    bool ReconstructH(std::vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
    void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
    void Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
    int CheckRT(const cv::Mat &R, const cv::Mat &t, const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2,
                const std::vector<Match> &vMatches12, std::vector<bool> &vbInliers,
                const cv::Mat &K, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, float &parallax);
    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
    int FindMatches(std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize=10, int TH_LOW = 50, float mfNNratio = 0.9, bool CheckOrientation = true);
    bool FindTransformation(const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D,
                            std::vector<bool> &vbTriangulated);
    float ComputeSceneMedianDepth(const int q, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);
    bool OptimizeDirect(std::vector<cv::Point3f> &mvIniP3D, std::vector<bool> &vbTriangulated, SE3 &thisToNext);
    void resetPoints(std::vector<std::shared_ptr<Pnt>>& Points); //int lvl
    void doStep(std::vector<std::shared_ptr<Pnt>>& Points, float lambda, Vec8f inc); //int lvl 
    void applyStep(std::vector<std::shared_ptr<Pnt>>& Points); //int lvl
    Vec3f calcResAndGS(std::vector<std::shared_ptr<Pnt>>& Points, Mat88f &H_out, Vec8f &b_out, Mat88f &H_out_sc, Vec8f &b_out_sc, const SE3 &refToNew, AffLight refToNew_aff, bool plot); //int lvl 
    Vec3f calcEC(std::vector<std::shared_ptr<Pnt>>& Points); //int lvl 
    void optReg(std::vector<std::shared_ptr<Pnt>> & Points); //int lvl 
    void debugPlot(std::vector<std::shared_ptr<Pnt>>&Points);
    void StructureOnlyDirectOptimization(std::vector<std::shared_ptr<Pnt>>& Points, std::vector<cv::Point3f> &mvIniP3D, std::vector<bool> &vbTriangulated, SE3 &thisToNext);
    double linearizeResidual(std::shared_ptr<Pnt> Points, const float outlierTHSlack, float &Hdd, float &bd, float idepth);
    EIGEN_STRONG_INLINE bool projectPoint(const float &u_pt, const float &v_pt, const float &idepth, const int &dx, const int &dy,
													 std::shared_ptr<CalibData> const &Calib, const Mat33f &R, const Vec3f &t, float &drescale,
													 float &u, float &v, float &Ku, float &Kv, Vec3f &KliP, float &new_idepth);
    std::shared_ptr<ORBDetector> Detector;
    std::shared_ptr<GUI> displayhandler;
    //Camera calibration information
    std::shared_ptr<CalibData> Calib;

    // Current Matches from Reference to Current
    std::vector<Match> mvMatches12;
    std::vector<bool> mvbMatched1;

    std::vector<int> mvIniMatches;
    std::vector<cv::Point3f> mvIniP3D;

    // Standard Deviation and Variance
    float mSigma, mSigma2;
    // Ransac max iterations
    int mMaxIterations;
    // Ransac sets
    std::vector<std::vector<size_t> > mvSets;   
    std::vector<cv::Point2f> mvbPrevMatched;

    Eigen::DiagonalMatrix<float, 8> wM;
    // temporary buffers for H and b.
	Vec10f* JbBuffer;			// 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
	Vec10f* JbBuffer_new;
    bool fixAffine;
    AffLight thisToNext_aff;
	Accumulator9 acc9;
	Accumulator9 acc9SC;

    std::vector<int> maxIterations; // increase the size of this according to number of pyramids used
    int GNDirStrucOnlytMaxIter;
	float alphaK ;
	float alphaW;
	float regWeight;
	float couplingWeight;

	bool snapped;
	int snappedAt;
    int frameID;

    std::shared_ptr<cv::RNG> randomGen;


    static const int HISTO_LENGTH = 30;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    IndirectInitializer(std::shared_ptr<CalibData> _Calib, std::shared_ptr<ORBDetector> _Detector, std::shared_ptr<GUI>_DisplayHandler);
    ~IndirectInitializer();
    bool Initialize(std::shared_ptr<Frame> _Frame);
    
    //structures containing extracted features
    std::shared_ptr<Frame> FirstFrame;
    std::shared_ptr<Frame> SecondFrame;
};



} // namespace FSLAM

#endif