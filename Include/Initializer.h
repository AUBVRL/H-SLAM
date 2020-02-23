#ifndef __INITIALIZER__
#define __INITIALIZER__

#include "Settings.h"
#include "MatrixAccumulators.h"

namespace boost
{
    class mutex;
}

namespace FSLAM
{

class FrameShell;
class CalibData;
class GUI;
template<typename Type> class IndexThreadReduce;
struct Pnt;
class DirectRefinement;


struct CheckRTIn
{
public:
    float fx, fy, cx, cy;
    cv::Mat R, t, P1, P2, O1, O2;
    vector<bool> *vbMatchesInliers, *vbGood;
    vector<cv::Point2f> *vKeys1, *vKeys2;
    vector<cv::Point3f> *vP3D;
    vector<float> *vCosParallax;
    float th2;
    int nGood;
    CheckRTIn(){};
    ~CheckRTIn(){};
    shared_ptr<boost::mutex> thPoolLock;
};

class Initializer
{
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
    void Triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
    void Normalize(const std::vector<cv::Point2f> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
    int CheckRT(const cv::Mat &R, const cv::Mat &t, std::vector<cv::Point2f> &vKeys1, std::vector<cv::Point2f> &vKeys2,
                std::vector<bool> &vbInliers,
                const cv::Mat &K, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, float &parallax);
    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
    int FindMatches(int windowSize = 10, int maxL1Error = 7);
    bool FindTransformation(cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);
    float ComputeSceneMedianDepth(const int q, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);
    float ComputeSceneMeanDepth(std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);
    float ComputeMeanOpticalFlow(std::vector<cv::Point2f> &Prev, std::vector<cv::Point2f> &New);
    void ParallelCheckRT (std::shared_ptr<CheckRTIn> In, int min, int max);
    void Deliver();

    //Indirect Matching
    int MatchIndirect(std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize, int TH_LOW, float mfNNratio, bool CheckOrientation);
    static const int HISTO_LENGTH = 30;
    std::vector<cv::Point2f> mvbIndPrevMatched;
    std::vector<int> mvIniMatches;



    std::shared_ptr<CalibData> Calib;
    std::shared_ptr<GUI> displayhandler;

    float mSigma, mSigma2; // Standard Deviation and Variance
    int mMaxIterations;    // Ransac max iterations
    std::shared_ptr<cv::RNG> randomGen;

    std::vector<std::vector<size_t>> mvSets; // Ransac sets
    
    cv::Mat TransitImage;
    std::vector<cv::Point2f> FirstFramePts;
    std::vector<cv::Point2f> mvbPrevMatched; //p0 updated
    std::vector<cv::Point2f> MatchedPts;
    std::vector<cv::Point2f> MatchedPtsBkp; //used to detect stationary camera
    std::vector<bool> MatchedStatus;
    std::vector<cv::Scalar> ColorVec; 
    std::vector<cv::Point3f> mvIniP3D;
    
    std::shared_ptr<IndexThreadReduce<Vec10>> thPool;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    std::vector<float> videpth;
    std::shared_ptr<FrameShell> FirstFrame;
    std::shared_ptr<FrameShell> SecondFrame;
    SE3 Pose;

    bool Initialize(std::shared_ptr<FrameShell> _Frame);
    Initializer(std::shared_ptr<CalibData> _Calib,  std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft, std::shared_ptr<GUI>_DisplayHandler);
    ~Initializer(){};

};

class DirectRefinement
{
private:
    float alphaK;
	float alphaW;
	float regWeight;
	float couplingWeight;
    bool fixAffine;

    Eigen::DiagonalMatrix<float, 8> wM;
    Vec10f* JbBuffer;			// 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
	Vec10f* JbBuffer_new;
    Accumulator9 acc9;
	Accumulator9 acc9SC;

    std::shared_ptr<FrameShell> FirstFrame;
    std::shared_ptr<FrameShell> SecondFrame;
    std::shared_ptr<CalibData> Calib;
    std::shared_ptr<cv::RNG> randomGen;


	int numPoints;
    bool snapped;

    std::vector<cv::Point3f>* Pts3D;
    std::vector<bool>* Triangulated;


    Vec3f calcResAndGS(int lvl, Mat88f &H_out, Vec8f &b_out, Mat88f &H_out_sc, Vec8f &b_out_sc, const SE3 &refToNew, AffLight refToNew_aff, bool plot);
	Vec3f calcEC(int lvl); // returns OLD NERGY, NEW ENERGY, NUM TERMS.
    void optReg(int lvl);
    void resetPoints(int lvl);
	void doStep(int lvl, float lambda, Vec8f inc);
	void applyStep(int lvl);
	// void makeNN();
    void Refine();
    void debugPlot(Pnt* Points);
    void trace(Pnt* _pl);

public:
    Pnt* points;
    SE3 thisToNext;
    AffLight thisToNext_aff;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    DirectRefinement(std::shared_ptr<CalibData> _Calib, std::vector<cv::Point3f> &Pts3D, std::vector<bool> &Triangulated, std::shared_ptr<FrameShell> _FirstFrame,
                     std::shared_ptr<FrameShell> _SecondFrame, SE3 &_Pose, std::vector<float> &_videpth);
    ~DirectRefinement();
    
};

struct Pnt
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	float u,v;
	float idepth;
	bool isGood;
	Vec2f energy;
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
};

struct FLANNPointcloud
{
    inline FLANNPointcloud() {num=0; points=0;}
    inline FLANNPointcloud(int n, Pnt* p) :  num(n), points(p) {}
	int num;
	Pnt* points;
	inline size_t kdtree_get_point_count() const { return num; }
	inline float kdtree_distance(const float *p1, const size_t idx_p2,size_t /*size*/) const
	{
		const float d0=p1[0]-points[idx_p2].u;
		const float d1=p1[1]-points[idx_p2].v;
		return d0*d0+d1*d1;
	}

	inline float kdtree_get_pt(const size_t idx, int dim) const
	{
		if (dim==0) return points[idx].u;
		else return points[idx].v;
	}
	template <class BBOX>
		bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

} // namespace FSLAM

#endif