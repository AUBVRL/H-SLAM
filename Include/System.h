#include "Settings.h"
#include <boost/thread.hpp>
#include "IndexThreadReduce.h"

namespace FSLAM
{

class ImageData;
class OnlineCalibrator;
class FeatureDetector;
class FrameShell;
class CalibData;
class GeometricUndistorter;
class PhotometricUndistorter;
class GUI;
class Initializer;

//----------------begin dso------------------
class CoarseTracker;
struct MapPoint;
class ImmaturePoint;
struct ImmaturePointTemporaryResidual;
class CoarseDistanceMap;
class EnergyFunctional;
class PointFrameResidual;

//----------------end dso------------------


class System
{

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    System(std::shared_ptr<GeometricUndistorter> GeomUndist, std::shared_ptr<PhotometricUndistorter>  PhoUndistL, std::shared_ptr<PhotometricUndistorter> PhoUndistR,
            std::shared_ptr<GUI> _DisplayHandler);
    ~System();
    void ProcessNewFrame(std::shared_ptr<ImageData> DataIn);
	
    shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft;
    shared_ptr<IndexThreadReduce<Vec10>> BackEndThreadPool;


    //----------------begin dso------------------

    shared_ptr<EnergyFunctional> ef;
	// float* selectionMap;
	CoarseDistanceMap* coarseDistanceMap;

    vector<shared_ptr<FrameShell>> frameHessians;	// ONLY changed in marginalizeFrame and addFrame.
	vector<pair<shared_ptr<MapPoint>, shared_ptr<PointFrameResidual>>> activeResiduals;
	float currentMinActDist;
    
    boost::mutex trackMutex;
	vector<shared_ptr<FrameShell>> allFrameHistory;
	Vec5 lastCoarseRMSE;
	// ================== changed by mapper-thread. protected by mapMutex ===============
	boost::mutex mapMutex;
	vector<shared_ptr<FrameShell>> allKeyFramesHistory;


	IndexThreadReduce<Vec10> treadReduce;
	vector<float> allResVec;

	// mutex etc. for tracker exchange.
	boost::mutex coarseTrackerSwapMutex;			// if tracker sees that there is a new reference, tracker locks [coarseTrackerSwapMutex] and swaps the two.
	CoarseTracker* coarseTracker_forNewKF;			// set as as reference. protected by [coarseTrackerSwapMutex].
	CoarseTracker* coarseTracker;					// always used to track new frames. protected by [trackMutex].
    int lastRefStopID;

    // marginalizes a frame. drops / marginalizes points & residuals.
	void marginalizeFrame(shared_ptr<FrameShell> frame);
    float optimize(int mnumOptIts);
    // opt single point
	int optimizePoint(shared_ptr<MapPoint> &point, int minObs, bool flagOOB);
	shared_ptr<MapPoint> optimizeImmaturePoint(shared_ptr<ImmaturePoint>& point, int minObs, vector<shared_ptr<ImmaturePointTemporaryResidual>>& residuals);
    Vec4 trackNewCoarse(shared_ptr<FrameShell> &fh);
	void traceNewCoarse(shared_ptr<FrameShell> fh);
	void activatePoints();
	void activatePointsMT();
    void flagPointsForRemoval();
	void makeNewTraces(shared_ptr<FrameShell> newFrame);
    void flagFramesForMarginalization();
    void removeOutliers();

	void setPrecalcValues();
	void solveSystem(int iteration, double lambda);
	Vec3 linearizeAll(bool fixLinearization);
	bool doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD);
	void backupState(bool backupLastStep);
	void loadSateBackup();
	double calcLEnergy();
	double calcMEnergy();
	void linearizeAll_Reductor(bool fixLinearization, vector< pair< shared_ptr<MapPoint>, shared_ptr<PointFrameResidual>>>* toRemove, int min, int max, Vec10* stats, int tid);
	void activatePointsMT_Reductor(vector<shared_ptr<MapPoint>>* optimized, vector<shared_ptr<ImmaturePoint>>* toOptimize,int min, int max, Vec10* stats, int tid);
	void applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid);
   	void printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b);

    vector<VecX> getNullspaces(vector<VecX> &nullspaces_pose, vector<VecX> &nullspaces_scale, vector<VecX> &nullspaces_affA, vector<VecX> &nullspaces_affB);
	void setNewFrameEnergyTH();

    bool isLost;
    bool initFailed;
	bool initialized;
	bool linearizeOperation;
    //----------------end dso------------------
    


private:
    void DrawImages(shared_ptr<ImageData> DataIn, std::shared_ptr<FrameShell> CurrentFrame);
    void AddKeyframe(shared_ptr<FrameShell> fh);
    void ProcessNonKeyframe(shared_ptr<FrameShell> Frame);
    void BlockUntilMappingIsFinished();
    void MappingThread();
    void InitFromInitializer(shared_ptr<Initializer> _cInit);

    bool Initialized;
    shared_ptr<CalibData> Calib; //Calibration data that is used for projection and optimization
    shared_ptr<GUI> DisplayHandler;

    shared_ptr<Initializer> cInitializer;

    // std::shared_ptr<OnlineCalibrator> OnlinePhCalibL;
    // std::shared_ptr<OnlineCalibrator> OnlinePhCalibR;

    shared_ptr<FeatureDetector> Detector;

    //stored here without being used within the system.
    shared_ptr<PhotometricUndistorter> PhoUndistL; //The input photometric undistorter 
    shared_ptr<PhotometricUndistorter> PhoUndistR; //The input photometric undistorter 
    shared_ptr<GeometricUndistorter> GeomUndist;     //geometric calib data used for undistorting the images. Only used to initialize the calib ptr.

    
    boost::thread tMappingThread;
    boost::mutex MapThreadMutex;
    boost::condition_variable TrackedFrameSignal;
	boost::condition_variable MappedFrameSignal;
	deque<shared_ptr<FrameShell>> UnmappedTrackedFrames;
    bool RunMapping;
    bool NeedToCatchUp;
    int NeedNewKFAfter;
    boost::mutex shellPoseMutex;




    /* data */
};

} // namespace FSLAM