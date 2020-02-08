#include "Settings.h"
#include <boost/thread.hpp>
#include "IndexThreadReduce.h"

namespace FSLAM
{
class ImageData;
class OnlineCalibrator;
class FeatureDetector;
class Frame;
class CalibData;
class GeometricUndistorter;
class PhotometricUndistorter;
class GUI;
class Map;
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

    System(std::shared_ptr<GeometricUndistorter> GeomUndist, std::shared_ptr<PhotometricUndistorter> 
            PhoUndistL, std::shared_ptr<PhotometricUndistorter> PhoUndistR, std::shared_ptr<GUI> _DisplayHandler);
    ~System();
    void ProcessNewFrame(std::shared_ptr<ImageData> DataIn);
	
    std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft;
    std::shared_ptr<IndexThreadReduce<Vec10>> BackEndThreadPool;


    //----------------begin dso------------------

    std::shared_ptr<EnergyFunctional> ef;
	// float* selectionMap;
	CoarseDistanceMap* coarseDistanceMap;

    std::vector<std::shared_ptr<Frame>> frameHessians;	// ONLY changed in marginalizeFrame and addFrame.
	std::vector<std::shared_ptr<PointFrameResidual>> activeResiduals;
	float currentMinActDist;
    
    boost::mutex trackMutex;
	std::vector<std::shared_ptr<Frame>> allFrameHistory;
	Vec5 lastCoarseRMSE;
	// ================== changed by mapper-thread. protected by mapMutex ===============
	boost::mutex mapMutex;
	std::vector<std::shared_ptr<Frame>> allKeyFramesHistory;


	IndexThreadReduce<Vec10> treadReduce;
	std::vector<float> allResVec;

	// mutex etc. for tracker exchange.
	boost::mutex coarseTrackerSwapMutex;			// if tracker sees that there is a new reference, tracker locks [coarseTrackerSwapMutex] and swaps the two.
	CoarseTracker* coarseTracker_forNewKF;			// set as as reference. protected by [coarseTrackerSwapMutex].
	CoarseTracker* coarseTracker;					// always used to track new frames. protected by [trackMutex].
    int lastRefStopID;

    // marginalizes a frame. drops / marginalizes points & residuals.
	void marginalizeFrame(std::shared_ptr<Frame> frame);
    float optimize(int mnumOptIts);
    // opt single point
	int optimizePoint(std::shared_ptr<MapPoint> point, int minObs, bool flagOOB);
	std::shared_ptr<MapPoint> optimizeImmaturePoint(std::shared_ptr<ImmaturePoint> point, int minObs, std::vector<std::shared_ptr<ImmaturePointTemporaryResidual>>& residuals);
    Vec4 trackNewCoarse(std::shared_ptr<Frame> fh);
	void traceNewCoarse(std::shared_ptr<Frame> fh);
	void activatePoints();
	void activatePointsMT();
    void flagPointsForRemoval();
	void makeNewTraces(std::shared_ptr<Frame> newFrame);
    void flagFramesForMarginalization(std::shared_ptr<Frame> newFH);
    void removeOutliers();

	void setPrecalcValues();
	void solveSystem(int iteration, double lambda);
	Vec3 linearizeAll(bool fixLinearization);
	bool doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD);
	void backupState(bool backupLastStep);
	void loadSateBackup();
	double calcLEnergy();
	double calcMEnergy();
	void linearizeAll_Reductor(bool fixLinearization, std::vector<std::shared_ptr<PointFrameResidual>>* toRemove, int min, int max, Vec10* stats, int tid);
	void activatePointsMT_Reductor(std::vector<std::shared_ptr<MapPoint>>* optimized, std::vector<std::shared_ptr<ImmaturePoint>>* toOptimize,int min, int max, Vec10* stats, int tid);
	void applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid);
   	void printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b);

    std::vector<VecX> getNullspaces(std::vector<VecX> &nullspaces_pose, std::vector<VecX> &nullspaces_scale, std::vector<VecX> &nullspaces_affA, std::vector<VecX> &nullspaces_affB);
	void setNewFrameEnergyTH();

    bool isLost;
    bool initFailed;
	bool initialized;
	bool linearizeOperation;
    //----------------end dso------------------
    


private:
    void DrawImages(std::shared_ptr<ImageData> DataIn,std::shared_ptr<Frame> CurrentFrame);
    void AddKeyframe(std::shared_ptr<Frame> fh);
    void ProcessNonKeyframe(std::shared_ptr<Frame> Frame);
    void BlockUntilMappingIsFinished();
    void MappingThread();
    void InitFromInitializer(std::shared_ptr<Initializer> _cInit);

    bool Initialized;
    std::shared_ptr<CalibData> Calib; //Calibration data that is used for projection and optimization
    std::shared_ptr<GUI> DisplayHandler;

    std::shared_ptr<Map> SlamMap;

    std::shared_ptr<Initializer> cInitializer;

    // std::shared_ptr<OnlineCalibrator> OnlinePhCalibL;
    // std::shared_ptr<OnlineCalibrator> OnlinePhCalibR;

    std::shared_ptr<FeatureDetector> Detector;

    //stored here without being used within the system.
    std::shared_ptr<PhotometricUndistorter> PhoUndistL; //The input photometric undistorter 
    std::shared_ptr<PhotometricUndistorter> PhoUndistR; //The input photometric undistorter 
    std::shared_ptr<GeometricUndistorter> GeomUndist;     //geometric calib data used for undistorting the images. Only used to initialize the calib ptr.

    
    boost::thread tMappingThread;
    boost::mutex MapThreadMutex;
    boost::condition_variable TrackedFrameSignal;
	boost::condition_variable MappedFrameSignal;
	std::deque<std::shared_ptr<Frame>> UnmappedTrackedFrames;
    bool RunMapping;
    bool NeedToCatchUp;
    int NeedNewKFAfter;
    boost::mutex shellPoseMutex;




    /* data */
};

} // namespace FSLAM