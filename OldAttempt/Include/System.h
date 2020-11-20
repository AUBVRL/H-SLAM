#ifndef __SYSTEM_H_
#define __SYSTEM_H_
#pragma once

#include "Settings.h"

namespace SLAM
{
    class geomUndistorter;
    class photoUndistorter;
    class FeatureDetector;
    class CalibData;
    class FrameShell;
    class Initializer;
    class EnergyFunctional;
    class PointFrameResidual;
    class MapPoint;
    class ImmaturePoint;
    class ImmaturePointTemporaryResidual;
    class CoarseTracker;
    class CoarseDistanceMap;
    class GUI;

    template<typename Type> class IndexThreadReduce;


    class System
    {
    private:
        shared_ptr<geomUndistorter> gUndist;
        shared_ptr<photoUndistorter> pUndist;
        shared_ptr<FeatureDetector> detector;
        shared_ptr<CalibData> geomCalib;
        shared_ptr<IndexThreadReduce<Vec10>> frontEndThreadPool;
        shared_ptr<IndexThreadReduce<Vec10>> backEndThreadPool;

        shared_ptr<EnergyFunctional> ef;

        
        //Initialization data
        shared_ptr<Initializer> initializer;


        //Local Map data
        vector<shared_ptr<FrameShell>> frameHessians;	// ONLY changed in marginalizeFrame and addFrame.
	    
        
        //-- only modified in mapper --
        vector<shared_ptr<FrameShell>> allKeyFramesHistory; //entire set of KeyFrames 
	    vector<shared_ptr<FrameShell>> allFrameHistory;

        bool isLost; // if lost skip frame from both tracking and mapping!! --potentially dangerous though

        int NeedNewKFAfter; // id after which to add a new keyframe
        deque<shared_ptr<FrameShell>> UnmappedTrackedFrames;

        float currentMinActDist; //distance from map to add new data


        mutex shellPoseMutex;
        mutex mapMutex;
        mutex trackMapSyncMutex;

        condition_variable TrackedFrameSignal;
        condition_variable MappedFrameSignal;

        thread mappingThread;
        void MappingThread();
        
         void InitFromInitializer(shared_ptr<Initializer> _cInit);    

        //mapping 
        void toMapping(shared_ptr<FrameShell> currentFrame, bool needKeyframe);
        void AddKeyframe(shared_ptr<FrameShell> currentFrame);
        void BlockUntilMappingIsFinished();

        //Optimization structs
        void setPrecalcValues();
        float optimize(int mnumOptIts);
        vector<shared_ptr<PointFrameResidual>> activeResiduals;

        //optimization infrastructure
        Vec3 linearizeAll(bool fixLinearization);
	    void linearizeAll_Reductor(bool fixLinearization,  vector<shared_ptr<PointFrameResidual>>* toRemove, int min, 
                                   int max, Vec10* stats, int tid);
        void setNewFrameEnergyTH();
        vector<float> allResVec;
        void applyRes_Reductor(bool copyJacobians, int min, int max, Vec10 *stats, int tid);
	    void backupState(bool backupLastStep);
        vector<VecX> getNullspaces(vector<VecX> &nullspaces_pose, vector<VecX> &nullspaces_scale,
                                   vector<VecX> &nullspaces_affA, vector<VecX> &nullspaces_affB);
        bool doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD);
        void loadSateBackup();

        //marginalization
        void flagFramesForMarginalization();
        void marginalizeFrame(shared_ptr<FrameShell> frame);
        void flagPointsForRemoval();

        //convert Immature points to active points
        void makeNewTraces(shared_ptr<FrameShell> newFrame);
        void activatePoints();
        void activateImmaturePts(vector<shared_ptr<MapPoint>> *optimized, vector<shared_ptr<ImmaturePoint>> *toOptimize, 
                                     int min, int max, Vec10 *stats, int tid);
        shared_ptr<MapPoint> optimizeImmaturePoint(shared_ptr<ImmaturePoint> &point, int minObs,
                                                       vector<shared_ptr<ImmaturePointTemporaryResidual>> &residuals);
        void removeOutliers();


        //Tracking data
        void ProcessNonKeyframe(shared_ptr<FrameShell> currentFrame);
        void traceNewCoarse(shared_ptr<FrameShell> currentFrame);
        Vec4 trackNewCoarse(shared_ptr<FrameShell> &fh);

        bool RunMapping;

        //Tracker data
        shared_ptr<CoarseDistanceMap> coarseDistanceMap;
        mutex coarseTrackerSwapMutex;                     // if tracker sees that there is a new reference, tracker locks [coarseTrackerSwapMutex] and swaps the two.
        shared_ptr<CoarseTracker> coarseTracker_forNewKF; // set as as reference. protected by [coarseTrackerSwapMutex].
        shared_ptr<CoarseTracker> coarseTracker;          // always used to track new frames. protected by [trackMutex].
        int lastRefStopID;
        Vec5 lastCoarseRMSE;

        shared_ptr<GUI> displayHandler; 

        void getAllFrameMPs (shared_ptr<FrameShell> _In, vector<shared_ptr<MapPoint>>& _Out);


    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        System(shared_ptr<geomUndistorter> GeomUndist, shared_ptr<photoUndistorter> PhoUndist,
               shared_ptr<GUI> _DisplayHandler);
        ~System();

        void ProcessFrame(shared_ptr<ImageData> dataIn);

        bool initFailed;
        bool isInitialized;



        //Indirect methods
        bool TrackIndirect(shared_ptr<FrameShell> currentFrame, SE3 MotionModel);
        enum indirectState {OK=0, lost, notInit} indState ;
    };

} // namespace SLAM

#endif