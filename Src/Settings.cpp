#include "Settings.h"
// #include <omp.h>

namespace FSLAM
{
    // int NumProcessors = std::max(omp_get_num_threads()/2,6);
    Sensor Sensortype = Emptys;
    PhotoUnDistMode PhoUndistMode = Emptyp;

    int WidthOri; 
    int HeightOri;

    //feature Detector params (settings here are overriden by explicitly changing the input to software)
    int IndPyrLevels = 1;
    float IndPyrScaleFactor = 1.2;
    int IndNumFeatures = 3000;
    int minThFAST = 8;
    float tolerance = 0.1; //SSC tolerance 
    int EnforcedMinDist = 5;
    bool UseFAST = true;
    bool DoSubPix = false;
    bool DrawDetected = true;
    bool DrawDepthKfTest = false;
    bool DrawEpipolarMatching = false;
    bool Pause = false;

    //Direct data dector
    int DirPyrLevels = 6;
    float setting_minGradHistCut = 0.5;
    float setting_minGradHistAdd = 7;
    float setting_gradDownweightPerLevel = 0.75;
    bool setting_selectDirectionDistribution = true;

    float setting_keyframesPerSecond = 0; // if !=0, takes a fixed number of KF per second.
    bool setting_realTimeMaxKF = false;   // if true, takes as many KF's as possible (will break the system if the camera stays stationary)
    float setting_maxShiftWeightT = 0.04f * (640 + 480);
    float setting_maxShiftWeightR = 0.0f * (640 + 480);
    float setting_maxShiftWeightRT = 0.02f * (640 + 480);
    float setting_kfGlobalWeight = 1; // general weight on threshold, the larger the more KF's are taken (e.g., 2 = double the amount of KF's).
    float setting_maxAffineWeight = 2;

    bool SequentialOperation = true;

    //Display options
    bool DisplayOn = true;
    bool ShowInitializationMatches = false;
    bool ShowInitializationMatchesSideBySide = true;
    bool show_gradient_image = false;
    bool settings_show_InitDepth = true;


    float setting_maxLogAffFacInWindow = 0.7; // marg a frame if factor between intensities to current frame is larger than 1/X or X.
    float setting_minPointsRemaining = 0.05;
    int setting_GNItsOnPointActivation = 3;


    int setting_minFrames = 5; // min frames in window.
    int setting_maxFrames = 7; // max frames in window.
    int setting_minFrameAge = 1;
    int setting_maxOptIterations = 6;    // max GN iterations.
    int setting_minOptIterations = 1;    // min GN iterations.
    float setting_thOptIterations = 1.2; // factor on break threshold for GN iteration (larger = break earlier)

    float setting_outlierTHSumComponent = 50 * 50; // higher -> less strong gradient-based reweighting .
    float setting_outlierTH = 12*12; // higher -> less strict
    float setting_overallEnergyTHWeight = 1;

    float setting_huberTH = 9; //Huber Threshold!
    float benchmark_initializerSlackFactor = 1;

    
    // parameters controlling adaptive energy threshold computation.
    float setting_frameEnergyTHN = 0.7f;
    float setting_frameEnergyTHFacMedian = 1.5;
    float setting_frameEnergyTHConstWeight = 0.5;

    float setting_coarseCutoffTH = 20;


    float setting_margWeightFac = 0.5*0.5;          // factor on hessian when marginalizing, to account for inaccurate linearization points.

    float setting_reTrackThreshold = 1.5; // (larger = re-track more often)

    //Immatureure point tracking
    float setting_maxPixSearch = 0.027f; 
    float setting_minTraceQuality = 3;
    float setting_trace_slackInterval = 1.5f;
    float setting_trace_stepsize = 1.0f;
    float setting_trace_minImprovementFactor = 2;
    int setting_minTraceTestRadius = 2;
    int setting_trace_GNIterations = 3;
    float setting_trace_GNThreshold = 0.1f;
    float setting_trace_extraSlackOnTH = 1.2;

    //MapPoint settings
    int setting_minGoodActiveResForMarg = 3;
    int setting_minGoodResForMarg = 4;

    //Fixing priors on unobservable/initial dimensions
    float setting_idepthFixPrior = 50*50;
    float setting_idepthFixPriorMargFac = 600*600;
    float setting_initialRotPrior = 1e11;
    float setting_initialTransPrior = 1e10;
    float setting_initialAffBPrior = 1e14;
    float setting_initialAffAPrior = 1e14;
    float setting_initialCalibHessian = 5e9;


    float setting_affineOptModeA = 1e12; //-1: fix. >=0: optimize (with prior, if > 0).
    float setting_affineOptModeB = 1e8; //-1: fix. >=0: optimize (with prior, if > 0).
    
    
    //Solver Settings
    int setting_solverMode = SOLVER_FIX_LAMBDA | SOLVER_ORTHOGONALIZE_X_LATER;
    double setting_solverModeDelta = 0.00001;
    bool setting_forceAceptStep = true;

    float setting_minIdepthH_act = 100;
    float setting_minIdepthH_marg = 50;

    float setting_desiredPointDensity = 2000; // aimed total points in the active window.



    bool multiThreading = true;

}