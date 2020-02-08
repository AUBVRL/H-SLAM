#ifndef __Settings__
#define __Settings__
#include <memory>
#include "GlobalTypes.h"

namespace FSLAM
{
//GLOBAL VARIABLES
extern PhotoUnDistMode PhoUndistMode;
extern Sensor Sensortype;

// extern int NumProcessors;

extern int WidthOri;
extern int HeightOri;

//Detector settings
extern int IndPyrLevels;
extern float IndPyrScaleFactor;
extern int IndNumFeatures;
extern int minThFAST;
extern float tolerance; //Ssc telerance ratio
extern int EnforcedMinDist;
extern bool DoSubPix;
extern bool DrawDetected;
extern bool DrawDepthKfTest;
extern bool DrawEpipolarMatching;

//Direct data dector
extern int DirPyrLevels;

extern bool Pause;

extern bool SequentialOperation;

//Display options
extern bool DisplayOn;
extern bool ShowInitializationMatches;
extern bool ShowInitializationMatchesSideBySide;
extern bool show_gradient_image;
extern bool settings_show_InitDepth;

extern float setting_maxLogAffFacInWindow; // marg a frame if factor between intensities to current frame is larger than 1/X or X.
extern float setting_minPointsRemaining;  // marg a frame if less than X% points remain.

extern int setting_minFrames; // min frames in window.
extern int setting_maxFrames; // max frames in window.
extern int setting_minFrameAge;
extern int setting_maxOptIterations;  // max GN iterations.
extern int setting_minOptIterations;  // min GN iterations.
extern float setting_thOptIterations; // factor on break threshold for GN iteration (larger = break earlier)

extern float setting_desiredPointDensity;
extern float setting_outlierTHSumComponent; // higher -> less strong gradient-based reweighting .
extern float setting_outlierTH; // higher -> less strict
extern float setting_overallEnergyTHWeight;

extern float setting_huberTH; //Huber Threshold!

extern float benchmark_initializerSlackFactor;


// parameters controlling adaptive energy threshold computation.
extern float setting_frameEnergyTHN;
extern float setting_frameEnergyTHFacMedian;
extern float setting_frameEnergyTHConstWeight;

extern float setting_coarseCutoffTH;


//Immature point tracking
extern float setting_maxPixSearch;
extern float setting_trace_slackInterval;
extern float setting_trace_stepsize;
extern float setting_trace_minImprovementFactor;
extern int setting_minTraceTestRadius;
extern int setting_trace_GNIterations;
extern float setting_trace_GNThreshold;
extern float setting_trace_extraSlackOnTH;


extern float setting_margWeightFac; // factor on hessian when marginalizing, to account for inaccurate linearization points.
extern int setting_GNItsOnPointActivation;

extern float setting_minTraceQuality;


//MapPoint settings
extern int   setting_minGoodActiveResForMarg;
extern int   setting_minGoodResForMarg;

//Fixing priors on unobservable/initial dimensions
extern float setting_idepthFixPrior;
extern float setting_idepthFixPriorMargFac;
extern float setting_initialRotPrior;
extern float setting_initialTransPrior;
extern float setting_initialAffBPrior;
extern float setting_initialAffAPrior;
extern float setting_initialCalibHessian;


extern float setting_affineOptModeA; 
extern float setting_affineOptModeB;

//Solver Settings
extern int setting_solverMode;
extern double setting_solverModeDelta;
extern bool setting_forceAceptStep;

extern float setting_minIdepthH_act;
extern float setting_minIdepthH_marg;



extern bool multiThreading;

} // namespace FSLAM

#endif