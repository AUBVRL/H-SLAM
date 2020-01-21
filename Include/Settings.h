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
extern bool DrawDepthKf;

//Direct data dector
extern int DirPyrLevels;
extern float DirPyrScaleFactor;

extern bool Pause;

extern bool SequentialOperation;

//Display options
extern bool DisplayOn;
extern bool ShowInitializationMatches;
extern bool show_gradient_image;
extern bool settings_show_InitDepth;

extern int patternNum;
extern std::vector<std::vector<int>> patternP;
extern int patternPadding;

extern float setting_outlierTHSumComponent; // higher -> less strong gradient-based reweighting .
extern float setting_outlierTH; // higher -> less strict
extern float setting_overallEnergyTHWeight;

extern float setting_huberTH; //Huber Threshold!

//Immature point tracking
extern float setting_maxPixSearch;
extern float setting_trace_slackInterval;
extern float setting_trace_stepsize;
extern float setting_trace_minImprovementFactor;
extern int setting_minTraceTestRadius;
extern int setting_trace_GNIterations;
extern float setting_trace_GNThreshold;
extern float setting_trace_extraSlackOnTH;

//Fixing priors on unobservable/initial dimensions

extern float setting_initialRotPrior;
extern float setting_initialTransPrior;
extern float setting_initialAffBPrior;
extern float setting_initialAffAPrior;

extern float setting_affineOptModeA; 
extern float setting_affineOptModeB;

//Solver Settings
extern int setting_solverMode;

} // namespace FSLAM

#endif