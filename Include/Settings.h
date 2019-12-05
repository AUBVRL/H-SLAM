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
extern bool DoSubPix;
extern bool DrawDetected;

//Direct data dector
extern int DirPyrLevels;
extern float DirPyrScaleFactor;

extern bool Pause;

//Display options
extern bool DisplayOn;

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


} // namespace FSLAM

#endif