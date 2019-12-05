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
    int IndPyrLevels = 8;
    float IndPyrScaleFactor = 1.2;
    int IndNumFeatures = 2000;
    int minThFAST = 4;
    float tolerance = 0.1; //SSC tolerance - no longer used consider removing SSC

    bool DoSubPix = false;
    bool DrawDetected = true;
    bool Pause = false;

    //Direct data dector
    int DirPyrLevels = 4;
    float DirPyrScaleFactor = 2;


    //Display options
    bool DisplayOn = true;

    //feature pattern settings
    int patternNum = 8;
    std::vector<std::vector<int>> patternP = staticPattern[8];
    int patternPadding = 2;

    float setting_outlierTHSumComponent = 50*50; // higher -> less strong gradient-based reweighting .
    float setting_outlierTH = 12*12; // higher -> less strict
    float setting_overallEnergyTHWeight = 1;

    float setting_huberTH = 9; //Huber Threshold!

    //Immatureure point tracking
    float setting_maxPixSearch = 0.027f;
    float setting_trace_slackInterval = 1.5f;
    float setting_trace_stepsize = 1.0f;
    float setting_trace_minImprovementFactor = 2;
    int setting_minTraceTestRadius = 2;
    int setting_trace_GNIterations = 3;
    float setting_trace_GNThreshold = 0.1f;
    float setting_trace_extraSlackOnTH = 1.2;

}