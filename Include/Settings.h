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

} // namespace FSLAM

#endif