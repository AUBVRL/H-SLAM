#ifndef __Settings__
#define __Settings__
#include <memory>
#include <GlobalTypes.h>
namespace FSLAM
{


//GLOBAL VARIABLES
extern PhotoUnDistMode PhoUndistMode;
extern Sensor Sensortype;

// extern int NumProcessors;

extern int WidthOri;
extern int HeightOri;

//Detector settings
extern int PyrLevels;
extern float PyrScaleFactor;
extern int numFeatures;

}

#endif