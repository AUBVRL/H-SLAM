#ifndef __Settings__
#define __Settings__
#include <memory>
#include "GlobalTypes.h"
namespace FSLAM
{

#define SCALE_IDEPTH 1.0f		// scales internal value to idepth.
#define SCALE_XI_ROT 1.0f
#define SCALE_XI_TRANS 0.5f
#define SCALE_F 50.0f
#define SCALE_C 50.0f
#define SCALE_W 1.0f
#define SCALE_A 10.0f
#define SCALE_B 1000.0f

#define SCALE_IDEPTH_INVERSE (1.0f / SCALE_IDEPTH)
#define SCALE_XI_ROT_INVERSE (1.0f / SCALE_XI_ROT)
#define SCALE_XI_TRANS_INVERSE (1.0f / SCALE_XI_TRANS)
#define SCALE_F_INVERSE (1.0f / SCALE_F)
#define SCALE_C_INVERSE (1.0f / SCALE_C)
#define SCALE_W_INVERSE (1.0f / SCALE_W)
#define SCALE_A_INVERSE (1.0f / SCALE_A)
#define SCALE_B_INVERSE (1.0f / SCALE_B)

#if ThreadCount != 0 //Number of thread is set from cmake, cant use for some built in std::thread to detect number of cores!
#define NUM_THREADS (ThreadCount) // const int NUM_THREADS = boost::thread::hardware_concurrency();
#else
#define NUM_THREADS (6)
#endif


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
extern int minThFAST;
extern float tolerance; //Ssc telerance ratio
extern bool DoSubPix;
extern bool DrawDetected;


extern bool Pause;

//Display options
extern bool DisplayOn;

}

#endif