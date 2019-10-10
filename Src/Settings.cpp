#include "Settings.h"
// #include <omp.h>

namespace FSLAM
{
    // int NumProcessors = std::max(omp_get_num_threads()/2,6);
    Sensor Sensortype = Emptys;
    PhotoUnDistMode PhoUndistMode = Emptyp;

    int WidthOri; 
    int HeightOri;

    //Detector params (settings here are overriden by explicitly changing the input to software)
    int PyrLevels = 8;
    float PyrScaleFactor = 1.2;
    int numFeatures = 2000;
    int minThFAST = 4;
    float tolerance = 0.1; //SSC tolerance - no longer used consider removing SSC

    bool DoSubPix = false;
    bool DrawDetected = true;
    bool Pause = false;

    //Display options
    bool DisplayOn = true;


}