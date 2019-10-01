#include "Settings.h"
// #include <omp.h>

namespace FSLAM
{
    // int NumProcessors = std::max(omp_get_num_threads()/2,6);
    Sensor Sensortype = Emptys;
    PhotoUnDistMode PhoUndistMode = Emptyp;

    int WidthOri; 
    int HeightOri;

    //Detector params
    int PyrLevels = 8;
    float PyrScaleFactor = 1.2;
    int numFeatures = 1500;

}