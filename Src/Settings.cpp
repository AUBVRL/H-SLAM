#include "Settings.h"

namespace FSLAM
{
    std::shared_ptr<Undistorter> UndistorterL;
    std::shared_ptr<Undistorter> UndistorterR;
    Sensor Sensortype = Emptys;
    int WidthOri; 
    int HeightOri; 
}