#ifndef __Settings__
#define __Settings__
#include "Undistorter.h"
#include <memory>
namespace FSLAM
{


//GLOBAL VARIABLES
extern std::shared_ptr<Undistorter> UndistorterL;
extern std::shared_ptr<Undistorter> UndistorterR;
extern Sensor Sensortype;


extern int WidthOri;
extern int HeightOri;
}

#endif