#ifndef __PIXELSELECTOR__
#define __PIXELSELECTOR__
 
#include "GlobalTypes.h"

namespace FSLAM
{

enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};

class Frame;
class CalibData;


class PixelSelector
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	int makeMaps(
			std::vector<Vec3f*>&DirPyr, int id, std::vector<float*>& GradPyr,
			float* map_out, float density, int recursionsLeft=1, bool plot=false, float thFactor=1);

	PixelSelector(std::shared_ptr<CalibData>_Calib);
	~PixelSelector();
	int currentPotential;
    std::shared_ptr<CalibData> Calib;

	bool allowFast;
	void makeHists(std::vector<float*>& GradPyr, int id);
    int area;
private:

	Eigen::Vector3i select(std::vector<Vec3f*>&DirPyr, int id, std::vector<float*>& GradPyr,
			float* map_out, int pot, float thFactor=1);
    

	unsigned char* randomPattern;


	int* gradHist;
	float* ths;
	float* thsSmoothed;
	int thsStep;
	int gradHistFrame;
};




}

#endif

