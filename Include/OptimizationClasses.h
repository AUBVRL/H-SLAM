#ifndef __OptimizationClasses__
#define __OptimizationClasses__

#include "GlobalTypes.h"

namespace FSLAM
{

class Frame;
class CalibData;

struct FrameFramePrecalc
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    // static values
    // static int instanceCounter;
    std::weak_ptr<Frame> host;   // defines row
    std::weak_ptr<Frame> target; // defines column

    // precalc values
    Mat33f PRE_RTll;
    Mat33f PRE_KRKiTll;
    Mat33f PRE_RKiTll;
    Mat33f PRE_RTll_0;

    Vec2f PRE_aff_mode;
    float PRE_b0_mode;

    Vec3f PRE_tTll;
    Vec3f PRE_KtTll;
    Vec3f PRE_tTll_0;

    float distanceLL;

    inline ~FrameFramePrecalc() {}
    inline FrameFramePrecalc() { }
    void set(std::shared_ptr<Frame> host, std::shared_ptr<Frame> target, std::shared_ptr<CalibData> HCalib);

};

} // namespace FSLAM

#endif