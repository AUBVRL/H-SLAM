#ifndef __MAPPOINT_H__
#define __MAPPOINT_H__

#include "GlobalTypes.h"

namespace FSLAM
{
class CalibData;
class Frame;
class ImmaturePoint;
class EFPoint;
class PointFrameResidual;

struct MapPoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    static int instanceCounter;
    EFPoint* efPoint;

    // static values
    float color[MAX_RES_PER_POINT];   // colors in host frame
    float weights[MAX_RES_PER_POINT]; // host-weights for respective residuals.

    float u, v;
    int idx;
    float energyTH;
    Frame* host;
    std::shared_ptr<CalibData> Calib;
    bool hasDepthPrior;

    float my_type;

    float idepth_scaled;
    float idepth_zero_scaled;
    float idepth_zero;
    float idepth;
    float step;
    float step_backup;
    float idepth_backup;

    float nullspaces_scale;
    float idepth_hessian;
    float maxRelBaseline;
    int numGoodResiduals;

    enum PtStatus {ACTIVE = 0, OOB=1, OUTLIER=2, INACTIVE, MARGINALIZED};
    PtStatus status;

    inline void setPointStatus(PtStatus s) { status = s; }

    inline void setIdepth(float idepth)
    {
        this->idepth = idepth;
        this->idepth_scaled = SCALE_IDEPTH * idepth;
    }

    inline void setIdepthScaled(float idepth_scaled)
    {
        this->idepth = SCALE_IDEPTH_INVERSE * idepth_scaled;
        this->idepth_scaled = idepth_scaled;
    }

    inline void setIdepthZero(float idepth)
    {
        idepth_zero = idepth;
        idepth_zero_scaled = SCALE_IDEPTH * idepth;
        nullspaces_scale = -(idepth * 1.001 - idepth / 1.001) * 500;
    }

    std::vector<PointFrameResidual*> residuals;                // only contains good residuals (not OOB and not OUTLIER). Arbitrary order.
    std::pair<PointFrameResidual*, ResState> lastResiduals[2]; // contains information about residuals to the last two (!) frames. ([0] = latest, [1] = the one before).

    void release();

    MapPoint(const ImmaturePoint* const rawPoint, std::shared_ptr<CalibData> Hcalib);
   
    inline ~MapPoint()
    {
        // assert(efPoint == 0);
        release();
        instanceCounter--;
    }

    bool isOOB(const std::vector<Frame*> &toMarg) const;
    bool isInlierNew();

};

} // namespace FSLAM


#endif