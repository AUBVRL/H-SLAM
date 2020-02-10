#ifndef __MAPPOINT_H__
#define __MAPPOINT_H__

#include "Settings.h"

namespace FSLAM
{
class CalibData;
class Frame;
class ImmaturePoint;
// class EFPoint;
class PointFrameResidual;

struct MapPoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    // static int instanceCounter;
    // EFPoint* efPoint;

    // static values
    float color[MAX_RES_PER_POINT];   // colors in host frame
    float weights[MAX_RES_PER_POINT]; // host-weights for respective residuals.

    float u, v;
    // int idx;
    float energyTH;
    std::weak_ptr<Frame> host;
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

    bool WasMarginalized;

    enum PtStatus {ACTIVE = 0, OUTLIER = 1, INACTIVE = 2, MARGINALIZED = 3};
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

    std::vector<std::shared_ptr<PointFrameResidual>> residuals;                // only contains good residuals (not OOB and not OUTLIER). Arbitrary order.
    std::pair<std::shared_ptr<PointFrameResidual>, ResState> lastResiduals[2]; // contains information about residuals to the last two (!) frames. ([0] = latest, [1] = the one before).

    MapPoint(std::shared_ptr<ImmaturePoint> rawPoint, std::shared_ptr<CalibData> Hcalib);
   
    inline ~MapPoint() {}

    bool isOOB(const std::vector<std::shared_ptr<Frame>> &toMarg) const;
    bool isInlierNew();

    void takeData()
    {
        priorF = hasDepthPrior ? setting_idepthFixPrior * SCALE_IDEPTH * SCALE_IDEPTH : 0;
        if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR)
            priorF = 0;
        deltaF = idepth - idepth_zero;
    }

    float priorF = 0;
    float deltaF = 0;

    // H and b blocks
    float bdSumF = 0;
    float HdiF = 0;
    float Hdd_accLF = 0;
    VecCf Hcd_accLF = VecCf::Zero();
    float bd_accLF = 0;
    float Hdd_accAF = 0;
    VecCf Hcd_accAF = VecCf::Zero();
    float bd_accAF = 0;
};


} // namespace FSLAM


#endif