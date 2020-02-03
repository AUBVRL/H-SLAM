#ifndef __MAPPOINT_H__
#define __MAPPOINT_H__

#include "GlobalTypes.h"

namespace FSLAM
{
class CalibData;
class Frame;
class ImmaturePoint;
class EFPoint;

struct MapPoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    static int instanceCounter;
    std::shared_ptr<EFPoint> efPoint;

    // static values
    float color[MAX_RES_PER_POINT];   // colors in host frame
    float weights[MAX_RES_PER_POINT]; // host-weights for respective residuals.

    float u, v;
    int idx;
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

    enum PtStatus
    {
        ACTIVE = 0,
        INACTIVE,
        OUTLIER,
        OOB,
        MARGINALIZED
    };
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

    void release();

    MapPoint(const std::shared_ptr<MapPoint> const rawPoint, std::shared_ptr<CalibData> Hcalib);
   
    inline ~PointHessian()
    {
        // assert(efPoint == 0);
        release();
        instanceCounter--;
    }

    inline bool isOOB(const std::vector<std::shared_ptr<FrameHessian>> &toMarg) const
    {

        int visInToMarg = 0;
        for (auto r : residuals)
        {

            if (r->state_state != ResState::IN)
                continue;
            for (auto k : toMarg)
                if (r->target == k)
                    visInToMarg++;
        }
        if ((int)residuals.size() >= setting_minGoodActiveResForMarg &&
            numGoodResiduals > setting_minGoodResForMarg + 10 &&
            (int)residuals.size() - visInToMarg < setting_minGoodActiveResForMarg)
            return true;

        if (lastResiduals[0].second == ResState::OOB)
            return true;
        if (residuals.size() < 2)
            return false;
        if (lastResiduals[0].second == ResState::OUTLIER && lastResiduals[1].second == ResState::OUTLIER)
            return true;
        return false;
    }

    inline bool isInlierNew()
    {
        return (int)residuals.size() >= setting_minGoodActiveResForMarg && numGoodResiduals >= setting_minGoodResForMarg;
    }
};

} // namespace FSLAM


#endif