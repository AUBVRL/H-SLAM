#ifndef __MAPPOINT_H__
#define __MAPPOINT_H__

#include "Settings.h"
#include "ImmaturePoint.h"
#include "OptimizationClasses.h"

namespace FSLAM
{
class CalibData;
class FrameShell;
class ImmaturePoint;
struct MapPointOptimizationData;
// class PointFrameResidual;

struct MapPointOptimizationData
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    float idepth_zero = NAN;
    float step;
    float step_backup;
    float idepth_backup;

    float nullspaces_scale;
    MapPointOptimizationData(){};
    ~MapPointOptimizationData(){};

    inline void setIdepthZero(float _idepth)
    {
        idepth_zero = _idepth;
        nullspaces_scale = -(_idepth * 1.001 - _idepth / 1.001) * 500;
    }

    void takeData(bool hasDepthPrior, float idepth)
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
    energyStatus stateFlag = energyStatus::Good;
};

struct MapPoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    shared_ptr<MapPointOptimizationData> efPoint;

    // static values
    float color[MAX_RES_PER_POINT];   // colors in host frame
    float weights[MAX_RES_PER_POINT]; // host-weights for respective residuals.

    float u, v;
    int idx;
    float energyTH;
    shared_ptr<FrameShell> host;
    bool hasDepthPrior;

    float my_type;
    float idepth;

    float idepth_hessian;
    float maxRelBaseline;
    int numGoodResiduals;

    PtStatus status;

    inline void setPointStatus(PtStatus s) { status = s; }
    inline PtStatus getPointStatus() { return status; }

    inline void setIdepth(float _idepth)
    {
        idepth = _idepth;
    }

    vector<shared_ptr<PointFrameResidual>> residuals;                // only contains good residuals (not OOB and not OUTLIER). Arbitrary order.
    pair<shared_ptr<PointFrameResidual>, ResState> lastResiduals[2]; // contains information about residuals to the last two (!) frames. ([0] = latest, [1] = the one before).

    inline MapPoint(shared_ptr<ImmaturePoint> &rawPoint, shared_ptr<CalibData> &Hcalib)
    {
        host = rawPoint->host;
        hasDepthPrior = false;

        u = rawPoint->u;
        v = rawPoint->v;

        idepth_hessian = 0;
        maxRelBaseline = 0;
        numGoodResiduals = 0;

        assert(std::isfinite(rawPoint->idepth_max));

        my_type = rawPoint->my_type;

        setIdepth((rawPoint->idepth_max + rawPoint->idepth_min) * 0.5);
        setPointStatus(INACTIVE);

        memcpy(color, rawPoint->color, sizeof(float) * patternNum);
        memcpy(weights, rawPoint->weights, sizeof(float) * patternNum);
        energyTH = rawPoint->energyTH;
        efPoint = shared_ptr<MapPointOptimizationData>(new MapPointOptimizationData()); //remove this during reduce essential for all points of a keyframe.
    }

    inline void Clear()
    {
        if(residuals.size()>0)
            residuals.clear();

        if (lastResiduals[0].first)
            lastResiduals[0].first.reset();
        if (lastResiduals[1].first)
            lastResiduals[1].first.reset();

        if (efPoint)
            efPoint.reset();
    }

    inline ~MapPoint() {}

    inline bool isOOB( const vector<shared_ptr<FrameShell>> &toMarg) const
    {
        int visInToMarg = 0;
        for (auto &r : residuals)
        {
            if (r->state_state != ResState::IN)
                continue;
            for (auto &k : toMarg)
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
        if (lastResiduals[0].second == ResState::OUT && lastResiduals[1].second == ResState::OUT)
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