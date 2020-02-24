#ifndef __FRAME__
#define __FRAME__

#include "Settings.h"
#include "OptimizationClasses.h"
#include "boost/thread.hpp"


namespace FSLAM
{

class FeatureDetector;
class ImageData;
class CalibData;
class Frame;
class ImmaturePoint;
class MapPoint;
struct FrameOptimizationData;

template<typename Type> class IndexThreadReduce;

class Frame //structure that contains all the data needed for keyframes
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    Frame(std::shared_ptr<ImageData>Img, int id, float ab_exposure, std::shared_ptr<FeatureDetector>_Detector, std::shared_ptr<CalibData>_Calib, std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft, bool ForInit = false);
    ~Frame();
    void CreateIndPyrs(cv::Mat& Img, std::vector<cv::Mat>& Pyr);    
    void CreateDirPyrs(std::vector<float>& Img, std::vector<Vec3f*> &DirPyr);
    void ReduceToEssential();
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
    void Extract(int id ,bool ForInit, std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft);
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;


    std::shared_ptr<FeatureDetector> Detector;    
    std::vector<cv::Mat> IndPyr; //temporary CV_8U pyramids to extract features
    std::vector<Vec3f*> DirPyr; //float representation of image pyramid with computation of dIx ad dIy
    std::vector<float*> absSquaredGrad;

    std::vector<std::vector<std::vector<unsigned short int>>> mGrid;
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<FrameFramePrecalc,Eigen::aligned_allocator<FrameFramePrecalc>> targetPrecalc;
    
    std::vector<std::shared_ptr<MapPoint>> pointHessians;				// contains all ACTIVE points.
	std::vector<std::shared_ptr<MapPoint>> pointHessiansMarginalized;	// contains all MARGINALIZED points (= fully marginalized, usually because point went OOB.)
	std::vector<std::shared_ptr<MapPoint>> pointHessiansOut;		// contains all OUTLIER points (= discarded.).
    std::vector<std::shared_ptr<ImmaturePoint>> ImmaturePoints;

    cv::Mat Descriptors;

    unsigned int idx; // frame number in the moving optimization window
    
    static int mnGridCols, mnGridRows;
    static float mnMinX, mnMaxX, mnMinY, mnMaxY, mfGridElementWidthInv, mfGridElementHeightInv;
    static bool GridStructInit;
    static int EDGE_THRESHOLD; 

    // statisitcs
	int statistics_outlierResOnThis;
	int statistics_goodResOnThis;


	double MovedByOpt;
    float frameEnergyTH;
    bool FlaggedForMarginalization;
    int nFeatures; 
    bool isReduced;
    bool NeedRefresh;
    std::shared_ptr<CalibData> Calib;
    shared_ptr<FrameOptimizationData> efFrame; //CLEAR THIS ONCE KeyFRAME IS MARGINALIZED in reducetoessential!!!
};

struct FrameShell
{
    public:
	    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    inline FrameShell(std::shared_ptr<ImageData> Img, std::shared_ptr<FeatureDetector> _Detector, std::shared_ptr<CalibData>_Calib, 
                std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft, bool ForInit)
    {
        id = Globalid; Globalid++;
        KfId = -1;
        MarginalizedAt = -1;
        camToWorld = SE3();
        camToTrackingRef = SE3();
        trackingRef = nullptr;
        aff_g2l = AffLight(0, 0); //Past to present affine model of left image
        TimeStamp = Img->timestamp;
        ab_exposure = Img->ExposureL; //Exposure time of the left image
        poseValid = false;
        isKeyFrame = false;
        frame = std::shared_ptr<Frame> (new Frame(Img, id, ab_exposure, _Detector, _Calib, FrontEndThreadPoolLeft, ForInit ));
    }

    ~FrameShell(){};
    shared_ptr<Frame> frame;
    double TimeStamp;

    size_t id; //frame id number
    int KfId;
    static int GlobalKfId;
    static size_t Globalid; //frame id number that starts counting from the first initializing keyframe as 0.
    
    bool poseValid;
    bool isKeyFrame;
    SE3 camToWorld;
    SE3 camToTrackingRef;
	shared_ptr<FrameShell> trackingRef;
    AffLight aff_g2l;
    float ab_exposure;
    int MarginalizedAt;
};

struct FrameOptimizationData //contains all data needed to perform gradient based optimization
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    int id;
    float ab_exposure;
    
    SE3 PRE_worldToCam;
    SE3 PRE_camToWorld;
    SE3 worldToCam_evalPT;
    Vec10 state_zero;
    Vec10 state_scaled;
    Vec10 state; // [0-5: worldToCam-leftEps. 6-7: a,b]
    Vec10 step;
    Vec10 step_backup;
    Vec10 state_backup;

    Mat66 nullspaces_pose;
    Mat42 nullspaces_affine;
    Vec6 nullspaces_scale;

    FrameOptimizationData(int _id, float _ab_exposure) : id(_id), ab_exposure(_ab_exposure){};
    ~FrameOptimizationData(){};
    
    inline Vec6 w2c_leftEps() const { return get_state_scaled().head<6>(); }
    inline AffLight aff_g2l() const { return AffLight(get_state_scaled()[6], get_state_scaled()[7]); }
    inline AffLight aff_g2l_0() const { return AffLight(get_state_zero()[6] * SCALE_A, get_state_zero()[7] * SCALE_B); }

    EIGEN_STRONG_INLINE const SE3 &get_worldToCam_evalPT() const { return worldToCam_evalPT; }
    EIGEN_STRONG_INLINE const Vec10 &get_state_zero() const { return state_zero; }
    EIGEN_STRONG_INLINE const Vec10 &get_state() const { return state; }
    EIGEN_STRONG_INLINE const Vec10 &get_state_scaled() const { return state_scaled; }
    EIGEN_STRONG_INLINE const Vec10 get_state_minus_stateZero() const { return get_state() - get_state_zero(); }

    inline void setStateZero(const Vec10 &state_zero)
    {
        assert(state_zero.head<6>().squaredNorm() < 1e-20);

        this->state_zero = state_zero;

        for (int i = 0; i < 6; i++)
        {
            Vec6 eps;
            eps.setZero();
            eps[i] = 1e-3;
            SE3 EepsP = Sophus::SE3::exp(eps);
            SE3 EepsM = Sophus::SE3::exp(-eps);
            SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
            SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
            nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);
        }

        // scale change
        SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
        w2c_leftEps_P_x0.translation() *= 1.00001;
        w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
        SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
        w2c_leftEps_M_x0.translation() /= 1.00001;
        w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
        nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);

        nullspaces_affine.setZero();
        nullspaces_affine.topLeftCorner<2, 1>() = Vec2(1, 0);
        assert(ab_exposure > 0);
        nullspaces_affine.topRightCorner<2, 1>() = Vec2(0, expf(aff_g2l_0().a) * ab_exposure);
    };

    inline void setState(const Vec10 &state)
    {

        this->state = state;
        state_scaled.segment<3>(0) = SCALE_XI_TRANS * state.segment<3>(0);
        state_scaled.segment<3>(3) = SCALE_XI_ROT * state.segment<3>(3);
        state_scaled[6] = SCALE_A * state[6];
        state_scaled[7] = SCALE_B * state[7];
        state_scaled[8] = SCALE_A * state[8];
        state_scaled[9] = SCALE_B * state[9];

        PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
        PRE_camToWorld = PRE_worldToCam.inverse();
    };

    inline void setStateScaled(const Vec10 &state_scaled)
    {
        this->state_scaled = state_scaled;
        state.segment<3>(0) = SCALE_XI_TRANS_INVERSE * state_scaled.segment<3>(0);
        state.segment<3>(3) = SCALE_XI_ROT_INVERSE * state_scaled.segment<3>(3);
        state[6] = SCALE_A_INVERSE * state_scaled[6];
        state[7] = SCALE_B_INVERSE * state_scaled[7];
        state[8] = SCALE_A_INVERSE * state_scaled[8];
        state[9] = SCALE_B_INVERSE * state_scaled[9];

        PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
        PRE_camToWorld = PRE_worldToCam.inverse();
    };

    inline void setEvalPT(const SE3 &worldToCam_evalPT, const Vec10 &state)
    {
        this->worldToCam_evalPT = worldToCam_evalPT;
        setState(state);
        setStateZero(state);
    };

    inline void setEvalPT_scaled(const SE3 &worldToCam_evalPT, const AffLight &aff_g2l)
    {
        Vec10 initial_state = Vec10::Zero();
        initial_state[6] = aff_g2l.a;
        initial_state[7] = aff_g2l.b;
        this->worldToCam_evalPT = worldToCam_evalPT;
        setStateScaled(initial_state);
        setStateZero(this->get_state());
    };

    inline Vec10 getPrior()
    {
        Vec10 p = Vec10::Zero();
        if (id == 0)
        {
            p.head<3>() = Vec3::Constant(setting_initialTransPrior);
            p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);
            if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR)
                p.head<6>().setZero();

            p[6] = setting_initialAffAPrior;
            p[7] = setting_initialAffBPrior;
        }
        else
        {
            if (setting_affineOptModeA < 0)
                p[6] = setting_initialAffAPrior;
            else
                p[6] = setting_affineOptModeA;

            if (setting_affineOptModeB < 0)
                p[7] = setting_initialAffBPrior;
            else
                p[7] = setting_affineOptModeB;
        }
        p[8] = setting_initialAffAPrior;
        p[9] = setting_initialAffBPrior;
        return p;
    }

    inline Vec10 getPriorZero()
    {
        return Vec10::Zero();
    }

    Vec8 prior = Vec8::Zero();       // prior hessian (diagonal)
    Vec8 delta_prior = Vec8::Zero(); // = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
    Vec8 delta = Vec8::Zero();       // state - state_zero.
    inline void takeData()
    {
        prior = getPrior().head<8>();
        delta = get_state_minus_stateZero().head<8>();
        delta_prior = (get_state() - getPriorZero()).head<8>();
        return;
    }
};


} // namespace FSLAM









#endif