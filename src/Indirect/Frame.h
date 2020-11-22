#pragma once
#include "util/settings.h"
#include "util/NumType.h"

namespace HSLAM
{
    class CalibHessian;
    class FeatureDetector;
    class FrameShell;
    struct FrameHessian;

    template <typename Type> class IndexThreadReduce;

    class Frame //structure that contains all the data needed for keyframes
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        

        Frame(float* Img, std::shared_ptr<FeatureDetector> detector, CalibHessian *_HCalib, FrameHessian *_fh, FrameShell *_fs);
        ~Frame();
      
        void ReduceToEssential();
        bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
        // void Extract(std::shared_ptr<FeatureDetector> _Detector, int id, bool ForInit, std::shared_ptr<IndexThreadReduce<Vec10>> ThreadPool);
        void Extract();

        std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r) const;
        void ComputeBoVW();
        void assignFeaturesToGrid();

        cv::Mat Image;
        cv::Mat Occupancy;
        std::vector<std::vector<unsigned short int>> mGrid;
        std::vector<cv::KeyPoint> mvKeys;

        // std::vector<MapPoint> Mps;

        cv::Mat Descriptors;
        //BoW
        DBoW3::BowVector mBowVec;
        DBoW3::FeatureVector mFeatVec;

        int nFeatures;
        bool isReduced;
        bool NeedRefresh;


        CalibHessian *HCalib;
        FrameHessian *fh;
        FrameShell *fs;
    };

} // namespace HSLAM
