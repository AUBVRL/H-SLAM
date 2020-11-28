#pragma once
#include "util/settings.h"
#include "util/NumType.h"
#include <boost/thread.hpp>

namespace HSLAM
{
    class CalibHessian;
    class FeatureDetector;
    class FrameShell;
    class MapPoint;
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

        std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r) const;
        void ComputeBoVW();
        void assignFeaturesToGrid();
        
        inline std::shared_ptr<MapPoint> getMapPoint(int idx)
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            return mvpMapPoints[idx];
        }

        inline std::vector<std::shared_ptr<MapPoint>> getMapPointsV()
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            return mvpMapPoints;
        }

        void addMapPoint(std::shared_ptr<MapPoint>& Mp);
        void addMapPointMatch(std::shared_ptr<MapPoint> Mp, size_t index);

        cv::Mat Image;
        cv::Mat Occupancy;
        std::vector<std::vector<unsigned short int>> mGrid;
        std::vector<cv::KeyPoint> mvKeys;

        std::vector<std::shared_ptr<MapPoint>> mvpMapPoints;

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

        private:
   			boost::mutex _mtx;

    };

} // namespace HSLAM
