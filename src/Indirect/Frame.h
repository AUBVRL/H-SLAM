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

    class Frame : std::enable_shared_from_this<Frame> //structure that contains all the data needed for keyframes
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

        inline std::vector<std::shared_ptr<MapPoint>> getMapPointsV() //equivalent to getmappointmatches
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            return mvpMapPoints;
        }

        inline void EraseMapPointMatch(const size_t &idx)
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            mvpMapPoints[idx].reset();
        }

        void EraseMapPointMatch(std::shared_ptr<MapPoint> pMP);

        inline void ReplaceMapPointMatch(const size_t &idx, std::shared_ptr<MapPoint> pMP)
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            mvpMapPoints[idx] = pMP;
        }

        static bool weightComp(int a, int b)
        {
            return a > b;
        }

        void addMapPoint(std::shared_ptr<MapPoint>& Mp);
        void addMapPointMatch(std::shared_ptr<MapPoint> Mp, size_t index);


        void AddConnection(std::shared_ptr<Frame> pKF, const int &weight);
        void UpdateBestCovisibles();
        std::set<std::shared_ptr<Frame>> GetConnectedKeyFrames();
        std::vector<std::shared_ptr<Frame>> GetVectorCovisibleKeyFrames();
        std::vector<std::shared_ptr<Frame>> GetBestCovisibilityKeyFrames(const int &N);
        std::vector<std::shared_ptr<Frame>> GetCovisiblesByWeight(const int &w);
        int GetWeight(std::shared_ptr<Frame> pKF);
        void UpdateConnections();


        // Spanning tree functions
        void AddChild(std::shared_ptr<Frame> pKF);
        void EraseChild(std::shared_ptr<Frame> pKF);
        void ChangeParent(std::shared_ptr<Frame> pKF);
        std::set<std::shared_ptr<Frame>> GetChilds();
        std::shared_ptr<Frame> GetParent();
        bool hasChild(std::shared_ptr<Frame> pKF);

        // Loop Edges
        void AddLoopEdge(std::shared_ptr<Frame> pKF);
        std::set<std::shared_ptr<Frame>> GetLoopEdges();

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

        long unsigned int mnLoopQuery;
        int mnLoopWords;
        float mLoopScore;

        private:
   			boost::mutex _mtx;
            boost::mutex mMutexConnections;

            std::map<std::shared_ptr<Frame>, int> mConnectedKeyFrameWeights;
            std::vector<std::shared_ptr<Frame>> mvpOrderedConnectedKeyFrames;
            std::vector<int> mvOrderedWeights;

            std::shared_ptr<Frame> mpParent;
            std::set<std::shared_ptr<Frame>> mspChildrens;
            std::set<std::shared_ptr<Frame>> mspLoopEdges;
            bool mbFirstConnection;
            bool mbNotErase;
    };

} // namespace HSLAM
