#pragma once 

#include <boost/thread.hpp>
#include <memory>
#include <set>
#include "util/settings.h"
#include "util/NumType.h"

namespace HSLAM
{
    class MapPoint;
    class Frame;
    class KeyFrameDatabase;

    class Map
    {
    private:
        std::set<std::shared_ptr<MapPoint>> mspMapPoints;
        std::set<std::shared_ptr<Frame>> mspKeyFrames;

        std::vector<std::shared_ptr<MapPoint>> mvpReferenceMapPoints;

        long unsigned int mnMaxKFid;

        // Index related to a big change in the map (loop closure, global BA)
        int mnBigChangeIdx;

        boost::mutex mMutexMap;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Map();

        ~Map() {}
        void AddKeyFrame(std::shared_ptr<Frame> pKF);
        void AddMapPoint(std::shared_ptr<MapPoint> pMP);
        void EraseMapPoint(std::shared_ptr<MapPoint> pMP);
        void EraseKeyFrame(std::shared_ptr<Frame> pKF);
        void SetReferenceMapPoints(const std::vector<std::shared_ptr<MapPoint>> &vpMPs);
        void InformNewBigChange();
        int GetLastBigChangeIdx();

        std::vector<std::shared_ptr<Frame>> GetAllKeyFrames();
        std::vector<std::shared_ptr<MapPoint>> GetAllMapPoints();
        std::vector<std::shared_ptr<MapPoint>> GetReferenceMapPoints();

        long unsigned int MapPointsInMap();
        long unsigned KeyFramesInMap();

        long unsigned int GetMaxKFid();

        void clear();

        std::vector<std::shared_ptr<Frame>> mvpKeyFrameOrigins;
        std::shared_ptr<KeyFrameDatabase> KfDB;
        boost::mutex mMutexMapUpdate;
    };

    class KeyFrameDatabase
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        KeyFrameDatabase();

        void add(std::shared_ptr<Frame> pKF);

        void erase(std::shared_ptr<Frame> pKF);

        void clear();

        // Loop Detection
        std::vector<std::shared_ptr<Frame>> DetectLoopCandidates(std::shared_ptr<Frame> pKF, float minScore);

    protected:
        // Associated vocabulary
        // const ORBVocabulary *mpVoc;

        // Inverted file
        std::vector<std::list<std::shared_ptr<Frame>>> mvInvertedFile;

        // Mutex
        boost::mutex mMutex;
    };
} // namespace HSLAM