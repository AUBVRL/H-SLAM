#pragma once
#include "util/NumType.h"
#include "boost/thread.hpp"


namespace HSLAM
{
    class PointHessian;
    class PointHessian;
    class Frame;
    class Map;

    class MapPoint : public std::enable_shared_from_this<MapPoint>
    {
        

    private:
        mutable boost::mutex _mtx;
        std::map<std::shared_ptr<Frame>, size_t> mObservations; // Keyframes observing the point and associated index in keyframe
        Vec3f mNormalVector;// Mean viewing direction
        cv::Mat mDescriptor; // Best descriptor to fast matching
        int nObs;
        int mnFound;
        int mnVisible;
        
        Vec3f worldPose;
       
        float idepth;
        float idepthH;
        std::shared_ptr<MapPoint> mpReplaced;

        

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        MapPoint(PointHessian *_ph, std::shared_ptr<Map> _globalMap);
        ~MapPoint() {}

        std::shared_ptr<Frame> sourceFrame;
        PointHessian* ph;
        size_t id;  // the mappoint descriptor comes from the sourceId descriptor matrix.
        float index; //index of the point in the original keyframe = immPt.type-5
        Vec2f pt;
        float angle;
        bool mbBad;

        std::shared_ptr<MapPoint> getPtr();
        std::weak_ptr<Map> globalMap;

        //Tracking data
        bool mbTrackInView;
        float mTrackProjX;
        float mTrackProjY;
        float mTrackViewCos;
        long unsigned int mnTrackReferenceForFrame;
        long unsigned int mnLastFrameSeen;
        
        enum mpDirStatus {active =0, marginalized, removed } status;

        void EraseObservation(std::shared_ptr<Frame> &pKF);
        void SetBadFlag();

       
        void Replace(std::shared_ptr<MapPoint> pMP);
        
        
        inline std::shared_ptr<MapPoint> GetReplaced()
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            return mpReplaced;
        }

        void UpdateNormalAndDepth();
        // Vec3f getWorldPosewPose(SE3& pose);
        Vec3f getWorldPose();

        void updateGlobalPose();


        void ComputeDistinctiveDescriptors();
        static size_t idCounter;

        inline void setPointHessian(PointHessian *_ph)
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            ph = _ph;
        }

        void updateDepth();


        inline mpDirStatus getDirStatus()
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            return status;
        }


        inline void setDirStatus(mpDirStatus _status)
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            status = _status;
        }


        inline PointHessian *getPh()
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            return ph;
        }


        inline void setPh(PointHessian * _ph)
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            ph = _ph;
        }


        inline Vec3f GetNormal()
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            return mNormalVector;
        }


        inline std::shared_ptr<Frame> GetReferenceKeyFrame()
        {
            boost::lock_guard<boost::mutex> l(_mtx); //diff lock?
            return sourceFrame;
        }


        inline std::map<std::shared_ptr<Frame>,size_t> GetObservations()
        {
            boost::lock_guard<boost::mutex> l(_mtx); //diff lock?
            return mObservations;
        }


        inline bool isBad()
        {
            boost::lock_guard<boost::mutex> l(_mtx); //pose and diff lock?
            return mbBad;
        }


        inline int getNObservations()
        {
            boost::lock_guard<boost::mutex> l(_mtx); //diff lock?
            return nObs;
        }


        inline int getIndexInKF(std::shared_ptr<Frame> kf)
        {
            boost::lock_guard<boost::mutex> l(_mtx); //diff lock?
            if (mObservations.count(kf))
                return mObservations[kf];
            else
                return -1;
        }


        inline bool isInKeyframe(std::shared_ptr<Frame> kf)
        {
            boost::lock_guard<boost::mutex> l(_mtx); //diff lock?
            return (mObservations.count(kf));
        }


        inline void increaseVisible(int n = 1)
        {
            boost::lock_guard<boost::mutex> l(_mtx); //diff lock?
            mnVisible += n;
        }


        inline void increaseFound(int n=1)
        {
            boost::lock_guard<boost::mutex> l(_mtx); //diff lock?
            mnFound+=n;
        }


        inline float GetFoundRatio()
        {
            boost::lock_guard<boost::mutex> l(_mtx); //diff lock?
            return static_cast<float>(mnFound)/mnVisible;
        }


        inline int GetFound()
        {
            boost::lock_guard<boost::mutex> l(_mtx); //diff lock?
            return mnFound;
        }
        

        inline cv::Mat GetDescriptor()
        {
            boost::lock_guard<boost::mutex> l(_mtx); //diff lock?
            return mDescriptor.clone();
        }


        inline void setDescriptor(cv::Mat _descriptor)
        {
            boost::lock_guard<boost::mutex> l(_mtx); //diff lock?
            // if(_descriptor.empty())
            //     return;
            mDescriptor = _descriptor.clone();
        }


        inline void AddObservation(std::shared_ptr<Frame> &pKF, size_t idx)
        {
            boost::lock_guard<boost::mutex> l(_mtx); //diff lock?
            if (mObservations.count(pKF))
                return;
            mObservations[pKF] = idx;
            nObs++;
        }
    };

} // namespace HSLAM