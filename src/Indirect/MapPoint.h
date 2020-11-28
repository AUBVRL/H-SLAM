#pragma once
#include "util/NumType.h"
#include "boost/thread.hpp"


namespace HSLAM
{
    class PointHessian;
    class PointHessian;
    class Frame;

    class MapPoint
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    private:
        mutable boost::mutex _mtx;
        std::map<std::shared_ptr<Frame>, size_t> mObservations; // Keyframes observing the point and associated index in keyframe
        cv::Mat mNormalVector;// Mean viewing direction
        cv::Mat mDescriptor; // Best descriptor to fast matching
        int nObs;
        int mnFound;
        int mnVisible;

    public:

        MapPoint(PointHessian *_ph);
        ~MapPoint() {}

        std::shared_ptr<Frame> sourceFrame;
        PointHessian* ph;
        size_t id;  // the mappoint descriptor comes from the sourceId descriptor matrix.
        float index; //index of the point in the original keyframe = immPt.type-5
        float idepth;
        float idepthH;
        enum mpDirStatus {active =0, marginalized, removed } status;

        void EraseObservation(std::shared_ptr<Frame> &pKF);
        void SetBadFlag();
        bool isBad();
       
        // void Replace(MapPoint *pMP);
        // MapPoint *GetReplaced();
        void UpdateNormalAndDepth();
        Vec3f getWorldPosewPose(SE3& pose);
        Vec3f getWorldPose();


        void ComputeDistinctiveDescriptors();
        static size_t idCounter;

        inline void setPointHessian(PointHessian *_ph)
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            ph = _ph;
        }


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


        inline cv::Mat GetNormal()
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            return mNormalVector.clone();
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


        inline void increaseFound(int n)
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
            if(_descriptor.empty())
                return;
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