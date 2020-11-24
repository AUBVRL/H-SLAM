#pragma once

#include "util/NumType.h"
#include "algorithm"
#include <boost/thread.hpp>

namespace HSLAM
{

	class Frame;

	class FrameShell
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		int id;			  // INTERNAL ID, starting at zero.
		int incoming_id;  // ID passed into DSO
		double timestamp; // timestamp passed into DSO.
		

		int trackingRefId;

		// constantly adapted.
		
		AffLight aff_g2l;
		bool poseValid;

		// statisitcs
		int statistics_outlierResOnThis;
		int statistics_goodResOnThis;
		int marginalizedAt;
		double movedByOpt;

		bool isKeyframe;

		std::shared_ptr<Frame> frame;

		inline FrameShell()
		{
			id = 0;
			poseValid = true;
			camToWorld = SE3();
			aff_g2l = AffLight(0,0);
			worldToCamOpti = Sim3();
			timestamp = 0;
			marginalizedAt = -1;
			movedByOpt = 0;
			statistics_outlierResOnThis = statistics_goodResOnThis = 0;
			trackingRefId = 0;
			isKeyframe = false;
		}

		SE3 getPose() {
            boost::lock_guard<boost::mutex> l(shellPoseMutex);
            return camToWorld;
        }

        void setPose(const SE3 &_Twc) {
            boost::lock_guard<boost::mutex> l(shellPoseMutex);
            camToWorld = _Twc;
        }

		// get and write the optimized pose by loop closing
        Sim3 getPoseOpti() {
            boost::lock_guard<boost::mutex> l(shellPoseMutex);
            return worldToCamOpti;
        }

        void setPoseOpti(const Sim3 &Scw) {
            boost::lock_guard<boost::mutex> l(shellPoseMutex);
            worldToCamOpti = Scw;
        }
		

		private:
			boost::mutex shellPoseMutex;
			SE3 camToWorld; // Write: TRACKING, while frame is still fresh; MAPPING: only when locked [shellPoseMutex].
			Sim3 worldToCamOpti; //camToWorld.inverse
};


}

