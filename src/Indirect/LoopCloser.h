#pragma once

#include <util/NumType.h>
#include <boost/thread.hpp>

namespace HSLAM
{
    class Map;
    class Frame;

    class LoppCloser
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        LoppCloser();
        ~LoppCloser() {}

        void InsertKeyFrame(std::shared_ptr<Frame> frame);
        bool CorrectLoop(); //shared_ptr<CalibHessian> Hcalib
        void Run();
        void setFinish();


        std::shared_ptr<Map> globalMap;

    private:
        bool needFinish = false;
        bool needPoseGraph = false;
        boost::thread mainLoop;
        bool finished = false;
        std::deque<std::shared_ptr<Frame>> KFqueue;
        boost::mutex mutexKFQueue;
        double minScoreAccept = 0.06;
        int kfGap = 10;

    }
}