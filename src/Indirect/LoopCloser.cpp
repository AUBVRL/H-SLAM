#include "Indirect/LoopCloser.h"


namespace HSLAM
{
    using namespace std;
    LoopCloser::LoopCloser() : kfDB(new DBoW3::Database(*fullsystem->vocab)), voc(fullsystem->vocab),
                               globalMap(fullsystem->globalMap), Hcalib(fullsystem->Hcalib->mpCH),
                               coarseDistanceMap(fullsystem->GetDistanceMap()),
                               fullSystem(fullsystem)
    {

        mainLoop = thread(&LoopClosing::Run, this);
        idepthMap = new float[wG[0] * hG[0]];
    }

    void LoopCloser::InsertKeyFrame(shared_ptr<Frame> &frame)
    {
        boost::unique_lock<boost::mutex> lock(mutexKFQueue);
        KFqueue.push_back(frame);
    }

    void LoopCloser::SetFinish(bool finish = true)
    {

        needFinish = finish;
        mainLoop.join();
        while (globalMap && globalMap->Idle() == false)
        {
            usleep(10000);
        }

        if (needPoseGraph)
        {
            if (globalMap)
            {
                globalMap->OptimizeALLKFs();
                usleep(5000);
            }
            while (globalMap && globalMap->Idle() == false)
            {
                usleep(10000);
            }
        }
    }
} // namespace HSLAM