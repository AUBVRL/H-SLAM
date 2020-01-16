#include "System.h"
#include "Map.h"
#include "Frame.h"

namespace FSLAM
{
void System::AddKeyframe(std::shared_ptr<Frame> Frame)
{
    return;
}

void System::MappingThread()
{
    boost::unique_lock<boost::mutex> lock(MapThreadMutex);

    while (RunMapping)
    {
        while (UnmappedTrackedFrames.size() == 0)
        {
            TrackedFrameSignal.wait(lock);
            if (!RunMapping)
                return;
        }

        std::shared_ptr<Frame> frame = UnmappedTrackedFrames.front();
        UnmappedTrackedFrames.pop_front();

        // guaranteed to make a KF for the very first two tracked frames.
        if (SlamMap->KeyFrames.size() <= 2)
        {
            lock.unlock();
            AddKeyframe(frame);
            lock.lock();
            MappedFrameSignal.notify_all();
            continue;
        }

        if (UnmappedTrackedFrames.size() > 3)
            NeedToCatchUp = true;

        if (UnmappedTrackedFrames.size() > 0) // if there are other frames to track, do that first.
        {
            lock.unlock();
            ProcessNonKeyframe(frame);
            lock.lock();

            if (NeedToCatchUp && UnmappedTrackedFrames.size() > 0)
            {
                std::shared_ptr<Frame> frame = UnmappedTrackedFrames.front();
                UnmappedTrackedFrames.pop_front();
                {
                    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
                    assert(frame->trackingRef);
                    frame->camToWorld = frame->trackingRef->camToWorld * frame->camToTrackingRef;
                    frame->setEvalPT_scaled(frame->camToWorld.inverse(),frame->aff_g2l_internal);
                }
                frame.reset();
            }
        }
        else
        {
            if (NeedNewKFAfter >= SlamMap->LocalMap.back()->id)
            {
                lock.unlock();
                AddKeyframe(frame);
                NeedToCatchUp = false;
                lock.lock();
            }
            else
            {
                lock.unlock();
                ProcessNonKeyframe(frame);
                lock.lock();
            }
        }
        MappedFrameSignal.notify_all();
    }
    printf("MAPPING FINISHED!\n");
}

void System::BlockUntilMappingIsFinished()
{
    boost::unique_lock<boost::mutex> lock(MapThreadMutex);
    RunMapping = false;
    TrackedFrameSignal.notify_all();
    lock.unlock();

    tMappingThread.join();
}

} // namespace FSLAM