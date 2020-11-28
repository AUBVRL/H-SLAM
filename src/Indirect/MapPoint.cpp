#include "MapPoint.h"
#include "Indirect/Frame.h"
#include "FullSystem/HessianBlocks.h"
#include "Indirect/Matcher.h"

#include "util/FrameShell.h"

namespace HSLAM
{
    using namespace std;
    size_t MapPoint::idCounter = 0;

    MapPoint::MapPoint(PointHessian *_ph)
    {
        assert(ph->my_type > 4);
        ph = _ph;
        sourceFrame = ph->host->shell->frame;
        index = ph->my_type - 5;
        nObs = 0;
        mnVisible = 1;
        mnFound = 1;
        mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

        status = mpDirStatus::active;
        idepth = ph->idepth;
        idepthH = ph->idepth_hessian; //ph->efPoint->HdiF
        // OIdepth=pointhessian->idepth_zero_scaled;
        // OWeight = sqrt(1e-3/(pointhessian->efPoint->HdiF+1e-12));

        
        boost::lock_guard<boost::mutex> l(_mtx);  //just in case some other thread wants to create a mapPoint
        id = idCounter;
        idCounter++;
        
    }

    Vec3f MapPoint::getWorldPosewPose(SE3 &pose)
    {
        boost::lock_guard<boost::mutex> l(_mtx);

        if(idepth < 0)
            return Vec3f(0.0f, 0.0f, 0.0f);

        // float depth = 1.0f / idepth;
        auto calib = sourceFrame->HCalib;
        auto pt = sourceFrame->mvKeys[index].pt;

        // float x = (pt.x * calib->fxli() + calib->cxli()) * depth;
        // float y = (pt.y * calib->fyli() + calib->cyli()) * depth;
        // float z = depth;
        return pose.cast<float>() * (Vec3f((pt.x * calib->fxli() + calib->cxli()), (pt.y * calib->fyli() + calib->cyli()), 1.0f) * (1.0f/idepth));
        // SE3 Pose = sourceFrame->fs->getPose();
    }

    Vec3f MapPoint::getWorldPose()
    {
         boost::lock_guard<boost::mutex> l(_mtx);
         
        if (idepth < 0)
            return Vec3f(0.0f, 0.0f, 0.0f);

        auto pose = sourceFrame->fs->getPose();
        float depth = 1.0f / idepth;
        auto calib = sourceFrame->HCalib;
        auto pt = sourceFrame->mvKeys[index].pt;

        float x = (pt.x * calib->fxli() + calib->cxli()) * depth;
        float y = (pt.y * calib->fyli() + calib->cyli()) * depth;
        float z = depth;
        return pose.cast<float>() * Vec3f(x, y, z);
    }

    void MapPoint::ComputeDistinctiveDescriptors()
    {
        // Retrieve all observed descriptors
        vector<cv::Mat> vDescriptors;

        map<shared_ptr<Frame>, size_t> observations = GetObservations();

        if (observations.empty())
            return;

        vDescriptors.reserve(observations.size());

        for (map<shared_ptr<Frame>, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            // shared_ptr<Frame> pKF = mit->first;
            // if (!pKF->isBad())
            vDescriptors.push_back(mit->first->Descriptors.row(mit->second));
        }

        if (vDescriptors.empty())
            return;

        // Compute distances between them
        const size_t N = vDescriptors.size();

        float Distances[N][N];
        for (size_t i = 0; i < N; i++)
        {
            Distances[i][i] = 0;
            for (size_t j = i + 1; j < N; j++)
            {
                int distij = Matcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
                Distances[i][j] = distij;
                Distances[j][i] = distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for (size_t i = 0; i < N; i++)
        {
            vector<int> vDists(Distances[i], Distances[i] + N);
            sort(vDists.begin(), vDists.end());
            int median = vDists[0.5 * (N - 1)];

            if (median < BestMedian)
            {
                BestMedian = median;
                BestIdx = i;
            }
        }

        setDescriptor(vDescriptors[BestIdx]);

    }

} // namespace HSLAM