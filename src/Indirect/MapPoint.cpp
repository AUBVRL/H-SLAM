#include "MapPoint.h"
#include "Indirect/Frame.h"
#include "FullSystem/HessianBlocks.h"
#include "Indirect/Matcher.h"

namespace HSLAM
{
    using namespace std;
    size_t MapPoint::idCounter = 0;


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