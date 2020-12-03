#include "Frame.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include "Indirect/Detector.h"
#include "Indirect/MapPoint.h"
#include "opencv2/highgui.hpp"
#include "util/FrameShell.h"

namespace HSLAM
{

    using namespace std;

    Frame::Frame(float *Img, shared_ptr<FeatureDetector> detector, CalibHessian *_HCalib, FrameHessian *_fh, FrameShell *_fs) : nFeatures(0), isReduced(false), NeedRefresh(NeedRefresh), HCalib(_HCalib), fh(_fh), fs(_fs)
    {
        cv::Mat(hG[0], wG[0], CV_32FC1, Img).convertTo(Image, CV_8U);
        Occupancy = cv::Mat(hG[0], wG[0], CV_8U, cv::Scalar(0));
        detector->ExtractFeatures(Image, Occupancy, mvKeys, Descriptors, nFeatures, indFeaturesToExtract);
        assignFeaturesToGrid();
        mvpMapPoints.resize(nFeatures, nullptr);
        tMapPoints.resize(nFeatures, nullptr);
        mvbOutlier.resize(nFeatures, false);
        mnTrackReferenceForFrame = 0;
        mbFirstConnection = true;
        mnLoopQuery =0;
        mnLoopWords = 0;
        // ComputeBoVW();
    }
    Frame::~Frame()
    {
        Image.release();
        Occupancy.release();
        releaseVec(mvKeys);
        releaseVec(mGrid);
        Descriptors.release();
    };

    void Frame::ReduceToEssential()
    {
        if (isReduced)
            return;
        isReduced = true;
        {
            Image.release();
            Occupancy.release();
            releaseVec(tMapPoints);
            releaseVec(mvbOutlier);
            // NeedRefresh = true;
        }
        return;
    }

    void Frame::assignFeaturesToGrid()
    {
        mGrid.resize(mnGridCols * mnGridRows);
        for (unsigned short i = 0; i < nFeatures; ++i)
        {
            // assign feature to grid
            int gridX = mvKeys[i].pt.x / gridSize;
            int gridY = mvKeys[i].pt.y / gridSize;
            mGrid[gridY * mnGridCols + gridX].push_back(i);
        }

        return;
    }


        void Frame::addMapPoint(std::shared_ptr<MapPoint>& Mp )
        {
            //should only be used when creating a new map point from a pointHessian.
            boost::lock_guard<boost::mutex> l(_mtx);
            mvpMapPoints[Mp->index] = Mp;
            return;
        }

        void Frame::addMapPointMatch(std::shared_ptr<MapPoint> Mp, size_t index )
        {
            //should only be used when adding a mappoint match.
            boost::lock_guard<boost::mutex> l(_mtx);
            if(!mvpMapPoints[index])
                mvpMapPoints[index] = Mp;
            return;
        }

    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
    {
        posX = std::round((kp.pt.x) * mfGridElementWidthInv);
        posY = std::round((kp.pt.y) * mfGridElementHeightInv);
        if (posX < 0 || posX >= mnGridCols || posY < 0 || posY >= mnGridRows)
            return false;
        return true;
    }


    std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
    {
        std::vector<size_t> vIndices;
        // vIndices.reserve(nFeatures);

        const int nMinCellX = std::max(0, (int)std::floor((x - mnMinX - r) * mfGridElementWidthInv));
        if (nMinCellX >= mnGridCols)
            return vIndices;

        const int nMaxCellX = std::min((int)mnGridCols - 1, (int)std::ceil((x - mnMinX + r) * mfGridElementWidthInv));
        if (nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = std::max(0, (int)std::floor((y - mnMinY - r) * mfGridElementHeightInv));
        if (nMinCellY >= mnGridRows)
            return vIndices;

        const int nMaxCellY = std::min((int)mnGridRows - 1, (int)std::ceil((y - mnMinY + r) * mfGridElementHeightInv));
        if (nMaxCellY < 0)
            return vIndices;

        for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
        {
            for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
            {
                const std::vector<unsigned short int> vCell = mGrid[iy * mnGridCols + ix];
                for (size_t j = 0, jend = vCell.size(); j < jend; ++j)
                {
                    const cv::KeyPoint &kpUn = mvKeys[vCell[j]];
                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    if (fabs(distx) < r && fabs(disty) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    bool Frame::isInFrustum(shared_ptr<MapPoint> pMP, float viewingCosLimit)
    {
        pMP->mbTrackInView = false;

        // 3D in absolute coordinates
        Vec3f P = pMP->getWorldPose();
       

        // 3D in camera coordinates
        const Vec3f Pc =  fs->getPoseInverse().cast<float>() * P ;
        const float &PcX = Pc(0);
        const float &PcY = Pc(1);
        const float &PcZ = Pc(2);

        // Check positive depth
        if (PcZ < 0.0f)
            return false;

        // Project in image and check it is not outside
        const float invz = 1.0f / PcZ;
        const float u = HCalib->fxl() * PcX * invz + HCalib->cxl();
        const float v = HCalib->fyl() * PcY * invz + HCalib->cyl();

        if (u < mnMinX || u > mnMaxX)
            return false;
        if (v < mnMinY || v > mnMaxY)
            return false;

        // // Check distance is in the scale invariance region of the MapPoint
        // const float maxDistance = pMP->GetMaxDistanceInvariance();
        // const float minDistance = pMP->GetMinDistanceInvariance();
        const Vec3f PO = P - fs->getCameraCenter().cast<float>();
        const float dist = PO.norm();

        // if (dist < minDistance || dist > maxDistance)
        //     return false;

        // Check viewing angle
        Vec3f Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn) / dist;

        std::cout << viewCos << std::endl;
        if (viewCos < viewingCosLimit)
            return false;

        
        // Predict scale in the image
        // const int nPredictedLevel = pMP->PredictScale(dist, this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = u;
        // pMP->mTrackProjXR = u - mbf * invz;
        pMP->mTrackProjY = v;
        // pMP->mnTrackScaleLevel = nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }


    std::set<std::shared_ptr<MapPoint>> Frame::getMapPointsS()
    {
        boost::lock_guard<boost::mutex> l(_mtx);
        std::set<std::shared_ptr<MapPoint>> s;
        for (size_t i = 0, iend = mvpMapPoints.size(); i < iend; ++i)
        {
            if (!mvpMapPoints[i])
                continue;
            std::shared_ptr<MapPoint> pMP = mvpMapPoints[i];
            if (!pMP->isBad())
                s.insert(pMP);
        }
        return s;
    }


    void Frame::ComputeBoVW()
    {
        if (mBowVec.empty())
        {
            vector<cv::Mat> vDesc;
            vDesc.reserve(Descriptors.rows);
            for (int j = 0; j < Descriptors.rows; ++j)
                vDesc.push_back(Descriptors.row(j));
            Vocab.transform(vDesc, mBowVec, mFeatVec, 4);
        }
    }

    void Frame::EraseMapPointMatch(shared_ptr<MapPoint> pMP)
    {
        assert(!mpReferenceKF.expired());
        int idx = pMP->getIndexInKF(mpReferenceKF.lock());
        if (idx >= 0)
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            mvpMapPoints[idx].reset();
        }
    }

    void Frame::UpdateBestCovisibles()
    {
        boost::lock_guard<boost::mutex> l(mMutexConnections);
        vector<pair<int, shared_ptr<Frame>>> vPairs;
        vPairs.reserve(mConnectedKeyFrameWeights.size());
        for (map<shared_ptr<Frame>, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
            vPairs.push_back(make_pair(mit->second, mit->first));

        sort(vPairs.begin(), vPairs.end());
        list<shared_ptr<Frame>> lKFs;
        list<int> lWs;
        for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }

        mvpOrderedConnectedKeyFrames = vector<shared_ptr<Frame>>(lKFs.begin(), lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
    }


    void Frame::UpdateConnections()
    {
        assert(!mpReferenceKF.expired()); //check that we have set the mpreferencekf to point to the keyframe itself.
        std::shared_ptr<Frame> ptrThis = mpReferenceKF.lock();
        map<shared_ptr<Frame>, int> KFcounter;

        vector<shared_ptr<MapPoint>> vpMP = getMapPointsV();

        //For all map points in keyframe check in which other keyframes are they seen
        //Increase counter for those keyframes
        for (vector<shared_ptr<MapPoint>>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)
        {
            shared_ptr<MapPoint> pMP = *vit;

            if (!pMP)
                continue;

            if (pMP->isBad())
                continue;

            map<shared_ptr<Frame>, size_t> observations = pMP->GetObservations();

            for (map<shared_ptr<Frame> , size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
            {
                if (mit->first->fs->KfId == fs->KfId) // mnId == mnId)
                    continue;
                KFcounter[mit->first]++;
            }
        }

        // This should not happen
        if (KFcounter.empty())
            return;

        //If the counter is greater than threshold add connection
        //In case no keyframe counter is over threshold add the one with maximum counter
        int nmax = 0;
        shared_ptr<Frame> pKFmax = NULL;
        int th = 15;

        vector<pair<int, shared_ptr<Frame>>> vPairs;
        vPairs.reserve(KFcounter.size());
        for (map<shared_ptr<Frame> , int>::iterator mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
        {
            if (mit->second > nmax)
            {
                nmax = mit->second;
                pKFmax = mit->first;
            }
            if (mit->second >= th)
            {
                vPairs.push_back(make_pair(mit->second, mit->first));
                
                (mit->first)->AddConnection(ptrThis, mit->second);
            }
        }

        if (vPairs.empty())
        {
            vPairs.push_back(make_pair(nmax, pKFmax));
            pKFmax->AddConnection(ptrThis, nmax);
        }

        sort(vPairs.begin(), vPairs.end());
        list<shared_ptr<Frame> > lKFs;
        list<int> lWs;
        for (size_t i = 0; i < vPairs.size(); i++)
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }

        {
            boost::lock_guard<boost::mutex> l(mMutexConnections);
            // mspConnectedKeyFrames = spConnectedKeyFrames;
            mConnectedKeyFrameWeights = KFcounter;
            mvpOrderedConnectedKeyFrames = vector<shared_ptr<Frame>>(lKFs.begin(), lKFs.end());
            mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

            if (mbFirstConnection && fs->KfId != 0)
            {
                mpParent = mvpOrderedConnectedKeyFrames.front();
                mpParent->AddChild(ptrThis);
                mbFirstConnection = false;
            }
        }
    }


    void Frame::AddConnection(shared_ptr<Frame> pKF, const int &weight)
    {
        {
            boost::lock_guard<boost::mutex> l(mMutexConnections);
            if (!mConnectedKeyFrameWeights.count(pKF))
                mConnectedKeyFrameWeights[pKF] = weight;
            else if (mConnectedKeyFrameWeights[pKF] != weight)
                mConnectedKeyFrameWeights[pKF] = weight;
            else
                return;
        }

        UpdateBestCovisibles();
    }

    set<shared_ptr<Frame>> Frame::GetConnectedKeyFrames()
    {
        boost::lock_guard<boost::mutex> l(mMutexConnections);
        set<shared_ptr<Frame>> s;
        for (map<shared_ptr<Frame>, int>::iterator mit = mConnectedKeyFrameWeights.begin(); mit != mConnectedKeyFrameWeights.end(); mit++)
            s.insert(mit->first);
        return s;
    }
  
    vector<shared_ptr<Frame>> Frame::GetVectorCovisibleKeyFrames()
    {
        boost::lock_guard<boost::mutex> l(mMutexConnections);
        return mvpOrderedConnectedKeyFrames;
    }

    vector<shared_ptr<Frame>> Frame::GetBestCovisibilityKeyFrames(const int &N)
    {
        boost::lock_guard<boost::mutex> l(mMutexConnections);
        if ((int)mvpOrderedConnectedKeyFrames.size() < N)
            return mvpOrderedConnectedKeyFrames;
        else
            return vector<shared_ptr<Frame>>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);
    }

    vector<shared_ptr<Frame>> Frame::GetCovisiblesByWeight(const int &w)
    {
        boost::lock_guard<boost::mutex> l(mMutexConnections);

        if (mvpOrderedConnectedKeyFrames.empty())
            return vector<shared_ptr<Frame>>();

        vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, Frame::weightComp);
        if (it == mvOrderedWeights.end())
            return vector<shared_ptr<Frame>>();
        else
        {
            int n = it - mvOrderedWeights.begin();
            return vector<shared_ptr<Frame>>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + n);
        }
    }

    int Frame::GetWeight(shared_ptr<Frame> pKF)
    {
        boost::lock_guard<boost::mutex> l(mMutexConnections);
        if (mConnectedKeyFrameWeights.count(pKF))
            return mConnectedKeyFrameWeights[pKF];
        else
            return 0;
    }

    void Frame::AddChild(std::shared_ptr<Frame> pKF)
    {
        boost::lock_guard<boost::mutex> l(mMutexConnections);
        mspChildrens.insert(pKF);
    }

    void Frame::EraseChild(std::shared_ptr<Frame> pKF)
    {
        boost::lock_guard<boost::mutex> l(mMutexConnections);
	    mspChildrens.erase(pKF);
    }

    void Frame::ChangeParent(std::shared_ptr<Frame> pKF)
    {
        assert(!mpReferenceKF.expired());
        boost::lock_guard<boost::mutex> l(mMutexConnections);
        mpParent = pKF;
        pKF->AddChild(mpReferenceKF.lock());
    }

    set<shared_ptr<Frame>> Frame::GetChilds()
    {
        boost::lock_guard<boost::mutex> l(mMutexConnections);
        return mspChildrens;
    }

    shared_ptr<Frame> Frame::GetParent()
    {
        boost::lock_guard<boost::mutex> l(mMutexConnections);
	    return mpParent;
    }

    bool Frame::hasChild(shared_ptr<Frame> pKF)
    {
        boost::lock_guard<boost::mutex> l(mMutexConnections);
	    return mspChildrens.count(pKF);
    }

    // Loop Edges
    void Frame::AddLoopEdge(shared_ptr<Frame> pKF)
    {
        boost::lock_guard<boost::mutex> l(mMutexConnections);
	    mbNotErase = true;
	    mspLoopEdges.insert(pKF);
    }

    set<shared_ptr<Frame>> Frame::GetLoopEdges()
    {
        boost::lock_guard<boost::mutex> l(mMutexConnections);
	    return mspLoopEdges;
    }

} // namespace HSLAM
