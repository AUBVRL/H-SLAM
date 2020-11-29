#include "Matcher.h"
#include "Indirect/Frame.h"
#include "util/FrameShell.h"
#include "Indirect/MapPoint.h"
#include "DBoW3/DBoW3.h"
#include "FullSystem/HessianBlocks.h"

namespace HSLAM
{
    using namespace std;
    
    int Matcher::SearchByBoW(shared_ptr<Frame> pKF1, shared_ptr<Frame> pKF2, vector<pair<size_t, size_t>> &matches)
    {
        int nmatches = 0;

        return nmatches;
    }

    int Matcher::SearchByProjection(shared_ptr<Frame> &CurrentFrame, shared_ptr<Frame> &pKF, const set<shared_ptr<MapPoint>> &sAlreadyFound, const float th, const int ORBdist)
    {
        int nmatches = 0;
        auto CurrPoseInv = CurrentFrame->fs->getPoseInverse();
        auto currPose = CurrentFrame->fs->getPose();
        auto Rcw = CurrPoseInv.rotationMatrix();
        auto tcw = CurrPoseInv.translation();
        auto Ow = CurrentFrame->fs->getCameraCenter();
    

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f / HISTO_LENGTH;

        const vector<shared_ptr<MapPoint>> vpMPs = pKF->getMapPointsV();

        for (size_t i = 0, iend = vpMPs.size(); i < iend; ++i)
        {

            shared_ptr<MapPoint> pMP = vpMPs[i];

            if (pMP)
            {
                if (!pMP->isBad() && !sAlreadyFound.count(pMP))
                {
                    //Project
                    Vec3f x3Dw = pMP->getWorldPose(); //GetWorldPos();
                    Vec3f x3Dc = Rcw.cast<float>() * x3Dw + tcw.cast<float>();

                    const float xc = x3Dc(0);
                    const float yc = x3Dc(1);
                    const float invzc = 1.0 / x3Dc(2);

                    const float u =  CurrentFrame->HCalib->fxl() * xc * invzc + CurrentFrame->HCalib->cxl();
                    const float v = CurrentFrame->HCalib->fyl() * yc * invzc + CurrentFrame->HCalib->cyl();

                    if (u < mnMinX || u > mnMaxX)
                        continue;
                    if (v < mnMinY || v > mnMaxY)
                        continue;

                    // Compute predicted scale level
                    // Vec3f PO = x3Dw - Ow.cast<float>();
                    
                    // float dist3D = PO.norm();

                    // const float maxDistance = pMP->GetMaxDistanceInvariance();
                    // const float minDistance = pMP->GetMinDistanceInvariance();

                    // // Depth must be inside the scale pyramid of the image
                    // if (dist3D < minDistance || dist3D > maxDistance)
                    //     continue;

                    // int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);

                    // Search in a window
                    const float radius = th;

                    const vector<size_t> vIndices2 = CurrentFrame->GetFeaturesInArea(u, v, radius);

                    if (vIndices2.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for (vector<size_t>::const_iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
                    {
                        const size_t i2 = *vit;
                        if (CurrentFrame->getMapPoint(i2))
                            continue;

                        const cv::Mat &d = CurrentFrame->Descriptors.row(i2);

                        const int dist = DescriptorDistance(dMP, d);

                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx2 = i2;
                        }
                    }

                    if (bestDist <= ORBdist)
                    {
                        CurrentFrame->addMapPointMatch(pMP, bestIdx2);
                        nmatches++;

                        if (mbCheckOrientation)
                        {
                            float rot = pMP->angle - CurrentFrame->mvKeys[bestIdx2].angle; // pKF->mvKeys[i].angle
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }
                }
            }
        }

        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i != ind1 && i != ind2 && i != ind3)
                {
                    for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                    {
                        CurrentFrame->addMapPointMatch(nullptr, rotHist[i][j]);
                        nmatches--;
                    }
                }
            }
        }

        return nmatches;
    }

    int Matcher::searchWithEpipolar(shared_ptr<Frame> pKF1, shared_ptr<Frame> pKF2, vector<pair<size_t, size_t> > &vMatchedPairs)
    {
        const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

        //Compute epipole in second image
        Vec3 Cw = pKF1->fs->getCameraCenter();
        SE3 Kf1Pose = pKF1->fs->getPose();
        SO3 R1w = Kf1Pose.rotationMatrix();
        Vec3 t1w = Kf1Pose.translation();

        SE3 Kf2Pose = pKF2->fs->getPose();
        SO3 R2w = Kf2Pose.rotationMatrix();
        Vec3 t2w = Kf2Pose.translation();
        Vec3 C2 = R2w * Cw + t2w;

        SO3 R2wt = R2w.inverse();
        SO3 R12 = R1w * R2wt;
        Vec3 t12 = - (R1w * R2wt * t2w) + t1w;

        Mat33 t12x = Skew(t12);
       

        //compute fundamental matrix
        Mat33 K1ti = pKF1->HCalib->getInvCalibMatrix().transpose().cast<double>();
        Mat33 K2i = pKF1->HCalib->getInvCalibMatrix().cast<double>();
        Mat33f Fundamental = (K1ti * t12x * R12.matrix() * K2i).cast<float>();

        const float invz = 1.0f / C2(2);
        const float ex = pKF2->HCalib->fxl() * C2(0) * invz + pKF2->HCalib->cxl();
        const float ey = pKF2->HCalib->fyl() * C2(1) * invz + pKF2->HCalib->cyl();

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node

        int nmatches = 0;
        vector<bool> vbMatched2(pKF2->nFeatures, false);
        vector<int> vMatches12(pKF1->nFeatures, -1);

        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f / HISTO_LENGTH;

        DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while (f1it != f1end && f2it != f2end)
        {
            if (f1it->first == f2it->first)
            {
                for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];

                    shared_ptr<MapPoint> pMP1 = pKF1->getMapPoint(idx1);

                    // If there is already a MapPoint skip
                    if (pMP1)
                        continue;

                    const cv::KeyPoint &kp1 = pKF1->mvKeys[idx1];

                    const cv::Mat &d1 = pKF1->Descriptors.row(idx1);

                    int bestDist = TH_LOW;
                    int bestIdx2 = -1;

                    for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                    {
                        size_t idx2 = f2it->second[i2];

                        shared_ptr<MapPoint> pMP2 = pKF2->getMapPoint(idx2);

                        // If we have already matched or there is a MapPoint skip
                        if (vbMatched2[idx2] || pMP2)
                            continue;

                        const cv::Mat &d2 = pKF2->Descriptors.row(idx2);

                        int dist = DescriptorDistance(d1, d2);

                        if (dist > TH_LOW || dist > bestDist)
                            continue;

                        const cv::KeyPoint &kp2 = pKF2->mvKeys[idx2];

                       
                        float distex = ex - kp2.pt.x;
                        float distey = ey - kp2.pt.y;
                        if (distex * distex + distey * distey < 100 )
                                continue;
                        

                        if (CheckDistEpipolarLine(kp1, kp2, Fundamental))
                        {
                            bestIdx2 = idx2;
                            bestDist = dist;
                        }
                    }

                    if (bestIdx2 >= 0)
                    {
                        const cv::KeyPoint &kp2 = pKF2->mvKeys[bestIdx2];
                        vMatches12[idx1] = bestIdx2;
                        nmatches++;

                        if (mbCheckOrientation)
                        {
                            float rot = kp1.angle - kp2.angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if (f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    vMatches12[rotHist[i][j]] = -1;
                    nmatches--;
                }
            }
        }

        vMatchedPairs.clear();
        vMatchedPairs.reserve(nmatches);

        for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
        {
            if (vMatches12[i] < 0)
                continue;
            vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
        }

        return nmatches;
    }

    bool Matcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, Mat33f &F12)
    {
        // Epipolar line in second image l = x1'F12 = [a b c]
         float a = kp1.pt.x * F12(0, 0) + kp1.pt.y * F12(1, 0) + F12(2, 0);
        const float b = kp1.pt.x * F12(0, 1) + kp1.pt.y * F12(1, 1) + F12(2, 1);
        const float c = kp1.pt.x * F12(0, 2) + kp1.pt.y * F12(1, 2) + F12(2, 2);

        const float num = a * kp2.pt.x + b * kp2.pt.y + c;

        const float den = a * a + b * b;

        if (den == 0)
            return false;

        const float dsqr = num * num / den;

        return dsqr < 3.84;
    }



} // namespace HSLAM
