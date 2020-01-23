#include "IndirectInitializer.h"
#include "Frame.h"
#include "CalibData.h"
#include <opencv2/core.hpp>
#include "Detector.h"
#include <opencv2/core/eigen.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "Display.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
using namespace std;
namespace FSLAM
{

IndirectInitializer::IndirectInitializer(std::shared_ptr<CalibData> _Calib, std::shared_ptr<ORBDetector> _Detector, std::shared_ptr<GUI>_DisplayHandler): 
Calib(_Calib), Detector(_Detector), thisToNext_aff(0,0), displayhandler(_DisplayHandler)
{

    mSigma = 1.0;
    mSigma2 = mSigma * mSigma;
    mMaxIterations = 200; //ransac iterations
    
    JbBuffer = new Vec10f[Calib->wpyr[0]*Calib->hpyr[0]];
	JbBuffer_new = new Vec10f[Calib->wpyr[0]*Calib->hpyr[0]];
    wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
    fixAffine=true;

    randomGen = std::shared_ptr<cv::RNG>(new cv::RNG(0));

    maxIterations.push_back(200); //number of direct optimization iterations. increase the size of this according to number of pyramids used
	alphaK = 2.5*2.5;//*freeDebugParam1*freeDebugParam1;
	alphaW = 150*150;//*freeDebugParam2*freeDebugParam2;
	regWeight = 0.8;//*freeDebugParam4;
	couplingWeight = 1;//*freeDebugParam5;
    GNDirStrucOnlytMaxIter = 50;
}

IndirectInitializer::~IndirectInitializer()
{
    delete[] JbBuffer;
	delete[] JbBuffer_new;
}

bool IndirectInitializer::Initialize(std::shared_ptr<Frame> _Frame)
{
    if (Sensortype == Monocular)
    {
        if (!FirstFrame)
        {
            if (_Frame->nFeatures > 100)
            {
                FirstFrame = _Frame;
                mvbPrevMatched.resize(FirstFrame->nFeatures);
                for (size_t i = 0; i < FirstFrame->nFeatures; ++i)
                    mvbPrevMatched[i] = FirstFrame->mvKeys[i].pt;
                fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
                snapped = false;
                Frame::Globalid = 0;
	            frameID = snappedAt = 0;
            }
            return false;
        }

        //Processing Second Frame
        if (_Frame->nFeatures > 100)
        {
            SecondFrame = _Frame;
        }
        else
        {
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
            for (size_t i = 0; i < FirstFrame->nFeatures; ++i)
                    mvbPrevMatched[i] = FirstFrame->mvKeys[i].pt;
            return false;
        }
        int nmatches;
        
        nmatches = FindMatches(mvbPrevMatched, mvIniMatches, 30, 30, 0.9, true);
      
        // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
        // std::vector<std::vector<cv::DMatch>> knn_matches;
        // matcher->knnMatch(FirstFrame->Descriptors, SecondFrame->Descriptors, knn_matches, 2);
        // //-- Filter matches using the Lowe's ratio test
        // const float ratio_thresh = 0.7f;
        // std::vector<cv::DMatch> good_matches;
        // for (size_t i = 0; i < knn_matches.size(); ++i)
        // {
        //     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        //     {
        //         good_matches.push_back(knn_matches[i][0]);
        //         nmatches++;

        //     }
        // }
        // mvIniMatches = vector<int>(FirstFrame->mvKeys.size(), -1);

        // for(int i = 0; i < good_matches.size(); ++i)
        //     mvIniMatches[good_matches[i].queryIdx] = good_matches[i].trainIdx;

        

        if (ShowInitializationMatches)
        {
            std::vector<cv::DMatch> good_matches;

            for (int i = 0; i < mvIniMatches.size(); ++i)
                if (mvIniMatches[i] != -1)
                    good_matches.push_back(cv::DMatch(i, mvIniMatches[i], 1));

            cv::Mat MatchesImage;
            cv::drawMatches(FirstFrame->LeftIndPyr[0], FirstFrame->mvKeys, SecondFrame->LeftIndPyr[0], SecondFrame->mvKeys, good_matches, MatchesImage, cv::Scalar(0.0f, 255.0f, 0.0f));
            cv::namedWindow("Matches", cv::WindowFlags::WINDOW_GUI_NORMAL | cv::WindowFlags::WINDOW_KEEPRATIO);
            cv::imshow("Matches", MatchesImage);
            cv::waitKey(1);
        }

        if (nmatches < 100)
        {
            FirstFrame.reset();
            SecondFrame.reset();
            return false;
        }

        cv::Mat Rcw;                      // Current Camera Rotation
        cv::Mat tcw;                      // Current Camera Translation
        std::vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        if (FindTransformation(mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; ++i)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                {
                    mvIniMatches[i] = -1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            // mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
           
            float MedianInvDepth = 1.0f / ComputeSceneMedianDepth(2, mvIniP3D, vbTriangulated); //Initialize all untriangulated features to this inverse deph
            Tcw.col(3).rowRange(0,3) = Tcw.col(3).rowRange(0,3) * MedianInvDepth; // update baseline between first and Second Keyframes so that meandepth =1
            
            for (int i = 0; i < FirstFrame->nFeatures; ++i) // update points coordinates so that their inversedepth has a median of 1.
                if(vbTriangulated[i])
                    mvIniP3D[i] *= MedianInvDepth;
            
            FirstFrame->camToWorld = SE3();
            Mat44f Pose; cv::cv2eigen(Tcw,Pose);
            SecondFrame->camToWorld = SE3(Pose.cast<double>());
            SecondFrame->camToWorld = SecondFrame->camToWorld.inverse();
            
            
            std::vector<float> pts;
            for (int i = 0; i < FirstFrame->nFeatures; ++i)
            {
                if (mvIniMatches[i]>=0)
                {
                    pts.push_back(mvIniP3D[i].x); pts.push_back(mvIniP3D[i].y); pts.push_back(mvIniP3D[i].z);
                }
            }
            displayhandler->UploadPoints(pts);
            
            // OptimizeDirect(mvIniP3D, vbTriangulated, SecondFrame->camToWorld);
           
            // std::vector<std::shared_ptr<Pnt>> Points;
            // StructureOnlyDirectOptimization(Points, mvIniP3D, vbTriangulated, SecondFrame->camToWorld);

            // mCurrentFrame.SetPose(Tcw);

            // CreateInitialMapMonocular();
            std::cout << "found transofmration" << std::endl;
        }
    }
    else if (Sensortype == Stereo)
    {
    }
    else if (Sensortype == RGBD)
    {
    }

    return false;
}

bool IndirectInitializer::FindTransformation(const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    //  mvKeys2 = CurrentFrame.mvKeysUn;

    mvMatches12.clear();
    mvMatches12.reserve(SecondFrame->mvKeys.size());
    mvbMatched1.resize(FirstFrame->mvKeys.size());
    for (size_t i = 0, iend = vMatches12.size(); i < iend; ++i)
    {
        if (vMatches12[i] >= 0)
        {
            mvMatches12.push_back(make_pair(i, vMatches12[i]));
            mvbMatched1[i] = true;
        }
        else
            mvbMatched1[i] = false;
    }

    const int N = mvMatches12.size();

    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for (int i = 0; i < N; ++i)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    mvSets = vector<vector<size_t>>(mMaxIterations, vector<size_t>(8, 0));

    // DUtils::Random::SeedRandOnce(0);

    for (int it = 0; it < mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for (size_t j = 0; j < 8; j++)
        {
            int randi = randomGen->uniform((int)0, (int)vAvailableIndices.size() - 1);
            // int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];

            mvSets[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    float SH, SF;
    cv::Mat H, F;

    boost::thread threadH(&IndirectInitializer::FindHomography, this, boost::ref(vbMatchesInliersH), boost::ref(SH), boost::ref(H));
    // FindHomography(vbMatchesInliersH, SH, H);
    FindFundamental(vbMatchesInliersF, SF, F);

    // Wait until both threads have finished
    threadH.join();

    // Compute ratio of scores
    float RH = SH / (SH + SF);
    cv::Mat mK = Calib->GetCvK();
    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    if (RH > 0.40)
        return ReconstructH(vbMatchesInliersH, H, mK, R21, t21, vP3D, vbTriangulated, 1.0, 50);
    else
        return ReconstructF(vbMatchesInliersF, F, mK, R21, t21, vP3D, vbTriangulated, 1.0, 50);

    return false;
}

int IndirectInitializer::FindMatches(vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize, int TH_LOW, float mfNNratio, bool CheckOrientation)
{
    int nmatches = 0;
    vnMatches12 = vector<int>(FirstFrame->mvKeys.size(), -1);

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; ++i)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    vector<int> vMatchedDistance(SecondFrame->mvKeys.size(), INT_MAX);
    vector<int> vnMatches21(SecondFrame->mvKeys.size(), -1);

    for (size_t i1 = 0, iend1 = FirstFrame->mvKeys.size(); i1 < iend1; i1++)
    {
        cv::KeyPoint kp1 = FirstFrame->mvKeys[i1];
        int level1 = kp1.octave;
        if (level1 > 0)
            continue;

        vector<size_t> vIndices2 = SecondFrame->GetFeaturesInArea(vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize);
        if (vIndices2.empty())
            continue;

        cv::Mat d1 = FirstFrame->Descriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for (vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = SecondFrame->Descriptors.row(i2);

            int dist = Detector->DescriptorDistance(d1, d2);

            if (vMatchedDistance[i2] <= dist)
                continue;

            if (dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx2 = i2;
            }
            else if (dist < bestDist2)
            {
                bestDist2 = dist;
            }
        }

        if (bestDist <= TH_LOW)
        {
            if (bestDist < (float)bestDist2 * mfNNratio)
            {
                if (vnMatches21[bestIdx2] >= 0)
                {
                    vnMatches12[vnMatches21[bestIdx2]] = -1;
                    nmatches--;
                }
                vnMatches12[i1] = bestIdx2;
                vnMatches21[bestIdx2] = i1;
                vMatchedDistance[bestIdx2] = bestDist;
                nmatches++;

                if (CheckOrientation)
                {
                    float rot = FirstFrame->mvKeys[i1].angle - SecondFrame->mvKeys[bestIdx2].angle;
                    if (rot < 0.0)
                        rot += 360.0f;
                    int bin = round(rot * factor);
                    if (bin == HISTO_LENGTH)
                        bin = 0;
                    assert(bin >= 0 && bin < HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }
    }

    if (CheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        Detector->ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                int idx1 = rotHist[i][j];
                if (vnMatches12[idx1] >= 0)
                {
                    vnMatches12[idx1] = -1;
                    nmatches--;
                }
            }
        }
    }

    //Update prev matched
    for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
        if (vnMatches12[i1] >= 0)
            vbPrevMatched[i1] = SecondFrame->mvKeys[vnMatches12[i1]].pt;

    return nmatches;
}

void IndirectInitializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    const int N = mvMatches12.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(FirstFrame->mvKeys, vPn1, T1);
    Normalize(SecondFrame->mvKeys, vPn2, T2);
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N, false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N, false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for (int it = 0; it < mMaxIterations; it++)
    {
        // Select a minimum set
        for (size_t j = 0; j < 8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        cv::Mat Hn = ComputeH21(vPn1i, vPn2i);
        H21i = T2inv * Hn * T1;
        H12i = H21i.inv();

        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

        if (currentScore > score)
        {
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

void IndirectInitializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = vbMatchesInliers.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(FirstFrame->mvKeys, vPn1, T1);
    Normalize(SecondFrame->mvKeys, vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N, false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N, false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for (int it = 0; it < mMaxIterations; it++)
    {
        // Select a minimum set
        for (int j = 0; j < 8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        cv::Mat Fn = ComputeF21(vPn1i, vPn2i);

        F21i = T2t * Fn * T1;

        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        if (currentScore > score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

cv::Mat IndirectInitializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(2 * N, 9, CV_32F);

    for (int i = 0; i < N; ++i)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2 * i, 0) = 0.0;
        A.at<float>(2 * i, 1) = 0.0;
        A.at<float>(2 * i, 2) = 0.0;
        A.at<float>(2 * i, 3) = -u1;
        A.at<float>(2 * i, 4) = -v1;
        A.at<float>(2 * i, 5) = -1;
        A.at<float>(2 * i, 6) = v2 * u1;
        A.at<float>(2 * i, 7) = v2 * v1;
        A.at<float>(2 * i, 8) = v2;

        A.at<float>(2 * i + 1, 0) = u1;
        A.at<float>(2 * i + 1, 1) = v1;
        A.at<float>(2 * i + 1, 2) = 1;
        A.at<float>(2 * i + 1, 3) = 0.0;
        A.at<float>(2 * i + 1, 4) = 0.0;
        A.at<float>(2 * i + 1, 5) = 0.0;
        A.at<float>(2 * i + 1, 6) = -u2 * u1;
        A.at<float>(2 * i + 1, 7) = -u2 * v1;
        A.at<float>(2 * i + 1, 8) = -u2;
    }

    cv::Mat u, w, vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);
}

cv::Mat IndirectInitializer::ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(N, 9, CV_32F);

    for (int i = 0; i < N; ++i)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i, 0) = u2 * u1;
        A.at<float>(i, 1) = u2 * v1;
        A.at<float>(i, 2) = u2;
        A.at<float>(i, 3) = v2 * u1;
        A.at<float>(i, 4) = v2 * v1;
        A.at<float>(i, 5) = v2;
        A.at<float>(i, 6) = u1;
        A.at<float>(i, 7) = v1;
        A.at<float>(i, 8) = 1;
    }

    cv::Mat u, w, vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2) = 0;

    return u * cv::Mat::diag(w) * vt;
}

float IndirectInitializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float h11 = H21.at<float>(0, 0);
    const float h12 = H21.at<float>(0, 1);
    const float h13 = H21.at<float>(0, 2);
    const float h21 = H21.at<float>(1, 0);
    const float h22 = H21.at<float>(1, 1);
    const float h23 = H21.at<float>(1, 2);
    const float h31 = H21.at<float>(2, 0);
    const float h32 = H21.at<float>(2, 1);
    const float h33 = H21.at<float>(2, 2);

    const float h11inv = H12.at<float>(0, 0);
    const float h12inv = H12.at<float>(0, 1);
    const float h13inv = H12.at<float>(0, 2);
    const float h21inv = H12.at<float>(1, 0);
    const float h22inv = H12.at<float>(1, 1);
    const float h23inv = H12.at<float>(1, 2);
    const float h31inv = H12.at<float>(2, 0);
    const float h32inv = H12.at<float>(2, 1);
    const float h33inv = H12.at<float>(2, 2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 5.991;

    const float invSigmaSquare = 1.0 / (sigma * sigma);

    for (int i = 0; i < N; ++i)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = FirstFrame->mvKeys[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = SecondFrame->mvKeys[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2

        const float w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
        const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
        const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

        const float squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);

        const float chiSquare1 = squareDist1 * invSigmaSquare;

        if (chiSquare1 > th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1

        const float w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
        const float u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
        const float v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

        const float squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);

        const float chiSquare2 = squareDist2 * invSigmaSquare;

        if (chiSquare2 > th)
            bIn = false;
        else
            score += th - chiSquare2;

        if (bIn)
            vbMatchesInliers[i] = true;
        else
            vbMatchesInliers[i] = false;
    }

    return score;
}

float IndirectInitializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0, 0);
    const float f12 = F21.at<float>(0, 1);
    const float f13 = F21.at<float>(0, 2);
    const float f21 = F21.at<float>(1, 0);
    const float f22 = F21.at<float>(1, 1);
    const float f23 = F21.at<float>(1, 2);
    const float f31 = F21.at<float>(2, 0);
    const float f32 = F21.at<float>(2, 1);
    const float f33 = F21.at<float>(2, 2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0 / (sigma * sigma);

    for (int i = 0; i < N; ++i)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = FirstFrame->mvKeys[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = SecondFrame->mvKeys[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)

        const float a2 = f11 * u1 + f12 * v1 + f13;
        const float b2 = f21 * u1 + f22 * v1 + f23;
        const float c2 = f31 * u1 + f32 * v1 + f33;

        const float num2 = a2 * u2 + b2 * v2 + c2;

        const float squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

        const float chiSquare1 = squareDist1 * invSigmaSquare;

        if (chiSquare1 > th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11 * u2 + f21 * v2 + f31;
        const float b1 = f12 * u2 + f22 * v2 + f32;
        const float c1 = f13 * u2 + f23 * v2 + f33;

        const float num1 = a1 * u1 + b1 * v1 + c1;

        const float squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);

        const float chiSquare2 = squareDist2 * invSigmaSquare;

        if (chiSquare2 > th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if (bIn)
            vbMatchesInliers[i] = true;
        else
            vbMatchesInliers[i] = false;
    }

    return score;
}

bool IndirectInitializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                                       cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N = 0;
    for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; ++i)
        if (vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t() * F21 * K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    DecomposeE(E21, R1, R2, t);

    cv::Mat t1 = t;
    cv::Mat t2 = -t;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;
    float parallax1, parallax2, parallax3, parallax4;

    int nGood1 = CheckRT(R1, t1, FirstFrame->mvKeys, SecondFrame->mvKeys, mvMatches12, vbMatchesInliers, K, vP3D1, 4.0 * mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2, t1, FirstFrame->mvKeys, SecondFrame->mvKeys, mvMatches12, vbMatchesInliers, K, vP3D2, 4.0 * mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1, t2, FirstFrame->mvKeys, SecondFrame->mvKeys, mvMatches12, vbMatchesInliers, K, vP3D3, 4.0 * mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2, t2, FirstFrame->mvKeys, SecondFrame->mvKeys, mvMatches12, vbMatchesInliers, K, vP3D4, 4.0 * mSigma2, vbTriangulated4, parallax4);

    int maxGood = max(nGood1, max(nGood2, max(nGood3, nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = max(static_cast<int>(0.9 * N), minTriangulated);

    int nsimilar = 0;
    if (nGood1 > 0.7 * maxGood)
        nsimilar++;
    if (nGood2 > 0.7 * maxGood)
        nsimilar++;
    if (nGood3 > 0.7 * maxGood)
        nsimilar++;
    if (nGood4 > 0.7 * maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if (maxGood < nMinGood || nsimilar > 1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if (maxGood == nGood1)
    {
        if (parallax1 > minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood2)
    {
        if (parallax2 > minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood3)
    {
        if (parallax3 > minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood4)
    {
        if (parallax4 > minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

bool IndirectInitializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                                       cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N = 0;
    for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; ++i)
        if (vbMatchesInliers[i])
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    cv::Mat invK = K.inv();
    cv::Mat A = invK * H21 * K;

    cv::Mat U, w, Vt, V;
    cv::SVD::compute(A, w, U, Vt, cv::SVD::FULL_UV);
    V = Vt.t();

    float s = cv::determinant(U) * cv::determinant(Vt);

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    if (d1 / d2 < 1.00001 || d2 / d3 < 1.00001)
    {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
    float aux3 = sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
    float x1[] = {aux1, aux1, -aux1, -aux1};
    float x3[] = {aux3, -aux3, aux3, -aux3};

    //case d'=d2
    float aux_stheta = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);

    float ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    for (int i = 0; i < 4; ++i)
    {
        cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
        Rp.at<float>(0, 0) = ctheta;
        Rp.at<float>(0, 2) = -stheta[i];
        Rp.at<float>(2, 0) = stheta[i];
        Rp.at<float>(2, 2) = ctheta;

        cv::Mat R = s * U * Rp * Vt;
        vR.push_back(R);

        cv::Mat tp(3, 1, CV_32F);
        tp.at<float>(0) = x1[i];
        tp.at<float>(1) = 0;
        tp.at<float>(2) = -x3[i];
        tp *= d1 - d3;

        cv::Mat t = U * tp;
        vt.push_back(t / cv::norm(t));

        cv::Mat np(3, 1, CV_32F);
        np.at<float>(0) = x1[i];
        np.at<float>(1) = 0;
        np.at<float>(2) = x3[i];

        cv::Mat n = V * np;
        if (n.at<float>(2) < 0)
            n = -n;
        vn.push_back(n);
    }

    //case d'=-d2
    float aux_sphi = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);

    float cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for (int i = 0; i < 4; ++i)
    {
        cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
        Rp.at<float>(0, 0) = cphi;
        Rp.at<float>(0, 2) = sphi[i];
        Rp.at<float>(1, 1) = -1;
        Rp.at<float>(2, 0) = sphi[i];
        Rp.at<float>(2, 2) = -cphi;

        cv::Mat R = s * U * Rp * Vt;
        vR.push_back(R);

        cv::Mat tp(3, 1, CV_32F);
        tp.at<float>(0) = x1[i];
        tp.at<float>(1) = 0;
        tp.at<float>(2) = x3[i];
        tp *= d1 + d3;

        cv::Mat t = U * tp;
        vt.push_back(t / cv::norm(t));

        cv::Mat np(3, 1, CV_32F);
        np.at<float>(0) = x1[i];
        np.at<float>(1) = 0;
        np.at<float>(2) = x3[i];

        cv::Mat n = V * np;
        if (n.at<float>(2) < 0)
            n = -n;
        vn.push_back(n);
    }

    int bestGood = 0;
    int secondBestGood = 0;
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for (size_t i = 0; i < 8; ++i)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;
        int nGood = CheckRT(vR[i], vt[i], FirstFrame->mvKeys, SecondFrame->mvKeys, mvMatches12, vbMatchesInliers, K, vP3Di, 4.0 * mSigma2, vbTriangulatedi, parallaxi);

        if (nGood > bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if (nGood > secondBestGood)
        {
            secondBestGood = nGood;
        }
    }

    if (secondBestGood < 0.75 * bestGood && bestParallax >= minParallax && bestGood > minTriangulated && bestGood > 0.9 * N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}

void IndirectInitializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4, 4, CV_32F);

    A.row(0) = kp1.pt.x * P1.row(2) - P1.row(0);
    A.row(1) = kp1.pt.y * P1.row(2) - P1.row(1);
    A.row(2) = kp2.pt.x * P2.row(2) - P2.row(0);
    A.row(3) = kp2.pt.y * P2.row(2) - P2.row(1);

    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
}

void IndirectInitializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for (int i = 0; i < N; ++i)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX / N;
    meanY = meanY / N;

    float meanDevX = 0;
    float meanDevY = 0;

    for (int i = 0; i < N; ++i)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX / N;
    meanDevY = meanDevY / N;

    float sX = 1.0 / meanDevX;
    float sY = 1.0 / meanDevY;

    for (int i = 0; i < N; ++i)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    T = cv::Mat::eye(3, 3, CV_32F);
    T.at<float>(0, 0) = sX;
    T.at<float>(1, 1) = sY;
    T.at<float>(0, 2) = -meanX * sX;
    T.at<float>(1, 2) = -meanY * sY;
}

int IndirectInitializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                                 const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                                 const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters
    const float fx = K.at<float>(0, 0);
    const float fy = K.at<float>(1, 1);
    const float cx = K.at<float>(0, 2);
    const float cy = K.at<float>(1, 2);

    vbGood = vector<bool>(vKeys1.size(), false);
    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
    K.copyTo(P1.rowRange(0, 3).colRange(0, 3));

    cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3, 4, CV_32F);
    R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
    t.copyTo(P2.rowRange(0, 3).col(3));
    P2 = K * P2;

    cv::Mat O2 = -R.t() * t;

    int nGood = 0;

    for (size_t i = 0, iend = vMatches12.size(); i < iend; ++i)
    {
        if (!vbMatchesInliers[i])
            continue;

        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;

        Triangulate(kp1, kp2, P1, P2, p3dC1);

        if (!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first] = false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2) / (dist1 * dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if (p3dC1.at<float>(2) <= 0 && cosParallax < 0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R * p3dC1 + t;

        if (p3dC2.at<float>(2) <= 0 && cosParallax < 0.99998)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0 / p3dC1.at<float>(2);
        im1x = fx * p3dC1.at<float>(0) * invZ1 + cx;
        im1y = fy * p3dC1.at<float>(1) * invZ1 + cy;

        float squareError1 = (im1x - kp1.pt.x) * (im1x - kp1.pt.x) + (im1y - kp1.pt.y) * (im1y - kp1.pt.y);

        if (squareError1 > th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0 / p3dC2.at<float>(2);
        im2x = fx * p3dC2.at<float>(0) * invZ2 + cx;
        im2y = fy * p3dC2.at<float>(1) * invZ2 + cy;

        float squareError2 = (im2x - kp2.pt.x) * (im2x - kp2.pt.x) + (im2y - kp2.pt.y) * (im2y - kp2.pt.y);

        if (squareError2 > th2)
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
        nGood++;

        if (cosParallax < 0.99998)
            vbGood[vMatches12[i].first] = true;
    }

    if (nGood > 0)
    {
        sort(vCosParallax.begin(), vCosParallax.end());

        size_t idx = min(50, int(vCosParallax.size() - 1));
        parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
    }
    else
        parallax = 0;

    return nGood;
}

void IndirectInitializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u, w, vt;
    cv::SVD::compute(E, w, u, vt);

    u.col(2).copyTo(t);
    t = t / cv::norm(t);

    cv::Mat W(3, 3, CV_32F, cv::Scalar(0));
    W.at<float>(0, 1) = -1;
    W.at<float>(1, 0) = 1;
    W.at<float>(2, 2) = 1;

    R1 = u * W * vt;
    if (cv::determinant(R1) < 0)
        R1 = -R1;

    R2 = u * W.t() * vt;
    if (cv::determinant(R2) < 0)
        R2 = -R2;
}

bool IndirectInitializer::OptimizeDirect(std::vector<cv::Point3f> &mvIniP3D, std::vector<bool> &vbTriangulated, SE3 &thisToNext)
{
    std::vector<std::shared_ptr<Pnt>> Points;
    Points.resize(FirstFrame->nFeatures);
    for (int i = 0; i < FirstFrame->nFeatures; ++i)
    {
        std::shared_ptr<Pnt> Point = std::shared_ptr<Pnt>(new Pnt);
        Point->u = FirstFrame->mvKeys[i].pt.x + 0.1;
        Point->v = FirstFrame->mvKeys[i].pt.y + 0.1;
        Point->idepth = vbTriangulated[i] ? 1.0f / mvIniP3D[i].z : 1.0f; // if previously triangulated set it as the inv depth otherwise set it to 1.
        Point->iR = Point->idepth;//1;
        Point->isGood = true;
        Point->energy.setZero();
        Point->lastHessian = 0;
        Point->lastHessian_new = 0;
        Point->my_type = 1;
        Point->outlierTH = patternNum * setting_outlierTH;
        Points[i] = Point;
    }
    
	SE3 refToNew_current = thisToNext;
	AffLight refToNew_aff_current = thisToNext_aff;//SecondFrame->aff_g2l_internal;

    if (FirstFrame->ab_exposure > 0 && SecondFrame->ab_exposure > 0)
        refToNew_aff_current = AffLight(logf(SecondFrame->ab_exposure / FirstFrame->ab_exposure), 0); // coarse approximation.
    Vec3f latestRes = Vec3f::Zero();
   
    // for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--)
    // {
        // if (lvl < pyrLevelsUsed - 1)
        //     propagateDown(lvl + 1);

        Mat88f H, Hsc;
        Vec8f b, bsc;
        resetPoints(Points); //lvl
        Vec3f resOld = calcResAndGS(Points, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false); //lvl
        applyStep(Points); //lvl

        float lambda = 0.1;
        float eps = 1e-4;
        int fails = 0;
        int iteration = 0;
        while (true)
        {
            Mat88f Hl = H;
            for (int i = 0; i < 8; i++)
                Hl(i, i) *= (1 + lambda);
            Hl -= Hsc * (1 / (1 + lambda));
            Vec8f bl = b - bsc * (1 / (1 + lambda));

            Hl = wM * Hl * wM * (0.01f / (Calib->wpyr[0] *Calib->hpyr[0])); //w[lvl]* h[lvl]
            bl = wM * bl * (0.01f / (Calib->wpyr[0] *Calib->hpyr[0])); //w[lvl]* h[lvl]

            Vec8f inc;
            if (fixAffine)
            {
                inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6, 6>() * (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
                inc.tail<2>().setZero();
            }
            else
                inc = -(wM * (Hl.ldlt().solve(bl))); //=-H^-1 * b.

            // SE3 refToNew_new = (iteration > 10 && iteration < 25 )? SE3::exp(inc.head<6>().cast<double>()) * refToNew_current : refToNew_current;
            SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
            AffLight refToNew_aff_new = refToNew_aff_current;
            refToNew_aff_new.a += inc[6];
            refToNew_aff_new.b += inc[7];
            doStep(Points, lambda, inc); //lvl

            Mat88f H_new, Hsc_new;
            Vec8f b_new, bsc_new;
            Vec3f resNew = calcResAndGS(Points, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false); //lvl
            Vec3f regEnergy = calcEC(Points); //lvl

            float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
            float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);

            bool accept = eTotalOld > eTotalNew;

            // if (printDebug)
            // {
            //     printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
            //            lvl, iteration, lambda,
            //            (accept ? "ACCEPT" : "REJECT"),
            //            sqrtf((float)(resOld[0] / resOld[2])),
            //            sqrtf((float)(regEnergy[0] / regEnergy[2])),
            //            sqrtf((float)(resOld[1] / resOld[2])),
            //            sqrtf((float)(resNew[0] / resNew[2])),
            //            sqrtf((float)(regEnergy[1] / regEnergy[2])),
            //            sqrtf((float)(resNew[1] / resNew[2])),
            //            eTotalOld / resNew[2],
            //            eTotalNew / resNew[2],
            //            inc.norm());
            //     std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() << "\n";
            // }

            if (accept)
            {

                if (resNew[1] == alphaK * FirstFrame->nFeatures) //numPoints[lvl]
                    snapped = true;
                H = H_new;
                b = b_new;
                Hsc = Hsc_new;
                bsc = bsc_new;
                resOld = resNew;
                refToNew_aff_current = refToNew_aff_new;
                refToNew_current = refToNew_new;
                applyStep(Points); //lvl
                // optReg(Points); //lvl
                lambda *= 0.5;
                fails = 0;
                if (lambda < 0.0001)
                    lambda = 0.0001;
            }
            else
            {
                fails++;
                lambda *= 4;
                if (lambda > 10000)
                    lambda = 10000;
            }

            bool quitOpt = false;

            if ( !(inc.norm() > eps) || iteration >= maxIterations[0] || fails >= 4)  //maxIterations[lvl] 
            {
                Mat88f H, Hsc;
                Vec8f b, bsc;

                quitOpt = true;
            }

            if (quitOpt)
                break;
            iteration++;
        }
        latestRes = resOld;
    // }

    thisToNext = refToNew_current;
    SecondFrame->aff_g2l_internal = refToNew_aff_current;

    // for (int i = 0; i < pyrLevelsUsed - 1; i++)
    //     propagateUp(i);

    //UPDATE DEPTH DATA PER POINT:
    // for(int i=0;i<FirstFrame->nFeatures; ++i)
	// {
	// 	if(!Points[i]->isGood) continue;
	// 	Points[i]->iR += Points[i]->iR * Points[i]->lastHessian;
	// }
    // frameID++;
    // if (!snapped)
    //     snappedAt = 0;

    // if (snapped && snappedAt == 0)
    //     snappedAt = frameID;

    // debugPlot(0, wraps);
    // for (int i = 0 ; i < Points.size(); ++i)
    //     if(Points[i]->isGood)
    //         std::cout<<"Point: "<<i <<" depth= "<<Points[i]->idepth<<std::endl;
    
    if(settings_show_InitDepth)
        debugPlot(Points);

    std::vector<float> pts;
    for (int i = 0; i < FirstFrame->nFeatures; ++i)
    {
        if(Points[i]->isGood)
        {
            Vec3 Pt; Pt << (double)((Points[i]->u - Calib->cxl()) * Calib->fxli()/Points[i]->idepth), (double)((Points[i]->v - Calib->cyl()) * Calib->fyli()/Points[i]->idepth), (double)(1.0/Points[i]->idepth);
            Vec3 Pose = (SecondFrame->camToWorld.rotationMatrix().inverse() * Pt + SecondFrame->camToWorld.inverse().translation());
            // Vec3 Trans = SecondFrame->camToWorld.inverse().translation().transpose();//  translation().inverse();
            pts.push_back(Pose[0]); pts.push_back(Pose[1]); pts.push_back(Pose[2]);
        }
    }

    displayhandler->UploadPoints(pts);
    return snapped; //&& frameID > snappedAt + 5;
}

void IndirectInitializer::StructureOnlyDirectOptimization(std::vector<std::shared_ptr<Pnt>>& Points, std::vector<cv::Point3f> &mvIniP3D, std::vector<bool> &vbTriangulated, SE3 &thisToNext)
{
    bool print = false;
    Points.resize(FirstFrame->nFeatures);
    for (int i = 0; i < FirstFrame->nFeatures; ++i)
    {
        std::shared_ptr<Pnt> Point = std::shared_ptr<Pnt>(new Pnt);
        Point->u = FirstFrame->mvKeys[i].pt.x + 0.1;
        Point->v = FirstFrame->mvKeys[i].pt.y + 0.1;
        Point->idepth = vbTriangulated[i] ? 1.0f / mvIniP3D[i].z : 1.0f; // if previously triangulated set it as the inv depth otherwise set it to 1.
        Point->iR = Point->idepth;                                       //1;
        Point->isGood = true;
        Point->energy.setZero();
        Point->lastHessian = 0;
        Point->lastHessian_new = 0;
        Point->my_type = 1;
        Point->outlierTH = patternNum * setting_outlierTH;
        Points[i] = Point;

        Points[i]->Residual.state_NewEnergy = Points[i]->Residual.state_energy = 0;
        Points[i]->Residual.state_NewState = ResState::OUTLIER;
        Points[i]->Residual.state_state = ResState::IN;
        Points[i]->Residual.target = SecondFrame;

        for (int idx = 0; idx < patternNum; idx++)
        {
            int dx = patternP[idx][0];
            int dy = patternP[idx][1];
            Points[i]->color[idx] = getInterpolatedElement31(FirstFrame->LeftDirPyr[0], Points[i]->u + dx, Points[i]->v + dy, Calib->wpyr[0]);
        }
            

        float lastEnergy = 0;
        float lastHdd = 0;
        float lastbd = 0;
        float currentIdepth = Points[i]->idepth;

        lastEnergy += linearizeResidual(Points[i], 1000, lastHdd, lastbd, currentIdepth);
        Points[i]->Residual.state_state = Points[i]->Residual.state_NewState;
        Points[i]->Residual.state_energy = Points[i]->Residual.state_NewEnergy;

        if (!std::isfinite(lastEnergy))
            continue;

        float lambda = 0.1;
        for (int iteration = 0; iteration < GNDirStrucOnlytMaxIter; iteration++)
        {
            float H = lastHdd;
            H *= 1 + lambda;
            float step = (1.0 / H) * lastbd;
            float newIdepth = currentIdepth - step;

            float newHdd = 0;
            float newbd = 0;
            float newEnergy = 0;
            newEnergy += linearizeResidual(Points[i], 1, newHdd, newbd, newIdepth);
            // newEnergy += point->linearizeResidual(&Hcalib, 1, residuals + i, newHdd, newbd, newIdepth);

            if (!std::isfinite(lastEnergy) || newHdd < 100) //consider removing this threshold
            {
                if (print)
                    printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
                           2,
                           newHdd,
                           lastEnergy);
                continue;
            }

            if (print)
                printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
                       (true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
                       iteration,
                       log10(lambda),
                       "",
                       lastEnergy, newEnergy, newIdepth);

            if (newEnergy < lastEnergy)
            {
                currentIdepth = newIdepth;
                lastHdd = newHdd;
                lastbd = newbd;
                lastEnergy = newEnergy;
                Points[i]->Residual.state_state = Points[i]->Residual.state_NewState;
                Points[i]->Residual.state_energy = Points[i]->Residual.state_NewEnergy;                
                lambda *= 0.5;
            }
            else
            {
                lambda *= 5;
            }

            if (fabsf(step) < 0.0001 * currentIdepth)
                break;
        }

    // bool print = false;//rand()%50==0;

	
	if(!std::isfinite(currentIdepth) || Points[i]->Residual.state_state !=  ResState::IN)
	{
        Points[i]->isGood = false;
        continue;
	}
    Points[i]->idepth = currentIdepth;
    Points[i]->idepth_new = currentIdepth;
    Points[i]->iR = currentIdepth;

    }
    if(settings_show_InitDepth)
        debugPlot(Points);

    std::vector<float> pts;
    for (int i = 0; i < FirstFrame->nFeatures; ++i)
    {
        if(Points[i]->isGood)
        {
            Vec3 Pt; Pt << (double)((Points[i]->u - Calib->cxl()) * Calib->fxli()/Points[i]->idepth), (double)((Points[i]->v - Calib->cyl()) * Calib->fyli()/Points[i]->idepth), (double)(1.0/Points[i]->idepth);
            // Vec3 Pose = (SecondFrame->camToWorld.rotationMatrix().inverse() * Pt);
            // Vec3 Trans = SecondFrame->camToWorld.inverse().translation().transpose();//  translation().inverse();
            pts.push_back(Pt[0]); pts.push_back(Pt[1]); pts.push_back(Pt[2]);
        }
    }

    displayhandler->UploadPoints(pts);
	return;
}

double IndirectInitializer::linearizeResidual(std::shared_ptr<Pnt> Point, const float outlierTHSlack, float &Hdd, float &bd, float idepth)
{
	if (Point->Residual.state_state == ResState::OOB)
	{
		Point->Residual.state_NewState = ResState::OOB;
		return Point->Residual.state_energy;
	}

    float energyTH = patternNum*setting_outlierTH;

    SE3 leftToLeft_0 = SecondFrame->camToWorld.inverse() * FirstFrame->camToWorld.inverse();
	Mat33f PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
	Vec3f PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();



	// SE3 leftToLeft = SecondFrame->PRE_worldToCam * FirstFrame->PRE_camToWorld;
    // SE3 leftToLeft = SecondFrame->camToWorld*SecondFrame->camToWorld.inverse() * FirstFrame->camToWorld * FirstFrame->camToWorld.inverse();
    SE3 leftToLeft = SE3();//SecondFrame->camToWorld.inverse();
	Mat33f PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
	Vec3f PRE_tTll = (leftToLeft.translation()).cast<float>();
	float distanceLL = leftToLeft.translation().norm();


	Mat33f K = Mat33f::Zero();
	K(0,0) = Calib->fxl();
	K(1,1) = Calib->fyl();
	K(0,2) = Calib->cxl();
	K(1,2) = Calib->cyl();
	K(2,2) = 1;
	Mat33f PRE_KRKiTll = K * PRE_RTll * K.inverse();
	Mat33f PRE_RKiTll = PRE_RTll * K.inverse();
	Vec3f PRE_KtTll = K * PRE_tTll;


	Vec2f PRE_aff_mode = AffLight::fromToVecExposure(FirstFrame->ab_exposure, SecondFrame->ab_exposure, FirstFrame->aff_g2l(), SecondFrame->aff_g2l()).cast<float>();
	float PRE_b0_mode = FirstFrame->aff_g2l_0().b;
	

	// check OOB due to scale angle change.

	float energyLeft = 0;
	// const std::vector<Vec3f> *dIl =  &tmpRes->target.lock()->LeftDirPyr[0];
	//const Eigen::Vector3f *dIl = tmpRes->target.lock()->dI;
	// const Mat33f &PRE_RTll = precalc->PRE_RTll;
	// const Vec3f &PRE_tTll = precalc->PRE_tTll;

	Vec2f affLL = PRE_aff_mode;

	for (int idx = 0; idx < patternNum; idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		if (!projectPoint(Point->u, Point->v, idepth, dx, dy, Calib, PRE_RTll, PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth))
		{
			Point->Residual.state_NewState = ResState::OOB;
			return Point->Residual.state_energy;
		}

		Vec3f hitColor = (getInterpolatedElement33(SecondFrame->LeftDirPyr[0], Ku, Kv, Calib->wpyr[0]));

		if (!std::isfinite((float)hitColor[0]))
		{
			Point->Residual.state_NewState = ResState::OOB;
			return Point->Residual.state_energy;
		}
		float residual = hitColor[0] - (affLL[0] * Point->color[idx] + affLL[1]);

		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += Point->weights[idx] * Point->weights[idx] * hw * residual * residual * (2 - hw);

		// depth derivatives.
		float dxInterp = hitColor[1] * Calib->fxl();
		float dyInterp = hitColor[2] * Calib->fyl();
        float d_idepth = dxInterp * drescale * (PRE_tTll[0] - PRE_tTll[2] * u) + dyInterp * drescale * (PRE_tTll[1] - PRE_tTll[2] * v);
		// float d_idepth = derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale);

		hw *= Point->weights[idx] * Point->weights[idx];

		Hdd += (hw * d_idepth) * d_idepth;
		bd += (hw * residual) * d_idepth;
	}

	if (energyLeft > energyTH * outlierTHSlack)
	{
		energyLeft = energyTH * outlierTHSlack;
		Point->Residual.state_NewState = ResState::OUTLIER;
	}
	else
	{
		Point->Residual.state_NewState = ResState::IN;
	}

	Point->Residual.state_NewEnergy = energyLeft;
	return energyLeft;
}

EIGEN_STRONG_INLINE bool IndirectInitializer::projectPoint(const float &u_pt, const float &v_pt, const float &idepth, const int &dx, const int &dy,
													 std::shared_ptr<CalibData> const &Calib, const Mat33f &R, const Vec3f &t, float &drescale,
													 float &u, float &v, float &Ku, float &Kv, Vec3f &KliP, float &new_idepth)
{
	KliP = Vec3f((u_pt + dx - Calib->cxl()) * Calib->fxli(), (v_pt + dy - Calib->cyl()) * Calib->fyli(), 1);

	Vec3f ptp = R * KliP + t * idepth;
	drescale = 1.0f / ptp[2];
	new_idepth = idepth * drescale;

	if (!(drescale > 0))
		return false;

	u = ptp[0] * drescale;
	v = ptp[1] * drescale;
	Ku = u * Calib->fxl() + Calib->cxl();
	Kv = v * Calib->fyl() + Calib->cyl();

	return Ku > 1.1f && Kv > 1.1f && Ku < (Calib->wpyr[0] - 3) && Kv < (Calib->hpyr[0] - 3);
}

void IndirectInitializer::resetPoints(std::vector<std::shared_ptr<Pnt>> &Points) //int lvl
{
    // Pnt* pts = points[lvl];
    int npts = FirstFrame->nFeatures; // numPoints[lvl];
    for (int i = 0; i < npts; i++)
    {
        Points[i]->energy.setZero();
        Points[i]->idepth_new = Points[i]->idepth;

        // if(lvl==pyrLevelsUsed-1 && !pts[i].isGood)
        // {
        // 	float snd=0, sn=0;
        // 	for(int n = 0;n<10;n++)
        // 	{
        // 		if(pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue;
        // 		snd += pts[pts[i].neighbours[n]].iR;
        // 		sn += 1;
        // 	}

        // 	if(sn > 0)
        // 	{
        // 		pts[i].isGood=true;
        // 		pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd/sn;
        // 	}
        // }
    }
}

Vec3f IndirectInitializer::calcResAndGS( std::vector<std::shared_ptr<Pnt>>& Points, Mat88f &H_out, Vec8f &b_out, Mat88f &H_out_sc, Vec8f &b_out_sc, const SE3 &refToNew, AffLight refToNew_aff, bool plot)
{
    int wl = Calib->wpyr[0], hl = Calib->hpyr[0]; //w[lvl], h[lvl]

    // Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];
    // Eigen::Vector3f *colorNew = newFrame->dIp[lvl];
    
    Mat33f RKi = (refToNew.rotationMatrix() * Calib->pyrKi[0].cast<double>()).cast<float>(); //ki[lvl]
    Vec3f t = refToNew.translation().cast<float>();
    Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

    float fxl = Calib->pyrfx[0];//fx[lvl];
    float fyl = Calib->pyrfy[0];//fy[lvl];
    float cxl = Calib->pyrcx[0];//cx[lvl];
    float cyl = Calib->pyrcy[0];//cy[lvl];

    Accumulator11 E;
    acc9.initialize();
    E.initialize();

    int npts = FirstFrame->nFeatures;//numPoints[lvl];
    // Pnt *ptsl = points[lvl];
    for (int i = 0; i < npts; i++)
    {

        // Pnt *point = ptsl + i;

        Points[i]->maxstep = 1e10;
        if (!Points[i]->isGood)
        {
            E.updateSingle((float)(Points[i]->energy[0]));
            Points[i]->energy_new = Points[i]->energy;
            Points[i]->isGood_new = false;
            continue;
        }

        VecNRf dp0;
        VecNRf dp1;
        VecNRf dp2;
        VecNRf dp3;
        VecNRf dp4;
        VecNRf dp5;
        VecNRf dp6;
        VecNRf dp7;
        VecNRf dd;
        VecNRf r;
        JbBuffer_new[i].setZero();

        // sum over all residuals.
        bool isGood = true;
        float energy = 0;
        for (int idx = 0; idx < patternNum; idx++)
        {
            int dx = patternP[idx][0];
            int dy = patternP[idx][1];

            Vec3f pt = RKi * Vec3f(Points[i]->u + dx, Points[i]->v + dy, 1) + t * Points[i]->idepth_new;
            float u = pt[0] / pt[2];
            float v = pt[1] / pt[2];
            float Ku = fxl * u + cxl;
            float Kv = fyl * v + cyl;
            float new_idepth = Points[i]->idepth_new / pt[2];

            if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && new_idepth > 0))
            {
                isGood = false;
                break;
            }

            Vec3f hitColor = getInterpolatedElement33(SecondFrame->LeftDirPyr[0], Ku, Kv, wl); //colorNew
            //Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

            //float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
            float rlR = getInterpolatedElement31(FirstFrame->LeftDirPyr[0], Points[i]->u + dx, Points[i]->v + dy, wl);

            if (!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
            {
                isGood = false;
                break;
            }

            float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
            energy += hw * residual * residual * (2 - hw);

            float dxdd = (t[0] - t[2] * u) / pt[2];
            float dydd = (t[1] - t[2] * v) / pt[2];

            if (hw < 1)
                hw = sqrtf(hw);
            float dxInterp = hw * hitColor[1] * fxl;
            float dyInterp = hw * hitColor[2] * fyl;
            dp0[idx] = new_idepth * dxInterp;
            dp1[idx] = new_idepth * dyInterp;
            dp2[idx] = -new_idepth * (u * dxInterp + v * dyInterp);
            dp3[idx] = -u * v * dxInterp - (1 + v * v) * dyInterp;
            dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp;
            dp5[idx] = -v * dxInterp + u * dyInterp;
            dp6[idx] = -hw * r2new_aff[0] * rlR;
            dp7[idx] = -hw * 1;
            dd[idx] = dxInterp * dxdd + dyInterp * dydd;
            r[idx] = hw * residual;

            float maxstep = 1.0f / Vec2f(dxdd * fxl, dydd * fyl).norm();
            if (maxstep < Points[i]->maxstep)
                Points[i]->maxstep = maxstep;

            // immediately compute dp*dd' and dd*dd' in JbBuffer1.
            JbBuffer_new[i][0] += dp0[idx] * dd[idx];
            JbBuffer_new[i][1] += dp1[idx] * dd[idx];
            JbBuffer_new[i][2] += dp2[idx] * dd[idx];
            JbBuffer_new[i][3] += dp3[idx] * dd[idx];
            JbBuffer_new[i][4] += dp4[idx] * dd[idx];
            JbBuffer_new[i][5] += dp5[idx] * dd[idx];
            JbBuffer_new[i][6] += dp6[idx] * dd[idx];
            JbBuffer_new[i][7] += dp7[idx] * dd[idx];
            JbBuffer_new[i][8] += r[idx] * dd[idx];
            JbBuffer_new[i][9] += dd[idx] * dd[idx];
        }

        if (!isGood || energy > Points[i]->outlierTH * 20)
        {
            E.updateSingle((float)(Points[i]->energy[0]));
            Points[i]->isGood_new = false;
            Points[i]->energy_new = Points[i]->energy;
            continue;
        }

        // add into energy.
        E.updateSingle(energy);
        Points[i]->isGood_new = true;
        Points[i]->energy_new[0] = energy;

        // update Hessian matrix.
        for (int i = 0; i + 3 < patternNum; i += 4)
            acc9.updateSSE(
                _mm_load_ps(((float *)(&dp0)) + i),
                _mm_load_ps(((float *)(&dp1)) + i),
                _mm_load_ps(((float *)(&dp2)) + i),
                _mm_load_ps(((float *)(&dp3)) + i),
                _mm_load_ps(((float *)(&dp4)) + i),
                _mm_load_ps(((float *)(&dp5)) + i),
                _mm_load_ps(((float *)(&dp6)) + i),
                _mm_load_ps(((float *)(&dp7)) + i),
                _mm_load_ps(((float *)(&r)) + i));

        for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
            acc9.updateSingle(
                (float)dp0[i], (float)dp1[i], (float)dp2[i], (float)dp3[i],
                (float)dp4[i], (float)dp5[i], (float)dp6[i], (float)dp7[i],
                (float)r[i]);
    }

    E.finish();
    acc9.finish();

    // calculate alpha energy, and decide if we cap it.
    Accumulator11 EAlpha;
    EAlpha.initialize();
    for (int i = 0; i < npts; i++)
    {
        // Pnt *point = ptsl + i;
        if (!Points[i]->isGood_new)
        {
            E.updateSingle((float)(Points[i]->energy[1]));
        }
        else
        {
            Points[i]->energy_new[1] = (Points[i]->idepth_new - 1) * (Points[i]->idepth_new - 1);
            E.updateSingle((float)(Points[i]->energy_new[1]));
        }
    }
    EAlpha.finish();
    float alphaEnergy = alphaW * (EAlpha.A + refToNew.translation().squaredNorm() * npts);

    //printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);

    // compute alpha opt.
    float alphaOpt;
    if (alphaEnergy > alphaK * npts)
    {
        alphaOpt = 0;
        alphaEnergy = alphaK * npts;
    }
    else
    {
        alphaOpt = alphaW;
    }

    acc9SC.initialize();
    for (int i = 0; i < npts; i++)
    {
        // Pnt *point = ptsl + i;
        if (!Points[i]->isGood_new)
            continue;

        Points[i]->lastHessian_new = JbBuffer_new[i][9];

        JbBuffer_new[i][8] += alphaOpt * (Points[i]->idepth_new - 1);
        JbBuffer_new[i][9] += alphaOpt;

        if (alphaOpt == 0)
        {
            JbBuffer_new[i][8] += couplingWeight * (Points[i]->idepth_new - Points[i]->iR);
            JbBuffer_new[i][9] += couplingWeight;
        }

        JbBuffer_new[i][9] = 1 / (1 + JbBuffer_new[i][9]);
        acc9SC.updateSingleWeighted(
            (float)JbBuffer_new[i][0], (float)JbBuffer_new[i][1], (float)JbBuffer_new[i][2], (float)JbBuffer_new[i][3],
            (float)JbBuffer_new[i][4], (float)JbBuffer_new[i][5], (float)JbBuffer_new[i][6], (float)JbBuffer_new[i][7],
            (float)JbBuffer_new[i][8], (float)JbBuffer_new[i][9]);
    }
    acc9SC.finish();

    //printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
    H_out = acc9.H.topLeftCorner<8, 8>();       // / acc9.num;
    b_out = acc9.H.topRightCorner<8, 1>();      // / acc9.num;
    H_out_sc = acc9SC.H.topLeftCorner<8, 8>();  // / acc9.num;
    b_out_sc = acc9SC.H.topRightCorner<8, 1>(); // / acc9.num;

    H_out(0, 0) += alphaOpt * npts;
    H_out(1, 1) += alphaOpt * npts;
    H_out(2, 2) += alphaOpt * npts;

    Vec3f tlog = refToNew.log().head<3>().cast<float>();
    b_out[0] += tlog[0] * alphaOpt * npts;
    b_out[1] += tlog[1] * alphaOpt * npts;
    b_out[2] += tlog[2] * alphaOpt * npts;

    return Vec3f(E.A, alphaEnergy, E.num);
}

Vec3f IndirectInitializer::calcEC(std::vector<std::shared_ptr<Pnt>>& Points) //int lvl 
{
	// if(!snapped) return Vec3f(0,0,numPoints[lvl]);
    if(!snapped) return Vec3f(0,0, FirstFrame->nFeatures);

	AccumulatorX<2> E;
	E.initialize();
	int npts = FirstFrame->nFeatures; //numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		// Pnt* point = points[lvl]+i;
		if(!Points[i]->isGood_new) continue;
		float rOld = (Points[i]->idepth-Points[i]->iR);
		float rNew = (Points[i]->idepth_new-Points[i]->iR);
		E.updateNoWeight(Vec2f(rOld*rOld,rNew*rNew));

		//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
	}
	E.finish();

	//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
	return Vec3f(couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], E.num);
}
void IndirectInitializer::doStep(std::vector<std::shared_ptr<Pnt>>& Points, float lambda, Vec8f inc) //int lvl
{

	const float maxPixelStep = 5.0f;//0.25;
	const float idMaxStep = 1e10;
	// Pnt* pts = points[lvl];
	int npts = FirstFrame->nFeatures ;//numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!Points[i]->isGood) continue;


		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
		float step = - b * JbBuffer[i][9] / (1+lambda);


		float maxstep = maxPixelStep*Points[i]->maxstep;
		if(maxstep > idMaxStep) maxstep=idMaxStep;

		if(step >  maxstep) step = maxstep;
		if(step < -maxstep) step = -maxstep;

		float newIdepth = Points[i]->idepth + step;
		if(newIdepth < 1e-3 ) newIdepth = 1e-3;
		if(newIdepth > 50) newIdepth = 50;
		Points[i]->idepth_new = newIdepth;
	}

}
void IndirectInitializer::applyStep(std::vector<std::shared_ptr<Pnt>>& Points) //int lvl
{
	// Pnt* pts = points[lvl];
	int npts = FirstFrame->nFeatures; //numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!Points[i]->isGood)
		{
			Points[i]->idepth = Points[i]->idepth_new = Points[i]->iR;
			continue;
		}
		Points[i]->energy = Points[i]->energy_new;
		Points[i]->isGood = Points[i]->isGood_new;
		Points[i]->idepth = Points[i]->idepth_new;
		Points[i]->lastHessian = Points[i]->lastHessian_new;
	}
	std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

void IndirectInitializer::optReg(std::vector<std::shared_ptr<Pnt>> & Points) //int lvl 
{
	int npts = FirstFrame->nFeatures;//numPoints[lvl];
	// Pnt* ptsl = points[lvl];
	if(!snapped)
	{
		for(int i=0;i<npts;i++)
			Points[i]->iR = 1;
		return;
	}

    //KNN tree update
	// for(int i=0;i<npts;i++) 
	// {
	// 	Pnt* point = ptsl+i;
	// 	if(!point->isGood) continue;

	// 	float idnn[10];
	// 	int nnn=0;
	// 	for(int j=0;j<10;j++)
	// 	{
	// 		if(point->neighbours[j] == -1) continue;
	// 		Pnt* other = ptsl+point->neighbours[j];
	// 		if(!other->isGood) continue;
	// 		idnn[nnn] = other->iR;
	// 		nnn++;
	// 	}

	// 	if(nnn > 2)
	// 	{
	// 		std::nth_element(idnn,idnn+nnn/2,idnn+nnn);
	// 		point->iR = (1-regWeight)*point->idepth + regWeight*idnn[nnn/2];
	// 	}
	// }

}

void IndirectInitializer::debugPlot(std::vector<std::shared_ptr<Pnt>>&Points)
{
	int wl = Calib->wpyr[0], hl = Calib->hpyr[0];
	// Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
    FirstFrame->LeftDirPyr[0];
	// MinimalImageB3 iRImg(wl,hl);
    cv::Mat Depth; 
    cv::cvtColor(FirstFrame->LeftIndPyr[0], Depth, CV_GRAY2RGB);

	int npts = FirstFrame->nFeatures;

	float nid = 0, sid=0;
	for(int i=0;i<npts;i++)
	{
		// Pnt* point = points[lvl]+i;
		if(Points[i]->isGood)
		{
			nid++;
			sid += (Points[i]->idepth); //iR
		}
	}
	float fac = nid / sid;

	for(int i=0;i<npts;i++)
	{
		// Pnt* point = points[lvl]+i;
        Vec3b Color = Vec3b(0,0,0);
        if(Points[i]->isGood)
        {
            Color = makeRainbow3B(Points[i]->idepth *fac);
        }
        setPixel9(Depth, std::floor(Points[i]->v + 0.5f), std::floor(Points[i]->u + 0.5f), Color);
    }

    cv::namedWindow("InitDepthTest", cv::WINDOW_KEEPRATIO);
    cv::imshow("InitDepthTest", Depth);
    cv::waitKey(1);
}

float IndirectInitializer::ComputeSceneMedianDepth(const int q, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated)
{
    std::vector<float> vDepths;
    vDepths.reserve(FirstFrame->nFeatures);
   
    for(int i=0; i< FirstFrame->nFeatures; ++i)
    {
        if(vbTriangulated[i])
            vDepths.push_back(vP3D[i].z);
    }

    std::sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

} // namespace FSLAM