#include "Initializer.h"
#include "Frame.h"
#include "CalibData.h"
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include "IndexThreadReduce.h"
#include "boost/thread/mutex.hpp"
#include "Display.h"

#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

namespace FSLAM
{

Initializer::Initializer(std::shared_ptr<CalibData> _Calib, std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft, std::shared_ptr<GUI> _DisplayHandler) :
                                            thPool(FrontEndThreadPoolLeft), Calib(_Calib), displayhandler(_DisplayHandler)
{
    mSigma = 1.0;
    mSigma2 = mSigma * mSigma;
    mMaxIterations = 200; //ransac iterations
    randomGen = std::shared_ptr<cv::RNG>(new cv::RNG(0));
}

bool Initializer::Initialize(std::shared_ptr<Frame> _Frame)
{
    static std::vector<int> NumMatches;
    if (Sensortype == Monocular)
    {
        static int NumFails = 0;
        if (NumFails > 40)
        {
            FirstFrame.reset();
            NumFails = 0;
        }

        if (!FirstFrame)
        {
            if (_Frame->nFeatures > 100)
            {
                FirstFrame = _Frame;
                FirstFrame->IndPyr[0].copyTo(TransitImage);

                FirstFramePts.clear();
                FirstFramePts.reserve(FirstFrame->nFeatures);
                ColorVec.clear();
                ColorVec.reserve(FirstFrame->nFeatures);

                for (size_t i = 0; i < FirstFrame->nFeatures; ++i)
                {
                    FirstFramePts.push_back(FirstFrame->mvKeys[i].pt);
                    ColorVec.push_back(Scalar(randomGen->uniform((int)0, (int)255), randomGen->uniform((int)0, (int)255), randomGen->uniform((int)0, (int)255)));
                }

                mvbPrevMatched = FirstFramePts;

                if (!MatchedPts.empty())
                {
                    MatchedPts.clear();
                    MatchedPts.shrink_to_fit();
                }
                Frame::Globalid = 1;

                // //Testing IndirectMatcher
                // mvbIndPrevMatched.resize(FirstFrame->nFeatures);
                // for (size_t i = 0; i < FirstFrame->nFeatures; ++i)
                //     mvbIndPrevMatched[i] = FirstFrame->mvKeys[i].pt;
                // fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                // //end of testing
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
            FirstFrame.reset();
            return false;
        }


        // //Testing IndirectMatcher:
        // int indnmatches = MatchIndirect(mvbIndPrevMatched, mvIniMatches, 30, 30, 0.9, true);
        // NumMatches.push_back(indnmatches);

        // cout<<"average Numb Matches: "<< accumulate( NumMatches.begin(), NumMatches.end(), 0.0)/ NumMatches.size()<<endl;
        // if (true)
        // {
        //     std::vector<cv::DMatch> good_matches;

        //     for (int i = 0; i < mvIniMatches.size(); ++i)
        //         if (mvIniMatches[i] != -1)
        //             good_matches.push_back(cv::DMatch(i, mvIniMatches[i], 1));

        //     cv::Mat MatchesImage;
        //     cv::drawMatches(FirstFrame->IndPyr[0], FirstFrame->mvKeys, SecondFrame->IndPyr[0], SecondFrame->mvKeys, good_matches, MatchesImage, cv::Scalar(0.0f, 255.0f, 0.0f));
        //     cv::namedWindow("Matches", cv::WindowFlags::WINDOW_GUI_NORMAL | cv::WindowFlags::WINDOW_KEEPRATIO);
        //     cv::imshow("Matches", MatchesImage);
        //     cv::waitKey(1);
        // }
        // //endofTesting


        int nmatches = FindMatches(15, 7);

        if (nmatches < 0.2f * ((float)FirstFrame->nFeatures))
        {
            FirstFrame.reset();
            return false;
        }

        if(ComputeMeanOpticalFlow(MatchedPtsBkp, MatchedPts) < 2.0f) //camera might be stationary! don't attempt to initialize. but dont reset
            return false;

        cv::Mat Rcw;
        cv::Mat tcw;

        std::vector<bool> vbTriangulated;

        if (FindTransformation(Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            NumFails = 0;

            for (int i = 0, iend = vbTriangulated.size(); i < iend; ++i)
                if (!vbTriangulated[i])
                    nmatches--;

            if (nmatches < 0.1 * FirstFrame->nFeatures || nmatches < 150) //did not triangulate enough repeat the process!
            {
                FirstFrame.reset();
                return false;
            }

            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            float MedianInvDepth = 1.0f / ComputeSceneMedianDepth(2, mvIniP3D, vbTriangulated); //Initialize all untriangulated features to this inverse deph

            Tcw.col(3).rowRange(0, 3) = Tcw.col(3).rowRange(0, 3) * MedianInvDepth; // update baseline between first and Second Keyframes so that meandepth =1
            for (int i = 0; i < FirstFrame->nFeatures; ++i)                         // update points coordinates so that their inversedepth has a median of 1.
                if (vbTriangulated[i])
                    mvIniP3D[i] *= MedianInvDepth;

            FirstFrame->Globalid = 0;
            FirstFrame->camToWorld = SE3();
            Mat44f Pose;
            cv::cv2eigen(Tcw, Pose);
            SecondFrame->camToWorld = SE3(Pose.cast<double>());
            SecondFrame->camToWorld = SecondFrame->camToWorld.inverse();
            std::vector<float> pts;
            for (int i = 0; i < FirstFrame->nFeatures; ++i)
            {
                if (vbTriangulated[i])
                {
                    pts.push_back(mvIniP3D[i].x);
                    pts.push_back(mvIniP3D[i].y);
                    pts.push_back(mvIniP3D[i].z);
                }
            }
            displayhandler->UploadPoints(pts);
            std::cout << "found Initialization with " << nmatches <<" points"<< std::endl;
            // return true;
        }
        else
            NumFails++;
    }
    else if (Sensortype == Stereo)
    {
    }
    else if (Sensortype == RGBD)
    {
    }

    return false;
}

int Initializer::MatchIndirect(vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize, int TH_LOW, float mfNNratio, bool CheckOrientation)
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

            int dist = DescriptorDistance(d1, d2);

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

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

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

int Initializer::FindMatches(int windowSize, int maxL1Error)
{
    std::vector<unsigned char> status;
    std::vector<float> err;

    int flowFlag = (MatchedPts.empty() ? 0 : cv::OPTFLOW_USE_INITIAL_FLOW);
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 30, 0.01);
    
    MatchedPtsBkp = MatchedPts;
    if(MatchedPtsBkp.empty())
        MatchedPtsBkp = mvbPrevMatched;
    calcOpticalFlowPyrLK(TransitImage, SecondFrame->IndPyr[0], mvbPrevMatched, MatchedPts, status, err, Size(windowSize, windowSize), 2, criteria, flowFlag);

    MatchedStatus.clear();

    int count = 0;
    for (uint i = 0, iend = mvbPrevMatched.size(); i < iend; ++i)
    {
        if (status[i] && err[i] < maxL1Error) // Select good points
        {
            count++;
            MatchedStatus.push_back(1);
            // mvbPrevMatched[i] = MatchedPts[i];
        }
        else
        {
            mvbPrevMatched[i] = Point2f(NAN, NAN);
            MatchedStatus.push_back(0);
        }
    }

    
    SecondFrame->IndPyr[0].copyTo(TransitImage);
    mvbPrevMatched = MatchedPts;


    // MatchedPts = mvbPrevMatched;

    if (ShowInitializationMatches)
    {

        cv::Mat Image;

        if(ShowInitializationMatchesSideBySide)
            hconcat(FirstFrame->IndPyr[0], SecondFrame->IndPyr[0], Image);
        else 
            SecondFrame->IndPyr[0].copyTo(Image);

        cvtColor(Image, Image, CV_GRAY2RGB);
        for (int i = 0; i < FirstFramePts.size(); ++i)
            if (MatchedStatus[i])
                line(Image, FirstFramePts[i], cv::Point2f(ShowInitializationMatchesSideBySide ? FirstFrame->IndPyr[0].cols : 0, 0) +  MatchedPts[i], ColorVec[i], 1);

        imshow("InitMatches", Image);
        waitKey(1);
    }
    return count;
}


bool Initializer::FindTransformation(cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(MatchedPts.size());
    vector<size_t> vAvailableIndices;

    for (int i = 0, iend = MatchedStatus.size(); i < iend; ++i)
    {
        if (MatchedStatus[i])
            vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    mvSets = vector<vector<size_t>>(mMaxIterations, vector<size_t>(8, 0));

    for (int it = 0; it < mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for (size_t j = 0; j < 8; j++)
        {
            int randi = randomGen->uniform((int)0, (int)vAvailableIndices.size() - 1);
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

    boost::thread threadH(&Initializer::FindHomography, this, boost::ref(vbMatchesInliersH), boost::ref(SH), boost::ref(H));
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

void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    const int N = MatchedPts.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(FirstFramePts, vPn1, T1);
    Normalize(MatchedPts, vPn2, T2);
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

            vPn1i[j] = vPn1[idx];
            vPn2i[j] = vPn2[idx];
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

void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = vbMatchesInliers.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(FirstFramePts, vPn1, T1);
    Normalize(MatchedPts, vPn2, T2);
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

            vPn1i[j] = vPn1[idx];
            vPn2i[j] = vPn2[idx];
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

cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
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

cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
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

float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = MatchedPts.size();

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
        if(!MatchedStatus[i])
        {
            vbMatchesInliers[i] = false;
            continue;
        }
           
        bool bIn = true;

        const cv::Point2f &kp1 = FirstFramePts[i];
        const cv::Point2f &kp2 = MatchedPts[i];

        const float u1 = kp1.x;
        const float v1 = kp1.y;
        const float u2 = kp2.x;
        const float v2 = kp2.y;

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

float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = MatchedPts.size();

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
        if(!MatchedStatus[i])
        {
            vbMatchesInliers[i] = false;
            continue;
        }
        bool bIn = true;

        const cv::Point2f &kp1 = FirstFramePts[i];
        const cv::Point2f &kp2 = MatchedPts[i];

        const float u1 = kp1.x;
        const float v1 = kp1.y;
        const float u2 = kp2.x;
        const float v2 = kp2.y;

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

bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
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

    int nGood1 = CheckRT(R1, t1, FirstFramePts, MatchedPts, vbMatchesInliers, K, vP3D1, 4.0 * mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2, t1, FirstFramePts, MatchedPts, vbMatchesInliers, K, vP3D2, 4.0 * mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1, t2, FirstFramePts, MatchedPts, vbMatchesInliers, K, vP3D3, 4.0 * mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2, t2, FirstFramePts, MatchedPts, vbMatchesInliers, K, vP3D4, 4.0 * mSigma2, vbTriangulated4, parallax4);

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

bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
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
        int nGood = CheckRT(vR[i], vt[i], FirstFramePts, MatchedPts, vbMatchesInliers, K, vP3Di, 4.0 * mSigma2, vbTriangulatedi, parallaxi);
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

void Initializer::Triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4, 4, CV_32F);

    A.row(0) = kp1.x * P1.row(2) - P1.row(0);
    A.row(1) = kp1.y * P1.row(2) - P1.row(1);
    A.row(2) = kp2.x * P2.row(2) - P2.row(0);
    A.row(3) = kp2.y * P2.row(2) - P2.row(1);

    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
}

void Initializer::Normalize(const vector<cv::Point2f> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for (int i = 0; i < N; ++i)
    {
        meanX += vKeys[i].x;
        meanY += vKeys[i].y;
    }

    meanX = meanX / N;
    meanY = meanY / N;

    float meanDevX = 0;
    float meanDevY = 0;

    for (int i = 0; i < N; ++i)
    {
        vNormalizedPoints[i].x = vKeys[i].x - meanX;
        vNormalizedPoints[i].y = vKeys[i].y - meanY;

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

int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, vector<cv::Point2f> &vKeys1, vector<cv::Point2f> &vKeys2,
                                vector<bool> &vbMatchesInliers,
                                const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters    
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
    
    std::shared_ptr<CheckRTIn> ChRT = std::shared_ptr<CheckRTIn>(new CheckRTIn());
    ChRT->fx = K.at<float>(0, 0); ChRT->fy = K.at<float>(1, 1); ChRT->cx = K.at<float>(0, 2); ChRT->cy = K.at<float>(1, 2);
    ChRT->O1 = O1; ChRT->O2 = O2; ChRT->P1 = P1; ChRT->P2 = P2; ChRT->R = R; ChRT->t = t; ChRT->th2 = th2;
    ChRT->vbGood = &vbGood; ChRT->vbMatchesInliers = &vbMatchesInliers; ChRT->vCosParallax = &vCosParallax;
    ChRT->vKeys1 = &vKeys1; ChRT->vKeys2 = &vKeys2; ChRT->vP3D = &vP3D; ChRT->thPoolLock = std::shared_ptr<boost::mutex>(new boost::mutex);

    ChRT->nGood = 0;

    thPool->reduce( boost::bind( &Initializer::ParallelCheckRT, this, ChRT ,_1, _2), 0, vbMatchesInliers.size(), std::ceil(vbMatchesInliers.size() / NUM_THREADS));

    if (ChRT->nGood > 0)
    {
        sort(vCosParallax.begin(), vCosParallax.end());

        size_t idx = min(50, int(vCosParallax.size() - 1));
        parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
    }
    else
        parallax = 0;

    return ChRT->nGood;
}


void Initializer::ParallelCheckRT(std::shared_ptr<CheckRTIn> In, int min, int max)
{

    for (int i = min; i < max; ++i)
    {
        if (!In->vbMatchesInliers[0][i])
            continue;

        const cv::Point2f &kp1 = In->vKeys1[0][i];
        const cv::Point2f &kp2 = In->vKeys2[0][i];
        cv::Mat p3dC1;

        Triangulate(kp1, kp2, In->P1, In->P2, p3dC1);

        if (!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            In->vbGood[0][i] = false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - In->O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - In->O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2) / (dist1 * dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if (p3dC1.at<float>(2) <= 0 && cosParallax < 0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = In->R * p3dC1 + In->t;

        if (p3dC2.at<float>(2) <= 0 && cosParallax < 0.99998)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0 / p3dC1.at<float>(2);
        im1x = In->fx * p3dC1.at<float>(0) * invZ1 + In->cx;
        im1y = In->fy * p3dC1.at<float>(1) * invZ1 + In->cy;

        float squareError1 = (im1x - kp1.x) * (im1x - kp1.x) + (im1y - kp1.y) * (im1y - kp1.y);

        if (squareError1 > In->th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0 / p3dC2.at<float>(2);
        im2x = In->fx * p3dC2.at<float>(0) * invZ2 + In->cx;
        im2y = In->fy * p3dC2.at<float>(1) * invZ2 + In->cy;

        float squareError2 = (im2x - kp2.x) * (im2x - kp2.x) + (im2y - kp2.y) * (im2y - kp2.y);

        if (squareError2 > In->th2)
            continue;

        In->vP3D[0][i] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));

        {
            boost::unique_lock<boost::mutex> lock(*In->thPoolLock);
            In->vCosParallax[0].push_back(cosParallax);
             In->nGood++;
        }

        if (cosParallax < 0.99998)
            In->vbGood[0][i] = true;
    }
}

void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
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

float Initializer::ComputeSceneMedianDepth(const int q, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated)
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

float Initializer::ComputeMeanOpticalFlow(vector<Point2f> &Prev, vector<Point2f> &New)
{
    assert(Prev.size() == New.size());

    std::vector<float> Median;
    Median.reserve(Prev.size());

    for (int i = 0, iend = Prev.size(); i < iend; ++i)
    {
        if (MatchedStatus[i])
            Median.push_back(norm(Prev[i] - New[i]));
    }

    return accumulate(Median.begin(), Median.end(),0)/(float)Median.size();

}

} // namespace FSLAM