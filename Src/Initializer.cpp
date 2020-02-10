#include "Initializer.h"
#include "Frame.h"
#include "CalibData.h"
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include "IndexThreadReduce.h"
#include "boost/thread/mutex.hpp"
#include "Display.h"

// #include <opencv2/xfeatures2d.hpp>

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

            FirstFrame->id = 0;
            FirstFrame->camToWorld = SE3();

            Mat44 _Pose; cv::cv2eigen(Tcw, _Pose);
            Pose = SE3(_Pose);

            videpth.reserve(FirstFrame->nFeatures);
            for (int i = 0; i < FirstFrame->nFeatures; ++i)
                if(vbTriangulated[i])
                    videpth.push_back(1.0 / mvIniP3D[i].z);
                else videpth.push_back(-1.0);

            shared_ptr<DirectRefinement> DirOpti = shared_ptr<DirectRefinement> (new DirectRefinement(Calib, mvIniP3D, vbTriangulated, FirstFrame, SecondFrame, Pose, videpth));

            // //Direct Refinement statistics!!
            // std::vector<float> pts;
            // // std::vector<float> RepErrorAft;
            // for (int i = 0; i < FirstFrame->nFeatures; ++i)
            // {
            //     if(vbTriangulated[i])
            //         continue;
            //     if(!DirOpti->points[i].isGood)
            //         continue;
               
            //     //red points were not triangulated by Init algorithm
            //     float im1z = 1.0f/DirOpti->points[i].idepth;
            //     float  im1x =  (FirstFrame->mvKeys[i].pt.x - Calib->cxl())*Calib->fxli()* im1z;
            //     float  im1y =  (FirstFrame->mvKeys[i].pt.y - Calib->cyl())*Calib->fyli()* im1z;
                
            //     pts.push_back(im1x);
            //     pts.push_back(im1y);
            //     pts.push_back(im1z);
            //     pts.push_back(1);

            // //     float zz = 1.0 / im1z;
            // //     float xx = Calib->fxl() * im1x * zz + Calib->cxl();
            // //     float yy = Calib->fyl() * im1y * zz + Calib->cyl();

            // //     if(vbTriangulated[i])
            // //         RepErrorAft.push_back(norm(FirstFrame->mvKeys[i].pt - Point2f(xx, yy)));
            // }

            // // // std::vector<float> pts;
            // // std::vector<float> RepErrorBef;
            // for (int i = 0; i < FirstFrame->nFeatures; ++i)
            // {
            //     if (vbTriangulated[i])
            //     {
            //         pts.push_back(mvIniP3D[i].x);
            //         pts.push_back(mvIniP3D[i].y);
            //         pts.push_back(mvIniP3D[i].z);
            //         pts.push_back(0);


            // //         float invZ1 = 1.0 / mvIniP3D[i].z;
            // //         float im1x = Calib->fxl() * mvIniP3D[i].x * invZ1 + Calib->cxl();
            // //         float im1y = Calib->fyl() * mvIniP3D[i].y * invZ1 + Calib->cyl();
            // //         RepErrorBef.push_back(norm(FirstFrame->mvKeys[i].pt - Point2f(im1x, im1y)));
            //     }
            // }
            // // // std::cout<<"Mean Rep Error Befor: "<< std::accumulate(RepErrorBef.begin(), RepErrorBef.end(), 0.0)/RepErrorBef.size();
            // // // std::cout<<"  Mean Rep Error After: "<<std::accumulate(RepErrorAft.begin(), RepErrorAft.end(), 0.0)/RepErrorAft.size() <<std::endl;
            // displayhandler->UploadPoints(pts);
            return true;
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

DirectRefinement::DirectRefinement(shared_ptr<CalibData> _Calib, std::vector<cv::Point3f> &_Pts3D, std::vector<bool> &_Triangulated, std::shared_ptr<Frame> _FirstFrame,
                                   std::shared_ptr<Frame> _SecondFrame, SE3 &_Pose, std::vector<float> &_videpth)
{
    Calib = _Calib;
    thisToNext_aff = AffLight(0, 0);
    thisToNext = _Pose;
    FirstFrame = _FirstFrame;
    SecondFrame = _SecondFrame;
    alphaK = 2.5 * 2.5;
    alphaW = 150 * 150;
    regWeight = 0.8;
    couplingWeight = 1;
    points = 0;
    numPoints = 0;
    JbBuffer = new Vec10f[Calib->Width * Calib->Height];
    JbBuffer_new = new Vec10f[Calib->Width * Calib->Height];
    fixAffine = true;
    randomGen = std::shared_ptr<cv::RNG>(new cv::RNG(0));

    Triangulated = &_Triangulated;
    Pts3D = &_Pts3D;

    wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
    wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
    wM.diagonal()[6] = SCALE_A;
    wM.diagonal()[7] = SCALE_B;

    if (points != 0)
        delete[] points;

    points = new Pnt[FirstFrame->nFeatures];
    Pnt* pl = points;

    for (int i = 0; i < FirstFrame->nFeatures; i++)
    {
        pl[i].u = FirstFrame-> mvKeys[i].pt.x;
		pl[i].v = FirstFrame-> mvKeys[i].pt.y;

        if(Triangulated[0][i])
        {
            pl[i].idepth = 1.0 / Pts3D[0][i].z;
            pl[i].iR = pl[i].idepth;
        }
        else
        {
            pl[i].idepth = 1.0f;
            pl[i].iR = 1.0f;
        }
            
        pl[i].isGood=true;
		pl[i].energy.setZero();
		pl[i].lastHessian=0;
		pl[i].lastHessian_new=0;
		pl[i].my_type= 1;
        pl[i].outlierTH = patternNum*setting_outlierTH;
    }
    numPoints = FirstFrame->nFeatures;
    snapped = false;

    // trace(pl);
    Refine();
    //Update optimized data!!
    _Pose = thisToNext;
    for (int i = 0; i < FirstFrame->nFeatures; ++i)
    {
        if (!points[i].isGood || !Triangulated[0][i])
            continue;
        _videpth[i] = points[i].idepth;
        // _Triangulated[i] = true;
    }
    if(DrawDepthKfTest)
        debugPlot(points);
}

DirectRefinement::~DirectRefinement()
{
    if (points != 0)
        delete[] points;
    delete[] JbBuffer;
    delete[] JbBuffer_new;
}

void DirectRefinement::Refine()
{
    bool printDebug = false;
	int maxIterations[] = {1000,5,10,30,50};

	SE3 refToNew_current = thisToNext;
	AffLight refToNew_aff_current = thisToNext_aff;

	if(FirstFrame->ab_exposure>0 && SecondFrame->ab_exposure>0)
		refToNew_aff_current = AffLight(logf(SecondFrame->ab_exposure /  FirstFrame->ab_exposure),0); // coarse approximation.

	Vec3f latestRes = Vec3f::Zero();

	// if(lvl<pyrLevelsUsed-1)
	// 	propagateDown(lvl+1);

	Mat88f H,Hsc; Vec8f b,bsc;
	resetPoints(0);
	Vec3f resOld = calcResAndGS(0, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
	applyStep(0);

		float lambda = 0.1;
		float eps = 1e-4;
		int fails=0;

		if(printDebug)
		{
			printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
					0, 0, lambda,
					"INITIA",
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					(resOld[0]+resOld[1]) / resOld[2],
					(resOld[0]+resOld[1]) / resOld[2],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() <<"\n";
		}

		int iteration=0;
        int lvl = 0;
		while(true)
		{
			Mat88f Hl = H;
			for(int i=0;i<8;i++) Hl(i,i) *= (1+lambda);
			Hl -= Hsc*(1/(1+lambda));
			Vec8f bl = b - bsc*(1/(1+lambda));

			Hl = wM * Hl * wM * (0.01f/(Calib->Width*Calib->Height));
			bl = wM * bl * (0.01f/(Calib->Width*Calib->Height));

			Vec8f inc;
			if(fixAffine)
			{
				inc.head<6>() = - (wM.toDenseMatrix().topLeftCorner<6,6>() * (Hl.topLeftCorner<6,6>().ldlt().solve(bl.head<6>())));
				inc.tail<2>().setZero();
			}
			else
				inc = - (wM * (Hl.ldlt().solve(bl)));	//=-H^-1 * b.

			SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
			AffLight refToNew_aff_new = refToNew_aff_current;
			refToNew_aff_new.a += inc[6];
			refToNew_aff_new.b += inc[7];
			doStep(lvl, lambda, inc);


			Mat88f H_new, Hsc_new; Vec8f b_new, bsc_new;
			Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
			Vec3f regEnergy = calcEC(lvl);

			float eTotalNew = (resNew[0]+resNew[1]+regEnergy[1]);
			float eTotalOld = (resOld[0]+resOld[1]+regEnergy[0]);


			bool accept = eTotalOld > eTotalNew;

			if(printDebug)
			{
				printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						(accept ? "ACCEPT" : "REJECT"),
						sqrtf((float)(resOld[0] / resOld[2])),
						sqrtf((float)(regEnergy[0] / regEnergy[2])),
						sqrtf((float)(resOld[1] / resOld[2])),
						sqrtf((float)(resNew[0] / resNew[2])),
						sqrtf((float)(regEnergy[1] / regEnergy[2])),
						sqrtf((float)(resNew[1] / resNew[2])),
						eTotalOld / resNew[2],
						eTotalNew / resNew[2],
						inc.norm());
				std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() <<"\n";
			}

			if(accept)
			{

				if(resNew[1] == alphaK*numPoints) //numPoints[lvl]
					snapped = true;
				H = H_new;
				b = b_new;
				Hsc = Hsc_new;
				bsc = bsc_new;
				resOld = resNew;
				refToNew_aff_current = refToNew_aff_new;
				refToNew_current = refToNew_new;
				applyStep(0); //lvl
				optReg(0); //lvl
				lambda *= 0.5;
				fails=0;
				if(lambda < 0.0001) lambda = 0.0001;
			}
			else
			{
				fails++;
				lambda *= 4;
				if(lambda > 10000) lambda = 10000;
			}

			bool quitOpt = false;

			if(!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2)
			{
				Mat88f H,Hsc; Vec8f b,bsc;

				quitOpt = true;
			}


			if(quitOpt) break;
			iteration++;
		}
		latestRes = resOld;

	


	thisToNext = refToNew_current;
	thisToNext_aff = refToNew_aff_current;

	// for(int i=0;i<pyrLevelsUsed-1;i++)
	// 	propagateUp(i);

	// frameID++;
	// if(!snapped) snappedAt=0;

	// if(snapped && snappedAt==0)
	// 	snappedAt = frameID;

    // debugPlot(0,wraps);

}

void DirectRefinement::trace(Pnt* _pl)
{
	float maxPixSearch = (Calib->Width + Calib->Height) * setting_maxPixSearch ; // 640*480 * 0.027
    SE3 Temp = thisToNext;
    Mat33f KRKi = (Calib->pyrK[0].cast<double>() * Temp.rotationMatrix() * Calib->pyrKi[0].cast<double>()).cast<float>();
	Vec3f Kt = Calib->pyrK[0] * Temp.translation().cast<float>();

    Eigen::Vector3f* colorRef = FirstFrame->DirPyr[0];
    Eigen::Vector3f *colorNew = SecondFrame->DirPyr[0];

    cv::Mat Image;
    if (DrawEpipolarMatching)
    {
        cv::hconcat(FirstFrame->IndPyr[0], SecondFrame->IndPyr[0], Image);
        cvtColor(Image, Image, CV_GRAY2RGB);
        cv::namedWindow("epi trace", cv::WindowFlags::WINDOW_KEEPRATIO);
    }

    float d_min = 0.0f;
    for (int i = 0; i < FirstFrame->nFeatures; ++i)
    {
        if(Triangulated[0][i])
            continue;

        float u = FirstFrame->mvKeys[i].pt.x;
        float v = (float)FirstFrame->mvKeys[i].pt.y;

        Vec3f pr = KRKi * Vec3f(u, v, 1.0f);
        Vec3f ptpMin = pr + Kt*d_min;
        float uMin = ptpMin[0] / ptpMin[2];
        float vMin = ptpMin[1] / ptpMin[2];

        Vec2f lastTraceUV;
        float lastTracePixelInterval;
        if (!(uMin > 4 && vMin > 4 && uMin < Calib->Width - 5 && vMin < Calib->Height - 5))
        {
            lastTraceUV = Vec2f(-1, -1);
            lastTracePixelInterval = 0;
            continue;
        }

        Mat22f gradH;
        float color[patternNum];
        float weights[patternNum];
        float energyTH = patternNum*setting_outlierTH;
	    energyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;
        float quality = 10000.0f;
        float idepth_min = d_min;
        float idepth_max = NAN;
        for (int idx = 0; idx < patternNum; idx++)
        {
            int dx = patternP[idx][0];
            int dy = patternP[idx][1];
            Vec3f ptc = getInterpolatedElement33BiLin(colorRef, u + dx, v + dy, Calib->Width);
            
            color[idx] = ptc[0];
            if (!std::isfinite(color[idx]))
            {
                energyTH = NAN;
                continue;
            }
            gradH += ptc.tail<2>() * ptc.tail<2>().transpose();
            weights[idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
        }
        float dist;
        float uMax;
        float vMax;
        Vec3f ptpMax;
        dist = maxPixSearch;

        // project to arbitrary depth to get direction.
        ptpMax = pr + Kt * 0.01; //0.01
        uMax = ptpMax[0] / ptpMax[2];
        vMax = ptpMax[1] / ptpMax[2];
        
        // direction.
        float _dx = uMax - uMin;
        float _dy = vMax - vMin;
        float _d = 1.0f / sqrtf(_dx * _dx + _dy * _dy);

        // set to [setting_maxPixSearch].
        uMax = uMin + dist * _dx * _d;
        vMax = vMin + dist * _dy * _d;

        // may still be out!
        if (!(uMax > 4 && vMax > 4 && uMax < Calib->Width - 5 && vMax < Calib->Height - 5))
        {
            lastTraceUV = Vec2f(-1, -1);
            lastTracePixelInterval = 0;
            continue;
        }
        assert(dist > 0);
        if (!(d_min < 0 || (ptpMin[2] > 0.75 && ptpMin[2] < 1.5)))
        {
            lastTraceUV = Vec2f(-1, -1);
            lastTracePixelInterval = 0;
            continue;
        }

        float dx = setting_trace_stepsize * (uMax - uMin);
        float dy = setting_trace_stepsize * (vMax - vMin);

        float a = (Vec2f(dx, dy).transpose() * gradH * Vec2f(dx, dy));
        float b = (Vec2f(dy, -dx).transpose() * gradH * Vec2f(dy, -dx));
        float errorInPixel = 0.2f + 0.2f * (a + b) / a;

        if (errorInPixel > 10)
            errorInPixel = 10;

        // ============== do the discrete search ===================
        dx /= dist;
        dy /= dist;

        // if (debugPrint)
        //     printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f (%.1f %.1f) -> %f (%.1f %.1f)! ErrorInPixel %.1f!\n",
        //            u, v,
        //            host->shell->id, frame->shell->id,
        //            idepth_min, uMin, vMin,
        //            idepth_max, uMax, vMax,
        //            errorInPixel);

        if (dist > maxPixSearch)
        {
            uMax = uMin + maxPixSearch * dx;
            vMax = vMin + maxPixSearch * dy;
            dist = maxPixSearch;
        }

        int numSteps = 1.9999f + dist / setting_trace_stepsize;
        Mat22f Rplane = KRKi.topLeftCorner<2, 2>();

        float randShift = uMin * 1000 - floorf(uMin * 1000);
        float ptx = uMin - randShift * dx;
        float pty = vMin - randShift * dy;

        Vec2f rotatetPattern[MAX_RES_PER_POINT];
        for (int idx = 0; idx < patternNum; idx++)
            rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);

        if (!std::isfinite(dx) || !std::isfinite(dy))
        {
            lastTracePixelInterval = 0;
            lastTraceUV = Vec2f(-1, -1);
            continue;
        }

        float errors[100];
	    float bestU=0, bestV=0, bestEnergy=1e10;
	    int bestIdx=-1;
	    if(numSteps >= 100) numSteps = 99;

        for (int i = 0; i < numSteps; i++)
        {
            float energy = 0;
            for (int idx = 0; idx < patternNum; idx++)
            {
                float hitColor = getInterpolatedElement31(colorNew, (float)(ptx + rotatetPattern[idx][0]), (float)(pty + rotatetPattern[idx][1]), Calib->Width);

                if (!std::isfinite(hitColor))
                {
                    energy += 1e5;
                    continue;
                }
                float residual = hitColor - (float)( color[idx]); //removed affine model here
                float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
                energy += hw * residual * residual * (2 - hw);
            }

            // if (debugPrint)
                // printf("step %.1f %.1f (id %f): energy = %f!\n",
                //        ptx, pty, 0.0f, energy);

            errors[i] = energy;
            if (energy < bestEnergy)
            {
                bestU = ptx;
                bestV = pty;
                bestEnergy = energy;
                bestIdx = i;
            }

            ptx += dx;
            pty += dy;
        }

        // find best score outside a +-2px radius.
        float secondBest = 1e10;
        for (int i = 0; i < numSteps; i++)
        {
            if ((i < bestIdx - setting_minTraceTestRadius || i > bestIdx + setting_minTraceTestRadius) && errors[i] < secondBest)
                secondBest = errors[i];
        }
        float newQuality = secondBest / bestEnergy;
        if (newQuality < quality || numSteps > 10)
            quality = newQuality;

        // ============== do GN optimization ===================
        float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0;
        if (setting_trace_GNIterations > 0)
            bestEnergy = 1e5;
        int gnStepsGood = 0, gnStepsBad = 0;
        for (int it = 0; it < setting_trace_GNIterations; it++)
        {
            float H = 1, b = 0, energy = 0;
            for (int idx = 0; idx < patternNum; idx++)
            {
                Vec3f hitColor = getInterpolatedElement33(colorNew,
                                                          (float)(bestU + rotatetPattern[idx][0]),
                                                          (float)(bestV + rotatetPattern[idx][1]), Calib->Width);

                if (!std::isfinite((float)hitColor[0]))
                {
                    energy += 1e5;
                    continue;
                }
                float residual = hitColor[0] - (color[idx]); //removed affine model here!!
                float dResdDist = dx * hitColor[1] + dy * hitColor[2];
                float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

                H += hw * dResdDist * dResdDist;
                b += hw * residual * dResdDist;
                energy += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
            }

            if (energy > bestEnergy)
            {
                gnStepsBad++;

                // do a smaller step from old point.
                stepBack *= 0.5;
                bestU = uBak + stepBack * dx;
                bestV = vBak + stepBack * dy;
                // if (debugPrint)
                //     printf("GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
                //            it, energy, H, b, stepBack,
                //            uBak, vBak, bestU, bestV);
            }
            else
            {
                gnStepsGood++;

                float step = -gnstepsize * b / H;
                if (step < -0.5)
                    step = -0.5;
                else if (step > 0.5)
                    step = 0.5;

                if (!std::isfinite(step))
                    step = 0;

                uBak = bestU;
                vBak = bestV;
                stepBack = step;

                bestU += step * dx;
                bestV += step * dy;
                bestEnergy = energy;

                // if (debugPrint)
                //     printf("GN step %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
                //            it, energy, H, b, step,
                //            uBak, vBak, bestU, bestV);
            }

            if (fabsf(stepBack) < setting_trace_GNThreshold)
                break;
        }

        // ============== detect energy-based outlier. ===================
        //	float absGrad0 = getInterpolatedElement(frame->absSquaredGrad[0],bestU, bestV, wG[0]);
        //	float absGrad1 = getInterpolatedElement(frame->absSquaredGrad[1],bestU*0.5-0.25, bestV*0.5-0.25, wG[1]);
        //	float absGrad2 = getInterpolatedElement(frame->absSquaredGrad[2],bestU*0.25-0.375, bestV*0.25-0.375, wG[2]);
        if (!(bestEnergy < energyTH * setting_trace_extraSlackOnTH))
        //			|| (absGrad0*areaGradientSlackFactor < host->frameGradTH
        //		     && absGrad1*areaGradientSlackFactor < host->frameGradTH*0.75f
        //			 && absGrad2*areaGradientSlackFactor < host->frameGradTH*0.50f))
        {
            // if (debugPrint)
            //     printf("OUTLIER!\n");

            lastTracePixelInterval = 0;
            lastTraceUV = Vec2f(-1, -1);
            continue;
        }

        // ============== set new interval ===================
        if (dx * dx > dy * dy)
        {
            idepth_min = (pr[2] * (bestU - errorInPixel * dx) - pr[0]) / (Kt[0] - Kt[2] * (bestU - errorInPixel * dx));
            idepth_max = (pr[2] * (bestU + errorInPixel * dx) - pr[0]) / (Kt[0] - Kt[2] * (bestU + errorInPixel * dx));
        }
        else
        {
            idepth_min = (pr[2] * (bestV - errorInPixel * dy) - pr[1]) / (Kt[1] - Kt[2] * (bestV - errorInPixel * dy));
            idepth_max = (pr[2] * (bestV + errorInPixel * dy) - pr[1]) / (Kt[1] - Kt[2] * (bestV + errorInPixel * dy));
        }
        if (idepth_min > idepth_max)
            std::swap<float>(idepth_min, idepth_max);

        if (!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max < 0))
        {
            //printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min, idepth_max);

            lastTracePixelInterval = 0;
            lastTraceUV = Vec2f(-1, -1);
            continue;
        }

        lastTracePixelInterval = 2 * errorInPixel;
        lastTraceUV = Vec2f(bestU, bestV);
            
        _pl[i].idepth = (idepth_min + idepth_max) /2.0f;
        _pl[i].iR = _pl[i].idepth;

        if (DrawEpipolarMatching)
        {
            cv::Scalar COlor = cv::Scalar(randomGen->uniform((int)0, (int)255), randomGen->uniform((int)0, (int)255), randomGen->uniform((int)0, (int)255));
            cv::circle(Image, FirstFrame->mvKeys[i].pt, 3, COlor);
            cv::line(Image, cv::Point2f(Calib->Width, 0) + cv::Point2f(uMax, vMax), cv::Point2f(Calib->Width, 0) + cv::Point2f(uMin, vMin), COlor);
            cv::circle(Image, cv::Point2f(Calib->Width, 0) + cv::Point2f(bestU, bestV), 3, COlor);
        }
        // return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
    }
    if(DrawEpipolarMatching)
    {
        cv::imshow("epi trace", Image);
        cv::waitKey(1);
    }

}

void DirectRefinement::resetPoints(int lvl)
{
	Pnt* pts = points;
	int npts = numPoints;
	for(int i=0;i<npts;i++)
	{
		pts[i].energy.setZero();
		pts[i].idepth_new = pts[i].idepth;


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

Vec3f DirectRefinement::calcResAndGS( int lvl, Mat88f &H_out, Vec8f &b_out, Mat88f &H_out_sc, Vec8f &b_out_sc, const SE3 &refToNew, AffLight refToNew_aff, bool plot)
{
	int wl = Calib->Width;
    int hl = Calib->Height;
	Eigen::Vector3f* colorRef = FirstFrame->DirPyr[lvl];
	Eigen::Vector3f* colorNew = SecondFrame->DirPyr[lvl];

	Mat33f RKi = (refToNew.rotationMatrix() * Calib->pyrKi[lvl].cast<double>()).cast<float>();
	Vec3f t = refToNew.translation().cast<float>();
	Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);
    
	float fxl = Calib->pyrfx[lvl];
	float fyl = Calib->pyrfy[lvl];
	float cxl = Calib->pyrcx[lvl];
	float cyl = Calib->pyrcy[lvl];

	Accumulator11 E;
	acc9.initialize();
	E.initialize();

	int npts = numPoints;
	Pnt* ptsl = points;
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		point->maxstep = 1e10;
		if(!point->isGood)
		{
			E.updateSingle((float)(point->energy[0]));
			point->energy_new = point->energy;
			point->isGood_new = false;
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
		float energy=0;
		for(int idx=0;idx<patternNum;idx++)
		{
			int dx = patternP[idx][0];
			int dy = patternP[idx][1];
			Vec3f pt = RKi * Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;
			float u = pt[0] / pt[2];
			float v = pt[1] / pt[2];
			float Ku = fxl * u + cxl;
			float Kv = fyl * v + cyl;
			float new_idepth = point->idepth_new/pt[2];

			if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0))
			{
				isGood = false;
				break;
			}

			Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
			//Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

			//float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
			float rlR = getInterpolatedElement31(colorRef, point->u+dx, point->v+dy, wl);

			if(!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
			{
				isGood = false;
				break;
			}

			float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
            if(!Triangulated[0][i])
                hw *= 0.1;
			energy += hw *residual*residual*(2-hw);

			float dxdd = (t[0]-t[2]*u)/pt[2];
			float dydd = (t[1]-t[2]*v)/pt[2];

			if(hw < 1) hw = sqrtf(hw);
			float dxInterp = hw*hitColor[1]*fxl;
			float dyInterp = hw*hitColor[2]*fyl;
			dp0[idx] = new_idepth*dxInterp;
			dp1[idx] = new_idepth*dyInterp;
			dp2[idx] = -new_idepth*(u*dxInterp + v*dyInterp);
			dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp;
			dp4[idx] = (1+u*u)*dxInterp + u*v*dyInterp;
			dp5[idx] = -v*dxInterp + u*dyInterp;
			dp6[idx] = - hw*r2new_aff[0] * rlR;
			dp7[idx] = - hw*1;
			dd[idx] = dxInterp * dxdd  + dyInterp * dydd;
			r[idx] = hw*residual;

			float maxstep = 1.0f / Vec2f(dxdd*fxl, dydd*fyl).norm();
			if(maxstep < point->maxstep) point->maxstep = maxstep;

			// immediately compute dp*dd' and dd*dd' in JbBuffer1.
			JbBuffer_new[i][0] += dp0[idx]*dd[idx];
			JbBuffer_new[i][1] += dp1[idx]*dd[idx];
			JbBuffer_new[i][2] += dp2[idx]*dd[idx];
			JbBuffer_new[i][3] += dp3[idx]*dd[idx];
			JbBuffer_new[i][4] += dp4[idx]*dd[idx];
			JbBuffer_new[i][5] += dp5[idx]*dd[idx];
			JbBuffer_new[i][6] += dp6[idx]*dd[idx];
			JbBuffer_new[i][7] += dp7[idx]*dd[idx];
			JbBuffer_new[i][8] += r[idx]*dd[idx];
			JbBuffer_new[i][9] += dd[idx]*dd[idx];
		}

		if(!isGood || energy > point->outlierTH*20)
		{
			E.updateSingle((float)(point->energy[0]));
			point->isGood_new = false;
			point->energy_new = point->energy;
			continue;
		}

		// add into energy.
		E.updateSingle(energy);
		point->isGood_new = true;
		point->energy_new[0] = energy;

		// update Hessian matrix.
		for(int i=0;i+3<patternNum;i+=4)
			acc9.updateSSE(
					_mm_load_ps(((float*)(&dp0))+i),
					_mm_load_ps(((float*)(&dp1))+i),
					_mm_load_ps(((float*)(&dp2))+i),
					_mm_load_ps(((float*)(&dp3))+i),
					_mm_load_ps(((float*)(&dp4))+i),
					_mm_load_ps(((float*)(&dp5))+i),
					_mm_load_ps(((float*)(&dp6))+i),
					_mm_load_ps(((float*)(&dp7))+i),
					_mm_load_ps(((float*)(&r))+i));


		for(int i=((patternNum>>2)<<2); i < patternNum; i++)
			acc9.updateSingle(
					(float)dp0[i],(float)dp1[i],(float)dp2[i],(float)dp3[i],
					(float)dp4[i],(float)dp5[i],(float)dp6[i],(float)dp7[i],
					(float)r[i]);
	}

	E.finish();
	acc9.finish();

	// calculate alpha energy, and decide if we cap it.
	Accumulator11 EAlpha;
	EAlpha.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
		{
			E.updateSingle((float)(point->energy[1]));
		}
		else
		{
			point->energy_new[1] = (point->idepth_new-1)*(point->idepth_new-1);
			E.updateSingle((float)(point->energy_new[1]));
		}
	}
	EAlpha.finish();
	float alphaEnergy = alphaW*(EAlpha.A + refToNew.translation().squaredNorm() * npts);

	// compute alpha opt.
	float alphaOpt;
	if(alphaEnergy > alphaK*npts)
	{
		alphaOpt = 0;
		alphaEnergy = alphaK*npts;
	}
	else
	{
		alphaOpt = alphaW;
	}

	acc9SC.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
			continue;

		point->lastHessian_new = JbBuffer_new[i][9];

		JbBuffer_new[i][8] += alphaOpt*(point->idepth_new - 1);
		JbBuffer_new[i][9] += alphaOpt;

		if(alphaOpt==0)
		{
			JbBuffer_new[i][8] += couplingWeight*(point->idepth_new - point->iR);
			JbBuffer_new[i][9] += couplingWeight;
		}

		JbBuffer_new[i][9] = 1/(1+JbBuffer_new[i][9]);
		acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2],(float)JbBuffer_new[i][3],
				(float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7],
				(float)JbBuffer_new[i][8],(float)JbBuffer_new[i][9]);
	}
	acc9SC.finish();

	//printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
	H_out = acc9.H.topLeftCorner<8,8>();// / acc9.num;
	b_out = acc9.H.topRightCorner<8,1>();// / acc9.num;
	H_out_sc = acc9SC.H.topLeftCorner<8,8>();// / acc9.num;
	b_out_sc = acc9SC.H.topRightCorner<8,1>();// / acc9.num;

	H_out(0,0) += alphaOpt*npts;
	H_out(1,1) += alphaOpt*npts;
	H_out(2,2) += alphaOpt*npts;

	Vec3f tlog = refToNew.log().head<3>().cast<float>();
	b_out[0] += tlog[0]*alphaOpt*npts;
	b_out[1] += tlog[1]*alphaOpt*npts;
	b_out[2] += tlog[2]*alphaOpt*npts;

	return Vec3f(E.A, alphaEnergy ,E.num);
}

void DirectRefinement::doStep(int lvl, float lambda, Vec8f inc)
{

	const float maxPixelStep = 0.25;
	const float idMaxStep = 1e10;
	Pnt* pts = points; //[lvl]
	int npts = numPoints; //[lvl]
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood) continue;
		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
		float step = - b * JbBuffer[i][9] / (1+lambda);
        
        // if(!Triangulated[0][i])
        // {
        //     pts[i].maxstep *= 10;
        //     // step *= 10;
        // }
		
        float maxstep = maxPixelStep*pts[i].maxstep;
		if(maxstep > idMaxStep) maxstep=idMaxStep;

		if(step >  maxstep) step = maxstep;
		if(step < -maxstep) step = -maxstep;
        

		float newIdepth = pts[i].idepth + step;
		if(newIdepth < 1e-3 ) newIdepth = 1e-3;
		if(newIdepth > 50) newIdepth = 50;
		pts[i].idepth_new = newIdepth;
	}

}
void DirectRefinement::applyStep(int lvl)
{
	Pnt* pts = points; //[lvl]
	int npts = numPoints;//[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood)
		{
			pts[i].idepth = pts[i].idepth_new = pts[i].iR;
			continue;
		}
		pts[i].energy = pts[i].energy_new;
		pts[i].isGood = pts[i].isGood_new;
		pts[i].idepth = pts[i].idepth_new;
		pts[i].lastHessian = pts[i].lastHessian_new;
	}
	std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

Vec3f DirectRefinement::calcEC(int lvl)
{
	if(!snapped) return Vec3f(0,0,numPoints); //[lvl]
	AccumulatorX<2> E;
	E.initialize();
	int npts = numPoints;//[lvl];
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points+i; //points[lvl]
		if(!point->isGood_new) continue;
		float rOld = (point->idepth-point->iR);
		float rNew = (point->idepth_new-point->iR);
		E.updateNoWeight(Vec2f(rOld*rOld,rNew*rNew));

		//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
	}
	E.finish();

	//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
	return Vec3f(couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], E.num);
}

void DirectRefinement::optReg(int lvl)
{
	int npts = numPoints;//[lvl];
	Pnt* ptsl = points;//[lvl];
	if(!snapped)
	{
		for(int i=0;i<npts;i++)
        {
            if(Triangulated[0][i])
                ptsl[i].iR = 1.0 / Pts3D[0][i].z;
            else
                ptsl[i].iR = ptsl[i].idepth; //changed this from 1.0f
        }
			
		return;
	}


	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood) continue;

		// float idnn[10];
		// int nnn=0;
		// for(int j=0;j<10;j++)
		// {
		// 	if(point->neighbours[j] == -1) continue;
		// 	Pnt* other = ptsl+point->neighbours[j];
		// 	if(!other->isGood) continue;
		// 	idnn[nnn] = other->iR;
		// 	nnn++;
		// }

		// if(nnn > 2)
		// {
		// 	std::nth_element(idnn,idnn+nnn/2,idnn+nnn);
		// 	point->iR = (1-regWeight)*point->idepth + regWeight*idnn[nnn/2];
		// }
        point->iR = point->idepth;
	}
}

void DirectRefinement::debugPlot(Pnt* Points)
{
	int wl = Calib->wpyr[0], hl = Calib->hpyr[0];
	// Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
    // FirstFrame->LeftDirPyr[0];
	// MinimalImageB3 iRImg(wl,hl);
    cv::Mat Depth; 
    cv::cvtColor(FirstFrame->IndPyr[0], Depth, CV_GRAY2RGB);

	int npts = FirstFrame->nFeatures;

	float nid = 0, sid=0;
	for(int i=0;i<npts;i++)
	{
            if(!Triangulated[0][i])
            continue;
		// Pnt* point = points[lvl]+i;
		if(Points[i].isGood)
		{
			nid++;
			sid += (Points[i].idepth); //iR
		}
	}
	float fac = nid / sid;

	for(int i=0;i<npts;i++)
	{
		// Pnt* point = points[lvl]+i;
        if(!Triangulated[0][i])
            continue;
        Vec3b Color = Vec3b(0,0,0);
        if(Points[i].isGood)
        {
            Color = makeRainbow3B(Points[i].idepth *fac);
        }
        setPixel9(Depth, std::floor(Points[i].v + 0.5f), std::floor(Points[i].u + 0.5f), Color);
    }

    cv::namedWindow("InitDepthTest", cv::WINDOW_KEEPRATIO);
    cv::imshow("InitDepthTest", Depth);
    cv::waitKey(1);
}

} // namespace FSLAM