#include "Frame.h"
#include "CalibData.h"
#include "MapPoint.h"
#include "ImmaturePoint.h"
#include "photoUndistorter.h"
#include "Detector.h"
#include <chrono>
#include "DBoW3/DBoW3.h"
#include "Settings.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace SLAM
{
size_t FrameShell::Globalid = 0;
int FrameShell::GlobalKfId = 0;

bool Frame::GridStructInit = false;
int Frame::mnGridCols = 0;
int Frame::mnGridRows = 0;
float Frame::mnMinX = 0.0f;
float Frame::mnMaxX = 0.0f;
float Frame::mnMinY = 0.0f;
float Frame::mnMaxY = 0.0f;
float Frame::mfGridElementWidthInv = 0.0f;
float Frame::mfGridElementHeightInv = 0.0f;

int Frame::EDGE_THRESHOLD = 19; //15?

Frame::Frame(shared_ptr<ImageData> Img, int id, float ab_exposure,
             shared_ptr<CalibData> _Calib, shared_ptr<photoUndistorter> _phoUndist, bool ForInit) : 
             Calib(_Calib), phoUndist(_phoUndist)
{
    if (!GridStructInit)
    {
        mnGridCols = ceil(Img->cvImg.cols / 10);
        mnGridRows = ceil(Img->cvImg.rows / 10);
        mnMinX = 0.0f; mnMaxX = Img->cvImg.cols; mnMinY = 0.0f; mnMaxY = Img->cvImg.rows;
        mfGridElementWidthInv = static_cast<float>(mnGridCols) / static_cast<float>(mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(mnGridRows) / static_cast<float>(mnMaxY - mnMinY);
        GridStructInit = true;
    }

    NeedRefresh = false;
    MovedByOpt=0;
	statistics_outlierResOnThis = statistics_goodResOnThis = 0;
    
    FlaggedForMarginalization = false;
    frameEnergyTH = 8*8*patternNum;

    // CreateIndPyrs(Img->cvImgL, IndPyr);
    //for now I'm only extracting features from highest resolution image!!

    CreateDirPyrs(Img->fImg, DirPyr);


    Img->cvImg.copyTo(Image);
    nFeatures = 0;
    efFrame = shared_ptr<FrameOptimizationData>(new FrameOptimizationData(id, ab_exposure));
    isReduced = false;
}

void Frame::Extract(shared_ptr<FeatureDetector>_Detector, int id ,bool ForInit, shared_ptr<IndexThreadReduce<Vec10>> ThreadPool)
{
    _Detector->ExtractFeatures(Image, featType, DirPyr, id, absSquaredGrad,  mvKeys,
                 Descriptors, nFeatures, (ForInit ? featuresToExtract : featuresToExtract), ThreadPool); 
    // pointHessians.resize(nFeatures);
    // ImmaturePoints.resize(nFeatures);
    //Assign Features to Grid
        
    mGrid.resize(mnGridCols);
    for (int i = 0; i < mnGridCols; ++i)
    {
        mGrid[i].resize(mnGridRows);
        for (int j = 0, jend = mGrid[i].size(); j < jend; ++j)
            mGrid[i][j].reserve(5);
    }
    for (unsigned short int i = 0; i < nFeatures; ++i)
    {
        int nGridPosX, nGridPosY;
        if (PosInGrid(mvKeys[i], nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
    
    for (int i = 0, iend = mGrid.size(); i < iend; ++i)
        for (int j = 0, jend = mGrid[i].size(); j < jend; ++j)
            mGrid[i][j].shrink_to_fit();
    
    // ComputeBoVW();
}

void Frame::ComputeBoVW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vDesc;
        vDesc.reserve(Descriptors.rows);
        for (int j=0;j<Descriptors.rows;j++)
            vDesc.push_back(Descriptors.row(j));
        Vocab.transform(vDesc, mBowVec, mFeatVec, 4);
    }
}

void Frame::CreateDirPyrs(std::vector<float>& Img, std::vector<Vec3f*> &DirPyr)
{
    DirPyr.resize(pyramidSize);
    absSquaredGrad.resize(pyramidSize);

    for (int i = 0; i < pyramidSize; ++i)
    {
        absSquaredGrad[i] = new float [Calib->pyrImgSize[i]];
        DirPyr[i] = new Eigen::Vector3f[Calib->pyrImgSize[i]];
    }

    for (int i = 0; i < Calib->pyrImgSize[0]; ++i) //populate the data of the highest resolution pyramid level
        DirPyr[0][i][0] = Img[i]; // 0th pyr level, ith point, [0] (intensity), [1] (dx), [2] (dy)

    for (int lvl = 0; lvl < pyramidSize; ++lvl)
    {
        int wl = Calib->wpyr[lvl], hl = Calib->hpyr[lvl];
        Vec3f* &dI_l = DirPyr[lvl];

        if (lvl > 0)
        {
            int lvlm1 = lvl - 1;
            int wlm1 = Calib->wpyr[lvlm1];
            Vec3f* &dI_lm = DirPyr[lvlm1];

            for (int y = 0; y < hl; ++y)
                for (int x = 0; x < wl; ++x)
                {
                    //Interpolation to get the downscaled pyr level
                    dI_l[x + y*wl][0] = 0.25f * (dI_lm[2*x   + 2*y*wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1][0] +
												dI_lm[2*x   + 2*y*wlm1+wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1+wlm1][0]);
				}
		}

        int it = wl * (hl - 1);
        for (int idx = wl; idx < it; ++idx)
        {
            float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
            float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

            if(!std::isfinite(dx)) dx=0;
			if(!std::isfinite(dy)) dy=0;

            dI_l[idx][1] = dx;
            dI_l[idx][2] = dy;

            absSquaredGrad[lvl][idx] = dx * dx + dy * dy;
            if (phoUndist->GammaValid)
            {
                float gw = phoUndist->getBGradOnly((float)(dI_l[idx][0]));
                absSquaredGrad[lvl][idx] *= gw * gw; // convert to gradient of original color space (before removing response).
            }
        }
    }

    // cv::Mat Test = cv::Mat(Calib->Height, Calib->Width, CV_32FC3, DirPyr[0] );
    // cv::Mat Bands[3];
    // cv::split(Test, Bands);
    // Bands[0].convertTo(Test, CV_8U);    
    // cv::imshow("test", Test);
    // cv::waitKey(1);

    
    // if (show_gradient_image) //make sure this does not get called in stereo system (parallel thread- remove right image createDirPyr?)
    // {
    //     // cv::namedWindow("AbsSquaredGrad", cv::WindowFlags::WINDOW_KEEPRATIO);
    //     cv::Mat imGrad = cv::Mat(Calib->hpyr[0], Calib->wpyr[0], CV_32F, absSquaredGrad[0]);
    //     // float* dataptr = imGrad.ptr<float>(0);
    //     // for (int i = 0, iend = Calib->hpyr[0]* Calib->wpyr[0]; i < iend; ++i)
    //     //     dataptr[i] = absSquaredGrad[0][i];

    //     imGrad.convertTo(imGrad,CV_8U);
    //     // cv::imshow("AbsSquaredGrad", imGrad);
    //     // cv::waitKey(1);
    // }
}

Frame::~Frame() 
{
    ReduceToEssential();
    mvKeys.resize(0); mvKeys.shrink_to_fit();   
    mGrid.clear(); mGrid.shrink_to_fit();
    Descriptors.release();
}

void Frame::ReduceToEssential()
{
    if(isReduced)
        return;
    isReduced = true;
    Image.release();

    {
        std::lock_guard<std::mutex> l(_mtx);
        ImmaturePoints.clear();
        ImmaturePoints.shrink_to_fit();

        // mvKeys.resize(0); mvKeys.shrink_to_fit();
        // mGrid.clear(); mGrid.shrink_to_fit();
        // Descriptors.release();

        for (auto &it : pointHessians)
            it.reset();
        for (auto &it : pointHessiansOut)
            it.reset();
        for (auto &it : pointHessiansMarginalized)
            if (it->efPoint)
                it->Clear();

        pointHessians.clear();
        pointHessians.shrink_to_fit();
        pointHessiansOut.clear();
        pointHessiansOut.shrink_to_fit();
        // pointHessiansMarginalized.clear();
        // pointHessiansMarginalized.shrink_to_fit();

        NeedRefresh = true;
    }
    // for (auto &it : pointHessians)
    // {
    //     if (!it)
    //         continue;
    //     assert(it->getPointStatus() != ACTIVE);

    //     if (it->status != MARGINALIZED)
    //         it.reset();
    //     else
    //     {
    //         if (it->efPoint)
    //             it->efPoint.reset();
    //         for (auto &it2 : it->residuals)
    //             if (it2)
    //                 it2.reset();
    //         if (it->lastResiduals)
    //             if (it->lastResiduals->first)
    //                 it->lastResiduals->first.reset();
    //     }
    // }
    

    for (int i = 0, iend = DirPyr.size(); i < iend; ++i)
    {
            delete[] DirPyr[i];
            delete[]  absSquaredGrad[i];
    }
    
    DirPyr.resize(0); DirPyr.shrink_to_fit();
    absSquaredGrad.resize(0); absSquaredGrad.shrink_to_fit();
    targetPrecalc.clear(); targetPrecalc.shrink_to_fit();

    if(efFrame)
        efFrame.reset();
    return;
}

std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    std::vector<size_t> vIndices;
    vIndices.reserve(nFeatures);

    const int nMinCellX = std::max(0,(int)std::floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = std::min((int)mnGridCols-1,(int)std::ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = std::max(0,(int)std::floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = std::min((int)mnGridRows-1,(int)std::ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const std::vector<unsigned short int> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; ++j)
            {
                const cv::KeyPoint &kpUn = mvKeys[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = std::round((kp.pt.x ) * mfGridElementWidthInv);
    posY = std::round((kp.pt.y) * mfGridElementHeightInv);
    if (posX < 0 || posX >= mnGridCols || posY < 0 || posY >= mnGridRows)
        return false;
    return true;
}

}