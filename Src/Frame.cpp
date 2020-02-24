#include "Frame.h"
#include "Detector.h"
#include <opencv2/imgproc.hpp>
#include "IndexThreadReduce.h"
#include "CalibData.h"
#include "ImmaturePoint.h"
#include "photometricUndistorter.h"
#include "MapPoint.h"
// #include "EnergyFunctionalStructs.h"

#include <chrono>
#include <opencv2/highgui.hpp>

namespace FSLAM
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
int Frame::EDGE_THRESHOLD = 19;

Frame::Frame(std::shared_ptr<ImageData> Img, int id, float ab_exposure, std::shared_ptr<FeatureDetector> _Detector, std::shared_ptr<CalibData>_Calib, 
                std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft, bool ForInit): Detector(_Detector), Calib(_Calib)  
{
    if(!GridStructInit)
    {
        mnGridCols = std::ceil(Img->cvImgL.cols / 10);  mnGridRows = std::ceil(Img->cvImgL.rows / 10);
        mnMinX = 0.0f; mnMaxX = Img->cvImgL.cols; mnMinY = 0.0f; mnMaxY = Img->cvImgL.rows;
        mfGridElementWidthInv = static_cast<float>(mnGridCols) / static_cast<float>(mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(mnGridRows) / static_cast<float>(mnMaxY - mnMinY);
        GridStructInit = true;
    }

    NeedRefresh = false;
    MovedByOpt=0;
	statistics_outlierResOnThis = statistics_goodResOnThis = 0;
    
    FlaggedForMarginalization = false;
    frameEnergyTH = 8*8*patternNum;

    CreateIndPyrs(Img->cvImgL, IndPyr);

    //for now I'm only extracting features from highest resolution image!!
    CreateDirPyrs(Img->fImgL, DirPyr);
    nFeatures = 0;
    efFrame = shared_ptr<FrameOptimizationData>(new FrameOptimizationData(id, ab_exposure));
}

void Frame::CreateIndPyrs(cv::Mat& Img, std::vector<cv::Mat>& Pyr)
{
    Pyr.resize(IndPyrLevels);
    for (int i = 0; i < IndPyrLevels; ++i)
    {
        cv::Size wholeSize(Calib->IndPyrSizes[i].width + EDGE_THRESHOLD * 2, Calib->IndPyrSizes[i].height + EDGE_THRESHOLD * 2);
        cv::Mat temp(wholeSize, Img.type());
        Pyr[i] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, Calib->IndPyrSizes[i].width, Calib->IndPyrSizes[i].height));
        if(i != 0)
        {
            cv::resize(Pyr[i - 1], Pyr[i], Calib->IndPyrSizes[i], 0, 0, cv::INTER_LINEAR);
            copyMakeBorder(Pyr[i], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, cv::BORDER_REFLECT_101);

        }
        else
            copyMakeBorder(Img, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, cv::BORDER_REFLECT_101);
    }
}

void Frame::Extract(int id ,bool ForInit, std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft)
{
    Detector->ExtractFeatures(IndPyr[0], DirPyr, id, absSquaredGrad,  mvKeys, Descriptors, nFeatures, (ForInit ? IndNumFeatures : IndNumFeatures), FrontEndThreadPoolLeft); 
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
   
    isReduced = false;
}

void Frame::CreateDirPyrs(std::vector<float>& Img, std::vector<Vec3f*> &DirPyr)
{

    DirPyr.resize(DirPyrLevels);
    absSquaredGrad.resize(DirPyrLevels);

    for (int i = 0; i < DirPyrLevels; ++i)
    {
        size_t PyrSize = Calib->wpyr[i] * Calib->hpyr[i];
        absSquaredGrad[i] = new float [PyrSize];
        DirPyr[i] = new Eigen::Vector3f[PyrSize];
    }

    size_t imSize = Calib->wpyr[0] * Calib->hpyr[0];
    for (int i = 0; i < imSize; ++i) //populate the data of the highest resolution pyramid level
        DirPyr[0][i][0] = Img[i]; // 0th pyr level, ith point, 0 (intensity), 1 (dx), 2 (dy)

    for (int lvl = 0; lvl < DirPyrLevels; ++lvl)
    {
        int wl = Calib->wpyr[lvl], hl = Calib->hpyr[lvl];
        Vec3f* &dI_l = DirPyr[lvl];

        // float* dabs_l = absSquaredGrad[lvl];
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

            // if (lvl == 0)
            // {
                absSquaredGrad[lvl][idx] = dx * dx + dy * dy;
                if (Calib->PhotoUnDistL) //this only works in the left image for now! (consider removing dir pyrs for right images and only keeping highest res with no abssquaredgrad)
                    if (Calib->PhotoUnDistL->GammaValid)
                    {
                        float gw = Calib->PhotoUnDistL->getBGradOnly((float)(dI_l[idx][0]));
                        absSquaredGrad[lvl][idx] *= gw * gw; // convert to gradient of original color space (before removing response).
                    }
            // }
        }
    }
    if (show_gradient_image) //make sure this does not get called in stereo system (parallel thread- remove right image createDirPyr?)
    {
        cv::namedWindow("AbsSquaredGrad", cv::WindowFlags::WINDOW_KEEPRATIO);
        cv::Mat imGrad = cv::Mat(Calib->hpyr[0], Calib->wpyr[0],CV_32F, absSquaredGrad[0]);
        // float* dataptr = imGrad.ptr<float>(0);
        // for (int i = 0, iend = Calib->hpyr[0]* Calib->wpyr[0]; i < iend; ++i)
        //     dataptr[i] = absSquaredGrad[0][i];
        
        imGrad.convertTo(imGrad,CV_8U);
        cv::imshow("AbsSquaredGrad", imGrad);
        cv::waitKey(1);
    }
    
}

Frame::~Frame() 
{
    for (int i = 0, iend = DirPyr.size(); i < iend; ++i)
    {
        delete[] DirPyr[i];
        delete[] absSquaredGrad[i];
    }
    DirPyr.clear(); DirPyr.shrink_to_fit();
    absSquaredGrad.clear(); absSquaredGrad.shrink_to_fit();
}

void Frame::ReduceToEssential()
{
    if(isReduced)
        return;
    isReduced = true;
    
    ImmaturePoints.clear(); ImmaturePoints.shrink_to_fit();

    // mvKeys.resize(0); mvKeys.shrink_to_fit();   
    // mGrid.clear(); mGrid.shrink_to_fit();
    // Descriptors.release();
   
   for (auto &it : pointHessians)
        it.reset();
    for (auto &it: pointHessiansOut)
        it.reset();
    for(auto &it: pointHessiansMarginalized)
        if(it->efPoint)
            it->Clear();

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
    

    Detector.reset();

    IndPyr.clear(); IndPyr.shrink_to_fit();

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