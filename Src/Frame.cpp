#include "Frame.h"
#include "Detector.h"
#include <opencv2/imgproc.hpp>
#include "IndexThreadReduce.h"
#include "CalibData.h"
#include "ImmaturePoint.h"
#include "photometricUndistorter.h"

#include <chrono>
#include <opencv2/highgui.hpp>

namespace FSLAM
{
size_t Frame::Globalid = 0;
size_t Frame::GlobalIncoming_id = 0;


Frame::Frame(std::shared_ptr<ImageData> Img, std::shared_ptr<ORBDetector> _Detector, std::shared_ptr<CalibData>_Calib, std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft, bool ForInit):
Detector(_Detector), EDGE_THRESHOLD(19), Calib(_Calib)  
{
    frameNumb = GlobalIncoming_id; GlobalIncoming_id++; //keeps track of the number of frames processed
    id = Globalid; Globalid++; //Set the frame id (this might be reset to 0 in the initializer so that the first keyframe in the map has id= 0)
    poseValid=false;
    
    marginalizedAt=-1;
    movedByOpt=0;
	statistics_outlierResOnThis = statistics_goodResOnThis = 0;
    
    camToWorld = SE3();
    camToTrackingRef = SE3();
    aff_g2l_internal = AffLight(0,0); //Past to present affine model of left image
    aff_g2l_internalR = AffLight(0,0); //Left to right affine model
    
    ab_exposure = Img->ExposureL; //Exposure time of the left image
    ab_exposureR = 1;
    flaggedForMarginalization = false;
    frameEnergyTH = 8*8*patternNum;

    if (Sensortype == Stereo)
    {
        Img->cvImgR.copyTo(ImgR);
        ab_exposureR = Img->ExposureR;
        RightImageThread = boost::thread(&Frame::CreateDirPyrs, this, boost::ref(Img->fImgR), boost::ref(RightDirPyr));
    }

    CreateIndPyrs(Img->cvImgL, LeftIndPyr);

    //for now I'm only extracting features from highest resolution image!!
    CreateDirPyrs(Img->fImgL,LeftDirPyr);
    if (RightImageThread.joinable())
        RightImageThread.join();
    
    Detector->ExtractFeatures(LeftIndPyr[0], mvKeys, Descriptors, nFeatures, (ForInit ? 2*IndNumFeatures : IndNumFeatures), FrontEndThreadPoolLeft); 

    //Assign Features to Grid
    mnGridCols = std::ceil(Img->cvImgL.cols / 10);
    mnGridRows = std::ceil(Img->cvImgL.rows / 10);
    mnMinX = 0.0f; mnMaxX = Img->cvImgL.cols; mnMinY = 0.0f; mnMaxY = Img->cvImgL.rows;
    mfGridElementWidthInv = static_cast<float>(mnGridCols) / static_cast<float>(mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(mnGridRows) / static_cast<float>(mnMaxY - mnMinY);
    mGrid.resize(mnGridCols);
    for (int i = 0; i < mnGridCols; ++i)
        mGrid[i].resize(mnGridRows);
    for (unsigned short int i = 0; i < nFeatures; ++i)
    {
        int nGridPosX, nGridPosY;
        if (PosInGrid(mvKeys[i], nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
    
    isReduced = false;
    isKeyFrame = false;
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

void Frame::CreateDirPyrs(std::vector<float>& Img, std::vector<std::vector<Vec3f>> &DirPyr)
{
    DirPyr.resize(DirPyrLevels);
    absSquaredGrad.resize(Calib->wpyr[0] * Calib->hpyr[0]); // store the absolute squared intensity gradient per pixel
    for (int i = 0; i < DirPyrLevels; ++i)
        DirPyr[i].resize(Calib->wpyr[i] * Calib->hpyr[i]);

    size_t imSize = Calib->wpyr[0] * Calib->hpyr[0];
    for (int i = 0; i < imSize; ++i) //populate the data of the highest resolution pyramid level
        DirPyr[0][i][0] = Img[i]; // 0th pyr level, ith point, 0 (intensity), 1 (dx), 2 (dy)

    for (int lvl = 0; lvl < DirPyrLevels; ++lvl)
    {
        int wl = Calib->wpyr[lvl], hl = Calib->hpyr[lvl];
        std::vector<Vec3f> &dI_l = DirPyr[lvl];

        // float* dabs_l = absSquaredGrad[lvl];
        if (lvl > 0)
        {
            int lvlm1 = lvl - 1;
            int wlm1 = Calib->wpyr[lvlm1];
            std::vector<Vec3f> &dI_lm = DirPyr[lvlm1];

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

            if (lvl == 0)
            {
                absSquaredGrad[idx] = dx * dx + dy * dy;
                if (Calib->PhotoUnDistL) //this only works in the left image for now! (consider removing dir pyrs for right images and only keeping highest res with no abssquaredgrad)
                    if (Calib->PhotoUnDistL->GammaValid)
                    {
                        float gw = Calib->PhotoUnDistL->getBGradOnly((float)(dI_l[idx][0]));
                        absSquaredGrad[idx] *= gw * gw; // convert to gradient of original color space (before removing response).
                    }
            }
        }
    }
    if (show_gradient_image) //make sure this does not get called in stereo system (parallel thread- remove right image createDirPyr?)
    {
        cv::namedWindow("AbsSquaredGrad", cv::WindowFlags::WINDOW_KEEPRATIO);
        cv::Mat imGrad = cv::Mat(Calib->hpyr[0], Calib->wpyr[0],CV_32F, &absSquaredGrad[0]);
        imGrad.convertTo(imGrad,CV_8U);
        cv::imshow("AbsSquaredGrad", imGrad);
        cv::waitKey(1);
    }
    
}

Frame::~Frame() {}

void Frame::ComputeStereoDepth(std::shared_ptr<Frame> FramePtr, std::vector<std::shared_ptr<ImmaturePoint>>& ImPts, int min, int max)
{
    for (size_t i = min ; i < max; ++i)
    {
        std::shared_ptr<ImmaturePoint> impt = std::shared_ptr<ImmaturePoint>(new ImmaturePoint(FramePtr->mvKeys[i].pt.x, FramePtr->mvKeys[i].pt.y, i, FramePtr, 0, Calib));
	    if(std::isfinite(impt->energyTH))
            ImPts[i] = impt;
        impt->idepth_max_stereo = NAN;
        impt->idepth_min_stereo = 0;

        impt->traceStereo(FramePtr->RightDirPyr[0],Calib);
        // if(impt->lastTraceStatus == ImmaturePointStatus::IPS_GOOD)
        //     std::cout<<impt->idepth_stereo<<std::endl;
    }
   
	return ;
}

void Frame::ReduceToEssential(bool KeepIndirectData)
{
    isReduced = true;
    if(!KeepIndirectData) //if true (global keyframe) keep these
    {
        mvKeys.clear(); mvKeys.shrink_to_fit();   
        Descriptors.release();
    }
    Detector.reset();
    LeftIndPyr.clear(); LeftIndPyr.shrink_to_fit();
    LeftDirPyr.clear(); LeftDirPyr.shrink_to_fit();  
    RightDirPyr.clear(); RightDirPyr.shrink_to_fit();   
    absSquaredGrad.clear(); absSquaredGrad.shrink_to_fit();
    
    ImmaturePointsLeftRight.clear(); ImmaturePointsLeftRight.shrink_to_fit();
    targetPrecalc.clear(); targetPrecalc.shrink_to_fit();

    
    ImgR.release();
    Calib.reset();
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
            const std::vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
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