#include "Frame.h"
#include "Detector.h"
#include <opencv2/imgproc.hpp>
// #include "IndexThreadReduce.h"
#include "CalibData.h"
#include "ImmaturePoint.h"

#include <chrono>
#include <opencv2/highgui.hpp>

namespace FSLAM
{

Frame::Frame(std::shared_ptr<ImageData> Img, std::shared_ptr<ORBDetector> _Detector, std::shared_ptr<CalibData>_Calib): //std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft
Detector(_Detector), EDGE_THRESHOLD(19), Calib(_Calib)  
{
    static size_t Globalid = 0; id = Globalid; Globalid++; //Set frameId
    poseValid=true;
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
    FrameState = RegularFrame;
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

            // dabs_l[idx] = dx*dx+dy*dy;

            // if(setting_gammaWeightsPixelSelect==1 && HCalib!=0)
            // {
            // 	float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
            // 	dabs_l[idx] *= gw*gw;	// convert to gradient of original color space (before removing response).
            // }
        }
	}

}

Frame::~Frame() {}


void Frame::ComputeStereoDepth(std::shared_ptr<Frame> FramePtr, int min, int max)
{
    for (size_t i = min ; i < max; ++i)
    {
        std::shared_ptr<ImmaturePoint> impt = std::make_shared<ImmaturePoint>(FramePtr->mvKeysL[i].pt.x, FramePtr->mvKeysL[i].pt.y, i, FramePtr, 0, Calib);
	    if(std::isfinite(impt->energyTH))
            FramePtr->ImmaturePointsLeftRight[i] = impt;
    }
   
	return ;
}

void Frame::ReduceToEssential(bool KeepIndirectData)
{
    FrameState = ReducedFrame;
    if(!KeepIndirectData) //if true (global keyframe) keep these
    {
        mvKeysL.clear(); mvKeysL.shrink_to_fit();   
        DescriptorsL.release();
    }
    Detector.reset();
    LeftIndPyr.clear(); LeftIndPyr.shrink_to_fit();
    LeftDirPyr.clear(); LeftDirPyr.shrink_to_fit();  
    RightDirPyr.clear(); RightDirPyr.shrink_to_fit();   
    
    ImmaturePointsLeftRight.clear(); ImmaturePointsLeftRight.shrink_to_fit();
    targetPrecalc.clear(); targetPrecalc.shrink_to_fit();

    
    ImgR.release();
    Calib.reset();
    return;
}




}