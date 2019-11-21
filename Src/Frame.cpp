#include "Frame.h"
#include "Detector.h"
#include <opencv2/imgproc.hpp>
#include "IndexThreadReduce.h"
#include "CalibData.h"

#include <chrono>

namespace FSLAM
{

Frame::Frame(std::shared_ptr<ImageData> Img, std::shared_ptr<ORBDetector> _Detector, std::shared_ptr<CalibData>_Calib,
        std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft): //, std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolRight
        Detector(_Detector), EDGE_THRESHOLD(19), Calib(_Calib)//, FrontEndThreadPool(_FrontEndThreadPool)
{
    boost::thread RightImageThread;
    if (Sensortype == Stereo)
    {
        Img->cvImgR.copyTo(ImgR);
        RightImageThread = boost::thread(&Frame::CreateDirPyrs, this, boost::ref(Img->fImgR), boost::ref(RightDirPyr));
        // CreateDirPyrs(Img->fImgR,RightDirPyr);
        // Img->cvImgR.convertTo(ImgR,CV_8U);
        //RightImageThread = boost::thread(&Frame::RightThread, this, boost::ref(Img->cvImgR), boost::ref(RightPyr), boost::ref(mvKeysR), boost::ref(DescriptorsR), boost::ref(nFeaturesR), boost::ref(FrontEndThreadPoolRight));
    }

    CreateIndPyrs(Img->cvImgL, LeftIndPyr);
    //for now I'm only extracting features from highest resolution image!!
    Detector->ExtractFeatures(LeftIndPyr[0], mvKeysL, DescriptorsL, nFeaturesL, FrontEndThreadPoolLeft); 
    CreateDirPyrs(Img->fImgL,LeftDirPyr);
    if (RightImageThread.joinable())
        RightImageThread.join();
}

// void Frame::RightThread(cv::Mat &Img, std::vector<cv::Mat> &Pyr, std::vector<cv::KeyPoint> &mvKeysR, cv::Mat &DescriptorsR, int &nFeaturesR, std::shared_ptr<IndexThreadReduce<Vec10>> &FrontEndThreadPoolRight)
// {
//     CreatePyrs(Img, Pyr);
//     for (int i = 0; i < Pyr.size(); i++)
//         Pyr[i].convertTo(Pyr[i], CV_8U);
//     // Detector->ExtractFeatures(FeatureFrameR, mvKeysR, DescriptorsR, nFeaturesR, FrontEndThreadPoolRight);
// }

void Frame::CreateIndPyrs(cv::Mat& Img, std::vector<cv::Mat>& Pyr)
{
    Pyr.resize(IndPyrLevels);
    for (int i = 0; i < IndPyrLevels; ++i)
    {
        if (i == 0)
        {
            cv::Size sz = cv::Size(Img.cols, Img.rows);
            cv::Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
            cv::Mat temp(wholeSize, Img.type());
            Pyr[i] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));
            copyMakeBorder(Img, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           cv::BORDER_REFLECT_101);
        }
        else
        {
            cv::Size sz = cv::Size(cvRound((float)Pyr[i - 1].cols / IndPyrScaleFactor), cvRound((float)Pyr[i - 1].rows / IndPyrScaleFactor));
            cv::Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
            cv::Mat temp(wholeSize, Img.type());
            Pyr[i] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));
            cv::resize(Pyr[i - 1], Pyr[i], sz, 0, 0, cv::INTER_LINEAR);
            copyMakeBorder(Pyr[i], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           cv::BORDER_REFLECT_101 + cv::BORDER_ISOLATED);
        }
    }
}

void Frame::CreateDirPyrs(std::vector<float>& Img, std::vector<std::vector<Vec3f>> &DirPyr)
{
    DirPyr.resize(DirPyrLevels);
    for (int i = 0; i < DirPyrLevels; ++i)
        DirPyr[i].resize(Calib->wpyr[i] * Calib->hpyr[i]);

    size_t imSize = Calib->wpyr[0] * Calib->hpyr[0];
    for (int i = 0; i < imSize; ++i) //populate the data of the highest resolution pyramid level
        DirPyr[0][i][0] = Img[i];

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

Frame::~Frame()
{
}






}