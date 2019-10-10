#include "Frame.h"
#include "Detector.h"
#include <opencv2/imgproc.hpp>
#include "IndexThreadReduce.h"

namespace FSLAM
{

Frame::Frame(std::shared_ptr<ImageData> Img, std::shared_ptr<ORBDetector> _Detector,
        std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolLeft, std::shared_ptr<IndexThreadReduce<Vec10>> FrontEndThreadPoolRight): 
        Detector(_Detector), EDGE_THRESHOLD(19)//, FrontEndThreadPool(_FrontEndThreadPool)
{   
    boost::thread RightImageThread;
    if(Sensortype == Stereo)
        RightImageThread = boost::thread(&Frame::RightThread, this, boost::ref(Img->cvImgR), boost::ref(RightPyr), boost::ref(mvKeysR) ,boost::ref(DescriptorsR), boost::ref(nFeaturesR), boost::ref(FrontEndThreadPoolRight));
    // if(Sensortype == Stereo)
    // {
    //     CreatePyrs(Img->cvImgR, RightPyr); //This is faster than parallelizing it!!   
    //     Detector->ExtractFeatures(Img->cvImgR,mvKeysR,DescriptorsR,nFeaturesR, FrontEndThreadPoolRight);
    // }
    
    CreatePyrs(Img->cvImgL, LeftPyr);
    Detector->ExtractFeatures(Img->cvImgL,mvKeysL,DescriptorsL,nFeaturesL, FrontEndThreadPoolLeft);

    if(Sensortype == Stereo)
        if(RightImageThread.joinable())
            RightImageThread.join();
    
    // if(Sensortype == Stereo || Sensortype == RGBD)
    // {
    //     memcpy(vfImgR[0],Img->fImgR,Img->cvImgR.cols*Img->cvImgR.rows*sizeof(float));

    //     // if(Sensortype == Stereo)
    //     //     Detector->ExtractFeatures(Img->cvImgR,mvKeysR,DescriptorsR,nFeaturesR);
    // }
}

void Frame::RightThread(cv::Mat& Img, std::vector<cv::Mat>& Pyr, std::vector<cv::KeyPoint>& mvKeysR,cv::Mat& DescriptorsR, int& nFeaturesR, std::shared_ptr<IndexThreadReduce<Vec10>>& FrontEndThreadPoolRight)
{
    CreatePyrs(Img, Pyr);
    Detector->ExtractFeatures(Pyr[0],mvKeysR,DescriptorsR, nFeaturesR, FrontEndThreadPoolRight);
}

void Frame::CreatePyrs(cv::Mat& Img, std::vector<cv::Mat>& Pyr)
{
    Pyr.resize(PyrLevels);
    for (int i = 0; i < PyrLevels; ++i)
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
            cv::Size sz = cv::Size(cvRound((float)Pyr[i - 1].cols / PyrScaleFactor), cvRound((float)Pyr[i - 1].rows / PyrScaleFactor));
            cv::Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
            cv::Mat temp(wholeSize, Img.type());
            Pyr[i] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));
            cv::resize(Pyr[i - 1], Pyr[i], sz, 0, 0, cv::INTER_LINEAR);
            copyMakeBorder(Pyr[i], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           cv::BORDER_REFLECT_101 + cv::BORDER_ISOLATED);
        }
        // cv::GaussianBlur( Pyr[i], Pyr[i], cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

 
    }
    // Pyr.resize(PyrLevels);
    // Img.copyTo(Pyr[0]);
    // // cv::GaussianBlur( Img, Pyr[0], cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    // for (size_t i = 1; i < PyrLevels; ++i)
    // {      
    //     cv::Size sz(cvRound((float)Pyr[i-1].cols/PyrScaleFactor), cvRound((float)Pyr[i-1].rows/PyrScaleFactor));
    //     cv::resize(Pyr[i-1],Pyr[i],sz,0,0,CV_INTER_LINEAR);
    //     // cv::GaussianBlur( Pyr[i], Pyr[i], cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    // }
}

// void Frame::CreatePyrsAndExtractFeats(std::shared_ptr<ImageData> Img)
// {
    // vfImgL.resize(PyrSize);
    // vfImgR.resize(PyrSize);

    // vfImgL[0] = new float[Img->cvImgL.cols*Img->cvImgL.rows];
    // if(!Img->cvImgR.empty())
    //     vfImgR[0] = new float[Img->cvImgR.cols*Img->cvImgR.rows];

    // for(int i=1;i<PyrSize;i++)
	// {
		// vfImgL[i] = new float[wG[i]*hG[i]];
		// absSquaredGrad[i] = new float[wG[i]*hG[i]];
	// }

    // std::vector<cv::Mat> PyrsL;
    // std::vector<cv::Mat> PyrsR;

    // vfImgL.resize(PyrSize);
    // vfImgR.resize(PyrSize);

    // PyrsL.resize(PyrSize);
    // PyrsR.resize(PyrSize);

    // PyrsL[0] = Img->cvImgL;
    // if(!Img->cvImgR.empty())
    //     PyrsR[0] = Img->cvImgR;

    // for (int i = 1 ; i < PyrSize; i++ )
    // {
    //     cv::Size sz(cvRound((float)PyrsL[i-1].cols/ScaleFactor), cvRound((float)PyrsL[i-1].rows/ScaleFactor));
       
    //     cv::resize(PyrsL[i-1],PyrsL[i],sz,0,0,CV_INTER_LINEAR);
    //     if(!PyrsR[0].empty())
    //         cv::resize(PyrsR[i-1],PyrsR[i],sz,0,0,CV_INTER_LINEAR);
    // }

    // for (int i = 0 ; i < PyrSize; i++)
    // {
        // vfImgL[i] = new float [PyrsL[i].cols*PyrsL[i].rows];

        // for(int j = 0; j < PyrsL[i].cols; j ++)
        //     for (int k = 0; k < PyrsL[i].rows; k ++)
        //         vfImgL[i][k+k*j]= (float) PyrsL[i].at<uchar>(k,j);

        // memcpy(vfImgL[i], PyrsL[i].data, PyrsL[i].rows*PyrsL[i].cols*sizeof(float));
        // if(!PyrsR[i].empty())
        // {
        //     vfImgR[i] = new float [PyrsR[i].cols*PyrsR[i].rows];
        //     memcpy(vfImgR[i], PyrsR[i].data, PyrsR[i].rows*PyrsR[i].cols*sizeof(float));
        // }
    // }

        // cv::Mat Test = cv::Mat(Img->cvImgL.size().height,Img->cvImgL.size().width,CV_32F,Img->fImgL);
        // Test.convertTo(Test,CV_8U);
        // cv::imshow("test", Test);
        // cv::waitKey(1);

// }

Frame::~Frame()
{
    // for(size_t i =0; i < vfImgL.size(); i ++)
    //     if(vfImgL[i]) { delete vfImgL[i]; vfImgL[i] = NULL; }
    
    // for(size_t i =0; i < vfImgR.size(); i ++)
    //     if(vfImgR[i]) { delete vfImgR[i]; vfImgR[i] = NULL; }
}






}