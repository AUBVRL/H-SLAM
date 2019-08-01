#include "OnlineCalibrator.h"
#include <Settings.h>
#include "Detector.h"
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
namespace FSLAM
{

/* Online Calibrator class */

OnlineCalibrator::OnlineCalibrator()
{
    Detector = std::make_shared<ORBDetector>();
}

void OnlineCalibrator::AddFrame(cv::Mat Image)
{
    Frames.push_back(std::make_shared<Frame>(Image, Detector));

}


/* Frame class */
Frame::Frame(cv::Mat Image, std::shared_ptr<ORBDetector>& Detector_): image(Image), Detector(Detector_)
{
    Detector->ExtractFeatures(image, mvKeys, Descriptors, nOrb);
    features.resize(nOrb);

    for (int i = 0; i < nOrb; i ++)
    {
        features[i] = std::make_shared<Feature>(mvKeys[i]);
        features[i]->InterpolateFeaturePatch(image);
    }
        
}


/* Feature class */
Feature::Feature(cv::KeyPoint Kp_): Kp(Kp_)
{
}

void Feature::InterpolateFeaturePatch(cv::Mat Image)
{
    const uchar *center = &Image.at<uchar>(cvRound(Kp.pt.y), cvRound(Kp.pt.x));
    float angle = (float)Kp.angle * (float)(CV_PI / 180.f);
    float a = (float)cos(angle), b = (float)sin(angle);
    int step = (int)Image.step;
    cv::Mat Temp = cv::Mat::zeros(2*patchsize+1, 2*patchsize+1,CV_8U);
    // std::vector<double> result;
    
for(int x_offset = -patchsize;x_offset <= patchsize;x_offset++)
    {
        for(int y_offset = -patchsize;y_offset <= patchsize;y_offset++)
        {
            // double floor_x = std::floor(x);
            // double ceil_x  = std::ceil(x);
    
            // double floor_y = std::floor(y);
            // double ceil_y  = std::ceil(y);
    
            // // Normalize x,y to be in [0,1)
            // double x_normalized = x - floor_x;
            // double y_normalized = y - floor_y;
    
            // // Get bilinear interpolation weights
            // double w1 = (1-x_normalized)*(1-y_normalized);
            // double w2 = x_normalized*(1-y_normalized);
            // double w3 = (1-x_normalized)*y_normalized;
            // double w4 = x_normalized*y_normalized;
    
            // // Evaluate image locations
            // double i1 = static_cast<double>(image.at<uchar>(floor_y,floor_x));
            // double i2 = static_cast<double>(image.at<uchar>(floor_y,ceil_x));
            // double i3 = static_cast<double>(image.at<uchar>(ceil_y,floor_x));
            // double i4 = static_cast<double>(image.at<uchar>(ceil_y,ceil_x));
    
            // // Interpolate the result
            // return w1*i1 + w2*i2 + w3*i3 + w4*i4;
            uchar pixel = center[cvRound(y_offset * b + x_offset * a) * step + cvRound(y_offset * a - x_offset * b)];
            Temp.at<uchar>(x_offset+patchsize,y_offset+patchsize) = pixel;
            // result.push_back(pixel);
        }
    }
}



}