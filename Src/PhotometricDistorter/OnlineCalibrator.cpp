#include "OnlineCalibrator.h"
#include "Settings.h"
#include "Detector.h"

namespace FSLAM
{

// /* Online Calibrator class */

// OnlineCalibrator::OnlineCalibrator()
// {
//     Detector = std::make_shared<FeatureDetector>();
//     NumFrames = 100;
// }

// void OnlineCalibrator::ProcessFrame(cv::Mat Image)
// {
//     std::shared_ptr<Frame> NewFrame = std::make_shared<Frame>(Image,Detector);
    
//     //get frames from connectivity graph
//     //match their landmarks to newframe
//     //updateconnectivity
//     //perform the photometric optimization
//     //if not enough matches found create new landmarks
    
    
//     bool needNewKF = true; //(numMatches/numfeatures < 0.7)
//     if(needNewKF)
//     {
//         //Create new Landmarks and update connectivity
//         for (int i = 0 ; i < NewFrame->nOrb ; i ++)
//         {
//             if(!NewFrame->features[i])
//                 NewFrame->features[i] = std::make_shared<LandMark>(NewFrame->mvKeys[i]);
//             else
//             {
//                 std::pair<std::weak_ptr<Frame>,int> ConnectedTo = Connectivity[std::weak_ptr<Frame>(NewFrame)]; 
                
//             }

//         }

        

//         Frames.push(NewFrame);
//         // if(Frames.size() > NumFrames)
//         //     Frames.pop();

//     }
// }


// /* Frame class */
// Frame::Frame(cv::Mat Image, std::shared_ptr<FeatureDetector> Detector_): image(Image), Detector(Detector_)
// {
//     Detector->ExtractFeatures(image, mvKeys, Descriptors, nOrb);
//     for (int i = 0; i < nOrb; i ++)
//         InterpolateFeaturePatches(i);
//     features.resize(nOrb);
// }

// void Frame::InterpolateFeaturePatches(int keyInd)
// {
//     // const uchar *center = &Image.at<uchar>(cvRound(Kp.pt.y), cvRound(Kp.pt.x));
//     float angle = (float)mvKeys[keyInd].angle * (float)(CV_PI / 180.f);
//     float a = (float)cos(angle), b = (float)sin(angle);
//     std::vector<double> Patch;
//     // int step = (int)Image.step;
//     // cv::Mat Temp = cv::Mat::zeros(2*patchsize+1, 2*patchsize+1,CV_8U);
//     // std::vector<double> result;

//     // cv::Mat Disp;
//     // Image.copyTo(Disp); 
//     // cv::rectangle(Disp, Kp.pt - cv::Point2f(3,3), Kp.pt+cv::Point2f(3,3),cv::Scalar(0,255,0),1,8 );
//     // cv::imshow("ori",Disp);
//     // std::cout<<"angle: "<<Kp.angle<<std::endl;
// for(int x_offset = -patchsize;x_offset <= patchsize;x_offset++)
//     {
//         for(int y_offset = -patchsize;y_offset <= patchsize;y_offset++)
//         {
//             double xoffset = x_offset*a - y_offset * b;
//             double yoffset = x_offset * b + y_offset * a;

//             double floor_x = std::floor(mvKeys[keyInd].pt.x + xoffset);
//             double ceil_x  = std::ceil(mvKeys[keyInd].pt.x + xoffset);
            
//             double floor_y = std::floor(mvKeys[keyInd].pt.y + yoffset);
//             double ceil_y  = std::ceil(mvKeys[keyInd].pt.y + yoffset);

//             double x_normalized = (mvKeys[keyInd].pt.x + xoffset) - floor_x;
//             double y_normalized = (mvKeys[keyInd].pt.y + yoffset) - floor_y;
    
//             double w1 = (1-x_normalized)*(1-y_normalized);
//             double w2 = x_normalized*(1-y_normalized);
//             double w3 = (1-x_normalized)*y_normalized;
//             double w4 = x_normalized*y_normalized;
    
//             double i1 = static_cast<double>(image.at<uchar>(floor_y,floor_x));
//             double i2 = static_cast<double>(image.at<uchar>(floor_y,ceil_x));
//             double i3 = static_cast<double>(image.at<uchar>(ceil_y,floor_x));
//             double i4 = static_cast<double>(image.at<uchar>(ceil_y,ceil_x));
    
//             // uchar pixel = center[cvRound(x_offset * b + y_offset * a) * step + cvRound(x_offset * a - y_offset * b)];
//             // Temp.at<uchar>(y_offset+patchsize,x_offset+patchsize) = (w1*i1 + w2*i2 + w3*i3 + w4*i4);
//             Patch.push_back((w1*i1 + w2*i2 + w3*i3 + w4*i4));
//         }
//     }
//     mvKeysPatches.push_back(Patch);
//     // cv::namedWindow("patch", cv::WINDOW_KEEPRATIO|cv::WND_PROP_FULLSCREEN);
//     // cv::imshow("patch",Temp);
//     // cv::waitKey(0);
// }



}