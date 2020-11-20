#include "Matcher.h"
#include "Frame.h"

#include<limits.h>

#include<stdint-gcc.h>

using namespace std;

namespace SLAM
{

const int Matcher::TH_HIGH = 100;
const int Matcher::TH_LOW = 50;
const int Matcher::HISTO_LENGTH = 30;

Matcher::Matcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}



int Matcher::SearchLastFrameByProjection(shared_ptr<Frame> CurrentFrame, shared_ptr<Frame> LastFrame, const float th, const bool bMono = true)
{
    int nmatches = 0;

//     // Rotation Histogram (to check rotation consistency)
//     vector<int> rotHist[HISTO_LENGTH];
//     for(int i=0;i<HISTO_LENGTH;i++)
//         rotHist[i].reserve(500);
//     const float factor = 1.0f/HISTO_LENGTH;

//     const cv::Mat Rcw = CurrentFrame-> .mTcw.rowRange(0,3).colRange(0,3);
//     const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

//  //   const cv::Mat twc = -Rcw.t()*tcw;

// //    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
// //    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);

//    // const cv::Mat tlc = Rlw*twc+tlw;

//    // const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
//     //const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;


//     std::vector<cv::DMatch> DMatches;
//     for (int i = 0; i < LastFrame->nFeatures; ++i)
//     {
//         // shared_ptr<MapPoint> pMP = LastFrame->mv
//         MapPoint* pMP = LastFrame.mvpMapPoints[i];

//         if(pMP)
//         {
//              if(!LastFrame.mvbOutlier[i])
//              {
//                  // Project
//                  cv::Mat x3Dw = pMP->GetWorldPos();
//                  cv::Mat x3Dc = Rcw*x3Dw+tcw;

//                  const float xc = x3Dc.at<float>(0);
//                  const float yc = x3Dc.at<float>(1);
//                  const float invzc = 1.0/x3Dc.at<float>(2);

//                  if(invzc<0)
//                      continue;

//                  float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
//                  float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

//                  if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
//                      continue;
//                  if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
//                      continue;

//                  int nLastOctave = LastFrame.mvKeys[i].octave;

//                  // Search in a window. Size depends on scale
//                  float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

//                  vector<size_t> vIndices2;

//                  //if(bForward)
//                  //    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
//                  //else if(bBackward)
//                   //   vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
//                  //else
//                      vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);

//                  if(vIndices2.empty())
//                      continue;

//                  const cv::Mat dMP = pMP->GetDescriptor();

//                  int bestDist = 256;
//                  int bestIdx2 = -1;

//                  for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
//                  {
//                      const size_t i2 = *vit;
//                      if(CurrentFrame.mvpMapPoints[i2]) continue;
//                          //if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
//                          //    continue;

//                     // if(CurrentFrame.mvuRight[i2]>0)
//                     // {
//                      //    const float ur = u - CurrentFrame.mbf*invzc;
//                      //    const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
//                      //    if(er>radius)
//                      //        continue;
//                     // }
//                      const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);
//                       const int dist = DescriptorDistance(dMP,d);

//                       if(dist<bestDist)
//                       {
//                           bestDist=dist;
//                           bestIdx2=i2;
//                       }
//                   }

//                   if(bestDist<=TH_HIGH)
//                   {
//                       CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
//                       nmatches++;
//                       DMatches.push_back(cv::DMatch(bestIdx2,i,bestDist));
//                       if(mbCheckOrientation)
//                       {
//                           float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
//                           if(rot<0.0)
//                               rot+=360.0f;
//                           int bin = round(rot*factor);
//                           if(bin==HISTO_LENGTH)
//                               bin=0;
//                           assert(bin>=0 && bin<HISTO_LENGTH);
//                           rotHist[bin].push_back(bestIdx2);
//                       }
//                   }
//               }
//           }
//       }


//     //Apply rotation consistency
//     if(mbCheckOrientation)
//     {
//         int ind1=-1;
//         int ind2=-1;
//         int ind3=-1;

//         ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

//         for(int i=0; i<HISTO_LENGTH; i++)
//         {
//             if(i!=ind1 && i!=ind2 && i!=ind3)
//             {
//                 for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
//                 {
//                     CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
//                     nmatches--;
//                  //   DMatches[rotHist[i][j]]=cv::DMatch();
//                 }
//             }
//         }
//     }
//     // if(dso::DrawLocalMatches)
//     // {
//     //   cv::namedWindow("Matches",cv::WINDOW_FREERATIO);
//     //   cv::Mat MatchesImage;
//     //   cv::drawMatches( CurrentFrame.Image, CurrentFrame.mvKeysUn, LastFrame.Image, LastFrame.mvKeysUn,
//     //       DMatches, MatchesImage, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::DEFAULT ); //NOT_DRAW_SINGLE_POINTS
//     //   cv::imshow("Matches",MatchesImage);
//     //   cv::waitKey(0);
//     // }
//    // if(!dso::DrawLocalMatches && cv::getWindowProperty("Matches",cv::WND_PROP_VISIBLE))cv::destroyWindow("Matches"); //find a way to properly dispose of opencv window
    return nmatches;
}



} //namespace ORB_SLAM
