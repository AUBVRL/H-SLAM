#include "Frame.h"
#include "Detector.h"

namespace FSLAM
{

Frame::Frame(std::shared_ptr<ImageData>& Img, std::shared_ptr<ORBDetector> _Detector): Detector(_Detector)
{
    int PyrSize = 1;

    vfImgL.resize(PyrSize);
    vfImgR.resize(PyrSize);
    vfImgL[0] = new float [Img->cvImgL.cols*Img->cvImgL.rows];
    vfImgR[0] = new float [Img->cvImgR.cols*Img->cvImgR.rows];

    memcpy(vfImgL[0],Img->fImgL,Img->cvImgL.cols*Img->cvImgL.rows*sizeof(float));

    Detector->ExtractFeatures(Img->cvImgL,mvKeysL,DescriptorsL,nFeaturesL);

    if(Sensortype == Stereo || Sensortype == RGBD)
    {
        memcpy(vfImgR[0],Img->fImgR,Img->cvImgR.cols*Img->cvImgR.rows*sizeof(float));

        if(Sensortype == Stereo)
            Detector->ExtractFeatures(Img->cvImgR,mvKeysR,DescriptorsR,nFeaturesR);
    }

}

Frame::~Frame()
{
    for(size_t i =0; i < vfImgL.size(); i ++)
        if(vfImgL[i]) { delete vfImgL[i]; vfImgL[i] = NULL; }
    
    for(size_t i =0; i < vfImgR.size(); i ++)
        if(vfImgR[i]) { delete vfImgR[i]; vfImgR[i] = NULL; }
}






}