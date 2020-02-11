#ifndef __DISPLAY__
#define __DISPLAY__

#include <pangolin/pangolin.h>
#include "GlobalTypes.h"
#include <boost/thread.hpp>

static const std::string main_window_name = "FSLAM";

namespace cv
{
class Mat;
class KeyPoint;
}

namespace FSLAM
{

class Frame;

struct InternalImage
{
public:
    InternalImage() {}
    ~InternalImage() {}
    pangolin::GlTexture FeatureFrameTexture;
    bool IsTextureGood = false;
    bool HaveNewImage = false;
    unsigned char* Image;
    int Width = 0;
    int Height = 0;
};

struct FrameDisplayData
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FrameDisplayData(std::shared_ptr<ImageData> _ImgData, std::vector<cv::KeyPoint>& mvKeys, SE3 _Pose)
    {
        ImgData = _ImgData;
        Keys = mvKeys;
        Pose = _Pose.matrix();
    } 
    std::shared_ptr<ImageData> ImgData;
    std::vector<cv::KeyPoint> Keys;
    Mat44 Pose;
};

struct GUI
{
public:
    
    GUI();
    ~GUI();
    void setup();
    void run();
    // void UploadDepthKeyFrameImage(unsigned char* _In, int width, int height);
    void UploadPoints(std::vector<float> Points);
    void Reset();
    boost::thread render_loop;
    bool isDead = false;

// public:
    void ProcessInput();
    void RenderInputFrameImage(std::unique_ptr<InternalImage>& ImageToRender, pangolin::View* CanvasFrame);
    void InitializeImageData(std::unique_ptr<InternalImage> &IntImage, cv::Mat Img);

    int MenuWidth = 150; //pixel units
    pangolin::OpenGlRenderState scene_cam;
    pangolin::View* display_cam;

    //configuration panel settings
    pangolin::View* panel;
    pangolin::View* Nopanel;
    pangolin::Var<bool>* ShowPanel;
    pangolin::Var<bool>* HidePanel;
    pangolin::View* FeatureFrame;
    pangolin::View* DepthKeyFrame;

    pangolin::Var<bool>* ShowDetectedFeatures;
    pangolin::Var<bool>* ShowImages;
    pangolin::Var<bool>* ShowDepthKF;
    pangolin::Var<bool>* Show2D;
    pangolin::Var<bool>* Show3D;
    pangolin::Var<bool>* RecordScreen;
    pangolin::Var<bool>* _Pause;
    pangolin::Var<bool>* bFollow;
    // pangolin::Var<double> a_double;//("ui.A_Double",3,0,5);
    // pangolin::Var<int> a_int; //("ui.An_Int",2,0,5);
    pangolin::View* FramesPanel;

    boost::mutex mRenderThread;
    std::unique_ptr<InternalImage> FrameImage; 
    std::unique_ptr<InternalImage> DepthKfImage;
    std::vector<float> Pts;


    std::deque<std::shared_ptr<FrameDisplayData>> FramesToDraw;
    boost::mutex RunningFrameMutex;
    void UploadRunningFrameData(std::shared_ptr<ImageData> ImgIn, std::vector<cv::KeyPoint>&mvKeys, SE3 _Pose);
    void DrawRunningFrame();
    std::shared_ptr<FrameDisplayData> FrameDisp;
    pangolin::OpenGlMatrix Twc;
    std::deque<Mat44> SmoothMotion;
    void SetPointOfView();
    void drawCam(Mat44 Pose, float lineWidth, float* color, float sizeFactor);

    //Draw Trajectory
    std::vector<Vec3f,Eigen::aligned_allocator<Vec3f>> allFramePoses;

	

};

} // namespace FSLAM

#endif