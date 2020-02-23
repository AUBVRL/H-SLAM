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
static Vec3b blue = Vec3b(0, 0, 255);
static Vec3b red = Vec3b(255, 0, 0);
static Vec3b green = Vec3b(0, 255, 0);
static Vec3b black = Vec3b(0, 0, 0);
static Vec3b White = Vec3b(255, 255, 255);
static Vec3b yellow = Vec3b(255, 215, 0);
static Vec3b orange = Vec3b(255, 140, 0);

class FrameShell;

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

struct KFDisplay
{
    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    SE3 camToWorld;
    bool PoseValid;
    KFDisplay();
    void RefreshPC(std::shared_ptr<FrameShell> _In);
	pangolin::GlBuffer vertexBuffer;
	pangolin::GlBuffer colorBuffer;
    bool ValidBuffer;
    int numGLBufferPoints;
	int numGLBufferGoodPoints;
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
    pangolin::Var<bool>* ShowFullTrajectory;
    pangolin::Var<bool>* ShowAllKfs;

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
    cv::Mat Dest;
    std::shared_ptr<FrameDisplayData> FrameDisp;
    pangolin::OpenGlMatrix Twc;
    std::deque<Mat44> SmoothMotion;
    void SetPointOfView();
    void drawCam(Mat44 Pose, float lineWidth, Vec3b& color, float sizeFactor);

    //Draw Trajectory
    std::vector<Vec3f,Eigen::aligned_allocator<Vec3f>> allFramePoses;

    void UploadKeyFrame(std::shared_ptr<FrameShell> FrameIn);
    void DrawKeyFrames();
    std::vector< std::pair< std::shared_ptr<FrameShell>, std::shared_ptr<KFDisplay> > > AllKeyframes;
    boost::mutex KeyframesMutex;

};

} // namespace FSLAM

#endif