#ifndef __DISPLAY__
#define __DISPLAY__

#include <pangolin/pangolin.h>
#include <thread>
#include "globalTypes.h"

static const string main_window_name = "SLAM";

namespace SLAM
{
    class FrameShell;
    class System;
    static Vec3b blue = Vec3b(0, 0, 255);
    static Vec3b red = Vec3b(255, 0, 0);
    static Vec3b green = Vec3b(0, 255, 0);
    static Vec3b black = Vec3b(0, 0, 0);
    static Vec3b White = Vec3b(255, 255, 255);
    static Vec3b yellow = Vec3b(255, 215, 0);
    static Vec3b orange = Vec3b(255, 140, 0);
    struct InternalImage
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        InternalImage() {}
        ~InternalImage() { }
        pangolin::GlTexture FeatureFrameTexture;
        bool IsTextureGood = false;
        bool HaveNewImage = false;
        unsigned char *Image = 0;
        int Width = 0;
        int Height = 0;
    };

    struct FrameDisplayData
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        FrameDisplayData(shared_ptr<ImageData> _ImgData, vector<cv::KeyPoint> &mvKeys, SE3 _Pose)
        {
            ImgData = _ImgData;
            Keys = mvKeys;
            Pose = _Pose.matrix();
        }
        shared_ptr<ImageData> ImgData;
        vector<cv::KeyPoint> Keys;
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
        // void UploadPoints(std::vector<float> Points);
        void Reset();
        thread render_loop;
        bool isDead = false;

        // public:
        void ProcessInput();

        // void RenderInputFrameImage(unique_ptr<InternalImage> &ImageToRender, pangolin::View *CanvasFrame);
        // void InitializeImageData(unique_ptr<InternalImage> &IntImage, cv::Mat Img);

        int MenuWidth = 150; //pixel units

        //3D display window
        pangolin::OpenGlRenderState scene_cam;
        pangolin::View *display_cam;
        pangolin::OpenGlMatrix Twc;
        deque<Mat44> SmoothMotion;
        void SetPointOfView();

        //configuration panel settings
        pangolin::View *panel;
        pangolin::View *Nopanel;
        pangolin::Var<bool> *ShowPanel;
        pangolin::Var<bool> *HidePanel;


        pangolin::Var<bool> *ShowDetectedFeatures;
        pangolin::Var<bool> *ShowImages;
        pangolin::Var<bool> *ShowDepthKF;
        pangolin::Var<bool> *ShowFullTrajectory;
        pangolin::Var<bool> *ShowAllKfs;
        pangolin::Var<bool> *ShowFullConnectivity;
        pangolin::Var<bool> *RecordScreen;
        pangolin::Var<bool> *_Pause;
        pangolin::Var<bool> *_Reset;
        pangolin::Var<bool> *bFollow;
        // pangolin::Var<double> a_double;//("ui.A_Double",3,0,5);
        // pangolin::Var<int> a_int; //("ui.An_Int",2,0,5);
        
        //Images display
        pangolin::View *FramesPanel;
        void renderInternalFrame(std::unique_ptr<InternalImage> &ImageToRender, pangolin::View* CanvasFrame);
        void setInternalImageData(std::unique_ptr<InternalImage> &InternalImage, cv::Mat &Img);
        
        unique_ptr<InternalImage> FrameImage;
        pangolin::View *FeatureFrame;
        deque<shared_ptr<FrameDisplayData>> FramesToDraw;
        mutex RunningFrameMutex;
        shared_ptr<FrameDisplayData> FrameDisp;
        void UploadCurrentFrame(std::shared_ptr<ImageData> ImgIn, std::vector<cv::KeyPoint> &mvKeys, SE3 _Pose, bool poseValid);
        void CreateCurrentFrame();


        unique_ptr<InternalImage> DepthKfImage;
        pangolin::View *DepthKeyFrame;
        cv::Mat KfToDraw;
        vector<pair<float, int>> kfdepthmap;
        bool isKFImgChanged = false;
        mutex ActiveKFMutex;
        void UploadActiveKF(cv::Mat ActiveKF, vector<pair<float, int>>& dmap);
        void CreateActiveKF();

        void DrawTrajectory();
        void drawCam(Mat44 Pose, float lineWidth, Vec3b& color, float sizeFactor);



        mutex mRenderThread;


        
        
        vector<Vec3f, Eigen::aligned_allocator<Vec3f>> allFramePoses; //Draw Trajectory

        // vector<float> Pts;

        // 
        // void DrawRunningFrame();
        cv::Mat Dest;
        // shared_ptr<FrameDisplayData> FrameDisp;
    
        // void drawCam(Mat44 Pose, float lineWidth, Vec3b &color, float sizeFactor);



        void UploadKeyFrame(std::shared_ptr<FrameShell> FrameIn);
        void DrawKeyFrames();
        vector<pair<shared_ptr<FrameShell>, shared_ptr<KFDisplay>>> AllKeyframes;
        mutex KeyframesMutex;
    };

} // namespace SLAM

#endif