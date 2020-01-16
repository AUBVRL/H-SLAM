#ifndef __DISPLAY__
#define __DISPLAY__

#include <pangolin/pangolin.h>
#include <boost/thread.hpp>

static const std::string main_window_name = "FSLAM";

namespace FSLAM
{

struct InternalImage
{
public:
    InternalImage() {}
    ~InternalImage() {}
    pangolin::GlTexture FeatureFrameTexture;
    bool IsTextureGood = false;
    bool HaveNewImage = false;
    std::vector<unsigned char> Image;
    int Width = 0;
    int Height = 0;
    size_t SizeToCopy = 0;
};

struct GUI
{
public:
    
    GUI();
    ~GUI();
    void setup();
    void run();
    void UploadFrameImage(unsigned char* In_, int width, int height);
    void UploadDepthKeyFrameImage(unsigned char* _In, int width, int height);

    boost::thread render_loop;
    bool isDead = false;

// public:
    void ProcessInput();
    void RenderInputFrameImage(std::unique_ptr<InternalImage>& ImageToRender, pangolin::View* CanvasFrame);

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
    // pangolin::Var<double> a_double;//("ui.A_Double",3,0,5);
    // pangolin::Var<int> a_int; //("ui.An_Int",2,0,5);
    pangolin::View* FramesPanel;

    boost::mutex mSLAMThread;
    std::unique_ptr<InternalImage> FrameImage; 
    std::unique_ptr<InternalImage> DepthKfImage;

};

} // namespace FSLAM

#endif