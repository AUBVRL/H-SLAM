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
    boost::thread render_loop;
    bool isDead = false;

private:
    void ProcessInput();
    void RenderInputFrameImage(std::unique_ptr<InternalImage>& ImageToRender, std::unique_ptr<pangolin::View>& CanvasFrame);

    int MenuWidth = 150; //pixel units
    pangolin::OpenGlRenderState scene_cam;
    std::unique_ptr<pangolin::View> display_cam;

    //configuration panel settings
    std::unique_ptr<pangolin::View> panel;
    std::unique_ptr<pangolin::View> Nopanel;
    std::unique_ptr<pangolin::Var<bool>> ShowPanel;
    std::unique_ptr<pangolin::Var<bool>> HidePanel;
    std::unique_ptr<pangolin::View> FeatureFrame;
    std::unique_ptr<pangolin::Var<bool>> ShowFeatureFrames;
    std::unique_ptr<pangolin::Var<bool>> Show3D;
    std::unique_ptr<pangolin::Var<bool>> RecordScreen;
    // pangolin::Var<double> a_double;//("ui.A_Double",3,0,5);
    // pangolin::Var<int> a_int; //("ui.An_Int",2,0,5);
    std::unique_ptr<pangolin::View> FramesPanel;

    boost::mutex mSLAMThread;
    std::unique_ptr<InternalImage> FrameImage; 

};

} // namespace FSLAM

#endif