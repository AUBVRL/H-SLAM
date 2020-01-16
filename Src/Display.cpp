#include "Display.h"
#include "PangolinOverwrite.h"
#include "Settings.h"

namespace FSLAM
{

using namespace pangolin;

GUI::GUI()
{
    setup();
    render_loop = boost::thread(&GUI::run, this);
    render_loop.detach();
}

GUI::~GUI() {}

void GUI::setup()
{
    CreateWindowAndBind(main_window_name, 5000, 5000);
    glEnable(GL_DEPTH_TEST);

    FrameImage = std::unique_ptr<InternalImage>(new InternalImage());
    DepthKfImage = std::unique_ptr<InternalImage>(new InternalImage());

    scene_cam = OpenGlRenderState(ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100), ModelViewLookAt(-2, 2, -2, 0, 0, 0, AxisY));
    display_cam = &CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f).SetHandler(new pangolin::Handler3D(scene_cam));
    panel = &CreateNewPanel("ui").SetBounds(1.0, Attach::ReversePix(500), 0.0, Attach::Pix(MenuWidth));
    Nopanel = &CreateNewPanel("noui").SetBounds(1.0, Attach::ReversePix(35), 0.0, Attach::Pix(MenuWidth));
    ShowPanel = new Var<bool>("noui.Show Settings", false, false);
    HidePanel = new Var<bool>("ui.Hide Settings", false, false);
    Show2D = new Var<bool>("ui.Show 2D", true, true);
    ShowDepthKF = new Var<bool>("ui.Show Depth KF", true, true);
    ShowImages = new Var<bool>("ui.Show Images", true, true);
    ShowDetectedFeatures = new Var<bool>("ui.Show Features", DrawDetected, true);
    Show3D = new Var<bool>("ui.Show3D", true, true);
    RecordScreen = new Var<bool>("ui.Record Screen!Stop Recording", false, false);
    FeatureFrame = &Display("FeatureFrame");
    DepthKeyFrame = &Display("DepthKeyFrame");

    _Pause = new Var<bool>("ui.Pause!Resume", Pause, false);
    FramesPanel = &CreateDisplay().SetBounds(0.0, 0.2, 0.0, 1.0).SetLayout(LayoutEqual).AddDisplay(*DepthKeyFrame).AddDisplay(*FeatureFrame);
    FramesPanel->SetHandler(new HandlerResize());
    panel->Show(false); Nopanel->Show(true);

    GetBoundWindow()->RemoveCurrent();
}

void GUI::run()
{
    BindToContext(main_window_name);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    while (!ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        ProcessInput();

        if (Show3D->Get())
        {
            display_cam->Activate(scene_cam);
            pangolin::glDrawColouredCube();
        }

        if(Show2D->Get())
        {
            if (ShowImages->Get())
                RenderInputFrameImage(FrameImage, FeatureFrame);

            if (ShowDepthKF->Get())
                RenderInputFrameImage(DepthKfImage, DepthKeyFrame);

        }
        
        
        pangolin::FinishFrame();
        usleep(10000);
    }

    Pause = false;
    isDead = true;
}

void GUI::ProcessInput()
{
    if (Show2D->Get()) 
    {
        if (!FramesPanel->IsShown()) 
            FramesPanel->Show(true); 
    } else if (FramesPanel->IsShown()) 
        FramesPanel->Show(false);

    if(ShowDetectedFeatures->Get())
        DrawDetected = true;
    else
        DrawDetected = false;

    if (Pushed(*_Pause)) {Pause = !Pause;}
    if (Pushed(*ShowPanel)) {Nopanel->Show(false); panel->Show(true);}
    if (Pushed(*HidePanel)) {Nopanel->Show(true); panel->Show(false); }
    if (Pushed(*RecordScreen))
        DisplayBase().RecordOnRender("ffmpeg:[fps=30,bps=8388608,flip=true,unique_filename]//screencap.avi");
}

void GUI::UploadFrameImage(unsigned char* _In, int width, int height)
{
    if (Show2D->Get() && ShowImages->Get())
    {
        boost::unique_lock<boost::mutex> lock(mSLAMThread);
        if(FrameImage->Image.empty())
        {
            FrameImage->Width = width;
            FrameImage->Height = height;
            FrameImage->SizeToCopy = width * height * 3;
            FrameImage->Image.resize(FrameImage->SizeToCopy);
        }
        memcpy(&FrameImage->Image[0], _In, FrameImage->SizeToCopy);
        FrameImage->HaveNewImage = true;
    }
    
    return;
}

void GUI::RenderInputFrameImage(std::unique_ptr<InternalImage> &ImageToRender, View* CanvasFrame)
{
    if (!ImageToRender->IsTextureGood)
    {
        if (ImageToRender->Width + ImageToRender->Height != 0)
        {
            CanvasFrame->SetAspect((float)ImageToRender->Width / (float)ImageToRender->Height);
            ImageToRender->FeatureFrameTexture.Reinitialise(ImageToRender->Width, ImageToRender->Height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
            ImageToRender->IsTextureGood = true;
        }
    }

    if (ImageToRender->IsTextureGood)
    {
        boost::unique_lock<boost::mutex> lock(mSLAMThread);
        if (ImageToRender->HaveNewImage)
            ImageToRender->FeatureFrameTexture.Upload(&ImageToRender->Image[0], GL_BGR, GL_UNSIGNED_BYTE);
        CanvasFrame->Activate();
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        ImageToRender->FeatureFrameTexture.RenderToViewportFlipY();
    }
}

void GUI::UploadDepthKeyFrameImage(unsigned char* _In, int width, int height)
{
    if (Show2D->Get() && ShowDepthKF->Get())
    {
        boost::unique_lock<boost::mutex> lock(mSLAMThread);
        if(DepthKfImage->Image.empty())
        {
            DepthKfImage->Width = width;
            DepthKfImage->Height = height;
            DepthKfImage->SizeToCopy = width * height * 3;
            DepthKfImage->Image.resize(DepthKfImage->SizeToCopy);
        }
        memcpy(&DepthKfImage->Image[0], _In, DepthKfImage->SizeToCopy);
        DepthKfImage->HaveNewImage = true;
    }
    
    return;
}

} // namespace FSLAM
