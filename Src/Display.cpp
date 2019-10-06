#include "Display.h"
#include "PangolinOverwrite.h"

namespace FSLAM
{

Display::Display()
{
    setup();
    render_loop = std::thread(&Display::run, this);
}

Display::~Display()
{
    if (render_loop.joinable())
        render_loop.join();
}

void Display::setup()
{
    pangolin::CreateWindowAndBind(main_window_name, 5000, 5000);
    glEnable(GL_DEPTH_TEST);

    scene_cam = pangolin::OpenGlRenderState( pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100), 
                pangolin::ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin::AxisY));

    display_cam = pangolin::CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f).SetHandler(new pangolin::Handler3D(scene_cam));
    
    panel = std::unique_ptr<pangolin::View>(&pangolin::CreateNewPanel("ui").SetBounds(1.0, pangolin::Attach::ReversePix(500), 0.0, pangolin::Attach::Pix(MenuWidth)));
    Nopanel = std::unique_ptr<pangolin::View>(&pangolin::CreateNewPanel("noui").SetBounds(1.0, pangolin::Attach::ReversePix(35), 0.0, pangolin::Attach::Pix(MenuWidth)));
    Nopanel->Show(true); panel->Show(false);

    ShowPanel = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("noui.Show Settings", false, false));
    HidePanel = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("ui.Hide Settings", false, false));

    ShowInput = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("ui.ShowInput", false, true));
    Show3D = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("ui.Show3D", true, true));
   
    RecordScreen = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("ui.Record Screen!Stop Recording", false, false));


    pangolin::GetBoundWindow()->RemoveCurrent();

    // Video = pangolin::Display("image1").SetAspect(640.0f/480.0f); //change this to image width/height
    // FeatureFrame = pangolin::Display("image2").SetAspect(640.0f/480.0f); //change this to image width/height
}

void Display::run()
{

    pangolin::BindToContext(main_window_name);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        ProcessInput();

        if (*Show3D.get())
        {
            display_cam.Activate(scene_cam);
            pangolin::glDrawColouredCube();
        }

        pangolin::FinishFrame();
    }

    isDead = true;
    pangolin::GetBoundWindow()->RemoveCurrent();
}

void Display::ProcessInput()
{
    if (ShowInput->Get()) {}

    if (pangolin::Pushed(*ShowPanel.get())) { Nopanel->Show(false); panel->Show(true);}
    if (pangolin::Pushed(*HidePanel.get())) { Nopanel->Show(true); panel->Show(false);}

    if (pangolin::Pushed(*RecordScreen.get()))
        pangolin::DisplayBase().RecordOnRender("ffmpeg:[fps=30,bps=8388608,flip=true,unique_filename]//screencap.avi");
    
}

} // namespace FSLAM
