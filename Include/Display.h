#ifndef __DISPLAY__
#define __DISPLAY__

#include <pangolin/pangolin.h>
#include <thread>

static const std::string main_window_name = "FSLAM";

namespace FSLAM
{
struct View2;
struct Display
{
public:
    void setup ();
    void run();
    Display();
    ~Display();
    
    std::thread render_loop;
    bool isDead = false;

private:

    void ProcessInput();
    int MenuWidth = 150; //pixel units
    pangolin::OpenGlRenderState scene_cam;
    pangolin::View display_cam;

    //configuration panel settings
    std::unique_ptr<pangolin::View> panel;
    std::unique_ptr<pangolin::View> Nopanel;
    std::unique_ptr<pangolin::Var<bool>> ShowPanel;
    std::unique_ptr<pangolin::Var<bool>> HidePanel;



    pangolin::View Video;
    pangolin::View FeatureFrame;
    pangolin::View Image3;

    std::unique_ptr<pangolin::Var<bool>> ShowInput;
    std::unique_ptr<pangolin::Var<bool>> Show3D;

    std::unique_ptr<pangolin::Var<bool>> RecordScreen;
    // pangolin::Var<double> a_double;//("ui.A_Double",3,0,5);
    // pangolin::Var<int> a_int; //("ui.An_Int",2,0,5);


    
};


}


#endif