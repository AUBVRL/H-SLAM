#include "Display.h"
#include "PangolinOverwrite.h"
#include "Settings.h"
#include "Frame.h"
#include <opencv2/imgproc.hpp>
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

    // scene_cam = OpenGlRenderState(ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100), ModelViewLookAt(-2, 2, -2, 0, 0, 0, AxisY));
    scene_cam = OpenGlRenderState(ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100), ModelViewLookAt(-0, -0.1, -5, 0, 0.1, 0, 0.0, -100.0, 0.0));

    display_cam = &CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f).SetHandler(new pangolin::Handler3D(scene_cam));
    panel = &CreateNewPanel("ui").SetBounds(1.0, Attach::ReversePix(500), 0.0, Attach::Pix(MenuWidth));
    Nopanel = &CreateNewPanel("noui").SetBounds(1.0, Attach::ReversePix(35), 0.0, Attach::Pix(MenuWidth));
    ShowPanel = new Var<bool>("noui.Show Settings", false, false);
    HidePanel = new Var<bool>("ui.Hide Settings", false, false);
    bFollow = new Var<bool>("ui.Follow Camera", true, true);
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

    Twc.SetIdentity();

    GetBoundWindow()->RemoveCurrent();
}

void GUI::Reset()
{
    boost::unique_lock<boost::mutex> lock(mRenderThread);

    FrameImage.reset(); FrameImage = std::unique_ptr<InternalImage>(new InternalImage());
    DepthKfImage.reset(); DepthKfImage = std::unique_ptr<InternalImage>(new InternalImage());
    scene_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(-0, -0.1, -5, 0, 0.1, 0, 0.0, -100.0, 0.0));
    
    SmoothMotion.clear(); SmoothMotion.shrink_to_fit();
    FramesToDraw.clear(); FramesToDraw.shrink_to_fit();
    Mat44 Identity = Eigen::Matrix4d::Identity();
    Twc = pangolin::OpenGlMatrix(Identity);
    FrameDisp.reset();
    allFramePoses.clear();
    return;
}


void GUI::run()
{
    BindToContext(main_window_name);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    float blue[3] = {0.0f,0.0f,1.0f};
	float red[3] = {1.0f,0.0f,0.0f};
	float green[3] = {0.0f,1.0f,0.0f};
	glEnable(GL_POINT_SMOOTH);

    while (!ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        ProcessInput();

        boost::unique_lock<boost::mutex> lock(mRenderThread);
        
        DrawRunningFrame();
        
        if (Show3D->Get())
        {
            SetPointOfView();
            display_cam->Activate(scene_cam);
            if(FrameDisp)
                drawCam(FrameDisp->Pose, 1, red, 0.1);

            // if (settings_showFullTrajectory)
            // {
                glColor3f(green[0], green[1], green[2]);
                glLineWidth(3);
                glBegin(GL_LINE_STRIP);
                for (unsigned int i = 0; i < allFramePoses.size(); i++)
                {
                    glVertex3f((float)allFramePoses[i][0],
                               (float)allFramePoses[i][1],
                               (float)allFramePoses[i][2]);
                }
                glEnd();
            // }

            //     boost::unique_lock<boost::mutex> lock(mSLAMThread);
            //     glPointSize(5);
            //     glBegin(GL_POINTS);

            //     size_t sz = Pts.size();
            //     for (size_t i = 0; i < sz; i = i + 4)
            //     {
            //         if(Pts[i+3]==1)
            //             glColor3f(255.0, 0.0, 0.0);
            //         else
            //             glColor3f(0.0, 255.0, 0.0);
            //         glVertex3f(Pts[i], Pts[i + 1], Pts[i + 2]);
            //     }
            //     glEnd();
            // pangolin::glDrawColouredCube();
        }

        
       
        if(Show2D->Get())
        {
            // if (ShowImages->Get())
            //     RenderInputFrameImage(FrameImage, FeatureFrame);

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

void GUI::UploadPoints(std::vector<float> Points)
{
    // boost::unique_lock<boost::mutex> lock(mSLAMThread);
    Pts = Points;

}

void GUI::UploadRunningFrameData(std::shared_ptr<ImageData> ImgIn, std::vector<cv::KeyPoint>&mvKeys, SE3 _Pose)
{
    boost::unique_lock<boost::mutex> lock(RunningFrameMutex);
    FramesToDraw.push_back(std::shared_ptr<FrameDisplayData>(new FrameDisplayData(ImgIn, mvKeys, _Pose)));
}

void GUI::DrawRunningFrame()
{
    if (FramesToDraw.size() != 0)
    {
        boost::unique_lock<boost::mutex> lock(RunningFrameMutex);
        FrameDisp = FramesToDraw.back(); // no need to copy this as it will be the only holding pointer left once added here
        
        for (int i = 0; i < FramesToDraw.size(); ++i)  //If pushing images faster than rendering them, discard the old images and only draw the latest one
        {
            SmoothMotion.push_back(FramesToDraw[i]->Pose);
            while (SmoothMotion.size() > 40)
                SmoothMotion.pop_front();
            allFramePoses.push_back(FramesToDraw[i]->Pose.topRightCorner(3,1).cast<float>());
        }
        FramesToDraw.clear();
        lock.unlock();
        
        if (!Show2D->Get() || !ShowImages->Get())
            return;

        cv::Mat Dest;
        if (Sensortype == Stereo || Sensortype == RGBD)
            cv::hconcat(FrameDisp->ImgData->cvImgL, FrameDisp->ImgData->cvImgR, Dest);
        else
            Dest = FrameDisp->ImgData->cvImgL;

        cv::cvtColor(Dest, Dest, CV_GRAY2BGR);
        if (DrawDetected)
        {
            for (size_t i = 0, iend = FrameDisp->Keys.size(); i < iend; ++i)
                cv::circle(Dest, FrameDisp->Keys[i].pt, 3, cv::Scalar(255.0, 0.0, 0.0), -1, cv::LineTypes::LINE_8, 0);
        }

        //Initialize Image Data if not initialized
        InitializeImageData(FrameImage, Dest);
    }

    //Render Image
    RenderInputFrameImage(FrameImage, FeatureFrame);

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
        if (ImageToRender->HaveNewImage)
            ImageToRender->FeatureFrameTexture.Upload(&ImageToRender->Image[0], GL_BGR, GL_UNSIGNED_BYTE);
        CanvasFrame->Activate();
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        ImageToRender->FeatureFrameTexture.RenderToViewportFlipY();
    }
}

void GUI::InitializeImageData(std::unique_ptr<InternalImage> &InternalImage, cv::Mat Img)
{
    if (InternalImage->Width == 0 || InternalImage->Height == 0)
    {
        InternalImage->Width = Img.size().width;
        InternalImage->Height = Img.size().height;
    }
    InternalImage->Image = Img.data;
    InternalImage->HaveNewImage = true;
}

void GUI::SetPointOfView()
{
    boost::unique_lock<boost::mutex> lock(RunningFrameMutex);
    Eigen::Matrix4d AverageMotion = Eigen::Matrix4d::Zero();
    if (bFollow->Get())
    {
        if(bFollow->GuiChanged())
            scene_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(-0, -0.1, -5, 0, 0.1, 0, 0.0, -100.0, 0.0));
        if (SmoothMotion.size() > 5)
        {
            for (auto it : SmoothMotion)
            {
                AverageMotion = AverageMotion + it;
            }
            AverageMotion = AverageMotion / static_cast<double>(SmoothMotion.size());
            Twc = pangolin::OpenGlMatrix(AverageMotion);
        }
        else if (SmoothMotion.size() > 0)
            Twc = pangolin::OpenGlMatrix(SmoothMotion[0]);
        else
        {
            Mat44 Identity = Eigen::Matrix4d::Identity();
            Twc = pangolin::OpenGlMatrix(Identity);
        }
            
        scene_cam.Follow(Twc);
    }
}

void GUI::drawCam(Mat44 Pose, float lineWidth, float* color, float sizeFactor)
{
    if (!Show3D->Get())
        return;
    int width = 640; int height = 480;
    int cx =320; int cy = 240;
    int fx = 320; int fy = 320;

	if(width == 0)
		return;

	float sz=sizeFactor;

	glPushMatrix();

		Sophus::Matrix4f m = Pose.cast<float>();
		glMultMatrixf((GLfloat*)m.data());

		if(color == 0)
		{
			glColor3f(1,0,0);
		}
		else
			glColor3f(color[0],color[1],color[2]);

		glLineWidth(lineWidth);
		glBegin(GL_LINES);
		glVertex3f(0,0,0);
		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);

		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);
		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);

		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);

		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);

		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);
		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);

		glEnd();
	glPopMatrix();
}

} // namespace FSLAM
