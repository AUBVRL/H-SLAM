#include "Display.h"
#include "Settings.h"
#include "Frame.h"
#include "MapPoint.h"
#include "CalibData.h"
#include "ImmaturePoint.h"
#include "System.h"
#include "OverwritePangolin.h"
#include <unistd.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
namespace SLAM
{
using namespace pangolin;

GUI::GUI()
{
    setup();
    render_loop = thread(&GUI::run, this);
    render_loop.detach();
}

GUI::~GUI()
{
    unique_lock<mutex> lock(mRenderThread);

}

void GUI::setup()
{
    CreateWindowAndBind(main_window_name, 5000, 5000);
    glEnable(GL_DEPTH_TEST);

    //2D image structures that are to be drawn
    FrameImage = std::unique_ptr<InternalImage>(new InternalImage());
    DepthKfImage = std::unique_ptr<InternalImage>(new InternalImage());

    //setup 3D rendering display and handler
    // scene_cam = OpenGlRenderState(ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100), ModelViewLookAt(-2, 2, -2, 0, 0, 0, AxisY));
    scene_cam = OpenGlRenderState(ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000), ModelViewLookAt(-0, -0.1, -5, 0, 0.1, 0, 0.0, -100.0, 0.0));
    display_cam = &CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f).SetHandler(new Handler3D(scene_cam));
    
    //Create image displays
    FeatureFrame = &Display("FeatureFrame");
    DepthKeyFrame = &Display("DepthKeyFrame");
    FramesPanel = &CreateDisplay().SetBounds(0.0, 0.2, 0.0, 1.0).SetLayout(LayoutEqual).AddDisplay(*DepthKeyFrame).AddDisplay(*FeatureFrame);
    FramesPanel->SetHandler(new HandlerResize());

    //Create settings panel and append controls to it.
    panel = &CreateNewPanel("ui").SetBounds(1.0, Attach::ReversePix(500), 0.0, Attach::Pix(MenuWidth));
    Nopanel = &CreateNewPanel("noui").SetBounds(1.0, Attach::ReversePix(35), 0.0, Attach::Pix(MenuWidth));
    ShowPanel = new Var<bool>("noui.Show Settings", false, false);
    HidePanel = new Var<bool>("ui.Hide Settings", false, false);
    bFollow = new Var<bool>("ui.Follow Camera", true, true);
    ShowDepthKF = new Var<bool>("ui.Show Depth Map", true, true);
    ShowImages = new Var<bool>("ui.Show Current Frame", true, true);
    ShowDetectedFeatures = new Var<bool>("ui.Show matches", DrawDetected, true);
    ShowFullTrajectory = new Var<bool>("ui.Show Trajectory", true, true);
    ShowAllKfs = new Var<bool>("ui.Show All Kfs", true, true);
    ShowFullConnectivity = new Var<bool>("ui.Show Connectivity", true, true);
    RecordScreen = new Var<bool>("ui.Record Screen!Stop Recording", false, false);
   
    _Pause = new Var<bool>("ui.Pause!Resume", Pause, false);
    _Reset = new Var<bool>("ui.Reset", ResetRequest, false);

    panel->Show(false); Nopanel->Show(true);
    

    
    Twc.SetIdentity(); //For camera following
    kfdepthmap.reserve(5000);
    GetBoundWindow()->RemoveCurrent(); //detach rendering loop from main thread
}

void GUI::Reset()
{
    unique_lock<mutex> lock(mRenderThread);
    FrameImage.reset(); FrameImage = unique_ptr<InternalImage>(new InternalImage());
    DepthKfImage.reset(); DepthKfImage = unique_ptr<InternalImage>(new InternalImage());
    scene_cam.SetModelViewMatrix(ModelViewLookAt(-0, -0.1, -5, 0, 0.1, 0, 0.0, -100.0, 0.0));
    
    SmoothMotion.clear(); SmoothMotion.shrink_to_fit();
    FramesToDraw.clear(); FramesToDraw.shrink_to_fit();
    Mat44 Identity = Eigen::Matrix4d::Identity();
    Twc = OpenGlMatrix(Identity);
    FrameDisp.reset();
    Dest.release();
    allFramePoses.clear(); allFramePoses.shrink_to_fit(); allFramePoses.reserve(10000); 
    AllKeyframes.clear(); AllKeyframes.shrink_to_fit(); AllKeyframes.reserve(5000);
    return;
}

void GUI::run()
{
    BindToContext(main_window_name);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_POINT_SMOOTH);

    while (!ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        unique_lock<mutex> lock(mRenderThread);
        ProcessInput();
        

        CreateActiveKF(); //Draw current Active keyframe
        CreateCurrentFrame(); //Draw Current frame image + extracted features + matches?
        

        SetPointOfView();
        display_cam->Activate(scene_cam);
        DrawTrajectory();//draw trajectory
        if(FrameDisp)
            drawCam(FrameDisp->Pose, 1, red, 0.2);
        DrawKeyFrames();

        FinishFrame();
        mRenderThread.unlock();        
        usleep(1000);
    }

    Pause = false;
    isDead = true;
}

void GUI::ProcessInput()
{
    if(ShowDetectedFeatures->Get())
        DrawDetected = true;
    else
        DrawDetected = false;

    if (Pushed(*_Pause)) {Pause = !Pause;}
    if (Pushed(*_Reset)) {ResetRequest = !ResetRequest;}
    if (Pushed(*ShowPanel)) {Nopanel->Show(false); panel->Show(true);}
    if (Pushed(*HidePanel)) {Nopanel->Show(true); panel->Show(false); }
    if (Pushed(*RecordScreen))
        DisplayBase().RecordOnRender("ffmpeg:[fps=30,bps=45388608,flip=true,unique_filename]//screencap.avi"); //8388608
}

void GUI::UploadActiveKF(cv::Mat ActiveKF, vector<pair<float, int>>&dmap)
{
    if (!ShowDepthKF->Get())
        return;
    unique_lock<mutex> lock(ActiveKFMutex);
    ActiveKF.copyTo(KfToDraw);
    kfdepthmap = dmap;
    isKFImgChanged = true;
}

void GUI::CreateActiveKF()
{
    if (!ShowDepthKF->Get())
            return;
    unique_lock<mutex> lock(ActiveKFMutex);
    if (isKFImgChanged)
    {
        isKFImgChanged = false;
        
        cv::cvtColor(KfToDraw, KfToDraw, CV_GRAY2BGR);
        int npts = kfdepthmap.size() - 1;

        float nid = 0, sid=0;
        for(int i=0;i<npts;i++)
        {
        		nid++;
        		sid += (kfdepthmap[i].first); //iR
        }
        float fac = nid / sid;

        for(int i=0;i<npts; ++i)
        {
            Vec3b Color = Vec3b(0,0,0);
            Color = makeRainbow3B(kfdepthmap[i].first *fac);
            cv::Scalar color = cv::Scalar(Color(0),Color(1),Color(2));
         
            int u = floor(kfdepthmap[i].second / KfToDraw.cols);
            int v = floor(kfdepthmap[i].second % KfToDraw.cols);
            cv::circle(KfToDraw, cv::Point2f(v,u), 1, color, -1, cv::LineTypes::LINE_8, 0);

            // setPixel9(KfToDraw, floor(float(u)), floor(float (v)), Color);
        }

        setInternalImageData(DepthKfImage, KfToDraw);
    }
    renderInternalFrame(DepthKfImage, DepthKeyFrame);
}

void GUI::SetPointOfView()
{
    Eigen::Matrix4d AverageMotion = Eigen::Matrix4d::Zero();
    if (bFollow->Get())
    {
        if(bFollow->GuiChanged())
            scene_cam.SetModelViewMatrix(ModelViewLookAt(-0, -0.1, -5, 0, 0.1, 0, 0.0, -100.0, 0.0));
        if (SmoothMotion.size() > 5)
        {
            for (auto it : SmoothMotion)
            {
                AverageMotion = AverageMotion + it;
            }
            AverageMotion = AverageMotion / static_cast<double>(SmoothMotion.size());
            Twc = OpenGlMatrix(AverageMotion);
        }
        else if (SmoothMotion.size() > 0)
            Twc = OpenGlMatrix(SmoothMotion[0]);
        else
        {
            Mat44 Identity = Eigen::Matrix4d::Identity();
            Twc = OpenGlMatrix(Identity);
        }
        scene_cam.Follow(Twc);
    }
}


void GUI::UploadCurrentFrame(std::shared_ptr<ImageData> ImgIn, std::vector<cv::KeyPoint> &mvKeys, SE3 _Pose, bool poseValid)
{
    unique_lock<mutex> lock(RunningFrameMutex);
    if (!poseValid)
        return;
    FramesToDraw.push_front(std::shared_ptr<FrameDisplayData>(new FrameDisplayData(ImgIn, mvKeys, _Pose)));
}

void GUI::CreateCurrentFrame()
{
    unique_lock<mutex> lock(RunningFrameMutex);
    if (FramesToDraw.size() != 0)
    {
        
        FrameDisp = FramesToDraw.back();
        allFramePoses.push_back(FrameDisp->Pose.topRightCorner(3, 1).cast<float>());
        SmoothMotion.push_back(FrameDisp->Pose);
        while (SmoothMotion.size() > 40)
            SmoothMotion.pop_front();
        FramesToDraw.pop_back();

        if (FramesToDraw.size() > 3)
        {
            for (int i = 0; i < FramesToDraw.size(); ++i) //If pushing images faster than rendering them, discard the old images and only draw the latest one
            {
                SmoothMotion.push_back(FramesToDraw[i]->Pose);
                while (SmoothMotion.size() > 40)
                    SmoothMotion.pop_front();
                allFramePoses.push_back(FramesToDraw[i]->Pose.topRightCorner(3, 1).cast<float>());
            }
            FramesToDraw.clear();
        }
        if (!ShowImages->Get())
            return;

        cv::cvtColor(FrameDisp->ImgData->cvImg, Dest, CV_GRAY2BGR);

        if (ShowDetectedFeatures->Get())
            for (size_t i = 0, iend = FrameDisp->Keys.size(); i < iend; ++i)
                cv::circle(Dest, FrameDisp->Keys[i].pt, 3, cv::Scalar(255.0, 0.0, 0.0), -1, cv::LineTypes::LINE_8, 0);

        setInternalImageData(FrameImage, Dest);
    }
    
    if (ShowImages->Get())
        renderInternalFrame(FrameImage, FeatureFrame);

}

void GUI::setInternalImageData(std::unique_ptr<InternalImage> &InternalImage, cv::Mat &Img)
{
    if (InternalImage->Width == 0 || InternalImage->Height == 0)
    {
        InternalImage->Width = Img.size().width;
        InternalImage->Height = Img.size().height;
    }
    InternalImage->Image = Img.data;
    InternalImage->HaveNewImage = true;
}


void GUI::renderInternalFrame(std::unique_ptr<InternalImage> &ImageToRender, View* CanvasFrame)
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

void GUI::DrawTrajectory()
{
    if (ShowFullTrajectory->Get())
    {
        unique_lock<mutex> lock(RunningFrameMutex);
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
    }
}

void GUI::drawCam(Mat44 Pose, float lineWidth, Vec3b& color, float sizeFactor)
{
    int width = 640; int height = 480;
    int cx =320; int cy = 240;
    int fx = 320; int fy = 320;

	if(width == 0)
		return;

	float sz=sizeFactor;

	glPushMatrix();

		Sophus::Matrix4f m = Pose.cast<float>();
		glMultMatrixf((GLfloat*)m.data());

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

void GUI::UploadKeyFrame(shared_ptr<FrameShell> FrameIn) //this vector wont be modified from multiple threads. No need for mutex
{
    unique_lock<mutex> lock(KeyframesMutex); 
    AllKeyframes.push_back(std::make_pair(FrameIn, std::shared_ptr<KFDisplay>(new KFDisplay())));
}

void GUI::DrawKeyFrames()
{
    unique_lock<mutex> lock(KeyframesMutex); 
    for (auto it: AllKeyframes)
    {
        if(it.first->frame->NeedRefresh)
            it.second->RefreshPC(it.first);

        if(ShowAllKfs->Get())
            if(it.second->PoseValid) drawCam(it.second->camToWorld.matrix(), 1, blue, 0.1);
        
        if(!it.second->ValidBuffer) continue;
        glDisable(GL_LIGHTING);

        glPushMatrix();

        Sophus::Matrix4f m = it.second->camToWorld.matrix().cast<float>();
        glMultMatrixf((GLfloat *)m.data());

        glPointSize(1);
        it.second->colorBuffer.Bind();
        glColorPointer(it.second->colorBuffer.count_per_element, it.second->colorBuffer.datatype, 0, 0);
        glEnableClientState(GL_COLOR_ARRAY);

        it.second->vertexBuffer.Bind();
        glVertexPointer(it.second->vertexBuffer.count_per_element, it.second->vertexBuffer.datatype, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);
        glDrawArrays(GL_POINTS, 0, it.second->numGLBufferGoodPoints);
        glDisableClientState(GL_VERTEX_ARRAY);
        it.second->vertexBuffer.Unbind();

        glDisableClientState(GL_COLOR_ARRAY);
        it.second->colorBuffer.Unbind();

        glPopMatrix();
    }

    return;
}


KFDisplay::KFDisplay()
{
    ValidBuffer = false;
    numGLBufferPoints = 0;
}

void KFDisplay::RefreshPC(std::shared_ptr<FrameShell> _In)
{
    camToWorld = _In->camToWorld;
    PoseValid = true;
    float my_scaledTH = 0.001; //1e-10,1e10
	float my_absTH = 0.001; //1e-10,1e10
	float my_minRelBS = 0.1; //0,1
	int my_sparsifyFactor = 1 ; //from 1 to 20

    float fxi = _In->frame->Calib->fxli();
    float cxi = _In->frame->Calib->cxli();
    float fyi = _In->frame->Calib->fyli();
    float cyi = _In->frame->Calib->cyli();

    int numSparsePoints = _In->frame->pointHessians.size() + _In->frame->ImmaturePoints.size() + _In->frame->pointHessiansMarginalized.size();
    Vec3f* tmpVertexBuffer = new Vec3f[numSparsePoints*patternNum];
	Vec3b* tmpColorBuffer = new Vec3b[numSparsePoints*patternNum];
	int vertexBufferNumPoints=0;

    //Update MapPoints to draw.
    std::lock_guard<std::mutex> l(_In->frame->_mtx);

    for(auto it:_In->frame->pointHessians) //this thread locks the shared_ptr in case it was removed elsewhere!
    {
        if(!it) continue;
        if(it->getPointStatus() == INACTIVE) continue;
        if(it->idepth < 0) continue;
        float depth = 1.0f / it->idepth;
		float depth4 = depth*depth; depth4*= depth4;
		float var = (1.0f / (it->idepth_hessian+0.01));
        if(var * depth4 > my_scaledTH) continue;
		if(var > my_absTH) continue;
        if(it->maxRelBaseline < my_minRelBS) continue;
        for(int pnt=0;pnt<patternNum;pnt++)
		{
			if(my_sparsifyFactor > 1 && rand()%my_sparsifyFactor != 0) continue;
			int dx = patternP[pnt][0];
			int dy = patternP[pnt][1];

			tmpVertexBuffer[vertexBufferNumPoints][0] = ((it->u+dx)*fxi + cxi) * depth;
			tmpVertexBuffer[vertexBufferNumPoints][1] = ((it->v+dy)*fyi + cyi) * depth;
			tmpVertexBuffer[vertexBufferNumPoints][2] = depth*(1 + 2*fxi * (rand()/(float)RAND_MAX-0.5f));


            // if (it->getPointStatus() == ACTIVE)
                tmpColorBuffer[vertexBufferNumPoints] = blue;
            // else if (it->getPointStatus() == MARGINALIZED)
            //     tmpColorBuffer[vertexBufferNumPoints] = Vec3b(it->color[pnt], it->color[pnt], it->color[pnt]);
            // else if (it->getPointStatus() == OUTLIER)
            //     tmpColorBuffer[vertexBufferNumPoints] = red;   

            vertexBufferNumPoints++;
		}
    }

     for(auto it:_In->frame->pointHessiansMarginalized) //this thread locks the shared_ptr in case it was removed elsewhere!
    {
        if(!it) continue;
        // if(it->getPointStatus() == INACTIVE) continue;
        if(it->idepth < 0) continue;
        float depth = 1.0f / it->idepth;
		float depth4 = depth*depth; depth4*= depth4;
		float var = (1.0f / (it->idepth_hessian+0.01));
        if(var * depth4 > my_scaledTH) continue;
		if(var > my_absTH) continue;
        if(it->maxRelBaseline < my_minRelBS) continue;
        for(int pnt=0;pnt<patternNum;pnt++)
		{
			if(my_sparsifyFactor > 1 && rand()%my_sparsifyFactor != 0) continue;
			int dx = patternP[pnt][0];
			int dy = patternP[pnt][1];

			tmpVertexBuffer[vertexBufferNumPoints][0] = ((it->u+dx)*fxi + cxi) * depth;
			tmpVertexBuffer[vertexBufferNumPoints][1] = ((it->v+dy)*fyi + cyi) * depth;
			tmpVertexBuffer[vertexBufferNumPoints][2] = depth*(1 + 2*fxi * (rand()/(float)RAND_MAX-0.5f));

            tmpColorBuffer[vertexBufferNumPoints] = Vec3b(it->color[pnt], it->color[pnt], it->color[pnt]);
            vertexBufferNumPoints++;
		}
    }

    // //Update Immature Points to draw.
    // for (auto it: _In->ImmaturePoints)
    // {
    //     if (!it) continue;
    //     for (int pnt = 0; pnt < patternNum; pnt++)
    //     {
    //         if (my_sparsifyFactor > 1 && rand() % my_sparsifyFactor != 0) continue;

    //         int dx = patternP[pnt][0];
    //         int dy = patternP[pnt][1];
            
    //         float idepth = (it->idepth_max+it->idepth_min)*0.5f;
    //         if(idepth < 0) continue;
    //         float depth = 1.0f / idepth;
           
    //         tmpVertexBuffer[vertexBufferNumPoints][0] = ((it->u + dx) * fxi + cxi) * depth;
    //         tmpVertexBuffer[vertexBufferNumPoints][1] = ((it->v + dy) * fyi + cyi) * depth;
    //         tmpVertexBuffer[vertexBufferNumPoints][2] = depth * (1 + 2 * fxi * (rand() / (float)RAND_MAX - 0.5f));
    //         tmpColorBuffer[vertexBufferNumPoints] = orange;   
    //         vertexBufferNumPoints++;
    //     }
    // }

    assert(vertexBufferNumPoints <= numSparsePoints*patternNum);

    if(vertexBufferNumPoints==0)
	{
        ValidBuffer = false;
        _In->frame->NeedRefresh = false;
        numGLBufferGoodPoints = 0;
        numGLBufferPoints = 0;
		delete[] tmpColorBuffer;
		delete[] tmpVertexBuffer;
		return;
	}

    numGLBufferGoodPoints = vertexBufferNumPoints;
	if(numGLBufferGoodPoints > numGLBufferPoints)
	{
		numGLBufferPoints = vertexBufferNumPoints*1.3;
		vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW );
		colorBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW );
	}
	vertexBuffer.Upload(tmpVertexBuffer, sizeof(float)*3*numGLBufferGoodPoints, 0);
	colorBuffer.Upload(tmpColorBuffer, sizeof(unsigned char)*3*numGLBufferGoodPoints, 0);
    ValidBuffer = true;
	delete[] tmpColorBuffer;
	delete[] tmpVertexBuffer;
    _In->frame->NeedRefresh = false;


    return;
}


}