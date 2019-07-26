#include <fstream>
# include <sstream>
#include <opencv2/imgproc.hpp>

#include "GeometricUndistorter.h"

namespace FSLAM
{

void GeometricUndistorter::LoadGeometricCalibration(std::string GeomCalibPath)
{
    passthrough=false;
	remapX = 0;
	remapY = 0;

    cv::FileStorage CalibIn(GeomCalibPath, cv::FileStorage::READ);
    if(!CalibIn.isOpened())
    {printf("could not read Geometric calibration data from %s \n", GeomCalibPath.c_str()); exit(1); }
    
    wOrg = CalibIn["Input.width"]; hOrg = CalibIn["Input.height"];
    w = CalibIn["Output.width"]; h = CalibIn["Output.height"];
    WidthOri = wOrg; HeightOri = hOrg;
    cv::Mat K_l, distM;
    float atanDist;

    CalibIn["CameraL.K"] >> K_l;

    std::string Calibmodel = CalibIn["Calibration.model"];
    if(Calibmodel == "RadTan")
    { Cameramodel = CamModel::RadTan; CalibIn["CameraL.distM"] >> distM; }
    else if (Calibmodel == "Atan")
    { Cameramodel = CamModel::Atan; atanDist = CalibIn["CameraL.dist"];}
    else if (Calibmodel == "Pinhole")
    { Cameramodel = CamModel::Pinhole; atanDist = 0.0f; } 
    else if (Calibmodel == "EquiDistant")
    { Cameramodel = CamModel::EquiDistant; CalibIn["CameraL.distM"] >> distM; }
    else if (Calibmodel == "KannalaBrandt")
    { Cameramodel = CamModel::KannalaBrandt; CalibIn["CameraL.distM"] >> distM; }
    else {printf("Camera calibration model not specified! exit,\n"); exit(1);}
    
    if(K_l.empty() || wOrg <=0 || hOrg <= 0 || w <=0 || h<= 0) {printf("Reading camera calibration failed! exit.\n"); exit(1);}
    if( (Cameramodel == CamModel::EquiDistant || Cameramodel == CamModel::RadTan || Cameramodel== CamModel::KannalaBrandt) && distM.empty())
    {printf("Error reading camera distortion! exit.\n"); exit(1);}

    if(Cameramodel == CamModel::Atan || Cameramodel == CamModel::Pinhole)
        ic[4] = atanDist;
    else 
    {
        ic[4] = distM.at<float>(0, 0);
        ic[5] = distM.at<float>(0, 1);
        ic[6] = distM.at<float>(0, 2);
        ic[7] = distM.at<float>(0, 3);
    }
    if(K_l.at<float>(0,2) < 1 && K_l.at<float>(1,2) < 1)
    {
        K_l.at<float>(0,0) *= wOrg; //fx
        K_l.at<float>(1,1) *= hOrg; //fy
        K_l.at<float>(0,2) *= wOrg; //-0.5 //cx
        K_l.at<float>(1,2) *= hOrg; //-0.5 //cy
    }

    ic[0] = K_l.at<float>(0,0); ic[1] = K_l.at<float>(1,1);
    ic[2] = K_l.at<float>(0,2); ic[3] = K_l.at<float>(1,2);
    
    std::string CalibProcess = CalibIn["Calib.process"];
    CalibIn["Stereo.State"] >> StereoState;

    bool needStereoRect = (Sensortype == Stereo && StereoState == "rectify");

    if(Sensortype == Sensor::Stereo && StereoState == "prerectified")
    {
        baseline = CalibIn["Stereo.bf"];
        Cameramodel = CamModel::Pinhole;
        if(baseline<=0){printf("failed to read stereo baseline!\n"); exit(1);} 
        baseline/=K_l.at<float>(0,0); //this assumes horizonal stereo! if vertical stereo need to divide by fy
    }

    remapX = new float[w*h];
    remapY = new float[w*h];

    if (!needStereoRect)
    {
        if (CalibProcess == "crop")
        {
            makeOptimalK_crop();
        }
        else if (CalibProcess == "none")
        {
            if (w != wOrg || h != hOrg)
            {printf("ERROR: rectification mode none requires input and output dimenstions to match!\n\n"); exit(1); }
            K.setIdentity();
            ic[0] = K(0, 0) = K_l.at<float>(0, 0);
            ic[1] = K(1, 1) = K_l.at<float>(1, 1);
            ic[2] = K(0, 2) = K_l.at<float>(0, 2); //-0.5
            ic[3] = K(1, 2) = K_l.at<float>(1, 2); //-0.5
            passthrough = true;
        }
        else if (CalibProcess == "useK")
        {
            cv::Mat DesiredK;
            CalibIn["Calib.desiK"] >> DesiredK;
            if (DesiredK.empty() || DesiredK.at<float>(0,2) > 1 || DesiredK.at<float>(0,3) > 1)
            {printf("Error reading desired camera calibration! it should be fx fy cx cy relative to the width and height exit.\n"); exit(1);}

            K.setIdentity();
            ic[0] = K(0, 0) = DesiredK.at<float>(0, 0) * w;
            ic[1] = K(1, 1) = DesiredK.at<float>(0, 1) * h;
            ic[2] = K(0, 2) = DesiredK.at<float>(0, 2) * w; // - 0.5;
            ic[3] = K(1, 2) = DesiredK.at<float>(0, 3) * h; // - 0.5;
        }
    }
    else //rquire stereo rectification
    {
        if(Cameramodel == CamModel::Atan || Cameramodel == CamModel::Pinhole)
            {printf("Distortion parameters provided are not supported with unrectified images! \n"); exit(1);}

        Cameramodel = CamModel::Pinhole;
        CalibProcess = "none";
        
        cv::Mat R_L,IntR, DistR, R_R ,NewInt;
        CalibIn["CameraL.R"] >> R_L;
        CalibIn["CameraR.K"] >> IntR;
        CalibIn["CameraR.DistM"] >> DistR;
        CalibIn["CameraR.R"] >> R_R;
        CalibIn["New.Intrin"] >> NewInt;
        if(R_L.empty() || IntR.empty()|| DistR.empty() || R_R.empty() || NewInt.empty())
        {printf("failed to read required calibration for stereo rectification! exiting.\n");exit(1);}
        hOrg = h; wOrg=w;
        K.setIdentity();
        ic[0] = K(0, 0) = NewInt.at<float>(0,0);
        ic[1] = K(1, 1) = NewInt.at<float>(1,1);
        ic[2] = K(0, 2) = NewInt.at<float>(0,2);
        ic[3] = K(1, 2) = NewInt.at<float>(1,2);

        baseline = CalibIn["Stereo.bf"];
        if(baseline<=0){printf("failed to read stereo baseline!\n"); exit(1);} 
        baseline/=K_l.at<float>(0,0); //this assumes horizonal stereo! if vertical stereo need to divide by fy
        cv::initUndistortRectifyMap(K_l, distM, R_L, NewInt.rowRange(0, 3).colRange(0, 3), cv::Size(w, h), CV_32F, M1l, M2l);
        cv::initUndistortRectifyMap(IntR, DistR, R_R, NewInt.rowRange(0, 3).colRange(0, 3), cv::Size(w, h), CV_32F, M1r, M2r);
    }

    if(Sensortype == Stereo)
        baseline*=K(0,0); //for vertical baseline this should be K(1,1)

    for(int y=0;y<h;y++)
	    for(int x=0;x<w;x++)
		{
			remapX[x+y*w] = x;
			remapY[x+y*w] = y;
		}

	distortCoordinates(remapX, remapY, remapX, remapY, h*w);

	for(int y=0;y<h;y++)
		for(int x=0;x<w;x++)
		{
			// make rounding resistant.
			float ix = remapX[x+y*w];
			float iy = remapY[x+y*w];

			if(ix == 0) ix = 0.001;
			if(iy == 0) iy = 0.001;
			if(ix == wOrg-1) ix = wOrg-1.001;
			if(iy == hOrg-1) ix = hOrg-1.001;

			if(ix > 0 && iy > 0 && ix < wOrg-1 &&  iy < wOrg-1)
			{
				remapX[x+y*w] = ix;
				remapY[x+y*w] = iy;
			}
			else
			{
				remapX[x+y*w] = -1;
				remapY[x+y*w] = -1;
			}
		}
    std::cout << K << "\n\n";
}

void GeometricUndistorter::makeOptimalK_crop()
{
	// printf("finding CROP optimal new model!\n");
	K.setIdentity();

	// 1. stretch the center lines as far as possible, to get initial coarse quess.
	float* tgX = new float[100000];
	float* tgY = new float[100000];
	float minX = 0;
	float maxX = 0;
	float minY = 0;
	float maxY = 0;

	for(int x=0; x<100000;x++)
	{tgX[x] = (x-50000.0f) / 10000.0f; tgY[x] = 0;}
	distortCoordinates(tgX, tgY,tgX, tgY,100000);
	for(int x=0; x<100000;x++)
	{
		if(tgX[x] > 0 && tgX[x] < wOrg-1)
		{
			if(minX==0) minX = (x-50000.0f) / 10000.0f;
			maxX = (x-50000.0f) / 10000.0f;
		}
	}
	for(int y=0; y<100000;y++)
	{tgY[y] = (y-50000.0f) / 10000.0f; tgX[y] = 0;}
	distortCoordinates(tgX, tgY,tgX, tgY,100000);
	for(int y=0; y<100000;y++)
	{
		if(tgY[y] > 0 && tgY[y] < hOrg-1)
		{
			if(minY==0) minY = (y-50000.0f) / 10000.0f;
			maxY = (y-50000.0f) / 10000.0f;
		}
	}
	delete[] tgX;
	delete[] tgY;

	minX *= 1.01;
	maxX *= 1.01;
	minY *= 1.01;
	maxY *= 1.01;

	// printf("initial range: x: %.4f - %.4f; y: %.4f - %.4f!\n", minX, maxX, minY, maxY);

	// 2. while there are invalid pixels at the border: shrink square at the side that has invalid pixels,
	// if several to choose from, shrink the wider dimension.
	bool oobLeft=true, oobRight=true, oobTop=true, oobBottom=true;
	int iteration=0;
	while(oobLeft || oobRight || oobTop || oobBottom)
	{
		oobLeft=oobRight=oobTop=oobBottom=false;
		for(int y=0;y<h;y++)
		{
			remapX[y*2] = minX;
			remapX[y*2+1] = maxX;
			remapY[y*2] = remapY[y*2+1] = minY + (maxY-minY) * (float)y / ((float)h-1.0f);
		}
		distortCoordinates(remapX, remapY,remapX, remapY,2*h);
		for(int y=0;y<h;y++)
		{
			if(!(remapX[2*y] > 0 && remapX[2*y] < wOrg-1))
				oobLeft = true;
			if(!(remapX[2*y+1] > 0 && remapX[2*y+1] < wOrg-1))
				oobRight = true;
		}

		for(int x=0;x<w;x++)
		{
			remapY[x*2] = minY;
			remapY[x*2+1] = maxY;
			remapX[x*2] = remapX[x*2+1] = minX + (maxX-minX) * (float)x / ((float)w-1.0f);
		}
		distortCoordinates(remapX, remapY,remapX, remapY,2*w);


		for(int x=0;x<w;x++)
		{
			if(!(remapY[2*x] > 0 && remapY[2*x] < hOrg-1))
				oobTop = true;
			if(!(remapY[2*x+1] > 0 && remapY[2*x+1] < hOrg-1))
				oobBottom = true;
		}

		if((oobLeft || oobRight) && (oobTop || oobBottom))
		{
			if((maxX-minX) > (maxY-minY))
				oobBottom = oobTop = false;	// only shrink left/right
			else
				oobLeft = oobRight = false; // only shrink top/bottom
		}

		if(oobLeft) minX *= 0.995;
		if(oobRight) maxX *= 0.995;
		if(oobTop) minY *= 0.995;
		if(oobBottom) maxY *= 0.995;

		iteration++;

		// printf("iteration %05d: range: x: %.4f - %.4f; y: %.4f - %.4f!\n", iteration,  minX, maxX, minY, maxY);
		if(iteration > 500)
		{
			printf("FAILED TO COMPUTE GOOD CAMERA MATRIX - SOMETHING IS SERIOUSLY WRONG. ABORTING \n");
			exit(1);
		}
	}

	K(0,0) = ((float)w-1.0f)/(maxX-minX);
	K(1,1) = ((float)h-1.0f)/(maxY-minY);
	K(0,2) = -minX*K(0,0);
	K(1,2) = -minY*K(1,1);

}

void GeometricUndistorter::distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n)
{
    // current camera parameters
    float fx = ic[0];
    float fy = ic[1];
    float cx = ic[2];
    float cy = ic[3];

    float ofx = K(0, 0);
    float ofy = K(1, 1);
    float ocx = K(0, 2);
    float ocy = K(1, 2);

    if (Cameramodel == CamModel::Pinhole)
    {
        for (int i = 0; i < n; i++)
        {
            float x = in_x[i];
            float y = in_y[i];
            float ix = (x - ocx) / ofx;
            float iy = (y - ocy) / ofy;
            ix = fx * ix + cx;
            iy = fy * iy + cy;
            out_x[i] = ix;
            out_y[i] = iy;
        }
    }
    else if (Cameramodel == CamModel::Atan)
    {
        float dist = ic[4];
        float d2t = 2.0f * tan(dist / 2.0f);
        for (int i = 0; i < n; i++)
        {
            float x = in_x[i];
            float y = in_y[i];
            float ix = (x - ocx) / ofx;
            float iy = (y - ocy) / ofy;

            float r = sqrtf(ix * ix + iy * iy);
            float fac = (r == 0 || dist == 0) ? 1 : atanf(r * d2t) / (dist * r);

            ix = fx * fac * ix + cx;
            iy = fy * fac * iy + cy;

            out_x[i] = ix;
            out_y[i] = iy;
        }
    }
    else if (Cameramodel == CamModel::RadTan)
    {
        float k1 = ic[4];
        float k2 = ic[5];
        float r1 = ic[6];
        float r2 = ic[7];

        for (int i = 0; i < n; i++)
        {
            float x = in_x[i];
            float y = in_y[i];

            // RADTAN
            float ix = (x - ocx) / ofx;
            float iy = (y - ocy) / ofy;
            float mx2_u = ix * ix;
            float my2_u = iy * iy;
            float mxy_u = ix * iy;
            float rho2_u = mx2_u + my2_u;
            float rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
            float x_dist = ix + ix * rad_dist_u + 2.0 * r1 * mxy_u + r2 * (rho2_u + 2.0 * mx2_u);
            float y_dist = iy + iy * rad_dist_u + 2.0 * r2 * mxy_u + r1 * (rho2_u + 2.0 * my2_u);
            float ox = fx * x_dist + cx;
            float oy = fy * y_dist + cy;

            out_x[i] = ox;
            out_y[i] = oy;
        }
    }
    else if (Cameramodel == CamModel::EquiDistant)
    {
        float k1 = ic[4];
        float k2 = ic[5];
        float k3 = ic[6];
        float k4 = ic[7];
        for (int i = 0; i < n; i++)
        {
            float x = in_x[i];
            float y = in_y[i];

            // EQUI
            float ix = (x - ocx) / ofx;
            float iy = (y - ocy) / ofy;
            float r = sqrt(ix * ix + iy * iy);
            float theta = atan(r);
            float theta2 = theta * theta;
            float theta4 = theta2 * theta2;
            float theta6 = theta4 * theta2;
            float theta8 = theta4 * theta4;
            float thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
            float scaling = (r > 1e-8) ? thetad / r : 1.0;
            float ox = fx * ix * scaling + cx;
            float oy = fy * iy * scaling + cy;

            out_x[i] = ox;
            out_y[i] = oy;
        }
    }
    else if (Cameramodel == CamModel::KannalaBrandt)
    {
        const float k0 = ic[4];
        const float k1 = ic[5];
        const float k2 = ic[6];
        const float k3 = ic[7];
        for (int i = 0; i < n; i++)
        {
            float x = in_x[i];
            float y = in_y[i];

            // RADTAN
            float ix = (x - ocx) / ofx;
            float iy = (y - ocy) / ofy;

            const float Xsq_plus_Ysq = ix * ix + iy * iy;
            const float sqrt_Xsq_Ysq = sqrtf(Xsq_plus_Ysq);
            const float theta = atan2f(sqrt_Xsq_Ysq, 1);
            const float theta2 = theta * theta;
            const float theta3 = theta2 * theta;
            const float theta5 = theta3 * theta2;
            const float theta7 = theta5 * theta2;
            const float theta9 = theta7 * theta2;
            const float r = theta + k0 * theta3 + k1 * theta5 + k2 * theta7 + k3 * theta9;

            if (sqrt_Xsq_Ysq < 1e-6)
            {
                out_x[i] = fx * ix + cx;
                out_y[i] = fy * iy + cy;
            }
            else
            {
                out_x[i] = (r / sqrt_Xsq_Ysq) * fx * ix + cx;
                out_y[i] = (r / sqrt_Xsq_Ysq) * fy * iy + cy;
            }
        }
    }
    else
    { printf("something went horribly wrong!\n"); exit(1);}

}

void GeometricUndistorter::undistort(std::shared_ptr<ImageData>ImgData, float* In_L, float* In_R)
{
    bool doRight = !ImgData->cvImgR.empty();
    if (Sensortype == Stereo)
    {
        if (StereoState == "rectify")
        {

            ImgData->cvImgL = cv::Mat(cv::Size(w, h), CV_32F,  In_L);
            ImgData->cvImgR = cv::Mat(cv::Size(w, h), CV_32F,  In_R);
            
            cv::remap(ImgData->cvImgL, ImgData->cvImgL, M1l, M2l, cv::INTER_LINEAR);
            cv::remap(ImgData->cvImgR, ImgData->cvImgR, M1r, M2r, cv::INTER_LINEAR);
            
            for (int i = 0; i < w * h; i++)
            {
                ImgData->fImgL[i] = ImgData->cvImgL.data[i];
                ImgData->fImgR[i] = ImgData->cvImgR.data[i];
            }
            ImgData->cvImgL.convertTo(ImgData->cvImgL, CV_8U);
            ImgData->cvImgR.convertTo(ImgData->cvImgR, CV_8U);
            return;
        }
    }
    if (!passthrough)
    {
        int dim = ImgData->cvImgL.size().width * ImgData->cvImgL.size().height;

        float * in_data = In_L;
        float * in_data2 = In_R ;
        float *out_data;
        float *out_data2;
        
        out_data = ImgData->fImgL;
        out_data2 = ImgData->fImgR;

        for (int idx = w * h - 1; idx >= 0; idx--)
        {
            // get interp. values
            float xx = remapX[idx];
            float yy = remapY[idx];

            if (xx < 0)
            {
                out_data[idx] = 0;
                if(doRight)
                    out_data2[idx] = 0;
            }

            int xxi = xx;
            int yyi = yy;
            xx -= xxi;
            yy -= yyi;
            float xxyy = xx * yy;

            // get array base pointer
            const float *src = in_data + xxi + yyi * wOrg;
            const float *src2;
            if(doRight)
                src2 = in_data2 + xxi + yyi * wOrg;

            // interpolate (bilinear)
            out_data[idx] = xxyy * src[1 + wOrg] + (yy - xxyy) * src[wOrg] + (xx - xxyy) * src[1] + (1 - xx - yy + xxyy) * src[0];
            if(doRight)
                out_data2[idx] = xxyy * src2[1 + wOrg] + (yy - xxyy) * src2[wOrg] + (xx - xxyy) * src2[1] + (1 - xx - yy + xxyy) * src2[0];
        }


            ImgData->cvImgL = cv::Mat(cv::Size(w, h), CV_32F, out_data);
            ImgData->cvImgL.convertTo(ImgData->cvImgL, CV_8U);

        if(doRight )
        {
            ImgData->cvImgR = cv::Mat(cv::Size(w, h), CV_32F,out_data);
            ImgData->cvImgR.convertTo(ImgData->cvImgR, CV_8U);
        }
    }
    else
    {
        for (int i = 0; i < WidthOri * HeightOri; i++)
        {
            ImgData->fImgL[i] = ImgData->cvImgL.data[i];
            if(doRight)
                ImgData->fImgR[i] = ImgData->cvImgR.data[i];
        }
    }
}
}