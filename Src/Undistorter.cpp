#include "Undistorter.h"

#include <fstream>
# include <sstream>
#include <vector>
#include "Settings.h"
#include <opencv2/imgproc.hpp>

namespace FSLAM
{

void Undistorter::LoadGeometricCalibration(std::string GeomCalibPath)
{
    passthrough=false;
	remapX = 0;
	remapY = 0;

    Cameramodel = CamModel::Empty;
	std::ifstream f(GeomCalibPath.c_str());
	if (!f.good())
	{
		printf(" ... not found. Cannot operate without calibration, shutting down.\n");
		f.close();
		exit(1) ;
	}
    printf("calibration file found at %s!\n", GeomCalibPath.c_str());
    
    std::string line;
    while (std::getline(f, line))
    {
        std::stringstream ss(line);
        std::vector<std::string> vec;
        std::string word;
        while (ss >> word)
            vec.push_back(word);
        
        Words.push_back(vec);
    }
    f.close();

    try
    {
        //first word is a float
        for(int i = 0 ; i < Words[0].size();i++)
            ic[i] = std::stof(Words[0][i]);
        if(Words[0].size()==8)
            Cameramodel = CamModel::RadTan;
        else if (Words[0].size()==5)
        {
            if( ic[4] == 0)
                Cameramodel = CamModel::Pinhole;
            else
                Cameramodel = CamModel::Atan;
        }
    }
    catch(std::exception& e) //first word is a string
	{
        for(int i = 1 ; i < Words[0].size();i++)
            ic[i-1] = std::stof(Words[0][i]);
        
        if(Words[0][0] == "RadTan")
        {
            if(Words[0].size()-1 != 8)
                { printf("Invalid combination of model name and parameters!\n"); exit(1);}
            Cameramodel = CamModel::RadTan;
        }
        else if (Words[0][0] == "FOV")
        {
            if(Words[0].size()-1 != 5)
                { printf("Invalid combination of model name and parameters!\n"); exit(1);}
            Cameramodel = CamModel::Atan;
        }
        else if (Words[0][0] == "Pinhole")
        {
            if(Words[0].size()-1 != 5)
                { printf("Invalid combination of model name and parameters!\n"); exit(1);}
            Cameramodel = CamModel::Pinhole;
        }
        else if (Words[0][0] == "EquiDistant")
        {
             if(Words[0].size()-1 != 8)
                { printf("Invalid combination of model name and parameters!\n"); exit(1);}
            Cameramodel = CamModel::EquiDistant;
        }
        else if (Words[0][0] == "KannalaBrandt")
        {
             if(Words[0].size()-1 != 8)
                { printf("Invalid combination of model name and parameters!\n"); exit(1);}
            Cameramodel = CamModel::KannalaBrandt;
        }

        if(Cameramodel == Empty)
        {printf("Geometric calibration not valid!\n"); exit(1);}
    }

    try 
    {
        wOrg = std::stoi(Words[1][0]);
        hOrg = std::stoi(Words[1][1]);
        w = std::stoi(Words[3][0]);
        h = std::stoi(Words[3][1]);
        WidthOri = wOrg; HeightOri = hOrg;
    }
    catch(std::exception &e)
    { printf("Could not read input or output resolution! exiting.\n"); exit(1);}

    if(ic[2] < 1 && ic[3] < 1)
    {
        ic[0] *= wOrg; ic[1]*=hOrg;
        ic[2] = ic[2]* wOrg; //-0.5;
        ic[3] = ic[3]*hOrg; //-0.5
    }
    
    if(Sensortype == Stereo)
    {
        if(Words.size()<6) {printf("Stereo system requires more info than provided! \n"); exit(1);}
        if(Words[4][0] == "prerectified")
        {
            try {baseline = std::stof(Words[5][0])/ic[0];} catch(std::exception &e) {printf("could not read baseline*f! exiting\n"); exit(1);}
            Cameramodel = CamModel::Pinhole;
        }
        else if (Words[4][0]=="rectify") //rectification is required
        {
            if(Cameramodel == CamModel::Empty || Cameramodel == CamModel::Atan || Cameramodel == CamModel::Pinhole)
                {printf("Distortion parameters provided are not supported with unrectified images! \n"); exit(1);}
             if(Words.size()<11) {printf("Stereo rectification requires more info than provided! \n"); exit(1);}
            try {baseline = std::stof(Words[5][0])/ic[0];} catch(std::exception &e) {printf("could not read baseline*f! exiting\n"); exit(1);}
            
            //need to insert checks for these but so many !!
            cv::Mat DistL = (cv::Mat_<float>(4,1) << ic[4], ic[5],ic[6], ic[7]);
            cv::Mat IntL = (cv::Mat_<float>(3,3) << ic[0], 0, ic[2], 0, ic[1], ic[3],0,0,1);
            
            cv::Mat R_L = (cv::Mat_<float>(3,3) <<std::stof(Words[6][0]),std::stof(Words[6][1]),std::stof(Words[6][2]),std::stof(Words[6][3]),
                            std::stof(Words[6][4]),std::stof(Words[6][5]),std::stof(Words[6][6]),std::stof(Words[6][7]),std::stof(Words[6][8]));
            
            cv::Mat IntR = (cv::Mat_<float>(3,3) <<std::stof(Words[7][0]),std::stof(Words[7][1]),std::stof(Words[7][2]),std::stof(Words[7][3]),
                            std::stof(Words[7][4]),std::stof(Words[7][5]),std::stof(Words[7][6]),std::stof(Words[7][7]),std::stof(Words[7][8]));

            cv::Mat DistR = (cv::Mat_<float>(4,1) << std::stof(Words[8][0]),std::stof(Words[8][1]),std::stof(Words[8][2]),std::stof(Words[8][3]));

            cv::Mat R_R = (cv::Mat_<float>(3,3) <<std::stof(Words[9][0]),std::stof(Words[9][1]),std::stof(Words[9][2]),std::stof(Words[9][3]),
                            std::stof(Words[9][4]),std::stof(Words[9][5]),std::stof(Words[9][6]),std::stof(Words[9][7]),std::stof(Words[9][8]));

            cv::Mat NewInt = (cv::Mat_<float>(3,3) <<std::stof(Words[10][0]),std::stof(Words[10][1]),std::stof(Words[10][2]),std::stof(Words[10][3]),
                            std::stof(Words[10][4]),std::stof(Words[10][5]),std::stof(Words[10][6]),std::stof(Words[10][7]),std::stof(Words[10][8]));

            cv::initUndistortRectifyMap(IntL, DistL, R_L, NewInt.rowRange(0, 3).colRange(0, 3), cv::Size(w, h), CV_32F, M1l, M2l);
            cv::initUndistortRectifyMap(IntR, DistR, R_R, NewInt.rowRange(0, 3).colRange(0, 3), cv::Size(w, h), CV_32F, M1r, M2r);
            Words[2][0] = "none";
            hOrg = h; wOrg=w;
            Cameramodel = CamModel::Pinhole;
            ic[0] = NewInt.at<float>(0,0);
            ic[1] = NewInt.at<float>(1,1);
            ic[2] = NewInt.at<float>(0,2);
            ic[3] = NewInt.at<float>(1,2);
            std::cout<<NewInt<<std::endl;
        }
        else {printf("Your stereo calibration file is not right! exiting.\n");exit(1);}
    }
    
    remapX = new float[w*h];
    remapY = new float[w*h];

    float OutputCalibration[5];
    try
    {   // an output calibration is specified
        for(int i = 0 ; i < Words[2].size(); i ++)
            OutputCalibration[i] = std::stof(Words[2][i]);

        if(OutputCalibration[2] > 1 && OutputCalibration[3] > 1)
        {printf("calibration should be given relative to image width and height! exiting.\n"); exit(1);}
        
        K.setIdentity();
        K(0,0) = OutputCalibration[0] * w;
        K(1,1) = OutputCalibration[1] * h;
        K(0,2) = OutputCalibration[2] * w;// - 0.5;
        K(1,2) = OutputCalibration[3] * h;// - 0.5;
    }
    catch(const std::exception& e) // a crop model is specified
    {
        if(Words[2][0] == "crop")
        {
            makeOptimalK_crop();
        }
        else if (Words[2][0] == "full")
        {
            printf("Cannot handle full should use none instead!\n"); exit(1);
           // this kept for backward compatibility from DSO.
        }
        else if (Words[2][0] == "none")
        {
            if (w != wOrg || h != hOrg)
            {
                printf("ERROR: rectification mode none requires input and output dimenstions to match!\n\n");
                exit(1);
            }
            K.setIdentity();
            K(0, 0) = ic[0];
            K(1, 1) = ic[1];
            K(0, 2) = ic[2];
            K(1, 2) = ic[3];

            passthrough = true;
        }
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

void Undistorter::makeOptimalK_crop()
{
	printf("finding CROP optimal new model!\n");
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

void Undistorter::distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n)
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


}