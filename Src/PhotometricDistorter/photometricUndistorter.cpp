#include <fstream>
#include <sstream>
#include <iterator>
#include <opencv2/highgui/highgui.hpp>

#include "photometricUndistorter.h"
#include "Settings.h"
#include "Detector.h"
#include <chrono>


namespace FSLAM
{
PhotometricUndistorter::PhotometricUndistorter(std::string gamma_path, std::string vignetteImage, bool isRight)// int w_, int h_)
{
    vignetteMapInv = 0;
    GammaValid = false;
    VignetteValid = false;

    if(isRight)
    name = "Right";
    else name = "Left";

    if (PhoUndistMode == NoCalib)
    {
        printf("Photometric distortion is off!\n");
        return;
    } //vignette and gamma are not valid.

    if (gamma_path != "" && PhoUndistMode == HaveCalib)
    {
        std::ifstream f(gamma_path.c_str());
        printf("Reading Photometric Calibration from file %s\n", gamma_path.c_str());
        if (!f.good())
        {
            printf("PhotometricUndistorter: Could not open file!\n");
        }
        else
        {
            std::string line;
            std::getline(f, line);
            std::istringstream l1i(line);
            std::vector<float> Gvec = std::vector<float>(std::istream_iterator<float>(l1i), std::istream_iterator<float>());

            GDepth = Gvec.size();
            if (GDepth < 256)
            {
                printf("PhotometricUndistorter: invalid format! got %d entries in first line, expected at least 256!\n", (int)Gvec.size());
            }
            else
            {
                for (int i = 0; i < GDepth; i++)
                    G[i] = Gvec[i];
                bool goodG = true;
                for (int i = 0; i < GDepth - 1; i++)
                {
                    if (G[i + 1] <= G[i])
                    {
                        goodG = false;
                        printf("PhotometricUndistorter: G invalid! it has to be strictly increasing, but it isnt!\n");
                        break;
                    }
                }
                if (goodG)
                {
                    float min = G[0];
                    float max = G[GDepth - 1];
                    for (int i = 0; i < GDepth; i++)
                        G[i] = 255.0 * (G[i] - min) / (max - min); // make it to 0..255 => 0..255.
                    GammaValid = true;
                }
            }
        }
    }

    if (vignetteImage != "" && PhoUndistMode == HaveCalib)
    {
        cvVignette = cv::imread(vignetteImage, CV_LOAD_IMAGE_UNCHANGED);
        if (cvVignette.size().width != WidthOri || cvVignette.size().height != HeightOri)
        {
            printf("something wrong with vignetting! turning it off! \n");
            cvVignette.release();
        }
        else
        {
            int dim = cvVignette.size().width * cvVignette.size().height;
            vignetteMapInv = new float[dim];
            unsigned short * temp  = new unsigned short [dim];
            memcpy(temp, cvVignette.data, 2*dim);

            float maxV=0;
		    for(int i=0;i<dim;i++)
			    if(temp[i] > maxV) maxV = temp[i];

            for(int i=0;i<dim;i++)
			    vignetteMapInv[i] = maxV / temp[i];
            delete temp;
            VignetteValid = true;
        }
    }

    if(PhoUndistMode == OnlineCalib)
        Detector = std::make_shared<ORBDetector>();
}

void PhotometricUndistorter::undistort(cv::Mat &Image, float* fImage, bool isRightRGBD, float factor)
{
    int dim = Image.size().width * Image.size().height;

    if(isRightRGBD)
    {
        for(int i=0; i<dim;i++)
            fImage[i] = Image.data[i]*factor; //this is the RGBD magnitude factor
        return;
    }

    if (GammaValid && VignetteValid)
        for(int i=0; i<dim ;i++)
		    fImage[i] = G[Image.data[i]]*factor* vignetteMapInv[i];
    else if(GammaValid)
        for(int i=0; i<dim;i++)
            fImage[i] = G[Image.data[i]]*factor;
    else if(VignetteValid)
        for(int i=0; i<dim;i++)
		    fImage[i] = Image.data[i]*factor* vignetteMapInv[i];
    else
        for(int i=0; i<dim;i++)
            fImage[i] = Image.data[i]*factor;
}

void PhotometricUndistorter::undistort(float* fImg, cv::Mat& Img, int w, int h, bool isRightRGBD, float factor)
{
    int dim = w*h;

    if(isRightRGBD)
    {
        for(int i=0; i<dim;i++)
            fImg[i] = fImg[i]*factor; //this is the RGBD magnitude factor
        Img = cv::Mat(cv::Size(w, h), CV_32F, fImg);
        Img.convertTo(Img, CV_8U);
        return;
    }

    if (GammaValid && VignetteValid)
        for(int i=0; i<dim ;i++)
		    fImg[i] = G[(int)round(fImg[i])]*factor* vignetteMapInv[i];
    else if(GammaValid)
        for(int i=0; i<dim;i++)
            fImg[i] = G[(int)round(fImg[i])]*factor;
    else if(VignetteValid)
        for(int i=0; i<dim;i++)
		    fImg[i] = fImg[i]*factor* vignetteMapInv[i];
    else
        for(int i=0; i<dim;i++)
            fImg[i] = fImg[i]*factor;

    Img = cv::Mat(cv::Size(w, h), CV_32F, fImg);
    Img.convertTo(Img, CV_8U);

    std::vector<cv::KeyPoint> mvKeys;
    cv::Mat Descriptors;
    int nOrb;
    Detector->ExtractFeatures(Img, mvKeys, Descriptors, nOrb, name);
    return;
}

}