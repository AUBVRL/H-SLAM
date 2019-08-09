#include <fstream>
#include <sstream>
#include <iterator>
#include <opencv2/highgui/highgui.hpp>

#include "photometricUndistorter.h"
#include "Settings.h"


namespace FSLAM
{
PhotometricUndistorter::PhotometricUndistorter(std::string gamma_path, std::string vignetteImage)
{
    vignetteMapInv = 0;
    GammaValid = false;
    VignetteValid = false;


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

}

void PhotometricUndistorter::undistort(cv::Mat &Image, bool isRightRGBD, float factor)
{
    int dim = Image.size().width * Image.size().height;
    float *ptr = Image.ptr<float>();

    if (isRightRGBD)
    {
        Image *= factor;
    }
    else if (GammaValid && VignetteValid)
    {
        for (int i = 0; i < dim; i++)
            ptr[i] = G[cv::saturate_cast<uchar>(ptr[i])] * factor * vignetteMapInv[i];
    }
    else if (GammaValid)
    {
        for (int i = 0; i < dim; i++)
            ptr[i] = G[cv::saturate_cast<uchar>(ptr[i])] * factor;
    }
    else if (VignetteValid)
        for (int i = 0; i < dim; i++)
            ptr[i] = ptr[i] * factor * vignetteMapInv[i];
    else
        Image *= factor;
}
}