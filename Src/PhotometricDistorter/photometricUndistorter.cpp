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
    invignetteMapInv = NULL;
    vignetteMapInv = NULL;
    GDepth=0;
    GammaValid = false;
    VignetteValid = false;

    if (PhoUndistMode == NoCalib) //do not use vignette nor gamma.
    {
        printf("Photometric distortion is off!\n");
        for(int i=0;i<256;++i) inG[i]= (float)i;
        UpdateGamma(inG);
        //should reset vignette here too!!
        return;
    } 
    else if (PhoUndistMode == HaveCalib)
    {
        if (gamma_path != "")
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
                    for (int i = 0; i < GDepth; ++i)
                        inG[i] = Gvec[i];
                    bool goodG = true;
                    for (int i = 0; i < GDepth - 1; ++i)
                    {
                        if (inG[i + 1] <= inG[i])
                        {
                            goodG = false;
                            printf("PhotometricUndistorter: G invalid! it has to be strictly increasing, but it isnt!\n");
                            break;
                        }
                    }
                    if (goodG)
                    {
                        float min = inG[0];
                        float max = inG[GDepth - 1];
                        for (int i = 0; i < GDepth; ++i)
                            inG[i] = 255.0 * (inG[i] - min) / (max - min); // make it to 0..255 => 0..255.
                        GammaValid = true;
                        UpdateGamma(inG);
                    }
                }
            }
        }
        if (!GammaValid)
        {
            for(int i=0;i<256;++i) inG[i]= (float)i;
            UpdateGamma(inG);
        }

        /*-----------Load Vignette Model--------*/
        if (vignetteImage != "")
        {
            incvVignette = cv::imread(vignetteImage, CV_LOAD_IMAGE_UNCHANGED);
            if (incvVignette.size().width != WidthOri || incvVignette.size().height != HeightOri)
            {
                printf("something wrong with vignetting! turning it off! \n");
                incvVignette.release();
            }
            else
            {
                int dim = incvVignette.size().width * incvVignette.size().height;
                invignetteMapInv = new float[dim];
                unsigned short *temp = new unsigned short[dim];
                memcpy(temp, incvVignette.data, 2 * dim);

                float maxV = 0;
                for (int i = 0; i < dim; ++i)
                    if (temp[i] > maxV)
                        maxV = temp[i];

                for (int i = 0; i < dim; ++i)
                    invignetteMapInv[i] = maxV / temp[i];
                delete temp;
                VignetteValid = true;
                ResetVignette();
            }
        }
    }
    else if(PhoUndistMode == OnlineCalib)
    {
        //initialize data for online calib
        for(int i=0;i<256;++i) inG[i]= (float)i;
        UpdateGamma(inG);
        // initialize vignette here
        
    }
}

void PhotometricUndistorter::undistort(cv::Mat &Image, bool isRightRGBD, float factor)
{
    boost::unique_lock<boost::mutex> lock (mlock);
    int dim = Image.size().width * Image.size().height;
    float *ptr = Image.ptr<float>();

    if (isRightRGBD)
    {
        Image *= factor;
    }
    else if (GammaValid && VignetteValid)
    {
        for (int i = 0; i < dim; ++i)
            ptr[i] = Binv[cv::saturate_cast<uchar>(ptr[i])] * factor * vignetteMapInv[i];
    }
    else if (GammaValid)
    {
        for (int i = 0; i < dim; ++i)
            ptr[i] = Binv[cv::saturate_cast<uchar>(ptr[i])] * factor;
    }
    else if (VignetteValid)
        for (int i = 0; i < dim; ++i)
            ptr[i] = ptr[i] * factor * vignetteMapInv[i];
    else
        Image *= factor;
}

void PhotometricUndistorter::ResetVignette() //when system is reset, call this to reset vignette estimates
{
    boost::unique_lock<boost::mutex> lock (mlock);
    if(vignetteMapInv != 0)
        delete vignetteMapInv;

    if (PhoUndistMode == HaveCalib && VignetteValid)
    {
        int dim = incvVignette.size().width * incvVignette.size().height;
        vignetteMapInv = new float[dim];

        for (int i = 0; i < dim; ++i)
            vignetteMapInv[i] = invignetteMapInv[i];
    }
    else
    {
        //Reset vignette model here !!
    }
    
}


/* 
    Gamma should be updated once if we are running with havecalib
    or updated every time the global model changes when running onlinecalib.
    Binv is gamma and B is its inverse.
*/
void PhotometricUndistorter::UpdateGamma(float *_BInv)
{
    if (_BInv == 0) //should not happen!
    {
        printf("you tried updating gamma with nothing!!"); 
        return;
    }
    boost::unique_lock<boost::mutex> lock (mlock);

    memcpy(Binv, _BInv, sizeof(float) * 256);
    // invert binv to get B
    for (int i = 1; i < 255; ++i)
    {
        for (int s = 1; s < 255; ++s)
        {
            if (_BInv[s] <= i && _BInv[s + 1] >= i)
            {
                B[i] = s + (i - _BInv[s]) / (_BInv[s + 1] - _BInv[s]);
                break;
            }
        }
    }
    B[0] = 0;
    B[255] = 255;
    return;
}

void PhotometricUndistorter::Reset()
{
    if(PhoUndistMode == HaveCalib)
    {
        ResetGamma();
        ResetVignette();
    }
}

void PhotometricUndistorter::ResetGamma() //when system is reset, call this to reset gamma estimates
{
    boost::unique_lock<boost::mutex> lock (mlock);

    if(PhoUndistMode == HaveCalib && GammaValid)
    {
        for (int i =0; i < 256; ++i)
            Binv[i] = inG[i];
    }
    else 
    {
        for (int i =0; i < 256; ++i)
            Binv[i] = static_cast<float>(i);
    }
    // invert.
    for (int i = 1; i < 255; ++i)
    {
        for (int s = 1; s < 255; ++s)
        {
            if (Binv[s] <= i && Binv[s + 1] >= i)
            {
                B[i] = s + (i - Binv[s]) / (Binv[s + 1] - Binv[s]);
                break;
            }
        }
    }
    B[0] = 0;
    B[255] = 255;
}

}