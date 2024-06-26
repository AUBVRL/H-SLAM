#ifndef __CalibData__
#define __CalibData__
#pragma once

#include "GlobalTypes.h"
namespace cv
{
    class Mat;
}

namespace FSLAM
{

class PhotometricUndistorter;

class CalibData
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /*------Geometric Data------*/
    int Width;
    int Height;
    float mbf;

    std::shared_ptr<PhotometricUndistorter> PhotoUnDistL;
    std::shared_ptr<PhotometricUndistorter> PhotoUnDistR;

    VecC value_zero; //VecC contains the camera parameters (fx fy cx cy)
    VecC value_scaled;
    VecCf value_scaledf;
    VecCf value_scaledi; //contains the inverse of the camera parameters: 1/fx, 1/fy, -cx/fx, -cy/fy
    VecC value;
    VecC step;
    VecC step_backup;
    VecC value_backup;
    VecC value_minus_value_zero;

    std::vector<int> wpyr, hpyr;
    std::vector<double> pyrfx, pyrfxi, pyrfy, pyrfyi, pyrcx, pyrcxi, pyrcy, pyrcyi;
    
    std::vector<Mat33> pyrK, pyrKi;

    std::vector<cv::Size> IndPyrSizes;
    std::vector<float> IndScaleFactors;
    std::vector<float> IndInvScaleFactors;

    inline cv::Mat GetCvK()
    {
        float data[9] = { value_scaledf[0], 0, value_scaledf[2], 0.0f, value_scaledf[1], value_scaledf[3], 0.0f, 0.0f, 1.0f };
        return cv::Mat(3, 3, CV_32F, data);
    }

    inline cv::Mat GetCvInvK()
    {
        float data[9] = { value_scaledi[0], 0, value_scaledi[2], 0.0f, value_scaledi[1], value_scaledi[3], 0.0f, 0.0f, 1.0f };
        return cv::Mat(3, 3, CV_32F, data);
    }

    inline void setValueScaled(const VecC &_value_scaled)
    {
        value_scaled.setZero();
        value_scaled = _value_scaled;
        value_scaledf = value_scaled.cast<float>();
        value[0] = SCALE_F_INVERSE * _value_scaled[0];
        value[1] = SCALE_F_INVERSE * _value_scaled[1];
        value[2] = SCALE_C_INVERSE * _value_scaled[2];
        value[3] = SCALE_C_INVERSE * _value_scaled[3];

        value_minus_value_zero = value - value_zero;
        value_scaledi[0] = 1.0f / value_scaledf[0];
        value_scaledi[1] = 1.0f / value_scaledf[1];
        value_scaledi[2] = -value_scaledf[2] / value_scaledf[0];
        value_scaledi[3] = -value_scaledf[3] / value_scaledf[1];
    }

    inline void setValue(const VecC &_value)
    {
        value = _value;
        value_scaled[0] = SCALE_F * _value[0];
        value_scaled[1] = SCALE_F * _value[1];
        value_scaled[2] = SCALE_C * _value[2];
        value_scaled[3] = SCALE_C * _value[3];

        value_scaledf = value_scaled.cast<float>();
        value_scaledi[0] = 1.0f / value_scaledf[0];
        value_scaledi[1] = 1.0f / value_scaledf[1];
        value_scaledi[2] = -value_scaledf[2] / value_scaledf[0];
        value_scaledi[3] = -value_scaledf[3] / value_scaledf[1];
        value_minus_value_zero = value - value_zero;
    }

    inline float &fxl() { return value_scaledf[0]; }
    inline float &fyl() { return value_scaledf[1]; }
    inline float &cxl() { return value_scaledf[2]; }
    inline float &cyl() { return value_scaledf[3]; }
    inline float &fxli() { return value_scaledi[0]; }
    inline float &fyli() { return value_scaledi[1]; }
    inline float &cxli() { return value_scaledi[2]; }
    inline float &cyli() { return value_scaledi[3]; }

    inline CalibData(int _Width, int _Height, Mat33 K, float baseline, std::shared_ptr<PhotometricUndistorter> _PhoUndL,
                     std::shared_ptr<PhotometricUndistorter> _PhoUndR, int& DirPyrSize, int IndPyrLevels,
                     float IndPyrScaleFactor) : Width(_Width), Height(_Height), PhotoUnDistL(_PhoUndL), PhotoUnDistR(_PhoUndL), mbf(baseline)
    {

        int w_ = Width;
        int h_ = Height;
        int pyrLevels = 1;
        while (w_ % 2 == 0 && h_ % 2 == 0 && w_ * h_ > 5000 && pyrLevels < DirPyrSize)
        {
            w_ /= 2;
            h_ /= 2;
            pyrLevels++;
        }
        printf("using pyramid levels 0 to %d. coarsest resolution: %d x %d!\n",
               pyrLevels - 1, w_, h_);
        if (w_ > 100 && h_ > 100)
        {
            printf("\n\n===============WARNING!===================\n "
                   "using not enough pyramid levels.\n"
                   "Consider scaling to a resolution that is a multiple of a power of 2.\n");
        }
        if (pyrLevels < 3)
        {
            printf("\n\n===============WARNING!===================\n "
                   "I need higher resolution.\n"
                   "I will probably segfault.\n");
        }
        DirPyrSize = pyrLevels;

        VecC initial_value = VecC::Zero();
        initial_value[0] = (double)K(0, 0);
        initial_value[1] = (double)K(1, 1);
        initial_value[2] = (double)K(0, 2);
        initial_value[3] = (double)K(1, 2);

        setValueScaled(initial_value);
        value_zero = value;
        value_minus_value_zero.setZero();

        //Create pyramid calibration data used to create the direct pyramids
        pyrK.resize(DirPyrSize);
        pyrKi.resize(DirPyrSize);

        wpyr.push_back(_Width); hpyr.push_back(_Height);
        pyrfx.push_back(K(0, 0)); pyrfy.push_back(K(1, 1));
        pyrcx.push_back(K(0, 2)); pyrcy.push_back(K(1, 2));
        pyrK[0] = K;
        pyrKi[0] = pyrK[0].inverse();
        pyrfxi.push_back(pyrKi[0](0,0)); pyrfyi.push_back(pyrKi[0](1,1));
		pyrcxi.push_back(pyrKi[0](0,2)); pyrcyi.push_back(pyrKi[0](1,2));

        for (int i = 1; i < DirPyrSize; ++i)
        {
            wpyr.push_back(wpyr[i - 1] / 2); hpyr.push_back(hpyr[i - 1] / 2);
            pyrfx.push_back(pyrfx[i - 1] * 0.5); pyrfy.push_back(pyrfy[i - 1] * 0.5);
            pyrcx.push_back((pyrcx[0] + 0.5) / ((int)1 << i) - 0.5);
            pyrcy.push_back((pyrcy[0] + 0.5) / ((int)1 << i) - 0.5);
			pyrK[i] << pyrfx[i], 0.0f, pyrcx[i], 0.0f, pyrfy[i], pyrcy[i], 0.0f, 0.0f, 1.0f;

            pyrKi[i] = pyrK[i].inverse();
			pyrfxi[i] = pyrKi[i](0,0);
			pyrfyi[i] = pyrKi[i](1,1);
			pyrcxi[i] = pyrKi[i](0,2);
			pyrcyi[i] = pyrKi[i](1,2);

        }

        //Setup Indirect pyramid data
        IndPyrSizes.resize(IndPyrLevels);
        IndScaleFactors.resize(IndPyrLevels);
        IndInvScaleFactors.resize(IndPyrLevels);
        for (int i = 0 ; i < IndPyrLevels; ++i)
        {
            if(i == 0)
            {
                IndPyrSizes[i] = cv::Size(Width, Height);
                IndScaleFactors[i] = 1;
                IndInvScaleFactors[i] = 1;
            }
            else
            {
                IndScaleFactors[i] = IndScaleFactors[i-1] * IndPyrScaleFactor;
                IndInvScaleFactors[i] = IndInvScaleFactors[i-1] / IndPyrScaleFactor;
                IndPyrSizes[i] = cv::Size(cvRound((float) Width * IndInvScaleFactors[i]), cvRound((float)Height * IndInvScaleFactors[i]));
            }
        }
    }

    inline ~CalibData() {}

};

} // namespace FSLAM
#endif