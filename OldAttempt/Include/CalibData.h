#ifndef __CalibData__
#define __CalibData__
#pragma once

#include "globalTypes.h"
namespace cv
{
    class Mat;
}

namespace SLAM
{

class CalibData
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /*------Geometric Data------*/
    int Width;
    int Height;
    int pyrLevels;

    VecC value_zero; //VecC contains the camera parameters (fx fy cx cy)
    VecC value_scaled;
    VecCf value_scaledf;
    VecCf value_scaledi; //contains the inverse of the camera parameters: 1/fx, 1/fy, -cx/fx, -cy/fy
    VecC value;
    VecC step;
    VecC step_backup;
    VecC value_backup;
    VecC value_minus_value_zero;

    VecC initial_value;

    std::vector<int> wpyr, hpyr;
    vector<size_t> pyrImgSize;
    std::vector<Mat33> pyrK;
    std::vector<Mat33> pyrKi;

    int sumWH;
    int mulWH;

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

        UpdatePyramidalK();
    }

    inline float &fxl() { return value_scaledf[0]; }
    inline float &fyl() { return value_scaledf[1]; }
    inline float &cxl() { return value_scaledf[2]; }
    inline float &cyl() { return value_scaledf[3]; }
    inline float &fxli() { return value_scaledi[0]; }
    inline float &fyli() { return value_scaledi[1]; }
    inline float &cxli() { return value_scaledi[2]; }
    inline float &cyli() { return value_scaledi[3]; }

    inline CalibData(int _Width, int _Height, Mat33 K, int& DirPyrSize) : Width(_Width), Height(_Height)
    {
        pyrLevels = 1;
        while (_Width % 2 == 0 && _Height % 2 == 0 && _Width * _Height > 5000 && pyrLevels < DirPyrSize)
        {
            _Width /= 2;
            _Height /= 2;
            pyrLevels++;
        }
        cout << "using pyramid levels 0 to " << pyrLevels-1 << ". coarsest resolution: " << _Width << " x " << _Height << endl;

        if ( (_Width > 100 && _Height  > 100) || pyrLevels < 3 )
            cout<<"not enough pyramid levels! increase the number of pyramid levels in settings"<<endl;

        DirPyrSize = pyrLevels;

        initial_value = VecC::Zero();
        initial_value[0] = (double)K(0, 0);
        initial_value[1] = (double)K(1, 1);
        initial_value[2] = (double)K(0, 2);
        initial_value[3] = (double)K(1, 2);

        setValueScaled(initial_value);
        value_zero = value;
        value_minus_value_zero.setZero();

        //Create pyramid calibration data used to create the direct pyramids
        wpyr.push_back(Width); hpyr.push_back(Height);
        sumWH = Width + Height;
        mulWH = Width * Height;
        for (int i = 1; i < pyrLevels; ++i)
        {
            wpyr.push_back(wpyr[i - 1] / 2);
            hpyr.push_back(hpyr[i - 1] / 2);
        }

        pyrK.resize(pyrLevels); pyrKi.resize(pyrLevels);
        UpdatePyramidalK();
        for (int i = 0; i < pyrLevels; ++i)
            pyrImgSize.push_back(wpyr[i]*hpyr[i]);
    }

    inline void UpdatePyramidalK()
    {
        pyrK[0] <<  value_scaled[0], 0.0, value_scaled[2], 0.0, value_scaled[1], value_scaled[3], 0.0, 0.0, 1.0;
        pyrKi[0] = pyrK[0].inverse();
        for (int i = 1; i < pyrLevels; ++i)
        {
            pyrK[i] << pyrK[i-1](0,0) * 0.5, 0.0, (value_scaled[2] + 0.5) / ((int)1 << i) - 0.5, 
                       0.0, pyrK[i-1](1,1) * 0.5,  (value_scaled[3] + 0.5) / ((int)1 << i) - 0.5,
                       0.0, 0.0, 1.0;
            pyrKi[i] = pyrK[i].inverse();
        }
    }

    inline ~CalibData() {}

};

} // namespace SLAM
#endif