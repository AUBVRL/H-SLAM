#ifndef __CalibData__
#define __CalibData__
#pragma once

#include "GlobalTypes.h"

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
    VecCf value_scaledi; //This contains the inverse of the camera parameters: 1/fx, 1/fy, -cx/fx, -cy/fy
    VecC value;
    VecC step;
    VecC step_backup;
    VecC value_backup;
    VecC value_minus_value_zero;

    inline CalibData(int _Width, int _Height, Mat33f K, float baseline, std::shared_ptr<PhotometricUndistorter> _PhoUndL,
                     std::shared_ptr<PhotometricUndistorter> _PhoUndR) : Width(_Width), Height(_Height), PhotoUnDistL(_PhoUndL), 
                                                                         PhotoUnDistR(_PhoUndL), mbf(baseline)
    {
        // PyrWidth.push_back(Width);
        // PyrHeight.push_back(Height);
        // for (int i = 1; i < PyrSize; i++)
        // {r
        //     PyrWidth.push_back(cvRound((float)PyrWidth[i-1]/ScaleFactor));
        //     PyrHeight.push_back(cvRound((float)PyrHeight[i-1]/ScaleFactor));
        // }
        VecC initial_value = VecC::Zero();
        initial_value[0] = (double)K(0, 0);
        initial_value[1] = (double)K(1, 1);
        initial_value[2] = (double)K(0, 2);
        initial_value[3] = (double)K(1, 2);

        setValueScaled(initial_value);
        value_zero = value;
        value_minus_value_zero.setZero();
    }

    inline ~CalibData() {}

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
};

} // namespace FSLAM
#endif