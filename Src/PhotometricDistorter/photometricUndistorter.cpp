#include <fstream>
#include <sstream>

#include "photometricUndistorter.h"


namespace FSLAM
{
    PhotometricUndistorter::PhotometricUndistorter(std::string gamma_path, std::string vignetteImage, int w_, int h_)
    {
       

        //try to load class here, if it doesnt work initialize a parallel thread to estimate it if online calib until convergence (params don't change)
    }

    void PhotometricUndistorter::undistort(std::shared_ptr<ImageData> Img, float factor)
    {
        if (valid)
        {
            int dim = Img->cvImgL.size().width*Img->cvImgL.size().height ;
            for (int i = 0; i < dim; i++)
            {
                Img->fImgL[i] = G[(int)std::round(Img->fImgL[i])]; 
            }

            // if (setting_photometricCalibration == 2)
            // {
            //     for (int i = 0; i < wh; i++)
            //         data[i] *= vignetteMapInv[i];
            // }
        }
    }
}