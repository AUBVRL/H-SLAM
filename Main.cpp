#include <unistd.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "Main.h"
#include "DatasetLoader.h"
#include "System.h"
#include "Display.h"

#include <chrono>

using namespace FSLAM;

int main(int argc, char **argv) try
{
    //Get input data and initialize dataset reader
    std::shared_ptr<Input> Input_ = std::make_shared<Input>(argc, argv); //parse the arguments and set system settings
    std::shared_ptr<DatasetReader> DataReader = std::make_shared<DatasetReader>(Input_->IntrinCalib, 
    Input_->GammaL, Input_->GammaR, Input_->VignetteL, Input_->VignetteR, Input_->Path, Input_->timestampsL,
    Input_->dataset_);

    //Start gui
    std::shared_ptr<GUI> DisplayHandler;
    if(DisplayOn)
        DisplayHandler = std::make_shared<GUI>();
    
    //Configure image playback data
    if (Input_->Reverse)
    {
        int temp = Input_->Start;
        Input_->Start = Input_->End - 1;
        if (Input_->Start >= DataReader->nImgL)
            Input_->Start = DataReader->nImgL - 1;
        Input_->End = temp;
    }
    std::vector<int> idsToPlay;
    std::vector<double> timesToPlayAt;
    for (int i = Input_->Start; i >= 0 && i < DataReader->nImgL && Input_->linc * i < Input_->linc * Input_->End; i += Input_->linc)
    {
        idsToPlay.push_back(i);
        if (timesToPlayAt.size() == 0)
            timesToPlayAt.push_back((double)0);
        else
        {
            const int idsToPlayCount = idsToPlay.size();
            double tsThis = DataReader->getTimestamp(idsToPlay[idsToPlayCount-1]);
            double tsPrev = DataReader->getTimestamp(idsToPlay[idsToPlayCount-2]);
            timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/Input_->PlaybackSpeed);
        }
    }

    //Preload images if neccessary (reading from drive sometimes throttles!)
    std::vector<std::shared_ptr<ImageData>> Images;
    const int idsToPlayCount = idsToPlay.size();

    if (Input_->Prefetch && Images.empty())
    {
        printf("LOADING ALL IMAGES!\n");
        for (int ii = 0; ii < idsToPlayCount; ++ii)
        {
            int i = idsToPlay[ii];
            std::shared_ptr<ImageData> Img = std::make_shared<ImageData>(DataReader->GeomUndist->wOrg, DataReader->GeomUndist->hOrg);
            DataReader->getImage(Img, i);
            Images.push_back(Img);
        }
    }

    //Create a SLAM system instance
    std::shared_ptr<System> slam = std::make_shared<System>(DataReader->GeomUndist, DataReader->PhoUndistL, DataReader->PhoUndistR);


   //Initialize time
    struct timeval tv_start; gettimeofday(&tv_start, NULL); clock_t started = clock(); double sInitializerOffset = 0;

    for (int ii = 0; ii < idsToPlayCount; ++ii)
    {
        // if (!fullSystem->initialized) // if not initialized: reset start time.
        // {
        //     gettimeofday(&tv_start, NULL);
        //     started = clock();
        //     sInitializerOffset = timesToPlayAt[ii];
        // }

        int i = idsToPlay[ii];
        std::shared_ptr<ImageData> Img;
        if (!Images.empty())
            Img = Images[ii];
        else
        {
            Img = std::make_shared<ImageData>(DataReader->GeomUndist->wOrg, DataReader->GeomUndist->hOrg);
            DataReader->getImage(Img, i); 
        }

        bool skipFrame = false;
        if (Input_->PlaybackSpeed != 0)
        {
            struct timeval tv_now;
            gettimeofday(&tv_now, NULL);
            double sSinceStart = sInitializerOffset + ((tv_now.tv_sec - tv_start.tv_sec) + (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));
            if (sSinceStart < timesToPlayAt[ii])
                usleep((int)((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000)); 
            if (sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2))
            {
                printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                skipFrame = true;
            }
        }
        if (skipFrame)
            continue;

        slam->ProcessNewFrame(Img);

        cv::Mat Dest;
        if (Sensortype == Stereo || Sensortype == RGBD)
        {
            if (Input_->dataset_ != Kitti)
                cv::hconcat(Img->cvImgL, Img->cvImgR, Dest);
            else
                cv::vconcat(Img->cvImgL, Img->cvImgR, Dest);
        }
        else
            Dest = Img->cvImgL;

        cv::cvtColor(Dest,Dest, CV_GRAY2BGR);
        DisplayHandler->UploadFrameImage(Dest.data,Dest.size().width, Dest.size().height);

        if (DisplayHandler) if (DisplayHandler->isDead) break;
    }
    if (DisplayHandler) while (!DisplayHandler->isDead) usleep(50000);
    return 0;
}
catch(std::exception & e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
catch(...) {
    std::cerr << "Unknown exception" << std::endl;
    return 2;
}
