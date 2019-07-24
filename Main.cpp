#include <unistd.h>
#include <memory>
#include <chrono>

#include "Main.h"
#include "DatasetLoader.h"

using namespace FSLAM;

int main(int argc, char **argv)
{
    std::shared_ptr<Input> Input_ = std::make_shared<Input>(argc, argv);

     //Initialize undistorters
    std::shared_ptr<GeometricUndistorter> GeomUndistorter = std::make_shared<GeometricUndistorter>(Input_->IntrinCalib);

    std::shared_ptr<DatasetReader> DataReader = std::make_shared<DatasetReader>(Input_->dataset_, Input_->Path, Input_->timestampsL, GeomUndistorter);
    
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
        {
            timesToPlayAt.push_back((double)0);
        }
        else
        {
            double tsThis = DataReader->getTimestamp(idsToPlay[idsToPlay.size()-1]);
            double tsPrev = DataReader->getTimestamp(idsToPlay[idsToPlay.size()-2]);
            timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/Input_->PlaybackSpeed);
        }
    }

    std::vector<std::shared_ptr<ImageData>> Images;
    if (Input_->Prefetch && Images.empty())
    {
        printf("LOADING ALL IMAGES!\n");
        for (int ii = 0; ii < (int)idsToPlay.size(); ii++)
        {
            int i = idsToPlay[ii];
            std::shared_ptr<ImageData> Img = std::make_shared<ImageData>();
            DataReader->getImage(Img, i);
            Images.push_back(Img);
        }
    }
    struct timeval tv_start;
    gettimeofday(&tv_start, NULL);
    clock_t started = clock();
    double sInitializerOffset = 0;


    for (int ii = 0; ii < (int)idsToPlay.size(); ii++)
    {
        // if (!fullSystem->initialized) // if not initialized: reset start time.
        // {
        //     gettimeofday(&tv_start, NULL);
        //     started = clock();
        //     sInitializerOffset = timesToPlayAt[ii];
        // }
        int i = idsToPlay[ii];
        std::shared_ptr<ImageData> Img;
        if (Input_->Prefetch)
            Img = Images[ii];
        else
        {
            std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
            Img = std::make_shared<ImageData>();
            DataReader->getImage(Img, i);
            std::cout << "time: " << (float)(((std::chrono::duration<double>)(std::chrono::high_resolution_clock::now() - start)).count() * 1e3) << std::endl;
        }

        bool skipFrame = false;
            if (Input_->PlaybackSpeed != 0) {
                struct timeval tv_now;
                gettimeofday(&tv_now, NULL);
                double sSinceStart = sInitializerOffset + ((tv_now.tv_sec - tv_start.tv_sec) +
                                                           (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));
                
                if (sSinceStart < timesToPlayAt[ii]) {
                    usleep((int) ((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000));
                }
                if (sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2)) {
                    printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                    skipFrame = true;
                }
            }
            if (skipFrame)
                continue;

        cv::Mat Dest;
        if (Sensortype == Stereo)
        {
            if (Input_->dataset_ != Kitti)
                cv::hconcat(Img->ImageL, Img->ImageR, Dest);
            else
                cv::vconcat(Img->ImageL, Img->ImageR, Dest);
        }
        else if (Sensortype == RGBD)
            cv::hconcat(Img->ImageL, Img->Depth, Dest);
        else
            Dest = Img->ImageL;

        cv::imshow("Img", Dest);
        cv::waitKey(1);
    }

    return 0;
}
