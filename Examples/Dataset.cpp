
#include "Main.h"
#include <unistd.h>


int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
        parseargument(argv[i]);

    ValidateInput();

    std::shared_ptr<DatasetReader> DataReader = std::make_shared<DatasetReader>(dataset_, sensor_, Path, Path, Path, Path);

    if (Reverse)
    {
        int temp = Start;
        Start = End - 1;
        if (Start >= DataReader->nImgL)
            Start = DataReader->nImgL - 1;
        End = temp;
    }

    std::vector<int> idsToPlay;
    std::vector<double> timesToPlayAt;
    for (int i = Start; i >= 0 && i < DataReader->nImgL && linc * i < linc * End; i += linc)
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
            timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/PlaybackSpeed);
        }
    }

    std::vector<std::shared_ptr<ImageData>> Images;
    if (Prefetch && Images.empty())
    {
        printf("LOADING ALL IMAGES!\n");
        for (int ii = 0; ii < (int)idsToPlay.size(); ii++)
        {
            int i = idsToPlay[ii];
            std::shared_ptr<ImageData> Img = std::make_shared<ImageData>();
            DataReader->getImage(Img, sensor_, i);
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
        if (Prefetch)
            Img = Images[ii];
        else
        {
            Img = std::make_shared<ImageData>();
            DataReader->getImage(Img, sensor_, i);
        }

        bool skipFrame = false;
            if (PlaybackSpeed != 0) {
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
        if (sensor_ == Stereo)
        {
            if (dataset_ != Kitti)
                cv::hconcat(Img->ImageL, Img->ImageR, Dest);
            else
                cv::vconcat(Img->ImageL, Img->ImageR, Dest);
        }
        else if (sensor_ == RGBD)
            cv::hconcat(Img->ImageL, Img->Depth, Dest);
        else
            Dest = Img->ImageL;

        cv::imshow("Img", Dest);
        cv::waitKey(1);
    }

    return 0;
}