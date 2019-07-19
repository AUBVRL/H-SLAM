
#include "Main.h"

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
            // double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size()-1]);
            // double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size()-2]);
            // timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/playbackSpeed);
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

    for (int ii = 0; ii < (int)idsToPlay.size(); ii++)
    {
        int i = idsToPlay[ii];
        std::shared_ptr<ImageData> Img;
        if (Prefetch)
            Img = Images[ii];
        else
        {
            Img = std::make_shared<ImageData>();
            DataReader->getImage(Img, sensor_, i);
        }

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
        cv::waitKey(0);
    }

    return 0;
}
