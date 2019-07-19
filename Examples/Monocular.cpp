
#include "Main.h"


int main(int argc, char ** argv)
{
    for (int i =1; i <argc;i++)
        parseargument(argv[i]);
    ValidateInput();

    std::shared_ptr<DatasetReader> DataReader = std::make_shared<DatasetReader>(dataset_, sensor_, Path, Path, Path, Path);
    
    for (int i = 0; i < DataReader->nImgL; i++)
    {
        std::shared_ptr<ImageData> Img = std::make_shared<ImageData>();
        DataReader->getImage(Img, sensor_, i);



        // cv::Mat Dest;
        // cv::hconcat(Img->ImageL, Img->ImageR, Dest);
        cv::imshow("Img",Img->ImageL);
        cv::waitKey(1);
    }



    return 0;
}


