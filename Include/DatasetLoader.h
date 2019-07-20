#ifndef __DatasetLoader__
#define __DatasetLoader__

#include <fstream>
#include <dirent.h>
#include <opencv2/highgui/highgui.hpp>

#include "GlobalTypes.h"
#include "Settings.h"

#if HAS_ZIPLIB
#include "zip.h"
#endif

using namespace FSLAM;
enum CameraModel {Radtan, Opencv};

class DatasetReader
{
public:
    
    std::vector<std::string> filesL;
    std::vector<std::string> filesR;
    std::vector<std::string> filesD;
    std::vector<double> timestamps;
    std::vector<float> exposures;

    int nImgL;
    int nImgR;
    int nImgD;

    Sensor sensor;

    bool isZipped;
    
#if HAS_ZIPLIB
    zip_t *ziparchive;
    char *databuffer;
#endif

    DatasetReader(Dataset Dataset_t, Sensor Sensor_t,  std::string Path, std::string IntrCalib, std::string Gamma, std::string Vignette)
    {
        this->sensor = Sensor_t;
        #if HAS_ZIPLIB
        ziparchive = 0;
        databuffer = 0;
        #endif
        isZipped = (Path.length() > 4 && Path.substr(Path.length() - 4) == ".zip");
        if (isZipped)
        {
            
            #if HAS_ZIPLIB
            int ziperror = 0;
            ziparchive = zip_open(Path.c_str(), ZIP_RDONLY, &ziperror);
            if (ziperror != 0)
            {
                printf("ERROR %d reading archive %s!\n", ziperror, Path.c_str());
                exit(1);
            }

            filesL.clear();
            filesR.clear();
            filesD.clear();
            int numEntries;
            if (Dataset_t == Dataset::Tum_mono)
            {
                numEntries = zip_get_num_entries(ziparchive, 0);
                for (int k = 0; k < numEntries; k++)
                {
                    const char *name = zip_get_name(ziparchive, k, ZIP_FL_ENC_STRICT);
                    std::string nstr = std::string(name);
                    if (nstr == "." || nstr == "..")
                        continue;
                    filesL.push_back(name);
                }
                std::sort(filesL.begin(), filesL.end());
            }
            else if (Dataset_t == Dataset::Euroc)
            {
                numEntries = zip_get_num_entries(ziparchive, 0);
                std::string LeftDir = "mav0/cam0/data/";
                std::string RightDir = "mav0/cam1/data/";
                for (int k = 0; k < numEntries; k++)
                {
                    const char *name = zip_get_name(ziparchive, k, ZIP_FL_ENC_STRICT);
                    std::string nstr = std::string(name);
                    if (nstr == "." || nstr == ".." || nstr == LeftDir || nstr == RightDir)
                        continue;

                    if (LeftDir.compare(0, 15, nstr, 0, 15) == 0)
                        filesL.push_back(nstr);
                    else if (RightDir.compare(0, 15, nstr, 0, 15) == 0)
                        filesR.push_back(nstr);
                }
                if ((filesL.size() != filesR.size()) && Sensor_t == Stereo)
                {
                    printf("number of left images not equal number of right images!");
                    exit(-1);
                }
                std::sort(filesL.begin(), filesL.end());
                std::sort(filesR.begin(), filesR.end());
            }
            else
            {
                printf("Zip for the current dataset is not supported\n");
                exit(-1);
            }

            #else 
                printf("ERROR: cannot read .zip archive without ziplib!\n");
                exit(-1);
            #endif
                
        }
        else
        {
            if (Path.at(Path.length() - 1) != '/')
                Path = Path + "/";

            if (Dataset_t == Dataset::Tum_mono)
                getdir(Path, filesL);
            else if (Dataset_t == Dataset::Euroc)
            {
                getdir(Path + "mav0/cam0/data/", filesL);
                getdir(Path + "mav0/cam1/data/", filesR);
            }
            else if (Dataset_t == Dataset::Kitti)
            {
                getdir(Path + "image_0/", filesL);
                getdir(Path + "image_1/", filesR);
            }
        }

        nImgL = filesL.size(); nImgR = filesR.size(); nImgD = filesD.size();

        if (Sensor_t == Stereo)
            if (nImgL == 0 || nImgR == 0 || nImgL != nImgR)
            {
                printf("There is something wrong with the loaded stereo data");
                exit(-1);
            }
        if (Sensor_t == Monocular)
            if (filesL.size() == 0)
            {
                printf("There is something wrong with the loaded monocular images");
                exit(-1);
            }
    }

    ~DatasetReader() 
    {
        #if HAS_ZIPLIB
            if (ziparchive != 0) zip_close(ziparchive);
            if (databuffer != 0) delete databuffer;
        #endif
    }

    inline void loadtimestamps()
    {
    }

    inline int getdir(std::string dir, std::vector<std::string> &files)
    {
        DIR *dp;
        struct dirent *dirp;
        if ((dp = opendir(dir.c_str())) == NULL)
        {
            printf("FAILED to open path %s",dir.c_str());
            exit(-1);
        }

        while ((dirp = readdir(dp)) != NULL)
        {
            std::string name = std::string(dirp->d_name);
            if (name != "." && name != ".." && ( name.substr(name.size() - 3, name.size()) == "jpg" || name.substr(name.size() - 3, name.size()) == "png") )
                files.push_back(name);
        }
        closedir(dp);
        std::sort(files.begin(), files.end());
        if (dir.at(dir.length() - 1) != '/')
            dir = dir + "/";
        for (unsigned int i = 0; i < files.size(); i++)
        {
            if (files[i].at(0) != '/')
                files[i] = dir + files[i];
        }

        return files.size();
    }

    inline void getImage(std::shared_ptr<ImageData> ImgData, Sensor sensor, int id )
    {
        if (!isZipped)
        {
            if(sensor == Stereo)
            {
                ImgData->ImageL = cv::imread(filesL[id], cv::IMREAD_GRAYSCALE);
                ImgData->ImageR = cv::imread(filesR[id], cv::IMREAD_GRAYSCALE);
            }
            else if(sensor == Monocular)
                ImgData->ImageL = cv::imread(filesL[id], cv::IMREAD_GRAYSCALE);
        }
        else
        {
            #if HAS_ZIPLIB
            if(sensor == Stereo)
            {
                long readsize = ReadZipBuffer(filesL[id]);
                ImgData->ImageL = cv::imdecode(cv::Mat(readsize,1,CV_8U, databuffer), cv::IMREAD_GRAYSCALE);
                readsize = ReadZipBuffer(filesR[id]);
                ImgData->ImageR = cv::imdecode(cv::Mat(readsize,1,CV_8U, databuffer), cv::IMREAD_GRAYSCALE);
            }
            else if(sensor == Monocular)
            {
                long readsize = ReadZipBuffer(filesL[id]);
                ImgData->ImageL =  cv::imdecode(cv::Mat(readsize,1,CV_8U, databuffer), cv::IMREAD_GRAYSCALE); 
            }
            #else
                printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
                exit(1);
            #endif
        }
        return;
    }

    inline long ReadZipBuffer(std::string In_)
    {
        if (databuffer == 0) databuffer = new char[WidthOri * HeightOri * 6 + 10000];
            zip_file_t *fle = zip_fopen(ziparchive, In_.c_str(), 0);
            long readbytes = zip_fread(fle, databuffer, (long) WidthOri * HeightOri * 6 + 10000);
            if (readbytes > (long) WidthOri * HeightOri * 6) {
                printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes,
                       (long) WidthOri * HeightOri * 6 + 10000, In_.c_str());
                delete[] databuffer;
                databuffer = new char[(long) WidthOri * HeightOri * 30];
                fle = zip_fopen(ziparchive, In_.c_str(), 0);
                readbytes = zip_fread(fle, databuffer, (long) WidthOri * HeightOri * 30 + 10000);
               
                if (readbytes > (long) WidthOri * HeightOri * 30) {
                    printf("buffer still to small (read %ld/%ld). abort.\n", readbytes,
                           (long) WidthOri * HeightOri * 30 + 10000);
                    exit(1);
                }
            }
            zip_fclose(fle);
            return readbytes;
    }
};

#endif