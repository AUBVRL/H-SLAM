#ifndef __DatasetLoader__
#define __DatasetLoader__

#include <fstream>
#include <dirent.h>
#include <opencv2/highgui/highgui.hpp>

#include "GeometricUndistorter.h"
#include "photometricUndistorter.h"
#include "Main.h"

#if HAS_ZIPLIB
#include "zip.h"
#endif

using namespace FSLAM;

class DatasetReader
{
public:
    
    std::vector<std::string> filesL;
    std::vector<std::string> filesR;
    std::vector<std::string> filesD;
    std::vector<double> timestamps;
    std::vector<float> exposuresL;
    std::vector<float> exposuresR;

    int nImgL;
    int nImgR;
    int nImgD;

    bool isZipped;
    Dataset dataset;

    std::shared_ptr<GeometricUndistorter> GeomUndist;
    std::shared_ptr<PhotometricUndistorter> PhoUndistL;
    std::shared_ptr<PhotometricUndistorter> PhoUndistR;

    float *tPhoCalibL;
    float *tPhoCalibR;

    
#if HAS_ZIPLIB
    zip_t *ziparchive;
    char *databuffer;
#endif

    DatasetReader(std::shared_ptr<Input> Input_): dataset(Input_->dataset_)
    {
        tPhoCalibL = tPhoCalibR = NULL;
        //Initialize undistorters
        GeomUndist= std::make_shared<GeometricUndistorter>(Input_->IntrinCalib);

        PhoUndistL = std::make_shared<PhotometricUndistorter>(Input_->GammaL, Input_->VignetteL, GeomUndist->wOrg, GeomUndist->hOrg);
        tPhoCalibL = new float[GeomUndist->wOrg * GeomUndist->hOrg];
        if (Sensortype == Stereo)
        {
            PhoUndistR = std::make_shared<PhotometricUndistorter>(Input_->GammaR, Input_->VignetteR, GeomUndist->wOrg, GeomUndist->hOrg);
            tPhoCalibR = new float[GeomUndist->wOrg * GeomUndist->hOrg];
        }

        std::string ImPath = Input_->Path;
        std::string timestampPath = Input_->timestampsL;

#if HAS_ZIPLIB
        ziparchive = 0;
        databuffer = 0;
        #endif
        isZipped = (ImPath.length() > 4 && ImPath.substr(ImPath.length() - 4) == ".zip");

        if (isZipped)
        {
            #if HAS_ZIPLIB
            int ziperror = 0;
            ziparchive = zip_open(ImPath.c_str(), ZIP_RDONLY, &ziperror);
            if (ziperror != 0)
            {printf("ERROR %d reading archive %s!\n", ziperror, ImPath.c_str()); exit(1);}

            filesL.clear();
            filesR.clear();
            filesD.clear();
            int numEntries;
            if (dataset == Dataset::Tum_mono)
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
            else if (dataset == Dataset::Euroc)
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
                if ((filesL.size() != filesR.size()) && Sensortype == Stereo )
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
            if (ImPath.at(ImPath.length() - 1) != '/')
                ImPath = ImPath + "/";

            if (dataset == Dataset::Tum_mono)
                getdir(ImPath, filesL);
            else if (dataset == Dataset::Euroc)
            {
                getdir(ImPath + "mav0/cam0/data/", filesL);
                if(Sensortype == Stereo)
                    getdir(ImPath + "mav0/cam1/data/", filesR);
            }
            else if (dataset == Dataset::Kitti)
            {
                getdir(ImPath + "image_0/", filesL);
                if(Sensortype == Stereo)
                    getdir(ImPath + "image_1/", filesR);
            }
        }

        nImgL = filesL.size(); nImgR = filesR.size(); nImgD = filesD.size();

        if (Sensortype == Stereo)
            if (nImgL == 0 || nImgR == 0 || nImgL != nImgR)
            {
                printf("There is something wrong with the loaded stereo data: didn't load any or Left and Right images does not match!\n");
                exit(-1);
            }
        if (Sensortype == Monocular)
            if (filesL.size() == 0)
            {
                printf("There is something wrong with the images - didn't load any!\n");
                exit(-1);
            }
        loadtimestamps(timestampPath);
    }

    ~DatasetReader() 
    {
        #if HAS_ZIPLIB
            if (ziparchive != 0) zip_close(ziparchive);
            if (databuffer != 0) delete databuffer;
        #endif
        if(tPhoCalibL) delete tPhoCalibL;
        if(tPhoCalibR) delete tPhoCalibR;
    }

    inline void loadtimestamps(std::string path)
    {
        if (dataset == Kitti)
        {
            std::ifstream fTimes;
            std::string strPathTimeFile = path ;
            fTimes.open(strPathTimeFile.c_str());
            if (!fTimes)
            {
                printf("could not find timestamps file at %s - turning off real timestamps!\n",strPathTimeFile.c_str());
                return;
            }
            while (!fTimes.eof())
            {
                std::string s;
                getline(fTimes, s);
                if (!s.empty())
                {
                    std::stringstream ss;
                    ss << s;
                    double t;
                    ss >> t;
                    timestamps.push_back(t);
                }
            }
            fTimes.close();
        }
        else if (dataset == Euroc)
        {
            std::ifstream fTimes;
            std::string strPathTimeFile = path ;
   
            fTimes.open(strPathTimeFile.c_str());
            if (!fTimes)
            {
                printf("could not find timestamps file at %s - turning off real timestamps!\n",strPathTimeFile.c_str());
                return;
            }

            while (!fTimes.eof())
            {
                std::string line;
                char buf[1000];
                fTimes.getline(buf, 1000);

                double stamp;
                char filename[256];
                if (line[0] == '#')
                    continue;

                if (2 == sscanf(buf, "%lf,%s", &stamp, filename))
                    timestamps.push_back(stamp * 1e-9);
            }
            fTimes.close();
            if(timestamps.size() != nImgL)
            { printf("timestamps don't match number of images. disabling timestamps!\n"); timestamps.clear(); return;}
        }
        else if (dataset == Tum_mono)
        {
            std::ifstream tr;
            std::string timesFile = path;
            tr.open(timesFile.c_str());

            while (!tr.eof())
            {
                std::string line;
                char buf[1000];
                tr.getline(buf, 1000);

                int id;
                double stamp;
                float exposure = 0;

                if (3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure))
                {
                    timestamps.push_back(stamp);
                    exposuresL.push_back(exposure);
                }
                else if (2 == sscanf(buf, "%d %lf", &id, &stamp))
                {
                    timestamps.push_back(stamp);
                    exposuresL.push_back(exposure);
                }
            }
            tr.close();

            // check if exposures are correct, (possibly skip)
            bool exposuresGood = ((int)exposuresL.size() == nImgL);
            for (int i = 0; i < (int)exposuresL.size(); i++)
            {
                if (exposuresL[i] == 0)
                {
                    // fix!
                    float sum = 0, num = 0;
                    if (i > 0 && exposuresL[i - 1] > 0)
                    {
                        sum += exposuresL[i - 1];
                        num++;
                    }
                    if (i + 1 < (int)exposuresL.size() && exposuresL[i + 1] > 0)
                    {
                        sum += exposuresL[i + 1];
                        num++;
                    }

                    if (num > 0)
                        exposuresL[i] = sum / num;
                }

                if (exposuresL[i] == 0)
                    exposuresGood = false;
            }

            if (nImgL != (int)timestamps.size())
            {
                printf("set timestamps and exposures to zero!\n");
                exposuresL.clear();
                timestamps.clear();
            }

            if (nImgL != (int)exposuresL.size() || !exposuresGood)
            {
                printf("set EXPOSURES to zero!\n");
                exposuresL.clear();
            }
        }

        printf("got %d images and %d timestamps and %d exposures.!\n", nImgL, (int)timestamps.size(), (int)exposuresL.size());
    }

    inline double getTimestamp(int id) 
    {
        if (timestamps.size() == 0) return id * 0.1f;
        if (id >= (int) timestamps.size()) return 0;
        if (id < 0) return 0;
        return timestamps[id];
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

    inline void getImage(std::shared_ptr<ImageData> ImgData, int id )
    {
        if (!isZipped)
        {
            
            
            ImgData->cvImgL = cv::imread(filesL[id], cv::IMREAD_GRAYSCALE);
            if(ImgData->cvImgL.size().width != WidthOri || ImgData->cvImgL.size().height != HeightOri)
                {printf("Input resolution does not correspond to image read! something might be wrong in your intrinsics file!\n"); exit(1);}
            if(Sensortype == Stereo || Sensortype == RGBD)
            {
                ImgData->cvImgR = cv::imread(filesR[id], cv::IMREAD_GRAYSCALE);
                if(ImgData->cvImgL.size()!=ImgData->cvImgR.size() )
                    {printf("Right and Left image resolutions are not equal! exit!\n"); exit(1);}
            }
                undistort(ImgData);
        }
        else
        {
            #if HAS_ZIPLIB

            long readsize = ReadZipBuffer(filesL[id]);
            ImgData->cvImgL =  cv::imdecode(cv::Mat(readsize,1,CV_8U, databuffer), cv::IMREAD_GRAYSCALE); 
            if(ImgData->cvImgL.size().width != WidthOri || ImgData->cvImgL.size().height != HeightOri)
                {printf("Input resolution does not correspond to image read! something might be wrong in your intrinsics file!\n"); exit(1);}

            if(Sensortype == Stereo || Sensortype == RGBD)
            {
                readsize = ReadZipBuffer(filesR[id]);
                ImgData->cvImgR = cv::imdecode(cv::Mat(readsize,1,CV_8U, databuffer), cv::IMREAD_GRAYSCALE);
                if(ImgData->cvImgL.size()!=ImgData->cvImgR.size() )
                    {printf("Right and Left image resolutions are not equal! exit!\n"); exit(1);}
            }

            undistort(ImgData);
        
            #else
                printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
                exit(1);
            #endif
        }
        
        ImgData->timestamp = getTimestamp(id);
        ImgData->ExposureL =  exposuresL.size() == 0 ? 1.0f : exposuresL[id];
        ImgData->ExposureR = exposuresR.size() == 0 ? 1.0f : exposuresR[id];

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

    inline void undistort(std::shared_ptr<ImageData> ImgData)
    {
        PhoUndistL->undistort(ImgData->cvImgL, tPhoCalibL);
        if (PhoUndistR)
            PhoUndistR->undistort(ImgData->cvImgR, tPhoCalibR);

        GeomUndist->undistort(ImgData, tPhoCalibL, tPhoCalibR);

        return;
    }
};

#endif