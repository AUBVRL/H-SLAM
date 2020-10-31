#ifndef __DATASETLOADER_H
#define __DATASETLOADER_H
#pragma once

#include <fstream>
#include <dirent.h>
#include <opencv2/imgcodecs.hpp>

#include "geomUndistorter.h"
#include "photoUndistorter.h"
#include "globalTypes.h"

#if HAS_ZIPLIB
#include "zip.h"
#endif

namespace SLAM
{
class datasetReader
{
public:
    
    vector<string> files;
    vector<double> timestamps;
    vector<float> exposures;

    int nImg;

    bool isZipped;
    Dataset dataset;

    shared_ptr<geomUndistorter> gUndist;
    shared_ptr<photoUndistorter> pUndist;

#if HAS_ZIPLIB
    zip_t *ziparchive;
    std::vector<char> databuffer;
#endif

    datasetReader(shared_ptr<photoUndistorter> phoUndist, shared_ptr<geomUndistorter> geomUndist, string imPath,
                  string stimestamp, string strdataset): 
    gUndist(geomUndist), pUndist(phoUndist)
    {
        if(strdataset == "Euroc")
            dataset = Dataset::Euroc;
        else if (strdataset == "TumMono")
            dataset = Dataset::Tum_mono;
        else if (strdataset == "TartanAir")
            dataset = Dataset::TartanAir;
        else if(strdataset == "Kitti")
            dataset = Dataset::Kitti;
        else if (strdataset == "Live")
            dataset = Dataset::Live;
        else
        {
            printf("Wrong dataset name specified! Exit\n");
            exit(1);
        }

        #if HAS_ZIPLIB
            ziparchive = 0;
        #endif
        isZipped = (imPath.length() > 4 && imPath.substr(imPath.length() - 4) == ".zip");

        if (isZipped)
        {
            #if HAS_ZIPLIB
                int ziperror = 0;
                ziparchive = zip_open(imPath.c_str(), ZIP_RDONLY, &ziperror);
                if (ziperror != 0)
                    throw runtime_error("ERROR " + to_string(ziperror) + " reading archive " + imPath.c_str() + " ! exit\n." );

                files.clear();
                int numEntries;
                if (dataset == Dataset::Tum_mono)
                {
                    numEntries = zip_get_num_entries(ziparchive, 0);
                    for (int k = 0; k < numEntries; ++k)
                    {
                        const char *name = zip_get_name(ziparchive, k, ZIP_FL_ENC_STRICT);
                        string nstr = string(name);
                        if (nstr == "." || nstr == "..")
                            continue;
                        files.push_back(name);
                    }
                    sort(files.begin(), files.end());
                }
                else if (dataset == Dataset::Euroc)
                {
                    numEntries = zip_get_num_entries(ziparchive, 0);
                    string Dir = "mav0/cam0/data/";
                    for (int k = 0; k < numEntries; ++k)
                    {
                        const char *name = zip_get_name(ziparchive, k, ZIP_FL_ENC_STRICT);
                        string nstr = std::string(name);
                        if (nstr == "." || nstr == ".." || nstr == Dir)
                            continue;

                        if (Dir.compare(0, 15, nstr, 0, 15) == 0)
                            files.push_back(nstr);
                    }
                    sort(files.begin(), files.end());
                }
                else
                    throw std::runtime_error("Zip for the current dataset is not supported\n" );
            #else 
                throw std::runtime_error("ERROR: cannot read .zip archive without ziplib!\n");
            #endif
        }
        else
        {
            if (imPath.at(imPath.length() - 1) != '/')
                imPath = imPath + "/";

            if (dataset == Dataset::Tum_mono)
                getdir(imPath, files);
            else if (dataset == Dataset::TartanAir)
                getdir(imPath, files);
            else if (dataset == Dataset::Euroc)
                getdir(imPath + "mav0/cam0/data/", files);
            
            else if (dataset == Dataset::Kitti)
                getdir(imPath + "image_0/", files);
            else if (dataset == Dataset::Live)
                {
                    printf("Live mode not yet implemented! exit\n"); exit(1);
                }
        }

        nImg = files.size();

        if (files.size() == 0)
        {
            cout<<"There is something wrong with the images - didn't load any!"<<endl;
            exit(1);
        }
        loadtimestamps(stimestamp);
    }

    ~datasetReader() 
    {
        #if HAS_ZIPLIB
            if (ziparchive != 0) zip_close(ziparchive);
        #endif
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
            exposures.clear();
        }
        else if (dataset == TartanAir)
        {
            printf("TartanAir Dataset does not provide TimeStamps! Turning them off.\n");
            timestamps.clear();
            exposures.clear();
            return;
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
                else if (1 == sscanf(buf, "%lf", &stamp))
                    timestamps.push_back(stamp * 1e-9);
            }
            fTimes.close();
            exposures.clear();
            if(timestamps.size() != nImg)
            { 
                printf("timestamps don't match number of images. disabling timestamps!\n"); 
                timestamps.clear(); 
                return;
            } 
        }
        else if (dataset == Tum_mono)
        {
            std::ifstream tr;
            std::string timesFile = path;
            tr.open(timesFile.c_str());
            if (!tr)
            {
                printf("could not find timestamps file at %s - turning off real timestamps!\n",timesFile.c_str());
                return;
            }
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
                    exposures.push_back(exposure);
                }
                else if (2 == sscanf(buf, "%d %lf", &id, &stamp))
                {
                    timestamps.push_back(stamp);
                    exposures.push_back(exposure);
                }
            }
            tr.close();

            // check if exposures are correct, (possibly skip)
            bool exposuresGood = ((int)exposures.size() == nImg);
            for (int i = 0; i < (int)exposures.size(); ++i)
            {
                if (exposures[i] == 0)
                {
                  
                    float sum = 0, num = 0;
                    if (i > 0 && exposures[i - 1] > 0)
                    {
                        sum += exposures[i - 1];
                        num++;
                    }
                    if (i + 1 < (int)exposures.size() && exposures[i + 1] > 0)
                    {
                        sum += exposures[i + 1];
                        num++;
                    }

                    if (num > 0)
                        exposures[i] = sum / num;
                }

                if (exposures[i] == 0)
                    exposuresGood = false;
            }

            if (nImg != (int)timestamps.size())
            {
                printf("set timestamps and exposures to zero!\n");
                exposures.clear();
                timestamps.clear();
            }

            if (nImg != (int)exposures.size() || !exposuresGood)
            {
                printf("set EXPOSURES to zero!\n");
                exposures.clear();
            }
        }

        printf("got %d images and %d timestamps and %d exposures.!\n", nImg, (int)timestamps.size(), (int)exposures.size());
    }

    inline double getTimestamp(int id) 
    {
        if (timestamps.size() == 0) return id * 0.05f; //if no timestamps assume capture rate of 20fps
        if (id >= (int) timestamps.size()) return 0;
        if (id < 0) return 0;
        return timestamps[id];
    }

    inline int getdir(std::string dir, std::vector<std::string> &files)
    {
        DIR *dp;
        struct dirent *dirp;
        if ((dp = opendir(dir.c_str())) == NULL)
            throw std::runtime_error("FAILED to open path " + dir);

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
        const unsigned int filescount = files.size();
        for (unsigned int i = 0; i < filescount; ++i)
        {
            if (files[i].at(0) != '/')
                files[i] = dir + files[i];
        }

        return files.size();
    }

    void readNonZippedImage(shared_ptr<ImageData> ImgData, int id)
    {

        ImgData->cvImg = cv::imread(files[id], cv::IMREAD_GRAYSCALE);
        if (ImgData->cvImg.size().width != WidthOri || ImgData->cvImg.size().height != HeightOri)
            throw runtime_error("Input resolution does not correspond to image read! something might be wrong in your intrinsics file!\n");
        undistort(ImgData);
        
    }

    void readZippedImage(shared_ptr<ImageData> ImgData, int id)
    {
        long readsize = ReadZipBuffer(files[id], databuffer);
        ImgData->cvImg =  cv::imdecode(cv::Mat(readsize,1,CV_8U, &databuffer[0]), cv::IMREAD_GRAYSCALE); 
        if (ImgData->cvImg.size().width != WidthOri || ImgData->cvImg.size().height != HeightOri)
            throw runtime_error("Input resolution does not correspond to image read! something might be wrong in your intrinsics file! exit.\n");
        undistort(ImgData);
        
    }

    inline void getImage(shared_ptr<ImageData> ImgData, int id )
    {
        ImgData->timestamp = getTimestamp(id);
        ImgData->Exposure =  exposures.size() == 0 ? 1.0f : exposures[id];

        if (!isZipped)
            readNonZippedImage(ImgData, id);
        else
        {
            #if HAS_ZIPLIB
                readZippedImage(ImgData, id);
            #else
                throw runtime_error("ERROR: cannot read .zip archive, as compile without ziplib!\n");
            #endif
        }

        return;
    }

    inline long ReadZipBuffer(string In_, vector<char>& _databuffer)
    {
        static long int imageSizeBuf = WidthOri * HeightOri * 6 + 1000;
        if (_databuffer.empty()) _databuffer.resize(imageSizeBuf); //; = new char[WidthOri * HeightOri * 6 + 10000];
            zip_file_t *fle = zip_fopen(ziparchive, In_.c_str(), 0);
            if (!fle)
                throw runtime_error("Dataset file could not be read");
            long readbytes = zip_fread(fle, &_databuffer[0], _databuffer.size());
            if (readbytes > imageSizeBuf) 
            {
                printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes, imageSizeBuf, In_.c_str());
                imageSizeBuf = (long) WidthOri * HeightOri * 30 + 10000;
                _databuffer.resize(imageSizeBuf);
                fle = zip_fopen(ziparchive, In_.c_str(), 0);
                readbytes = zip_fread(fle, &_databuffer[0], imageSizeBuf);
               
                if (readbytes > (imageSizeBuf-10000))
                    throw runtime_error("buffer still to small (read" + to_string(readbytes) + " in " + 
                                        to_string(imageSizeBuf-10000) + "abort.\n");
            }
            zip_fclose(fle);
            if(readbytes == -1)
                throw runtime_error("zip_fread failed to read dataset file");
            return readbytes;
    }

    inline void undistort(shared_ptr<ImageData> ImgData)
    {
        cv::Mat Output = cv::Mat(gUndist->hOrg, gUndist->wOrg, CV_32F);
        pUndist->undistort(ImgData->cvImg, Output, 1.0f);
        gUndist->undistort(Output);
        memcpy(&ImgData->fImg[0], Output.data, sizeof(float)*gUndist->w * gUndist->h);

        //convert to CV_8U and remove gamma correction for opencv version of the image (better for feature matching)
        Output.convertTo(ImgData->cvImg, CV_8U);
        int dim = Output.cols * Output.rows;
        uchar *cvPtr = ImgData->cvImg.ptr<uchar>(0);
        if (pUndist->GammaValid)
            for (int i = 0; i < dim; ++i)
                cvPtr[i] = pUndist->getB(cvPtr[i]);

        return;
    }
};
}

#endif
