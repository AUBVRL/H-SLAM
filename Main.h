#ifndef __MAIN__
#define __MAIN__
#include <sstream>
#include <string>

#include "GlobalTypes.h"
#include "Settings.h"

#ifdef MSVC
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 
#include <time.h>

typedef struct timeval {
    long tv_sec;
    long tv_usec;
} timeval;

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
 
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
    return 0;
}
#else
#include <sys/time.h>
#endif

namespace FSLAM
{
class Input
{
public:
    Input(int argc, char** In_): 
            dataset_(Emptyd), IntrinCalib(""), GammaL(""), GammaR(""), timestampsL(""),
            VignetteL(""), VignetteR(""), Vocabulary(""), Path(""), Reverse(false), Nogui(false), 
            Prefetch(false), PlaybackSpeed(0), Start(0), End(9999999), Mode(0), linc(1)
           
    {
        for (int i = 1; i < argc; i++)
            parseargument(In_[i]);
        ValidateInput();

    }
    Dataset dataset_;
    std::string IntrinCalib;
    std::string GammaL;
    std::string GammaR;
    std::string timestampsL;
    std::string VignetteL;
    std::string VignetteR;
    std::string Vocabulary;
    std::string Path;
    bool Reverse;
    bool Nogui;
    bool Prefetch;
    float PlaybackSpeed;
    int Start;
    int End;
    int Mode; //not used for now!
    int linc;

    inline void ValidateInput()
    {
    //Mandatory Input
    if(dataset_ == Emptyd) throw std::runtime_error("you did not specify a dataset\n");
    if(Sensortype == Emptys) throw std::runtime_error("you did not specify a sensor type\n");
    if(PhoUndistMode== Emptyp) throw std::runtime_error("you did not specify a photometric distortion mode\n");
    if(IntrinCalib=="") throw std::runtime_error("you did not specify an Intrinsics calib path\n");
    if(Vocabulary=="") throw std::runtime_error("you did not specify a Vocabulary path\n");
    if(Path=="") throw std::runtime_error("you did not specify an image path\n");
    if(PhoUndistMode== PhotoUnDistMode::HaveCalib)
    {
        if(GammaL == "" || VignetteL =="")
            throw std::runtime_error("Gamma correction or vignette image for the left image not provided!exit.\n");
        if(Sensortype == Sensor::Stereo && (GammaR == "" || VignetteR ==""))
            throw std::runtime_error("photometric calibration for right image is not provided!exit.\n");
    }
    if(dataset_ == Tum_mono && Sensortype != Monocular) throw std::runtime_error("Tum_mono is a monocular dataset only\n");
    if(dataset_ == Euroc && Sensortype != Monocular && Sensortype != Stereo)
        throw std::runtime_error("invalid sensor type for the dataset chosen!\n");
    if( (dataset_ == Kitti) && (Sensortype != Monocular) && (Sensortype != Stereo) )
        throw std::runtime_error("invalid sensor type for the dataset chosen!\n");
    }

    void parseargument(char *arg)
    {
        int option;
        float foption;
        char buf[1000];

        if (1 == sscanf(arg, "path=%s", buf))
        {
            Path = buf;
            printf("loading data from %s!\n", Path.c_str());
            return;
        }
        else if (1 == sscanf(arg, "intrinsics=%s", buf))
        {
            IntrinCalib = buf;
            if (IntrinCalib.length() > 4 && IntrinCalib.substr(IntrinCalib.length() - 5) != ".yaml")
                throw std::runtime_error("Intrinsics have to be a .yaml file!\n");
            printf("loading Intrinsics from %s!\n", IntrinCalib.c_str());
            return;
        }
        else if (1 == sscanf(arg, "timestampsL=%s", buf))
        {
            timestampsL = buf;
            printf("loading left camera timestamps from %s!\n", timestampsL.c_str());
            return;
        }
        else if (1 == sscanf(arg, "gammaL=%s", buf))
        {
            GammaL = buf;
            printf("loading Gamma Left from %s!\n", GammaL.c_str());
            return;
        }
        else if (1 == sscanf(arg, "gammaR=%s", buf))
        {
            GammaR = buf;
            printf("loading Gamma Right from %s!\n", GammaR.c_str());
            return;
        }
        else if (1 == sscanf(arg, "vignetteL=%s", buf))
        {
            VignetteL = buf;
            printf("loading vignette left from %s!\n", VignetteL.c_str());
            return;
        }
        else if (1 == sscanf(arg, "vignetteR=%s", buf))
        {
            VignetteR = buf;
            printf("loading vignette right from %s!\n", VignetteR.c_str());
            return;
        }
        else if (1 == sscanf(arg, "vocabulary=%s", buf))
        {
            Vocabulary = buf;
            printf("loading Vocabulary from %s!\n", Vocabulary.c_str());
            return;
        }
        else if (1 == sscanf(arg, "start=%d", &option))
        {
            Start = option;
            printf("START AT %d!\n", Start);
            return;
        }
        else if (1 == sscanf(arg, "end=%d", &option))
        {
            End = option;
            printf("End AT %d!\n", End);
            return;
        }
        else if (1 == sscanf(arg, "reverse=%d", &option))
        {
            if (option == 1)
            {
                Reverse = true;
                linc = -1;
                printf("REVERSE!\n");
            }
            return;
        }
        else if (1 == sscanf(arg, "nogui=%d", &option))
        {
            if (option == 1)
            {
                Nogui = true;
                printf("No GUI!\n");
            }
            return;
        }
        else if (1 == sscanf(arg, "prefetch=%d", &option))
        {
            if (option == 1)
            {
                Prefetch = true;
                printf("Preload images!\n");
            }
            return;
        }
        else if (1 == sscanf(arg, "playbackspeed=%f", &foption))
        {
            PlaybackSpeed = foption;
            printf("playback speed %f!\n", PlaybackSpeed);
            return;
        }
        else if (1 == sscanf(arg, "dataset=%s", buf))
        {
            std::string data = buf;
            if (data == "Euroc")
                dataset_ = Euroc;
            else if (data == "TumMono")
                dataset_ = Tum_mono;
            else if (data == "Kitti")
                dataset_ = Kitti;
            return;
        }
        else if (1 == sscanf(arg, "sensor=%s", buf))
        {
            std::string data = buf;
            if (data == "Monocular")
                Sensortype = Monocular;
            else if (data == "Stereo")
                Sensortype = Stereo;
            else if (data == "RGBD")
                Sensortype = RGBD;
            return;
        }
        else if (1 == sscanf(arg, "photomCalibmodel=%s", buf))
        {
            std::string data = buf;
            if (data == "HaveCalib")
                PhoUndistMode = HaveCalib;
            else if (data == "OnlineCalib")
                PhoUndistMode = OnlineCalib;
            else if (data == "NoCalib")
                PhoUndistMode = NoCalib;
            return;
        }
        else if (1 == sscanf(arg, "PyrLevels=%i", &option))
        {
            if(option >= 1 && option <= 10)
            {
                PyrLevels = option;
                printf("Using %i pyramid levels\n", PyrLevels);
            }
            else
                printf("PyrLevel chosen is invalid, using default %i levels\n", PyrLevels);

            return;
        }
        else if (1 == sscanf(arg, "PyrScaleFactor=%f", &foption))
        {
            if (foption >= 1.0f && foption <= 4.0f)
            {
                PyrScaleFactor = foption;
                printf("Using %f as a scale factor\n", PyrScaleFactor);
            }
            else
                printf("Scale factor chosen is invalid, using default %f\n", PyrScaleFactor);

            return;
        }
        else if (1 == sscanf(arg, "numFeatures=%i", &option))
        {
            if (option >= 500 && option <= 5000)
            {
                numFeatures = option;
                printf("Extracting %i features\n", numFeatures);
            }
            else
                printf("Number of features chosen is invalid, using default %i features\n", numFeatures);

            return;
        }
    }
};
}
#endif