#ifndef __MAIN__
#define __MAIN__
#include <sstream>
#include <string>

#include "DatasetLoader.h"
#include "GlobalTypes.h"

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

Dataset dataset_= Emptyd;
Sensor sensor_= Emptys;
PhotoUnDistMode phoDistMode_ = Emptyp;
std::string IntrinCalib = "";
std::string Gamma = "";
std::string Vignette = "";
std::string Vocabulary = "";
std::string Path = "";

bool Reverse = false;
bool Nogui = false;
bool Prefetch = false;
float PlaybackSpeed = 0;
int Start = 0;
int End = 9999999;
int Mode = 0;
int linc = 1;

void parseargument(char * arg)
{
    int option;
	float foption;
	char buf[1000];
    
    if(1==sscanf(arg,"path=%s",buf))
	{
		Path = buf;
		printf("loading data from %s!\n", Path.c_str());
		return;
	}
    else if(1==sscanf(arg,"intrinsics=%s",buf))
	{
		IntrinCalib = buf;
		printf("loading Intrinsics from %s!\n", IntrinCalib.c_str());
		return;
	}
    else if(1==sscanf(arg,"gamma=%s",buf))
	{
		Gamma = buf;
		printf("loading Gamma from %s!\n", Gamma.c_str());
		return;
	}
    else if(1==sscanf(arg,"vignette=%s",buf))
	{
		Vignette = buf;
		printf("loading Gamma from %s!\n", Vignette.c_str());
		return;
	}
    else if(1==sscanf(arg,"vocabulary=%s",buf))
	{
		Vocabulary = buf;
		printf("loading Vocabulary from %s!\n", Vocabulary.c_str());
		return;
	}
    else if(1==sscanf(arg,"start=%d",&option))
    {
        Start = option;
        printf("START AT %d!\n",Start);
		return;
    }
    else if(1==sscanf(arg,"end=%d",&option))
    {
        End = option;
        printf("End AT %d!\n",End);
		return;
    }
    else if(1==sscanf(arg,"reverse=%d",&option))
	{
		if(option==1)
		{
			Reverse = true;
            linc = -1;
			printf("REVERSE!\n");
		}
		return;
	}
    else if(1==sscanf(arg,"nogui=%d",&option))
	{
		if(option==1)
		{
			Nogui = true;
			printf("No GUI!\n");
		}
		return;
	}
    else if(1==sscanf(arg,"prefetch=%d",&option))
	{
		if(option==1)
		{
			Prefetch = true;
			printf("Preload images!\n");
		}
		return;
	}
    else if(1==sscanf(arg,"playbackspeed=%f",&foption))
	{
		PlaybackSpeed = foption;
		printf("playback speed %f!\n",PlaybackSpeed);
		return;
	}
    else if (1==sscanf(arg,"dataset=%s",buf))
    {
        std::string data = buf;
        if(data == "Euroc")
            dataset_ = Euroc;
        else if(data == "TumMono")
            dataset_ = Tum_mono;
        else if (data == "Kitti")
            dataset_ = Kitti;
    }
    else if (1==sscanf(arg,"sensor=%s",buf))
    {
        std::string data = buf;
        if(data == "Monocular")
            sensor_ = Monocular;
        else if (data == "Stereo")
            sensor_ = Stereo;
        else if (data == "RGBD")
            sensor_= RGBD;
    }
    else if (1==sscanf(arg,"photoCalibmodel=%s",buf))
    {
        std::string data = buf;
        if(data == "HaveCalib")
            phoDistMode_ = HaveCalib;
        else if (data == "OnlineCalib")
            phoDistMode_ = OnlineCalib;
        else if (data == "NoCalib")
            phoDistMode_= NoCalib;
    }
}

void ValidateInput()
{
    if(dataset_ == Emptyd)
    {
        printf("you did not specify a dataset\n");
        exit(1);
    }
    if(sensor_ == Emptys)
    {
        printf("you did not specify a sensor type\n");
        exit(1);
    }
    if(Path=="")
    {
        printf("you did not specify an image path\n");
        exit(1);
    }
    if(IntrinCalib=="")
    {
        printf("you did not specify an Intrinsics calib path\n");
        exit(1);
    }
    if(Vocabulary=="")
    {
        printf("you did not specify a Vocabulary path\n");
        exit(1);
    }
    if(phoDistMode_== Emptyp)
    {
        printf("you did not specify a photometric distortion mode\n");
        exit(1);
    }
    if(Gamma=="" || Vignette == "" && phoDistMode_ == HaveCalib )
    {
        phoDistMode_= NoCalib;
        printf("Turning off photometric undistortion as the required data is not available\n");

    }
    if(dataset_ == Tum_mono && sensor_ != Monocular)
    {
        printf("Tum_mono is a monocular dataset only\n");
        exit(1);
    }
    if( (dataset_ == Tum_mono || dataset_ == Euroc || dataset_ == Kitti) && (sensor_ == RGBD))
    {
        printf("the dataset used does not contain RGBD data\n");
        exit(1);
    }    
}


#endif