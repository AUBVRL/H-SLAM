#include <iostream>
#include "Settings.h"
#include "datasetReader.h"
#include "System.h"
#include "Display.h"
#include "DBoW3/Vocabulary.h"
#include <unistd.h>
#include <chrono>
#include <csignal>

static bool exitSignal = false;

void signal_callback_handler(int signum) {
    exitSignal = true;
}

using namespace cv;
using namespace SLAM;

const cv::String keys =
    "{help h usage ?|      | print this message}"
    "{imagePath     |<none>| Input images path}"
    "{intrinsics    |<none>| Camera intrinsic callibration}"
    "{vocabPath     |<none>| Vocabulary path}"
    "{datasetName   |<none>| Dataset name: Tum_mono, Euroc, Kitti, TartanAir, Live }"
    "{timeStamps    |      | Path to timestamps}"
    "{vignetteModel |      | Path to Vignette model}"
    "{gammaModel    |      | Path to gamma response Model}"
    "{startIndex    | 0    | Image index to start from}"
    "{endIndex      |999999| Last image to be processed }"
    "{onlPhotoCalib | False| Perform online photometric calibration}"
    "{quiet         | True | disable message printing }"
    "{disableGUI    | False| disable GUI}"
    "{onPhotoCalib  | False| Enable online photometric calibration}"
    "{playbackSpeed | 0.0  | Enforce playback Speed to real-time}"
    "{pyramidSize   | 6    | Number of Pyramid levels used}";

int main(int argc, char ** argv)
{
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help") || parser.has("h") || parser.has("?")) {parser.printMessage(); return 0;}
    if(!parser.has("imagePath") || !parser.has("intrinsics") || !parser.has("vocabPath") || !parser.has("datasetName"))
    { 
        printf("One of the following inputs are required but were not provided:\n imagePath, intrinsics, vocabPath, datasetName");
        return (0);
    }
    signal(SIGINT, signal_callback_handler);

    string imagePath = parser.get<string>("imagePath");
    string intrinsics = parser.get<string>("intrinsics");
    string vocabPath = parser.get<string>("vocabPath");
    string datasetName = parser.get<string>("datasetName");
    string timeStamps = parser.get<string>("timeStamps");
    string vignetteModel = parser.get<string>("vignetteModel");
    string gammaModel = parser.get<string>("gammaModel");
    int startIndex = parser.get<int>("startIndex");
    int endIndex = parser.get<int>("endIndex");
    bool onlPhotoCalib = parser.get<bool>("onlPhotoCalib");
    bool quiet = parser.get<bool>("quiet");
    DisplayOn = !parser.get<bool>("disableGUI");
    bool onPhotoCalib = parser.get<bool>("onPhotoCalib");
    float playbackSpeed = parser.get<float>("playbackSpeed");
    pyramidSize = parser.get<int>("pyramidSize");
    
    if (!parser.check()) {parser.printErrors(); return 0; }

    shared_ptr<GUI> DisplayHandler;
    if(DisplayOn)
        DisplayHandler = make_shared<GUI>();
    
    Vocab.load(vocabPath);
    if(Vocab.empty()) {printf("failed to load vocabulary! Exit\n"); exit(1);}

    shared_ptr<geomUndistorter> gUndist = shared_ptr<geomUndistorter>(new geomUndistorter(intrinsics));
    shared_ptr<photoUndistorter> pUndist = shared_ptr<photoUndistorter>(new photoUndistorter(gammaModel, vignetteModel, onPhotoCalib));
    datasetReader dataReader = datasetReader(pUndist, gUndist, imagePath, timeStamps, datasetName);
    shared_ptr<System> slamSystem = shared_ptr<System>(new System(gUndist, pUndist, DisplayHandler));

    //if dataset
    vector<pair<int, double>> timesToPlayAt;
    for (int i = startIndex; i < dataReader.nImg && i < endIndex; ++i)
    {
        if(timesToPlayAt.size() == 0)
            timesToPlayAt.push_back( make_pair(i , 0.0));
        else
        {
            double tsThis = dataReader.getTimestamp(i);
            double tsPrev = dataReader.getTimestamp(i-1);
            timesToPlayAt.push_back( make_pair(i , timesToPlayAt.back().second +  fabs(tsThis-tsPrev) / playbackSpeed));        
        }
    }

    struct timeval tv_start; gettimeofday(&tv_start, NULL); clock_t started = clock(); double sInitializerOffset = 0.0f;

    for (int i = 0, size = timesToPlayAt.size(); i < size; ++i)
    {
        if( exitSignal || ( DisplayHandler != nullptr && DisplayHandler->isDead)  )
        {
            if(DisplayHandler != nullptr)
                DisplayHandler->isDead = true;
            break;
        }
            
        while(Pause)
            usleep(5000);

        if (!slamSystem->isInitialized) // if not initialized: reset start time.
        {
            gettimeofday(&tv_start, NULL);
            started = clock();
            sInitializerOffset = timesToPlayAt[i].second;
        }

        bool skipFrame = false;
        if (playbackSpeed != 0.0)
        {
            struct timeval tv_now; gettimeofday(&tv_now, NULL);
            double sSinceStart = sInitializerOffset + ((tv_now.tv_sec - tv_start.tv_sec) + (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));
            if (sSinceStart < timesToPlayAt[i].second)
                usleep((int)((timesToPlayAt[i].second - sSinceStart) * 1000 * 1000)); 
            if (sSinceStart > timesToPlayAt[i].second + 0.5 + 0.1 * (i % 2))
            {
                printf("SKIPFRAME %d (play at %f, now it is %f)!\n", i, timesToPlayAt[i].second, sSinceStart);
                skipFrame = true;
            }
        }
        if (skipFrame)
            continue;

        
        shared_ptr<ImageData> Img = make_shared<ImageData>(WidthOri, HeightOri);
        dataReader.getImage(Img, timesToPlayAt[i].first);

        slamSystem->ProcessFrame(Img);

        if (slamSystem->initFailed || ResetRequest)
        {
            cout << "RESETTING" << endl;
            ResetRequest = false;
            if (DisplayHandler)
                DisplayHandler->Reset();
            slamSystem.reset();
            usleep(10000);
            pUndist->Reset();
            slamSystem = shared_ptr<System>(new System(gUndist, pUndist, DisplayHandler));
        }
    }

    if (DisplayHandler != nullptr) //prevent closing when processing is done
        while (!DisplayHandler->isDead)
            usleep(50000);
    return 0;
}