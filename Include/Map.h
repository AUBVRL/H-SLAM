#include "Settings.h"

using namespace std;
namespace FSLAM
{
class Frame;

class Map
{

public:
    Map();
    ~Map(){};
    set<shared_ptr<Frame>> KeyFrames;
    std::vector<std::shared_ptr<Frame>> LocalMap;

};


}