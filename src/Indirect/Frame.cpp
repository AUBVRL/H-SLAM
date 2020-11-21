#include "Frame.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"

#include "opencv2/highgui.hpp"

namespace HSLAM
{

    using namespace std;

    Frame::Frame(float *Img, CalibHessian *_HCalib, FrameHessian *_fh, FrameShell *_fs) : nFeatures(0), isReduced(false), NeedRefresh(NeedRefresh), HCalib(_HCalib), fh(_fh), fs(_fs)
    {
        cv::Mat(hG[0], wG[0], CV_32FC1, Img).convertTo(Image, CV_8U);

        Extract();
        // CreateIndPyrs(Img->cvImgL, IndPyr);
        //for now I'm only extracting features from highest resolution image!!

    }
    Frame::~Frame()
    {
        Image.release();
        releaseVec(mvKeys);
        releaseVec(mGrid);
        Descriptors.release();
    };

    void Frame::ReduceToEssential()
    {
        if (isReduced)
            return;
        isReduced = true;
        {
            Image.release();
            NeedRefresh = true;
        }
        return;
    }

    void Frame::assignFeaturesToGrid()
    {
 
        mGrid.resize(mnGridCols * mnGridRows);
        for (unsigned short i = 0; i < nFeatures; ++i)
        {
            // assign feature to grid
            int gridX = mvKeys[i].pt.x / gridSize;
            int gridY = mvKeys[i].pt.y / gridSize;
            mGrid[gridY * mnGridCols + gridX].push_back(i);
        }

        return;
    }

    bool Frame::PosInGrid(const keypoint &kp, int &posX, int &posY)
    {
        posX = std::round((kp.pt.x) * mfGridElementWidthInv);
        posY = std::round((kp.pt.y) * mfGridElementHeightInv);
        if (posX < 0 || posX >= mnGridCols || posY < 0 || posY >= mnGridRows)
            return false;
        return true;
    }
    
    // void Frame::Extract(shared_ptr<FeatureDetector> _Detector, int id, bool ForInit, shared_ptr<IndexThreadReduce<Vec10>> ThreadPool)
    void Frame::Extract()
    {
        // _Detector->ExtractFeatures(Image, featType, DirPyr, id, absSquaredGrad, mvKeys,
        //                            Descriptors, nFeatures, (ForInit ? featuresToExtract : featuresToExtract), ThreadPool);
        // for (int i = 0; i < 2000; ++i)
        //     mvKeys.push_back(keypoint(320, 320));
            //Assign Features to Grid

        
        // assignFeaturesToGrid();
        


        // ComputeBoVW();
    }

    std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
    {
        std::vector<size_t> vIndices;
        // vIndices.reserve(nFeatures);

        const int nMinCellX = std::max(0, (int)std::floor((x - mnMinX - r) * mfGridElementWidthInv));
        if (nMinCellX >= mnGridCols)
            return vIndices;

        const int nMaxCellX = std::min((int)mnGridCols - 1, (int)std::ceil((x - mnMinX + r) * mfGridElementWidthInv));
        if (nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = std::max(0, (int)std::floor((y - mnMinY - r) * mfGridElementHeightInv));
        if (nMinCellY >= mnGridRows)
            return vIndices;

        const int nMaxCellY = std::min((int)mnGridRows - 1, (int)std::ceil((y - mnMinY + r) * mfGridElementHeightInv));
        if (nMaxCellY < 0)
            return vIndices;

        for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
        {
            for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
            {
                const std::vector<unsigned short int> vCell = mGrid[iy * mnGridCols + ix];
                for (size_t j = 0, jend = vCell.size(); j < jend; ++j)
                {
                    const keypoint &kpUn = mvKeys[vCell[j]];
                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    if (fabs(distx) < r && fabs(disty) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    void Frame::ComputeBoVW()
    {
        if (mBowVec.empty())
        {
            vector<cv::Mat> vDesc;
            vDesc.reserve(Descriptors.rows);
            for (int j = 0; j < Descriptors.rows; j++)
                vDesc.push_back(Descriptors.row(j));
            Vocab.transform(vDesc, mBowVec, mFeatVec, 4);
        }
    }


} // namespace HSLAM
