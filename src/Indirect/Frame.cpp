#include "Frame.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include "Indirect/Detector.h"
#include "Indirect/MapPoint.h"
#include "opencv2/highgui.hpp"

namespace HSLAM
{

    using namespace std;

    Frame::Frame(float *Img, shared_ptr<FeatureDetector> detector, CalibHessian *_HCalib, FrameHessian *_fh, FrameShell *_fs) : nFeatures(0), isReduced(false), NeedRefresh(NeedRefresh), HCalib(_HCalib), fh(_fh), fs(_fs)
    {
        cv::Mat(hG[0], wG[0], CV_32FC1, Img).convertTo(Image, CV_8U);
        Occupancy = cv::Mat(hG[0], wG[0], CV_8U, cv::Scalar(0));
        detector->ExtractFeatures(Image, Occupancy, mvKeys, Descriptors, nFeatures, indFeaturesToExtract);
        assignFeaturesToGrid();
        mvpMapPoints.resize(nFeatures, nullptr);
        // ComputeBoVW();
    }
    Frame::~Frame()
    {
        Image.release();
        Occupancy.release();
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
            Occupancy.release();
            // NeedRefresh = true;
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


        void Frame::addMapPoint(std::shared_ptr<MapPoint>& Mp )
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            mvpMapPoints[Mp->index] = Mp;
            return;
        }

        void Frame::addMapPointMatch(std::shared_ptr<MapPoint> Mp, size_t index )
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            mvpMapPoints[index] = Mp;
            return;
        }

    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
    {
        posX = std::round((kp.pt.x) * mfGridElementWidthInv);
        posY = std::round((kp.pt.y) * mfGridElementHeightInv);
        if (posX < 0 || posX >= mnGridCols || posY < 0 || posY >= mnGridRows)
            return false;
        return true;
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
                    const cv::KeyPoint &kpUn = mvKeys[vCell[j]];
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
            for (int j = 0; j < Descriptors.rows; ++j)
                vDesc.push_back(Descriptors.row(j));
            Vocab.transform(vDesc, mBowVec, mFeatVec, 4);
        }
    }


} // namespace HSLAM
