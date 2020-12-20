#include "Indirect/Optimizer.h"
#include "core/block_solver.h"
#include "core/optimization_algorithm_levenberg.h"
#include "core/robust_kernel_impl.h"
#include "solvers/dense/linear_solver_dense.h"

#include <boost/thread.hpp>
#include "Indirect/Frame.h"
#include "util/FrameShell.h"
#include "Indirect/MapPoint.h"


namespace HSLAM
{
    using namespace std;
    using namespace OptimizationStructs;

    int PoseOptimization(std::shared_ptr<Frame> pFrame, CalibHessian *calib)
    {
        g2o::SparseOptimizer optimizer;
        auto linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));

        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // Set Frame vertex
        vertexSE3 *vSE3 = new vertexSE3(); //VertexSE3Expmap Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setEstimate(pFrame->fs->getPoseInverse());
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);
        // Set MapPoint vertices
        const int N = pFrame->nFeatures;

        vector<edgeSE3XYZPoseOnly *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);
        vector<double> vInformation;
        vInformation.reserve(N);
        double maxInfo = 1e7;
        double stdDev=1e7;

        const float deltaMono = sqrt(5.991);
        {
            // boost::unique_lock<boost::mutex> lock(MapPoint::mGlobalMutex); //this would lock ALL map points poses from changing!

            for (int i = 0; i < N; i++)
            {
                std::shared_ptr<MapPoint> pMP = pFrame->tMapPoints[i];
                if (pMP)
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    edgeSE3XYZPoseOnly *e = new edgeSE3XYZPoseOnly();
                    e->setCamera(calib->fxl(), calib->fyl(), calib->cxl(), calib->cyl());
                    e->setXYZ(pMP->getWorldPose().cast<double>());
                    e->setMeasurement(Vec2((double)pFrame->mvKeys[i].pt.x, (double)pFrame->mvKeys[i].pt.y));
                    e->setId(i);
                    
                    e->setVertex(0, vSE3);
                    // e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    //const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];

                    double Info = (1.0/double(pMP->getVariance()));
                    vInformation.push_back(Info);
                    if (Info > maxInfo)
                        maxInfo = Info;

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }

            //compute information vector distribution:
            if (normalizeInfoWithVariance)
                stdDev = getStdDev(vInformation);
            
            for (int i = 0, iend = vpEdgesMono.size(); i < iend; ++i)
            {
                if(normalizeInfoWithVariance)
                    vpEdgesMono[i]->setInformation(Eigen::Matrix2d::Identity()* (vInformation[i]/(stdDev+0.001))); //* invSigma2); //set this to take into account depth variance!
                else //normalizing by the maximum
                    vpEdgesMono[i]->setInformation(Eigen::Matrix2d::Identity()* (vInformation[i]/(maxInfo+0.001))); //* invSigma2); //set this to take into account depth variance!
            }
        }
        if (nInitialCorrespondences < 3 || optimizer.edges().size() < 10)
            return 0;

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        for (size_t it = 0; it < 4; it++)
        {
            vSE3->setEstimate(pFrame->fs->getPoseInverse());
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);
            nBad = 0;
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
            {
                edgeSE3XYZPoseOnly *e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if (pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }
                

                const float chi2 = e->chi2();
                if (chi2 > chi2Mono[it])
                {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers

        pFrame->fs->setPose(vSE3->estimate().inverse());// SetPose(pose);

        return nInitialCorrespondences - nBad;
    }
} // namespace HSLAM