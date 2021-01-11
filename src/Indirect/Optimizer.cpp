#include "Indirect/Optimizer.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>

#include <g2o/solvers/dense/linear_solver_dense.h>
// #include "g2o/solvers/linear_solver_dense.h"

#include <g2o/solvers/eigen/linear_solver_eigen.h>
// #include <g2o/solvers/linear_solver_eigen.h>

#include <boost/thread.hpp>
#include "Indirect/Frame.h"
#include "util/FrameShell.h"
#include "Indirect/MapPoint.h"
#include "Indirect/Map.h"
#include "FullSystem/FullSystem.h"

#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"

// #include "g2o/types/types_seven_dof_expmap.h"


namespace HSLAM
{
    

    using namespace std;
    using namespace OptimizationStructs;


    bool PoseOptimization(std::shared_ptr<Frame> pFrame, CalibHessian *calib, bool updatePose)
    {

        // g2o::SparseOptimizer optimizer;
        // g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
        // linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
        // g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        // g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        // optimizer.setAlgorithm(solver);

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
        vector<double> initErr;
        double initScale = 1.0;
        initErr.reserve(2 * N);
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);
        vector<double> vInformation;
        vInformation.reserve(N);
        double maxInfo = 1e7;
        double stdDev=1e7;

        const float deltaMono = sqrt(1.345); //sqrt(1.345); // sqrt(5.991);
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
                    e->setInformation(Eigen::Matrix2d::Identity());
                    e->setMeasurement(Vec2((double)pFrame->mvKeys[i].pt.x, (double)pFrame->mvKeys[i].pt.y));
                    e->setId(i);
                    
                    e->setVertex(0, vSE3);

                    e->computeError();
                    initErr.push_back(e->error()[0]);
                    initErr.push_back(e->error()[1]);

                    double Info = pMP->getidepthHessian();
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

            initScale = getStdDev(initErr); //computeScale(initErr); 

            for (int i = 0, iend = vpEdgesMono.size(); i < iend; ++i)
            {
                vpEdgesMono[i]->setScale(initScale);

                if (normalizeInfoWithVariance)
                    vpEdgesMono[i]->setInformation(Eigen::Matrix2d::Identity()* (vInformation[i]/(stdDev+0.00001))); //* invSigma2); //set this to take into account depth variance!
                else //normalizing by the maximum
                    vpEdgesMono[i]->setInformation(Eigen::Matrix2d::Identity()* (vInformation[i]/(maxInfo+0.00001))); //* invSigma2); //set this to take into account depth variance!
            }
        }
        if (nInitialCorrespondences < 10 || optimizer.edges().size() < 10)
            return false;

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        // const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991}; //1.345
        const float chi2Mono[4] = {1.345, 1.345, 1.345, 1.345}; //5.991
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

            if(!updatePose)
                break;
        }

        // number_t *hessianData = vSE3->hessianData();
        // Vec6 vHessian;
        // vHessian<< hessianData[0], hessianData[7], hessianData[14], hessianData[21], hessianData[28], hessianData[35];
        
        // bool isUsable = vHessian.norm() > 1e6;
        // // Recover optimized pose
        // if(isUsable && updatePose)
        //     pFrame->fs->setPose(vSE3->estimate().inverse());
        bool isUsable = false;
        return isUsable;
        }

int OptimizeSim3(std::shared_ptr<Frame> pKF1, std::shared_ptr<Frame> pKF2, std::vector<std::shared_ptr<MapPoint>> &vpMatches1, Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    // g2o::SparseOptimizer optimizer;
    // g2o::BlockSolverX::LinearSolverType *linearSolver;
    // linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    // g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
    // g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    // optimizer.setAlgorithm(solver);

    g2o::SparseOptimizer optimizer;
    auto linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver)));
    optimizer.setAlgorithm(solver);

    // // Calibration
    // const cv::Mat &K1 = pKF1->mK;
    // const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    
    auto PKF1Pose = pKF1->fs->getPoseOpti();
    auto R1w = PKF1Pose.rotationMatrix();
    auto t1w = PKF1Pose.translation();

    auto PKF2Pose = pKF2->fs->getPoseOpti();  
    auto R2w = PKF2Pose.rotationMatrix();
    auto t2w = PKF2Pose.translation();

    // Set Sim3 vertex
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();    
    vSim3->_fix_scale=bFixScale;
   
    vSim3->setEstimate(g2o::Sim3(g2oS12.rotationMatrix(), g2oS12.translation(), g2oS12.scale()));
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = pKF1->HCalib->cxl();//  K1.at<float>(0,2);
    vSim3->_principle_point1[1] = pKF1->HCalib->cyl(); //K1.at<float>(1,2);
    vSim3->_focal_length1[0] = pKF1->HCalib->fxl();//K1.at<float>(0,0);
    vSim3->_focal_length1[1] = pKF1->HCalib->fyl();// K1.at<float>(1,1);
    vSim3->_principle_point2[0] = pKF2->HCalib->cxl();// K2.at<float>(0,2);
    vSim3->_principle_point2[1] = pKF2->HCalib->cyl(); //K2.at<float>(1,2);
    vSim3->_focal_length2[0] = pKF2->HCalib->fxl(); //K2.at<float>(0, 0);
    vSim3->_focal_length2[1] = pKF2->HCalib->fyl(); //K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<std::shared_ptr<MapPoint>> vpMapPoints1 = pKF1->getMapPointsV();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        std::shared_ptr<MapPoint> pMP1 = vpMapPoints1[i];
        std::shared_ptr<MapPoint> pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = pMP2->getIndexInKF(pKF2);

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                Vec3 P3D1w = pMP1->getWorldPose().cast<double>();
                Vec3 P3D1c = R1w * P3D1w + t1w;
              
                vPoint1->setEstimate(P3D1c);
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                Vec3 P3D2w = pMP2->getWorldPose().cast<double>();
                Vec3 P3D2c = R2w * P3D2w + t2w;
                vPoint2->setEstimate(P3D2c);
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Vec2 obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeys[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0,optimizer.vertex(id2)); // dynamic_cast<g2o::OptimizableGraph::Vertex*>()
        e12->setVertex(1, optimizer.vertex(0)); //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
        e12->setMeasurement(obs1);

        //const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()); //*invSigmaSquare1

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Vec2 obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeys[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, optimizer.vertex(id1)); //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
        e21->setVertex(1, optimizer.vertex(0)); //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
        e21->setMeasurement(obs2);
        // float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()); //*invSigmaSquare2

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization(0);
    optimizer.optimize(5);

    // Check inliers
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = nullptr;
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization(0);
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = nullptr;
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2o::Sim3 Out = vSim3_recov->estimate();

    SE3 Pose = SE3(Out.rotation().toRotationMatrix(), Out.translation());
    g2oS12 = Sim3(Pose.matrix());
    g2oS12.setScale(Out.scale());

    return nIn;
}


void OptimizeEssentialGraph(std::vector<std::shared_ptr<Frame>> &vpKFs, std::vector<std::shared_ptr<MapPoint>> &vpMPs, std::vector<std::shared_ptr<Frame>> &actKFAtCand, std::vector<std::shared_ptr<MapPoint>> &actMpAtCand, std::shared_ptr<Frame> pLoopKF, std::shared_ptr<Frame> pCurKF,
           const KeyFrameAndPose &NonCorrectedSim3, const KeyFrameAndPose &CorrectedSim3, 
           const std::map<std::shared_ptr<Frame>, std::set<std::shared_ptr<Frame>, std::owner_less<std::shared_ptr<Frame>>>, std::owner_less<std::shared_ptr<Frame>>> &LoopConnections,
           const size_t nMaxKFid, const size_t maxKfIdatCand, const size_t maxMPIdatCand, const bool &bFixScale)
        {

            g2o::SparseOptimizer optimizer;
            g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_7_3>(g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>>()));
            solver->setUserLambdaInit(1e-16);
            optimizer.setAlgorithm(solver);

            // g2o::SparseOptimizer optimizer;
            // optimizer.setVerbose(false);
            // g2o::BlockSolver_7_3::LinearSolverType *linearSolver =
            //     new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
            // g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
            // g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
            // solver->setUserLambdaInit(1e-16);
            // optimizer.setAlgorithm(solver);

        
            std::vector<Sim3, Eigen::aligned_allocator<Sim3>> vScw(nMaxKFid + 1);
            vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
            vector<g2o::VertexSim3Expmap *, Eigen::aligned_allocator<g2o::VertexSim3Expmap *>> vpVertices(nMaxKFid + 1);

            const int minFeat = 100;

            // Set KeyFrame vertices
            for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
            {
                std::shared_ptr<Frame> pKF = vpKFs[i];
                if (pKF->isBad())
                    continue;
                g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

                const int nIDi = pKF->fs->KfId;

                KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

                if (it != CorrectedSim3.end()) //these will contain all keyframes that were active at the time 
                {
                    vScw[nIDi] = it->second;
                    VSim3->setEstimate(g2o::Sim3(it->second.rotationMatrix(), it->second.translation(), it->second.scale()));
                }
                else
                {
                    // Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
                    // Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());

                    //g2o::Sim3 Siw(Rcw,tcw,1.0);
                    auto TempSiw = pKF->fs->getPoseOpti();
                    // TempSiw.setScale(1.0);
                    vScw[nIDi] = TempSiw; //Siw
                    VSim3->setEstimate(g2o::Sim3(TempSiw.rotationMatrix(), TempSiw.translation(), TempSiw.scale()));
                }
                // Sim3 TempPose = pKF->fs->getPoseOpti();
                // VSim3->setEstimate(g2o::Sim3(TempPose.rotationMatrix(), TempPose.translation(), TempPose.scale()));
                // vScw[nIDi] = TempPose;

                if (pKF == pLoopKF || pKF->getState() == Frame::active || pKF->fs->KfId > maxKfIdatCand)
                    VSim3->setFixed(true);
                else
                    VSim3->setFixed(false);
                
                if(std::find(actKFAtCand.begin(), actKFAtCand.end(), pKF) != actKFAtCand.end()) //this KF was active when the candidate was detected need to fix its pose!
                    VSim3->setFixed(true);

                VSim3->setId(nIDi);
                VSim3->setMarginalized(false);
                VSim3->_fix_scale = false; //bFixScale;

                VSim3->_principle_point1[0] = pKF->HCalib->cxl(); //  K1.at<float>(0,2);
                VSim3->_principle_point1[1] = pKF->HCalib->cyl(); //K1.at<float>(1,2);
                VSim3->_focal_length1[0] = pKF->HCalib->fxl();    //K1.at<float>(0,0);
                VSim3->_focal_length1[1] = pKF->HCalib->fyl();    // K1.at<float>(1,1);
                VSim3->_principle_point2[0] = pKF->HCalib->cxl(); // K2.at<float>(0,2);
                VSim3->_principle_point2[1] = pKF->HCalib->cyl(); //K2.at<float>(1,2);
                VSim3->_focal_length2[0] = pKF->HCalib->fxl();    //K2.at<float>(0, 0);
                VSim3->_focal_length2[1] = pKF->HCalib->fyl();    //K2.at<float>(1,1);
                optimizer.addVertex(VSim3);

                vpVertices[nIDi] = VSim3;
            }

            std::set<pair<long unsigned int, long unsigned int>> sInsertedEdges;

            const Mat77 matLambda = Mat77::Identity();
            int index = nMaxKFid + 1;
            // Set Loop edges
            for (std::map<std::shared_ptr<Frame>, std::set<std::shared_ptr<Frame>, std::owner_less<std::shared_ptr<Frame>>>, std::owner_less<std::shared_ptr<Frame>>>::const_iterator mit = LoopConnections.begin(), mend = LoopConnections.end(); mit != mend; mit++)
            {
                std::shared_ptr<Frame> pKF = mit->first;

                const long unsigned int nIDi = pKF->fs->KfId;
                const std::set<std::shared_ptr<Frame>, std::owner_less<std::shared_ptr<Frame>>> &spConnections = mit->second;
                const g2o::Sim3 Siw = g2o::Sim3(vScw[nIDi].rotationMatrix(), vScw[nIDi].translation(), vScw[nIDi].scale());
                const g2o::Sim3 Swi = Siw.inverse();

                for (std::set<std::shared_ptr<Frame>>::const_iterator sit = spConnections.begin(), send = spConnections.end(); sit != send; sit++)
                {
                    const long unsigned int nIDj = (*sit)->fs->KfId;
                    if ((nIDi != pCurKF->fs->KfId || nIDj != pLoopKF->fs->KfId) && pKF->GetWeight(*sit) < minFeat)
                        continue;

                    const g2o::Sim3 Sjw = g2o::Sim3(vScw[nIDj].rotationMatrix(), vScw[nIDj].translation(), vScw[nIDj].scale());
                    const g2o::Sim3 Sji = Sjw * Swi;

                    g2o::EdgeSim3 *e = new g2o::EdgeSim3();
                    e->setId(index);
                    index++;
                    e->setVertex(1, optimizer.vertex(nIDj)); //dynamic_cast<g2o::OptimizableGraph::Vertex *>()
                    e->setVertex(0, optimizer.vertex(nIDi)); //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
                    e->setMeasurement(Sji);
                    //e->information() = matLambda;
                    e->setInformation(matLambda);

                    optimizer.addEdge(e);

                    sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
                }
            }

            // Set normal edges
            for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
            {
                std::shared_ptr<Frame> pKF = vpKFs[i];

                const int nIDi = pKF->fs->KfId;

                Sim3 tSwi;

                KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

                if (iti != NonCorrectedSim3.end())
                    tSwi = (iti->second).inverse();
                else
                    tSwi = vScw[nIDi].inverse();

                g2o::Sim3 Swi = g2o::Sim3(tSwi.rotationMatrix(), tSwi.translation(), tSwi.scale());

                std::shared_ptr<Frame> pParentKF = pKF->GetParent();

                // Spanning tree edge
                if (pParentKF)
                {
                    int nIDj = pParentKF->fs->KfId;

                    Sim3 tSjw;

                    KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

                    if (itj != NonCorrectedSim3.end())
                        tSjw = itj->second;
                    else
                        tSjw = vScw[nIDj];

                    g2o::Sim3 Sjw = g2o::Sim3(tSjw.rotationMatrix(), tSjw.translation(), tSjw.scale());

                    g2o::Sim3 Sji = Sjw * Swi;

                    g2o::EdgeSim3 *e = new g2o::EdgeSim3();
                    e->setId(index);
                    index++;
                    e->setVertex(1, optimizer.vertex(nIDj)); //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
                    e->setVertex(0, optimizer.vertex(nIDi)); //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
                    e->setMeasurement(Sji);

                    // e->information() = matLambda;
                    e->setInformation(matLambda);
                    optimizer.addEdge(e);
                }

                // Loop edges
                const set<std::shared_ptr<Frame>> sLoopEdges = pKF->GetLoopEdges();
                for (std::set<std::shared_ptr<Frame>>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++)
                {
                    std::shared_ptr<Frame> pLKF = *sit;
                    if (pLKF->fs->KfId < pKF->fs->KfId)
                    {
                        Sim3 tSlw;

                        KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                        if (itl != NonCorrectedSim3.end())
                            tSlw = itl->second;
                        else
                            tSlw = vScw[pLKF->fs->KfId];

                        g2o::Sim3 Slw = g2o::Sim3(tSlw.rotationMatrix(), tSlw.translation(), tSlw.scale());

                        g2o::Sim3 Sli = Slw * Swi;
                        g2o::EdgeSim3 *el = new g2o::EdgeSim3();
                        el->setId(index);
                        index++;
                        el->setVertex(1, optimizer.vertex(pLKF->fs->KfId)); //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
                        el->setVertex(0, optimizer.vertex(nIDi));           //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
                        el->setMeasurement(Sli);
                        el->setInformation(matLambda);

                        // el->information() = matLambda;
                        optimizer.addEdge(el);
                    }
                }

                // Covisibility graph edges
                const vector<std::shared_ptr<Frame>> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
                for (vector<std::shared_ptr<Frame>>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
                {
                    std::shared_ptr<Frame> pKFn = *vit;
                    if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
                    {
                        if (!pKFn->isBad() && pKFn->fs->KfId < pKF->fs->KfId)
                        {
                            if (sInsertedEdges.count(make_pair(std::min(pKF->fs->KfId, pKFn->fs->KfId), std::max(pKF->fs->KfId, pKFn->fs->KfId))))
                                continue;

                            Sim3 tSnw;

                            KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                            if (itn != NonCorrectedSim3.end())
                                tSnw = itn->second;
                            else
                                tSnw = vScw[pKFn->fs->KfId];

                            g2o::Sim3 Snw = g2o::Sim3(tSnw.rotationMatrix(), tSnw.translation(), tSnw.scale());

                            g2o::Sim3 Sni = Snw * Swi;

                            g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                            en->setId(index);
                            index++;
                            en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->fs->KfId)));
                            en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                            en->setMeasurement(Sni);
                            en->setInformation(matLambda);
                            // en->information() = matLambda;
                            optimizer.addEdge(en);
                        }
                    }
                }
            }

            // lock.unlock();
            // Optimize!
            optimizer.initializeOptimization(0);
            optimizer.optimize(25);
            // lock.lock();

            // boost::unique_lock<boost::mutex> lock(pMap->mMutexMap);
            // boost::unique_lock<boost::mutex> lock(_fs->mapMutex);
            // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
            for (size_t i = 0; i < vpKFs.size(); i++)
            {
                std::shared_ptr<Frame> pKFi = vpKFs[i];

                // if(pKFi->getState()== Frame::active)
                //     continue;

                const int nIDi = pKFi->fs->KfId;

                g2o::VertexSim3Expmap *VSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
                if (VSim3->fixed())
                    continue;
                g2o::Sim3 CorrectedSiw = VSim3->estimate();
                // vCorrectedSwc[nIDi]=CorrectedSiw.inverse();

                // Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
                // Eigen::Vector3d eigt = CorrectedSiw.translation();
                // double s = CorrectedSiw.scale();

                // eigt *=(1./s); //[R t/s;0 1]

                // cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

                // pKFi->SetPose(Tiw);
                SE3 se3Corrected = SE3(CorrectedSiw.rotation().toRotationMatrix(), CorrectedSiw.translation());
                Sim3 corrected = Sim3(se3Corrected.matrix());
                corrected.setScale(CorrectedSiw.scale());
                pKFi->fs->setPoseOpti(corrected);
            }

            // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
            for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
            {
                std::shared_ptr<MapPoint> pMP = vpMPs[i];
                
                // if(pMP->id > maxMPIdatCand)
                //     continue;
                
                if (pMP->getDirStatus() == MapPoint::active || pMP->getDirStatus() == MapPoint::removed)
                    continue;

                if (pMP->isBad())
                    continue;

                // int nIDr;
                // if(pMP->mnCorrectedByKF == pCurKF->fs->KfId)
                // {
                //     nIDr = pMP->mnCorrectedReference;
                // }
                // else
                // {
                //     std::shared_ptr<Frame> pRefKF = pMP->GetReferenceKeyFrame();
                //     nIDr = pRefKF->fs->KfId;
                // }

                //     // g2o::Sim3 Srw = vScw[nIDr];
                //     // g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

                //     // cv::Mat P3Dw = pMP->GetWorldPos();
                //     // Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                //     // Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

                //     // cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                //     // pMP->SetWorldPos(cvCorrectedP3Dw);
                pMP->updateGlobalPose();
                pMP->UpdateNormalAndDepth();
            }
            // lock.unlock();
}


void GlobalBundleAdjustemnt(std::shared_ptr<Map> pMap, int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust, const bool useSchurTrick)
{
    std::vector<std::shared_ptr<Frame>> vpKFs;
    pMap->GetAllKeyFrames(vpKFs);
    std::vector<std::shared_ptr<MapPoint>> vpMP;
    pMap->GetAllMapPoints(vpMP);
    BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust, useSchurTrick);
}

void BundleAdjustment(const std::vector<std::shared_ptr<Frame>> &vpKFs, const std::vector<std::shared_ptr<MapPoint>> &vpMP,
                                 int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust, const bool useSchurTrick)
{
    if(vpKFs.size() < 2  || vpMP.size() < 10)
        return;
    
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    g2o::SparseOptimizer optimizer;
    g2o::OptimizationAlgorithmLevenberg *solver;
    if (useSchurTrick)
        solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>()));
    else
        solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>>()));
    
    optimizer.setAlgorithm(solver);

 
    camParams *cam_params = new camParams(vpKFs[0]->HCalib->fxl(), vpKFs[0]->HCalib->fyl(), vpKFs[0]->HCalib->cxl(), vpKFs[0]->HCalib->cyl());
    cam_params->setId(0);
    if (!optimizer.addParameter(cam_params))
    {
        std::cout << "could not set up BA calibration! do not run BA!" << std::endl;
        return;
    }

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        std::shared_ptr<Frame> pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();

        auto Pose = pKF->fs->getPoseOpti();
        Pose.setScale(1.0);
        vSE3->setEstimate(g2o::SE3Quat(Pose.rotationMatrix(), Pose.translation())); //this removes the scale from the pose, remember to re-add it later

        vSE3->setId(pKF->fs->KfId);
        if(pKF->getState() == Frame::active)
            vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKF->fs->KfId > maxKFid)
            maxKFid = pKF->fs->KfId;
    }

    const float thHuber2D = sqrt(5.99);
    
    // Set MapPoint vertices
    for (size_t i = 0; i < vpMP.size(); i++)
    {
        std::shared_ptr<MapPoint> pMP = vpMP[i];
        if (pMP->isBad() || pMP->sourceFrame->isBad() )
            continue;
        
        VertexPointDepth *vPoint = new VertexPointDepth(); //g2o::vertexSBAPointXYZ..
        vPoint->setUV(pMP->pt.cast<double>());
        
        auto sourcePose = pMP->sourceFrame->fs->getPoseOpti();
        // sourcePose.setScale(1.0); //map points are already converted to the true scale, no need to rescale
        // vPoint->setEstimate(invert_depth(sourcePose * pMP->getWorldPose().cast<double>())); //MP in frame
        // auto ptinFrame = (sourcePose * pMP->getWorldPose().cast<double>());
        vPoint->setEstimate(pMP->getidepth() / sourcePose.scale());

        const int id = pMP->id + maxKFid + 1;
        vPoint->setId(id);

        if(useSchurTrick)
            vPoint->setMarginalized(true);

        if(pMP->getDirStatus() == MapPoint::active || pMP->sourceFrame->getState()==Frame::active)
            vPoint->setFixed(true);

        optimizer.addVertex(vPoint);

        const std::map<std::shared_ptr<Frame>, size_t> observations = pMP->GetObservations();
        
        auto vanchor = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertices().find(pMP->sourceFrame->fs->KfId)->second);
        int nEdges = 0;
        
        //SET EDGES
        for (std::map<std::shared_ptr<Frame>, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
        {

            std::shared_ptr<Frame> pKF = mit->first;
            if (pKF->isBad() || pKF->fs->KfId > maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeys[mit->second];

            Eigen::Matrix<double, 2, 1> obs;
            obs << kpUn.pt.x, kpUn.pt.y;


            EdgeProjectinvDepth *e = new EdgeProjectinvDepth();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vPoint));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertices().find(pKF->fs->KfId)->second));
            e->setVertex(2, vanchor);

            
            e->setMeasurement(obs);
            // const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
            e->setInformation(Eigen::Matrix2d::Identity()); // * pMP->getVariance();
           

            if (bRobust)
            {
                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber2D);
            }

            e->setParameterId(0, 0);
            optimizer.addEdge(e);
        }

        if (nEdges == 0 ) //|| pMP->getDirStatus() == MapPoint::active
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        }
        else
        {
            vbNotIncludedMP[i] = false;
        }
    }
            optimizer.setVerbose(true);

        // Optimize!
        optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    //Keyframes
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        std::shared_ptr<Frame> pKF = vpKFs[i];
        if (pKF->isBad() || pKF->getState() == Frame::active)
            continue;
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->fs->KfId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        // if (nLoopKF == 0)
        // {
       
        auto PoseSim3 = Sim3(SE3(SE3quat.rotation().toRotationMatrix(), SE3quat.translation()).matrix());
        PoseSim3.setScale(pKF->fs->getPoseOpti().scale());
        pKF->fs->setPoseOpti(PoseSim3);
        // }
        // else
        // {
        //     pKF->mTcwGBA.create(4, 4, CV_32F);
        //     Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
        //     pKF->mnBAGlobalForKF = nLoopKF;
        // }
    }

    //Points
    for (size_t i = 0; i < vpMP.size(); i++)
    {
        if (vbNotIncludedMP[i])
            continue;

        std::shared_ptr<MapPoint> pMP = vpMP[i];

        if (pMP->isBad() || pMP->sourceFrame->isBad() ||  pMP->getDirStatus() != MapPoint::marginalized)
            continue;

        VertexPointDepth *vPoint = static_cast<VertexPointDepth *>(optimizer.vertex(pMP->id + maxKFid + 1));

        // if (nLoopKF == 0)
        // {
            // auto scale = pMP->sourceFrame->fs->getPoseOpti().scale();
            pMP->updateDepthfromInd((float)vPoint->estimate()); //vPoint->estimate()(2)
            pMP->updateGlobalPose();
            // pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        // }
        // else
        // {
        //     pMP->mPosGBA.create(3, 1, CV_32F);
        //     Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
        //     pMP->mnBAGlobalForKF = nLoopKF;
        // }
    }
}

} // namespace HSLAM