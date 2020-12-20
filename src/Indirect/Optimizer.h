#ifndef __OPTIMIZER_H_
#define __OPTIMIZER_H_

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include "FullSystem/HessianBlocks.h"
#include "util/NumType.h"

namespace HSLAM
{
    class Frame;

    namespace OptimizationStructs
    {
        using namespace g2o;
        using namespace Eigen;

        class vertexSE3 : public BaseVertex<6, SE3>
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            vertexSE3() : BaseVertex<6, SE3>() { }
            bool read(std::istream &is) override { return true; }
            bool write(std::ostream &os) const override { return true; }

            virtual void setToOriginImpl() override {_estimate = SE3();}
            virtual void oplusImpl(const double *update_) override
            {
                Eigen::Map<const Vector6> update(update_);
                _estimate = SE3::exp(update) * _estimate;
            }
        };

        class edgeSE3XYZPoseOnly : public BaseUnaryEdge<2, Vector2, vertexSE3>
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
            edgeSE3XYZPoseOnly() : BaseUnaryEdge<2, Vector2d, vertexSE3>(){}

            bool read(std::istream &is) override { return true; }
            bool write(std::ostream &os) const override { return true; }

            void computeError() override
            {
                const vertexSE3 *v1 = static_cast<vertexSE3 *>(_vertices[0]);
                

                Vec3 PointinFrame = v1->estimate() * (Xw);
                PointinFrame = PointinFrame * (1.0 / PointinFrame[2]);
                if (PointinFrame[2] < 0) 
                {
                    std::cout << "negative projected depth should not occur! skip"<<std::endl;
                    return;
                }
                double u = fx * PointinFrame[0] + cx;
                double v = fy * PointinFrame[1] + cy;
                _error =  _measurement - Vec2(u,v) ; //- Vec2(0.5,0.5)
            }

            void setCamera(const number_t _fx, const number_t _fy, const number_t _cx, const number_t _cy) { fx = (double)_fx; fy = (double)_fy; cx = (double)_cx; cy = (double)_cy; }

            void setXYZ(const Vector3d &_Xw) {Xw = _Xw;}

            bool isDepthPositive()
            {
                const vertexSE3 *v1 = static_cast<const vertexSE3 *>(_vertices[0]);
                Vec3 PointinFrame = v1->estimate() * (Xw);
                PointinFrame = PointinFrame * (1.0 / PointinFrame[2]);
                return (PointinFrame[2] > 0);
            }
            void linearizeOplus() override
            {
                vertexSE3 *vi = static_cast<vertexSE3 *>(_vertices[0]);
                Vector3 xyz_trans = vi->estimate() * (Xw);

                number_t x = xyz_trans[0];
                number_t y = xyz_trans[1];
                number_t invz = 1.0 / xyz_trans[2];
                number_t invz_2 = invz * invz;

                _jacobianOplusXi(0, 0) = -invz * fx; 
                _jacobianOplusXi(0, 1) = 0;
                _jacobianOplusXi(0, 2) = x * invz_2 * fx;
                _jacobianOplusXi(0, 3) = x * y * invz_2 * fx;
                _jacobianOplusXi(0, 4) = -(1 + (x * x * invz_2)) * fx;
                _jacobianOplusXi(0, 5) = y * invz * fx;
        
                _jacobianOplusXi(1, 0) = 0;
                _jacobianOplusXi(1, 1) = -invz * fy;
                _jacobianOplusXi(1, 2) = y * invz_2 * fy;
                _jacobianOplusXi(1, 3) = (1 + y * y * invz_2) * fy;
                _jacobianOplusXi(1, 4) = -x * y * invz_2 * fy;
                _jacobianOplusXi(1, 5) = -x * invz * fy;
               
            }
            Vector3 Xw;
        private:
           
            number_t fx, fy, cx, cy;
        };
    } // namespace OptimizationStructs

    int PoseOptimization(std::shared_ptr<Frame> pFrame, CalibHessian *calib);
}


#endif