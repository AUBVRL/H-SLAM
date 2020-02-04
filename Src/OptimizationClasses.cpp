#include "OptimizationClasses.h"
#include "CalibData.h"
#include "Frame.h"
#include "DirectProjection.h"
#include "MapPoint.h"

#include "EnergyFunctional.h"
#include "EnergyFunctionalStructs.h"

namespace FSLAM
{

  void FrameFramePrecalc::set(std::shared_ptr<Frame> host, std::shared_ptr<Frame> target, std::shared_ptr<CalibData> HCalib)
    {
        this->host = host;
        this->target = target;

        SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
        PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
        PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();

        SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
        PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
        PRE_tTll = (leftToLeft.translation()).cast<float>();
        distanceLL = leftToLeft.translation().norm();

        Mat33f K = Mat33f::Zero();
        K(0, 0) = HCalib->fxl();
        K(1, 1) = HCalib->fyl();
        K(0, 2) = HCalib->cxl();
        K(1, 2) = HCalib->cyl();
        K(2, 2) = 1;
        PRE_KRKiTll = K * PRE_RTll * K.inverse();
        PRE_RKiTll = PRE_RTll * K.inverse();
        PRE_KtTll = K * PRE_tTll;

        PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<float>();
        PRE_b0_mode = host->aff_g2l_0().b;
    }



    double PointFrameResidual::linearize(std::shared_ptr<CalibData> HCalib)
    {
      state_NewEnergyWithOutlier = -1;

      if (state_state == ResState::OOB)
      {
        state_NewState = ResState::OOB;
        return state_energy;
      }

      std::shared_ptr<Frame> lHost = host.lock();
      std::shared_ptr<Frame> lTarget = target.lock();
      std::shared_ptr<MapPoint> lpoint = point.lock();

      FrameFramePrecalc *precalc = &(lHost->targetPrecalc[lTarget->idx]);
      float energyLeft = 0;
      const Eigen::Vector3f *dIl = lTarget->DirPyr[0];
      //const float* const Il = target->I;
      const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
      const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
      const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;
      const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
      const float *const color = lpoint->color;
      const float *const weights = lpoint->weights;

      Vec2f affLL = precalc->PRE_aff_mode;
      float b0 = precalc->PRE_b0_mode;

      Vec6f d_xi_x, d_xi_y;
      Vec4f d_C_x, d_C_y;
      float d_d_x, d_d_y;
      {
        float drescale, u, v, new_idepth;
        float Ku, Kv;
        Vec3f KliP;

        if (!projectPoint(lpoint->u, lpoint->v, lpoint->idepth_zero_scaled, 0, 0, HCalib,
                          PRE_RTll_0, PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth))
        {
          state_NewState = ResState::OOB;
          return state_energy;
        }

        centerProjectedTo = Vec3f(Ku, Kv, new_idepth);

        // diff d_idepth
        d_d_x = drescale * (PRE_tTll_0[0] - PRE_tTll_0[2] * u) * SCALE_IDEPTH * HCalib->fxl();
        d_d_y = drescale * (PRE_tTll_0[1] - PRE_tTll_0[2] * v) * SCALE_IDEPTH * HCalib->fyl();

        // diff calib
        d_C_x[2] = drescale * (PRE_RTll_0(2, 0) * u - PRE_RTll_0(0, 0));
        d_C_x[3] = HCalib->fxl() * drescale * (PRE_RTll_0(2, 1) * u - PRE_RTll_0(0, 1)) * HCalib->fyli();
        d_C_x[0] = KliP[0] * d_C_x[2];
        d_C_x[1] = KliP[1] * d_C_x[3];

        d_C_y[2] = HCalib->fyl() * drescale * (PRE_RTll_0(2, 0) * v - PRE_RTll_0(1, 0)) * HCalib->fxli();
        d_C_y[3] = drescale * (PRE_RTll_0(2, 1) * v - PRE_RTll_0(1, 1));
        d_C_y[0] = KliP[0] * d_C_y[2];
        d_C_y[1] = KliP[1] * d_C_y[3];

        d_C_x[0] = (d_C_x[0] + u) * SCALE_F;
        d_C_x[1] *= SCALE_F;
        d_C_x[2] = (d_C_x[2] + 1) * SCALE_C;
        d_C_x[3] *= SCALE_C;

        d_C_y[0] *= SCALE_F;
        d_C_y[1] = (d_C_y[1] + v) * SCALE_F;
        d_C_y[2] *= SCALE_C;
        d_C_y[3] = (d_C_y[3] + 1) * SCALE_C;

        d_xi_x[0] = new_idepth * HCalib->fxl();
        d_xi_x[1] = 0;
        d_xi_x[2] = -new_idepth * u * HCalib->fxl();
        d_xi_x[3] = -u * v * HCalib->fxl();
        d_xi_x[4] = (1 + u * u) * HCalib->fxl();
        d_xi_x[5] = -v * HCalib->fxl();

        d_xi_y[0] = 0;
        d_xi_y[1] = new_idepth * HCalib->fyl();
        d_xi_y[2] = -new_idepth * v * HCalib->fyl();
        d_xi_y[3] = -(1 + v * v) * HCalib->fyl();
        d_xi_y[4] = u * v * HCalib->fyl();
        d_xi_y[5] = u * HCalib->fyl();
      }

      {
        J->Jpdxi[0] = d_xi_x;
        J->Jpdxi[1] = d_xi_y;

        J->Jpdc[0] = d_C_x;
        J->Jpdc[1] = d_C_y;

        J->Jpdd[0] = d_d_x;
        J->Jpdd[1] = d_d_y;
      }

      float JIdxJIdx_00 = 0, JIdxJIdx_11 = 0, JIdxJIdx_10 = 0;
      float JabJIdx_00 = 0, JabJIdx_01 = 0, JabJIdx_10 = 0, JabJIdx_11 = 0;
      float JabJab_00 = 0, JabJab_01 = 0, JabJab_11 = 0;

      float wJI2_sum = 0;

      for (int idx = 0; idx < patternNum; idx++)
      {
        float Ku, Kv;
        if (!projectPoint(lpoint->u + patternP[idx][0], lpoint->v + patternP[idx][1], lpoint->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv, HCalib))
        {
          state_NewState = ResState::OOB;
          return state_energy;
        }

        projectedTo[idx][0] = Ku;
        projectedTo[idx][1] = Kv;

        Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, HCalib->Width));
        float residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]);

        float drdA = (color[idx] - b0);
        if (!std::isfinite((float)hitColor[0]))
        {
          state_NewState = ResState::OOB;
          return state_energy;
        }

        float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
        w = 0.5f * (w + weights[idx]);

        float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
        energyLeft += w * w * hw * residual * residual * (2 - hw);

        {
          if (hw < 1)
            hw = sqrtf(hw);
          hw = hw * w;

          hitColor[1] *= hw;
          hitColor[2] *= hw;

          J->resF[idx] = residual * hw;

          J->JIdx[0][idx] = hitColor[1];
          J->JIdx[1][idx] = hitColor[2];
          J->JabF[0][idx] = drdA * hw;
          J->JabF[1][idx] = hw;

          JIdxJIdx_00 += hitColor[1] * hitColor[1];
          JIdxJIdx_11 += hitColor[2] * hitColor[2];
          JIdxJIdx_10 += hitColor[1] * hitColor[2];

          JabJIdx_00 += drdA * hw * hitColor[1];
          JabJIdx_01 += drdA * hw * hitColor[2];
          JabJIdx_10 += hw * hitColor[1];
          JabJIdx_11 += hw * hitColor[2];

          JabJab_00 += drdA * drdA * hw * hw;
          JabJab_01 += drdA * hw * hw;
          JabJab_11 += hw * hw;

          wJI2_sum += hw * hw * (hitColor[1] * hitColor[1] + hitColor[2] * hitColor[2]);

          if (setting_affineOptModeA < 0)
            J->JabF[0][idx] = 0;
          if (setting_affineOptModeB < 0)
            J->JabF[1][idx] = 0;
        }
      }

      J->JIdx2(0, 0) = JIdxJIdx_00;
      J->JIdx2(0, 1) = JIdxJIdx_10;
      J->JIdx2(1, 0) = JIdxJIdx_10;
      J->JIdx2(1, 1) = JIdxJIdx_11;
      J->JabJIdx(0, 0) = JabJIdx_00;
      J->JabJIdx(0, 1) = JabJIdx_01;
      J->JabJIdx(1, 0) = JabJIdx_10;
      J->JabJIdx(1, 1) = JabJIdx_11;
      J->Jab2(0, 0) = JabJab_00;
      J->Jab2(0, 1) = JabJab_01;
      J->Jab2(1, 0) = JabJab_01;
      J->Jab2(1, 1) = JabJab_11;

      state_NewEnergyWithOutlier = energyLeft;

      if (energyLeft > std::max<float>(lHost->frameEnergyTH, lTarget->frameEnergyTH) || wJI2_sum < 2)
      {
        energyLeft = std::max<float>(lHost->frameEnergyTH, lTarget->frameEnergyTH);
        state_NewState = ResState::OUTLIER;
      }
      else
      {
        state_NewState = ResState::IN;
      }

      state_NewEnergy = energyLeft;
      return energyLeft;
    }

    void PointFrameResidual::applyRes(bool copyJacobians)
    {
      if (copyJacobians)
      {
        if (state_state == ResState::OOB)
        {
          return; // can never go back from OOB
        }
        if (state_NewState == ResState::IN) // && )
        {
          isActiveAndIsGoodNEW = true;
          takeData();
        }
        else
        {
          isActiveAndIsGoodNEW = false;
        }
      }

      setState(state_NewState);
      state_energy = state_NewEnergy;
    }

    void PointFrameResidual::fixLinearizationF(std::shared_ptr<EnergyFunctional> ef) {

            Vec8f dp = ef->adHTdeltaF[hostIDX + ef->nFrames * targetIDX];

            // compute Jp*delta
            __m128 Jp_delta_x = _mm_set1_ps(J->Jpdxi[0].dot(dp.head<6>())
                                            + J->Jpdc[0].dot(ef->cDeltaF)
                                            + J->Jpdd[0] * point.lock()->deltaF);
            __m128 Jp_delta_y = _mm_set1_ps(J->Jpdxi[1].dot(dp.head<6>())
                                            + J->Jpdc[1].dot(ef->cDeltaF)
                                            + J->Jpdd[1] * point.lock()->deltaF);

            __m128 delta_a = _mm_set1_ps((float) (dp[6]));
            __m128 delta_b = _mm_set1_ps((float) (dp[7]));

            for (int i = 0; i < patternNum; i += 4) {
                // PATTERN: rtz = resF - [JI*Jp Ja]*delta.
                __m128 rtz = _mm_load_ps(((float *) &J->resF) + i);
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JIdx)) + i), Jp_delta_x));
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JIdx + 1)) + i), Jp_delta_y));
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JabF)) + i), delta_a));
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JabF + 1)) + i), delta_b));
                _mm_store_ps(((float *) &res_toZeroF) + i, rtz);
            }

            isLinearized = true;
        }

    } // namespace FSLAM