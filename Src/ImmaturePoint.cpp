#include "ImmaturePoint.h"
#include "CalibData.h"
#include "Frame.h"
#include "OptimizationClasses.h"
// #include "util/FrameShell.h"
// #include "FullSystem/ResidualProjections.h"

namespace FSLAM
{
ImmaturePoint::ImmaturePoint(int u_, int v_, int index_, std::shared_ptr<Frame> host_, float type, std::shared_ptr<CalibData> Calib)
	: u(u_), v(v_), index(index_), hostFrame(host_), my_type(type), idepth_min(0), idepth_max(NAN), lastTraceStatus(IPS_UNINITIALIZED)
{
	std::shared_ptr<Frame> lHost = hostFrame.lock();
	if (!lHost)
		throw std::runtime_error("Host frame for immature point does not exists or was removed!");
	u_stereo = u; v_stereo = v;
	gradH.setZero();

	for (int idx = 0; idx < patternNum; idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		Vec3f ptc = getInterpolatedElement33BiLin(lHost->LeftDirPyr[0], u + dx, v + dy, Calib->wpyr[0]);

		color[idx] = ptc[0];
		if (!std::isfinite(color[idx]))
		{
			energyTH = NAN;
			return;
		}

		gradH += ptc.tail<2>() * ptc.tail<2>().transpose();

		weights[idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
	}

	energyTH = patternNum * setting_outlierTH;
	energyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;

	idepth_GT = 0;
	quality = 10000;
}

ImmaturePoint::~ImmaturePoint()
{
}

/*
 * returns
 * * OOB -> point is optimized and marginalized
 * * UPDATED -> point has been updated.
 * * SKIP -> point has not been updated.
 */
ImmaturePointStatus ImmaturePoint::traceOn(std::vector<Vec3f> &frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f &hostToFrame_affine,
										   std::shared_ptr<CalibData> Calib, bool debugPrint)
{
	if (lastTraceStatus == ImmaturePointStatus::IPS_OOB)
		return lastTraceStatus;

	debugPrint = false; //rand()%100==0;
	float maxPixSearch = (Calib->wpyr[0] + Calib->hpyr[0]) * setting_maxPixSearch;

	// if(debugPrint)
	// 	printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f -> %f. t %f %f %f!\n",
	// 			u,v,
	// 			host->shell->id, frame->shell->id,
	// 			idepth_min, idepth_max,
	// 			hostToFrame_Kt[0],hostToFrame_Kt[1],hostToFrame_Kt[2]);

	//	const float stepsize = 1.0;				// stepsize for initial discrete search.
	// ============== project min and max. return if one of them is OOB ===================
	Vec3f pr = hostToFrame_KRKi * Vec3f(u, v, 1);
	Vec3f ptpMin = pr + hostToFrame_Kt * idepth_min;
	float uMin = ptpMin[0] / ptpMin[2];
	float vMin = ptpMin[1] / ptpMin[2];

	if (!(uMin > 4 && vMin > 4 && uMin < Calib->wpyr[0] - 5 && vMin < Calib->hpyr[0] - 5))
	{
		if (debugPrint)
			printf("OOB uMin %f %f - %f %f %f (id %f-%f)!\n", u, v, uMin, vMin, ptpMin[2], idepth_min, idepth_max);
		lastTraceUV = Vec2f(-1, -1);
		lastTracePixelInterval = 0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

	float dist;
	float uMax;
	float vMax;
	Vec3f ptpMax;
	if (std::isfinite(idepth_max))
	{
		ptpMax = pr + hostToFrame_Kt * idepth_max;
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];

		if (!(uMax > 4 && vMax > 4 && uMax < Calib->wpyr[0] - 5 && vMax < Calib->hpyr[0] - 5))
		{
			if (debugPrint)
				printf("OOB uMax  %f %f - %f %f!\n", u, v, uMax, vMax);
			
			lastTraceUV = Vec2f(-1, -1);
			lastTracePixelInterval = 0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}

		// ============== check their distance. everything below 2px is OK (-> skip). ===================
		dist = (uMin - uMax) * (uMin - uMax) + (vMin - vMax) * (vMin - vMax);
		dist = sqrtf(dist);
		if (dist < setting_trace_slackInterval) // if pixel-interval is smaller than this, leave it be.
		{
			if (debugPrint)
				printf("TOO CERTAIN ALREADY (dist %f)!\n", dist);

			lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;
			lastTracePixelInterval = dist;
			return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
		}
		assert(dist > 0);
	}
	else
	{
		dist = maxPixSearch;

		// project to arbitrary depth to get direction.
		ptpMax = pr + hostToFrame_Kt * 0.01;
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];

		// direction.
		float dx = uMax - uMin;
		float dy = vMax - vMin;
		float d = 1.0f / sqrtf(dx * dx + dy * dy);

		// set to [setting_maxPixSearch].
		uMax = uMin + dist * dx * d;
		vMax = vMin + dist * dy * d;

		// may still be out!
		if (!(uMax > 4 && vMax > 4 && uMax < Calib->wpyr[0] - 5 && vMax < Calib->hpyr[0] - 5))
		{
			if (debugPrint)
				printf("OOB uMax-coarse %f %f %f!\n", uMax, vMax, ptpMax[2]);
			lastTraceUV = Vec2f(-1, -1);
			lastTracePixelInterval = 0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}
		assert(dist > 0);
	}

	// set OOB if scale change too big.
	if (!(idepth_min < 0 || (ptpMin[2] > 0.75 && ptpMin[2] < 1.5)))
	{
		if (debugPrint)
			printf("OOB SCALE %f %f %f!\n", uMax, vMax, ptpMin[2]);
		
		lastTraceUV = Vec2f(-1, -1);
		lastTracePixelInterval = 0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

	// ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
	float dx = setting_trace_stepsize * (uMax - uMin);
	float dy = setting_trace_stepsize * (vMax - vMin);

	float a = (Vec2f(dx, dy).transpose() * gradH * Vec2f(dx, dy));
	float b = (Vec2f(dy, -dx).transpose() * gradH * Vec2f(dy, -dx));
	float errorInPixel = 0.2f + 0.2f * (a + b) / a;

	if (errorInPixel * setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max)) // if pixel-interval is smaller than this, leave it be.
	{
		if (debugPrint)
			printf("NO SIGNIFICANT IMPROVMENT (%f)!\n", errorInPixel);
		
		lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;
		lastTracePixelInterval = dist;
		return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
	}

	if (errorInPixel > 10)
		errorInPixel = 10;

	// ============== do the discrete search ===================
	dx /= dist;
	dy /= dist;

	// if(debugPrint)
	// 	printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f (%.1f %.1f) -> %f (%.1f %.1f)! ErrorInPixel %.1f!\n",
	// 			u,v,
	// 			host->shell->id, frame->shell->id,
	// 			idepth_min, uMin, vMin,
	// 			idepth_max, uMax, vMax,
	// 			errorInPixel
	// 			);

	if (dist > maxPixSearch)
	{
		uMax = uMin + maxPixSearch * dx;
		vMax = vMin + maxPixSearch * dy;
		dist = maxPixSearch;
	}

	int numSteps = 1.9999f + dist / setting_trace_stepsize;
	Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2, 2>();

	float randShift = uMin * 1000 - floorf(uMin * 1000);
	float ptx = uMin - randShift * dx;
	float pty = vMin - randShift * dy;

	Vec2f rotatetPattern[MAX_RES_PER_POINT];
	for (int idx = 0; idx < patternNum; idx++)
		rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);

	if (!std::isfinite(dx) || !std::isfinite(dy))
	{
		//printf("COUGHT INF / NAN dxdy (%f %f)!\n", dx, dx);

		lastTracePixelInterval = 0;
		lastTraceUV = Vec2f(-1, -1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

	float errors[100];
	float bestU = 0, bestV = 0, bestEnergy = 1e10;
	int bestIdx = -1;
	if (numSteps >= 100)
		numSteps = 99;

	for (int i = 0; i < numSteps; ++i)
	{
		float energy = 0;
		for (int idx = 0; idx < patternNum; idx++)
		{
			float hitColor = getInterpolatedElement31(frame, (float)(ptx + rotatetPattern[idx][0]), (float)(pty + rotatetPattern[idx][1]), Calib->wpyr[0]);

			if (!std::isfinite(hitColor))
			{
				energy += 1e5;
				continue;
			}
			float residual = hitColor - (float)(hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw * residual * residual * (2 - hw);
		}

		if (debugPrint)
			printf("step %.1f %.1f (id %f): energy = %f!\n", ptx, pty, 0.0f, energy);

		errors[i] = energy;
		if (energy < bestEnergy)
		{
			bestU = ptx;
			bestV = pty;
			bestEnergy = energy;
			bestIdx = i;
		}

		ptx += dx;
		pty += dy;
	}

	// find best score outside a +-2px radius.
	float secondBest = 1e10;
	for (int i = 0; i < numSteps; ++i)
	{
		if ((i < bestIdx - setting_minTraceTestRadius || i > bestIdx + setting_minTraceTestRadius) && errors[i] < secondBest)
			secondBest = errors[i];
	}
	float newQuality = secondBest / bestEnergy;
	if (newQuality < quality || numSteps > 10)
		quality = newQuality;

	// ============== do GN optimization ===================
	float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0;
	if (setting_trace_GNIterations > 0)
		bestEnergy = 1e5;
	int gnStepsGood = 0, gnStepsBad = 0;
	for (int it = 0; it < setting_trace_GNIterations; it++) // max # GN iterations
	{
		float H = 1, b = 0, energy = 0;
		for (int idx = 0; idx < patternNum; idx++)
		{
			Vec3f hitColor = getInterpolatedElement33(frame, (float)(bestU + rotatetPattern[idx][0]), (float)(bestV + rotatetPattern[idx][1]), Calib->wpyr[0]);

			if (!std::isfinite((float)hitColor[0]))
			{
				energy += 1e5;
				continue;
			}
			float residual = hitColor[0] - (hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
			float dResdDist = dx * hitColor[1] + dy * hitColor[2];
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			H += hw * dResdDist * dResdDist;
			b += hw * residual * dResdDist;
			energy += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
		}

		if (energy > bestEnergy)
		{
			gnStepsBad++;

			// do a smaller step from old point.
			stepBack *= 0.5;
			bestU = uBak + stepBack * dx;
			bestV = vBak + stepBack * dy;
			if (debugPrint)
				printf("GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n", it, energy, H, b, stepBack, uBak, vBak, bestU, bestV);
		}
		else
		{
			gnStepsGood++;

			float step = -gnstepsize * b / H;
			if (step < -0.5) 
				step = -0.5;
			else if (step > 0.5)
				step = 0.5;

			if (!std::isfinite(step))
				step = 0;

			uBak = bestU;
			vBak = bestV;
			stepBack = step;

			bestU += step * dx;
			bestV += step * dy;
			bestEnergy = energy;

			if (debugPrint)
				printf("GN step %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n", it, energy, H, b, step, uBak, vBak, bestU, bestV);
		}

		if (fabsf(stepBack) < setting_trace_GNThreshold)
			break; // GN stop after this stepsize.
	}

	// ============== detect energy-based outlier. ===================
	//	float absGrad0 = getInterpolatedElement(frame->absSquaredGrad[0],bestU, bestV, wG[0]);
	//	float absGrad1 = getInterpolatedElement(frame->absSquaredGrad[1],bestU*0.5-0.25, bestV*0.5-0.25, wG[1]);
	//	float absGrad2 = getInterpolatedElement(frame->absSquaredGrad[2],bestU*0.25-0.375, bestV*0.25-0.375, wG[2]);
	if (!(bestEnergy < energyTH * setting_trace_extraSlackOnTH)) //// for energy-based outlier check, be slightly more relaxed by this factor.
																 //			|| (absGrad0*areaGradientSlackFactor < host->frameGradTH
																 //		     && absGrad1*areaGradientSlackFactor < host->frameGradTH*0.75f
																 //			 && absGrad2*areaGradientSlackFactor < host->frameGradTH*0.50f))
	{
		if (debugPrint)
			printf("OUTLIER!\n");

		lastTracePixelInterval = 0;
		lastTraceUV = Vec2f(-1, -1);
		if (lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		else
			return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

	// ============== set new interval ===================
	if (dx * dx > dy * dy)
	{
		idepth_min = (pr[2] * (bestU - errorInPixel * dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU - errorInPixel * dx));
		idepth_max = (pr[2] * (bestU + errorInPixel * dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU + errorInPixel * dx));
	}
	else
	{
		idepth_min = (pr[2] * (bestV - errorInPixel * dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV - errorInPixel * dy));
		idepth_max = (pr[2] * (bestV + errorInPixel * dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV + errorInPixel * dy));
	}
	if (idepth_min > idepth_max)
		std::swap<float>(idepth_min, idepth_max);

	if (!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max < 0))
	{
		//printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min, idepth_max);

		lastTracePixelInterval = 0;
		lastTraceUV = Vec2f(-1, -1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

	lastTracePixelInterval = 2 * errorInPixel;
	lastTraceUV = Vec2f(bestU, bestV);
	return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
}

EIGEN_STRONG_INLINE bool ImmaturePoint::projectPoint(const float &u_pt, const float &v_pt, const float &idepth, const int &dx, const int &dy,
													 std::shared_ptr<CalibData> const &Calib, const Mat33f &R, const Vec3f &t, float &drescale,
													 float &u, float &v, float &Ku, float &Kv, Vec3f &KliP, float &new_idepth)
{
	KliP = Vec3f((u_pt + dx - Calib->cxl()) * Calib->fxli(), (v_pt + dy - Calib->cyl()) * Calib->fyli(), 1);

	Vec3f ptp = R * KliP + t * idepth;
	drescale = 1.0f / ptp[2];
	new_idepth = idepth * drescale;

	if (!(drescale > 0))
		return false;

	u = ptp[0] * drescale;
	v = ptp[1] * drescale;
	Ku = u * Calib->fxl() + Calib->cxl();
	Kv = v * Calib->fyl() + Calib->cyl();

	return Ku > 1.1f && Kv > 1.1f && Ku < (Calib->wpyr[0] - 3) && Kv < (Calib->hpyr[0] - 3);
}

ImmaturePointStatus ImmaturePoint::traceStereo(std::vector<Vec3f> &frame, std::shared_ptr<CalibData> Calib){
	Mat33f K = Mat33f::Identity();
	K(0,0) = Calib->fxl();
	K(1,1) = Calib->fyl();
	K(0,2) = Calib->cxl();
	K(1,2) = Calib->cyl();
	float baseline = Calib->mbf;
	
	Mat33f KRKi = Mat33f::Identity().cast<float>();
	Vec3f Kt;
	Vec3f bl;
	Vec2f aff;
	aff << 1, 0;

	bl << -baseline, 0, 0;
	Kt = K*bl;
	
	Vec3f pr = KRKi * Vec3f(u_stereo,v_stereo, 1);
	Vec3f ptpMin = pr +Kt * idepth_min_stereo;
	
	float uMin = ptpMin[0] / ptpMin[2];
	float vMin = ptpMin[1] / ptpMin[2];

	if (!(uMin > 4 && vMin > 4 && uMin < Calib->wpyr[0] - 5 && vMin < Calib->hpyr[0] - 5))
	{
		lastTraceUV = Vec2f(-1,-1);
		lastTracePixelInterval=0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}
	
	float dist;
	float uMax;
	float vMax;
	Vec3f ptpMax;
	float maxPixSearch = (Calib->wpyr[0]+Calib->hpyr[0])*setting_maxPixSearch;
	if(std::isfinite(idepth_max_stereo))
	{
		ptpMax = pr + Kt*idepth_max_stereo;
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];
		
		if(!(uMax > 4 && vMax > 4 && uMax < Calib->wpyr[0]-5 && vMax < Calib->hpyr[0]-5))
		{
			lastTraceUV = Vec2f(-1,-1);
			lastTracePixelInterval=0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}

		// ============== check their distance. everything below 2px is OK (-> skip). ===================
		dist = (uMin-uMax)*(uMin-uMax) + (vMin-vMax)*(vMin-vMax);
		dist = sqrtf(dist);
		if(dist < setting_trace_slackInterval)
		{
			return lastTraceStatus = ImmaturePointStatus ::IPS_SKIPPED;

		}
		assert(dist>0);
	}
	else
	{
		dist = maxPixSearch;

		// project to arbitrary depth to get direction.
		ptpMax = pr + Kt*0.01;
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];

		// direction.
		float dx = uMax-uMin;
		float dy = vMax-vMin;
		float d = 1.0f / sqrtf(dx*dx+dy*dy);

		// set to [setting_maxPixSearch].
		uMax = uMin + dist*dx*d;
		vMax = vMin + dist*dy*d;

		// may still be out!
		if(!(uMax > 4 && vMax > 4 && uMax < Calib->wpyr[0]-5 && vMax < Calib->hpyr[0]-5))
		{
			lastTraceUV = Vec2f(-1,-1);
			lastTracePixelInterval=0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}
		assert(dist>0);
	}
	if(!(idepth_min<0 || (ptpMin[2]>0.75 && ptpMin[2]<1.5)))
	{
		lastTraceUV = Vec2f(-1, -1);
		lastTracePixelInterval = 0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}
	
	// ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
	float dx = setting_trace_stepsize*(uMax-uMin);
	float dy = setting_trace_stepsize*(vMax-vMin);

	float a = (Vec2f(dx,dy).transpose() * gradH * Vec2f(dx,dy));
	float b = (Vec2f(dy,-dx).transpose() * gradH * Vec2f(dy,-dx));
	float errorInPixel = 0.2f + 0.2f * (a+b) / a;
	
	if(errorInPixel*setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max_stereo))
	{
		return lastTraceStatus = ImmaturePointStatus ::IPS_BADCONDITION;
	}
	
	if(errorInPixel >10) errorInPixel=10;
	
	// ============== do the discrete search ===================
	dx /= dist;
	dy /= dist;
	
	if(dist>maxPixSearch)
	{
		uMax = uMin + maxPixSearch*dx;
		vMax = vMin + maxPixSearch*dy;
		dist = maxPixSearch;
	}
	
	int numSteps = 1.9999f + dist / setting_trace_stepsize;
	Mat22f Rplane = KRKi.topLeftCorner<2,2>();
	
	float randShift = uMin*1000-floorf(uMin*1000);
	float ptx = uMin-randShift*dx;
	float pty = vMin-randShift*dy;
	
	Vec2f rotatetPattern[MAX_RES_PER_POINT];
	for(int idx=0;idx<patternNum;idx++)
		rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);
	
	if(!std::isfinite(dx) || !std::isfinite(dy))
	{
		lastTraceUV = Vec2f(-1,-1);
		lastTracePixelInterval=0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}
	
	float errors[100];
	float bestU=0, bestV=0, bestEnergy=1e10;
	int bestIdx=-1;
	if(numSteps >= 100) numSteps = 99;
	
	for(int i=0;i<numSteps;i++)
	{
		float energy=0;
		for(int idx=0;idx<patternNum;idx++)
		{
			float hitColor = getInterpolatedElement31(frame, (float)(ptx + rotatetPattern[idx][0]), (float)(pty + rotatetPattern[idx][1]), Calib->wpyr[0]);


			if(!std::isfinite(hitColor)) {energy+=1e5; continue;}
			float residual = hitColor - (float)(aff[0] * color[idx] + aff[1]);
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw *residual*residual*(2-hw);
		}

		errors[i] = energy;
		if(energy < bestEnergy)
		{
			bestU = ptx;
			bestV = pty;
			bestEnergy = energy;
			bestIdx = i;
		}

		ptx+=dx;
		pty+=dy;
	}
	
	// find best score outside a +-2px radius.
	float secondBest=1e10;
	for(int i=0;i<numSteps;i++)
	{
		if((i < bestIdx-setting_minTraceTestRadius || i > bestIdx+setting_minTraceTestRadius) && errors[i] < secondBest)
			secondBest = errors[i];
	}
	float newQuality = secondBest / bestEnergy;
	if(newQuality < quality || numSteps > 10) quality = newQuality;
	
	// ============== do GN optimization ===================
	float uBak=bestU, vBak=bestV, gnstepsize=1, stepBack=0;
	if(setting_trace_GNIterations>0) bestEnergy = 1e5;
	int gnStepsGood=0, gnStepsBad=0;
	for(int it=0;it<setting_trace_GNIterations;it++)
	{
		float H = 1, b=0, energy=0;
		for(int idx=0;idx<patternNum;idx++)
		{
			Vec3f hitColor = getInterpolatedElement33(frame, (float)(bestU + rotatetPattern[idx][0]), (float)(bestV + rotatetPattern[idx][1]), Calib->wpyr[0]);

			if(!std::isfinite((float)hitColor[0])) {energy+=1e5; continue;}
			float residual = hitColor[0] - (aff[0] * color[idx] + aff[1]);
			float dResdDist = dx*hitColor[1] + dy*hitColor[2];
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			H += hw*dResdDist*dResdDist;
			b += hw*residual*dResdDist;
			energy += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
		}


		if(energy > bestEnergy)
		{
			gnStepsBad++;

			// do a smaller step from old point.
			stepBack*=0.5;
			bestU = uBak + stepBack*dx;
			bestV = vBak + stepBack*dy;
		}
		else
		{
			gnStepsGood++;

			float step = -gnstepsize*b/H;
			if(step < -0.5) step = -0.5;
			else if(step > 0.5) step=0.5;

			if(!std::isfinite(step)) step=0;

			uBak=bestU;
			vBak=bestV;
			stepBack=step;

			bestU += step*dx;
			bestV += step*dy;
			bestEnergy = energy;

		}

		if(fabsf(stepBack) < setting_trace_GNThreshold) break;
	}

	if(!(bestEnergy < energyTH*setting_trace_extraSlackOnTH))
	{

		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		if(lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		else
			return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}
	// ============== set new interval ===================
	if(dx*dx>dy*dy)
	{
		idepth_min_stereo = (pr[2]*(bestU-errorInPixel*dx) - pr[0]) / (Kt[0] - Kt[2]*(bestU-errorInPixel*dx));
		idepth_max_stereo = (pr[2]*(bestU+errorInPixel*dx) - pr[0]) / (Kt[0] - Kt[2]*(bestU+errorInPixel*dx));
	}
	else
	{
		idepth_min_stereo = (pr[2]*(bestV-errorInPixel*dy) - pr[1]) / (Kt[1] - Kt[2]*(bestV-errorInPixel*dy));
		idepth_max_stereo = (pr[2]*(bestV+errorInPixel*dy) - pr[1]) / (Kt[1] - Kt[2]*(bestV+errorInPixel*dy));
	}
	if(idepth_min_stereo > idepth_max_stereo) std::swap<float>(idepth_min_stereo, idepth_max_stereo);

//  printf("the idpeth_min is %f, the idepth_max is %f \n", idepth_min, idepth_max);

	if(!std::isfinite(idepth_min_stereo) || !std::isfinite(idepth_max_stereo) || (idepth_max_stereo<0))
	{
		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}
// 	if(idepth_min_stereo<0||idepth_max_stereo<0||(1/idepth_min_stereo-1/idepth_max_stereo)>30){
// 	      lastTracePixelInterval=0;
// 	      lastTraceUV = Vec2f(-1,-1);
// 	      return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
// 	}

	lastTracePixelInterval=2*errorInPixel;
	lastTraceUV = Vec2f(bestU, bestV);
	// baseline * fx
// 	double idepth0;
// 	double bf = -K(0,0)*bl[0];
// 	idepth_stereo = (u_stereo - bestU)/bf;
	  
	Eigen::Matrix<float,3,2> A;
	for(int i=0;i<3;++i){
	    A(i,0) = pr(i);
	}
	A(0,1) = -bestU;
	A(1,1) = -bestV;
	A(2,1) = -1;
	Vec2f depth_l_r = ((A.transpose()*A).inverse())*A.transpose()*(-Kt);
	idepth_stereo = 1/depth_l_r(0);
	if(idepth_stereo<0){
		  idepth_stereo = 0;
	      lastTracePixelInterval=0;
	      lastTraceUV = Vec2f(-1,-1);
	      return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}
	
	
// 	LOG(INFO)<<"KRKi: \n"<<KRKi;
// 	LOG(INFO)<<"K: \n"<<K;
// 	LOG(INFO)<<"K_right: \n"<<K_right;
// 	LOG(INFO)<<"depth: "<<1/idepth_stereo<<" depth0: "<<1/idepth0<<" 1/idepth_min: "<<1/idepth_min_stereo<<" 1/idepth_max: "<<idepth_max_stereo;
// 	exit(1);
	return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
  
}


double ImmaturePoint::linearizeResidual(std::shared_ptr<CalibData> Calib, const float outlierTHSlack, ImmaturePointTemporaryResidual *tmpRes,
										float &Hdd, float &bd, float idepth)
{
	if (tmpRes->state_state == ResState::OOB)
	{
		tmpRes->state_NewState = ResState::OOB;
		return tmpRes->state_energy;
	}

	if(hostFrame.expired())
		return 999999; // this should never happen, in case it does return very large energy so it gets removed.
	FrameFramePrecalc *precalc = &(hostFrame.lock()->targetPrecalc[tmpRes->target.lock()->idx]);

	// check OOB due to scale angle change.

	float energyLeft = 0;
	const std::vector<Vec3f> *dIl =  &tmpRes->target.lock()->LeftDirPyr[0];
	//const Eigen::Vector3f *dIl = tmpRes->target.lock()->dI;
	const Mat33f &PRE_RTll = precalc->PRE_RTll;
	const Vec3f &PRE_tTll = precalc->PRE_tTll;

	Vec2f affLL = precalc->PRE_aff_mode;

	for (int idx = 0; idx < patternNum; idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		if (!projectPoint(this->u, this->v, idepth, dx, dy, Calib, PRE_RTll, PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth))
		{
			tmpRes->state_NewState = ResState::OOB;
			return tmpRes->state_energy;
		}

		Vec3f hitColor = (getInterpolatedElement33(dIl[0], Ku, Kv, Calib->wpyr[0]));

		if (!std::isfinite((float)hitColor[0]))
		{
			tmpRes->state_NewState = ResState::OOB;
			return tmpRes->state_energy;
		}
		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);

		// depth derivatives.
		float dxInterp = hitColor[1] * Calib->fxl();
		float dyInterp = hitColor[2] * Calib->fyl();
		float d_idepth = derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale);

		hw *= weights[idx] * weights[idx];

		Hdd += (hw * d_idepth) * d_idepth;
		bd += (hw * residual) * d_idepth;
	}

	if (energyLeft > energyTH * outlierTHSlack)
	{
		energyLeft = energyTH * outlierTHSlack;
		tmpRes->state_NewState = ResState::OUTLIER;
	}
	else
	{
		tmpRes->state_NewState = ResState::IN;
	}

	tmpRes->state_NewEnergy = energyLeft;
	return energyLeft;
}

} // namespace FSLAM
