#include "GlobalTypes.h"

namespace FSLAM
{

struct CalibData
{
public:
    int Width;
    int Height;
    
    // std::vector<int> PyrWidth;
    // std::vector<int> PyrHeight;
    VecC value_zero;
    VecC value_scaled;
    VecCf value_scaledf;
    VecCf value_scaledi;
    VecC value;
    VecC step;
    VecC step_backup;
    VecC value_backup;
    VecC value_minus_value_zero;

    inline CalibData(int _Width, int _Height, Mat33f K , float* binvL, float* binvR ,float ScaleFactor) 
    {
        // PyrWidth.push_back(Width);
        // PyrHeight.push_back(Height);
        // for (int i = 1; i < PyrSize; i++)
        // {
        //     PyrWidth.push_back(cvRound((float)PyrWidth[i-1]/ScaleFactor));
        //     PyrHeight.push_back(cvRound((float)PyrHeight[i-1]/ScaleFactor));
        // }
        Width = _Width;
        Height = _Height;
        VecC initial_value = VecC::Zero();
		initial_value[0] = K(0,0);
		initial_value[1] = K(1,1);
		initial_value[2] = K(0,2);
		initial_value[3] = K(1,2);
		setValueScaled(initial_value);
        value_zero = value;
        value_minus_value_zero.setZero();

        for (int i = 0; i < 256; i++)
        {
            BinvL[i] = BL[i] = i; // set gamma function to identity
            BinvR[i] = BR[i] = i; // set gamma function to identity
        }

        UpdateGamma(binvL);
        UpdateGamma(binvR,false);
    };

    inline float& fxl() {return value_scaledf[0];}
    inline float& fyl() {return value_scaledf[1];}
    inline float& cxl() {return value_scaledf[2];}
    inline float& cyl() {return value_scaledf[3];}
    inline float& fxli() {return value_scaledi[0];}
    inline float& fyli() {return value_scaledi[1];}
    inline float& cxli() {return value_scaledi[2];}
    inline float& cyli() {return value_scaledi[3];}
    
    inline void setValueScaled(const VecC &value_scaled)
    {
        this->value_scaled = value_scaled;
        this->value_scaledf = this->value_scaled.cast<float>();
        value[0] = SCALE_F_INVERSE * value_scaled[0];
        value[1] = SCALE_F_INVERSE * value_scaled[1];
        value[2] = SCALE_C_INVERSE * value_scaled[2];
        value[3] = SCALE_C_INVERSE * value_scaled[3];

        this->value_minus_value_zero = this->value - this->value_zero;
        this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
        this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
        this->value_scaledi[2] = -this->value_scaledf[2] / this->value_scaledf[0];
        this->value_scaledi[3] = -this->value_scaledf[3] / this->value_scaledf[1];
    };

    inline void setValue(const VecC &value)
	{
		this->value = value;
		value_scaled[0] = SCALE_F * value[0];
		value_scaled[1] = SCALE_F * value[1];
		value_scaled[2] = SCALE_C * value[2];
		value_scaled[3] = SCALE_C * value[3];

		this->value_scaledf = this->value_scaled.cast<float>();
		this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
		this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
		this->value_scaledi[2] = - this->value_scaledf[2] / this->value_scaledf[0];
		this->value_scaledi[3] = - this->value_scaledf[3] / this->value_scaledf[1];
		this->value_minus_value_zero = this->value - this->value_zero;
	};

    float BinvR[256];
	float BR[256];

    float BinvL[256];
	float BL[256];

    inline void UpdateGamma(float *BInv, bool isRight = false)
    {
         if (BInv == 0)
            return;
        
        if (isRight)
        {
            memcpy(BinvR, BInv, sizeof(float) * 256);

            // invert.
            for (int i = 1; i < 255; i++)
            {
                //update gamma once if havecalib or update gamma used for slam whenever the gamma estimate changes.
                for (int s = 1; s < 255; s++)
                {
                    if (BInv[s] <= i && BInv[s + 1] >= i)
                    {
                        BR[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
                        break;
                    }
                }
            }
            BR[0] = 0;
            BR[255] = 255;
        }
        else
        {
            memcpy(BinvL, BInv, sizeof(float) * 256);

            // invert.
            for (int i = 1; i < 255; i++)
            {
                //update gamma once if havecalib or update gamma used for slam whenever the gamma estimate changes.
                for (int s = 1; s < 255; s++)
                {
                    if (BInv[s] <= i && BInv[s + 1] >= i)
                    {
                        BL[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
                        break;
                    }
                }
            }
            BL[0] = 0;
            BL[255] = 255;
        }
    }

    EIGEN_STRONG_INLINE float getBLGradOnly(float color)
	{
		int c = color+0.5f;
		if(c<5) c=5;
		if(c>250) c=250;
		return BL[c+1]-BL[c];
	}

	EIGEN_STRONG_INLINE float getBInvLGradOnly(float color)
	{
		int c = color+0.5f;
		if(c<5) c=5;
		if(c>250) c=250;
		return BinvL[c+1]-BinvL[c];
	}

        EIGEN_STRONG_INLINE float getBRGradOnly(float color)
	{
		int c = color+0.5f;
		if(c<5) c=5;
		if(c>250) c=250;
		return BR[c+1]-BR[c];
	}

	EIGEN_STRONG_INLINE float getBInvRGradOnly(float color)
	{
		int c = color+0.5f;
		if(c<5) c=5;
		if(c>250) c=250;
		return BinvR[c+1]-BinvR[c];
	}

    inline ~CalibData() {}
};

}