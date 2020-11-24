#pragma once
#include "util/NumType.h"

#if _WIN32 || _WIN64
#if _WIN64
#define Env64
#else
#define Env32
#endif
#endif

// Check GCC
#if __GNUC__
#if __x86_64__ || __ppc64__
#define Env64
#else
#define Env32
#endif
#endif

namespace HSLAM
{
    class Matcher
    {
    private:
        float mfNNratio;
        bool mbCheckOrientation;
        
        inline float RadiusByViewingCos(const float &viewCos)
        {
            if (viewCos > 0.998)
                return 2.5;
            else
                return 4.0;
        }

        inline void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3)
        {
            int max1=0;
            int max2=0;
            int max3=0;

            for(int i=0; i<L; i++)
            {
                const int s = histo[i].size();
                if(s>max1)
                {
                    max3=max2; max2=max1; max1=s;
                    ind3=ind2; ind2=ind1; ind1=i;
                }
                else if(s>max2)
                {
                    max3 = max2; max2 = s; ind3=ind2; ind2=i;
                }
                else if(s>max3)
                {
                    max3=s; ind3=i;
                }
            }

            if(max2<0.1f*(float)max1)
            {
                ind2=-1; 
                ind3=-1;
            }
            else if(max3<0.1f*(float)max1)
            {
                ind3=-1;
            }
        }

    public:
        inline Matcher(float nnratio = 0.6, bool checkOri = true)
        {
            mfNNratio = nnratio;
            mbCheckOrientation = checkOri;
        }

        ~Matcher() {}


        inline static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
        {
            int dist = 0;
#ifdef __SSE2__

#ifdef Env64
            const unsigned long long int *pa = a.ptr<unsigned long long int>();
            const unsigned long long int *pb = b.ptr<unsigned long long int>();

            for (int i = 0; i < 4; i++, pa++, pb++)
            {
                unsigned int v = *pa ^ *pb; //can't do this
                dist += _mm_popcnt_u64(v);
            }
            return dist;
#else
            const int *pa = a.ptr<int32_t>();
            const int *pb = b.ptr<int32_t>();
            for (int i = 0; i < 8; i++, pa++, pb++)
            {
                unsigned int v = *pa ^ *pb; //can't do this
                dist += _mm_popcnt_u32(v);
            }
            return dist;
#endif

#else
            const int *pa = a.ptr<int32_t>();
            const int *pb = b.ptr<int32_t>();

            for (int i = 0; i < 8; i++, pa++, pb++)
            {
                unsigned int v = *pa ^ *pb;
                v = v - ((v >> 1) & 0x55555555);
                v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
                dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
            }
            return dist;

#endif
        }

        static const int TH_LOW;
        static const int TH_HIGH;
        static const int HISTO_LENGTH;
    };
}