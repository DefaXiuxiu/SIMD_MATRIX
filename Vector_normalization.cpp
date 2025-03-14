#include <random>       // prng
#include <cstdint>      // uint32_t
#include <iostream>     // std::cout
#include <immintrin.h>  // AVX intrinsics

// timers distributed with this book
#include "include/hpc_helpers.hpp"

void aos_init(float * xyz, uint64_t length) 
{

    std::mt19937 engine(42);
    std::uniform_real_distribution<float> density(-1, 1);

    /*
    int 是带符号的整数类型，意味着它可以表示负数和正数。例如，32位的 int 可以表示从 -2,147,483,648 到 2,147,483,647 的整数。
    uint64_t 是无符号的 64 位整数类型，意味着它只表示非负数（零和正数）。它的取值范围是从 0 到 18,446,744,073,709,551,615。
    */
    for (uint64_t i = 0; i < 3*length; i++)
        xyz[i] = density(engine);
 
    /*
    for(int i=0;i<9;i++)
    {
        std::cout<<xyz[i]<<" ";
    }
    std::cout<<std::endl;
    */
}

void avx_aos_norm(float * xyz, uint64_t length) 
{

    for (uint64_t i = 0; i < 3*length; i += 3*8) 
    {

        /////////////////////////////////////////////////////////////////////
        // AOS2SOA: XYZXYZXY ZXYZXYZX YZXYZXYZ --> XXXXXXX YYYYYYY ZZZZZZZZ
        /////////////////////////////////////////////////////////////////////

        // registers: NOTE: M is an SSE pointer (length 4)
        // 假设xyz有96个数据 i是24的倍数
        // __m128* 是四个数据为一组的 
        __m128 *M = (__m128*) (xyz+i);
        __m256 M03;
        __m256 M14;
        __m256 M25;

        // load lower halves
        // 将四个浮点数的数组 拓展成8个浮点数的数组
        // _mm256_castps128_ps256
        M03 = _mm256_castps128_ps256(M[0]);//1-4个浮点数 M[0]=1231
        M14 = _mm256_castps128_ps256(M[1]);//2-5        M[1]=2312

        /*
                std::cout<<"X: "<<std::endl;
        if(i<9)
        {
            for(int j=0;j<8;j++)
            {
                std::cout<<M14[j]<<" ";
            }
            std::cout<<std::endl;
        }
        */

        M25 = _mm256_castps128_ps256(M[2]);//           M[2]=3123
                                                      //M[3]=1231
                                                      //M[4]=2312
        // load upper halves
        M03 = _mm256_insertf128_ps(M03 ,M[3],1);// 在M03的后四个位置 1231 1231
        M14 = _mm256_insertf128_ps(M14 ,M[4],1);// 在M14的后四个位置 2312 2312
        M25 = _mm256_insertf128_ps(M25 ,M[5],1);// 在M25的后四个位置 3123 3123
        //第一组：

        // everyday I am shuffeling...
        __m256 XY = _mm256_shuffle_ps(M14, M25, _MM_SHUFFLE( 2,1,3,2));//M14的第三个数，第二个数，M25的第四个数 M2的第3个数 组成XY
                                                                       //1332
        /*
           std::cout<<"XY: "<<std::endl;
            if(i<9)
            {
                for(int j=0;j<8;j++)
                {
                    std::cout<<XY[j]<<" ";
                }
            std::cout<<std::endl;
            }    
            */                                                           
        __m256 YZ = _mm256_shuffle_ps(M03, M14, _MM_SHUFFLE( 1,0,2,1));// 2113
        __m256 X  = _mm256_shuffle_ps(M03, XY , _MM_SHUFFLE( 2,0,3,0));//3112
        /*
        std::cout<<"X: "<<std::endl;
        if(i<9)
        {
            for(int j=0;j<8;j++)
            {
                std::cout<<X[j]<<" ";
            }
            std::cout<<std::endl;
        }
            */

        __m256 Y  = _mm256_shuffle_ps(YZ , XY , _MM_SHUFFLE( 3,1,2,0));//3131
        __m256 Z  = _mm256_shuffle_ps(YZ , M25, _MM_SHUFFLE( 3,0,3,1));//3231

        /////////////////////////////////////////////////////////////////////
        // SOA computation
        /////////////////////////////////////////////////////////////////////

        // R <- X*X+Y*Y+Z*Z
        __m256 R = _mm256_add_ps(_mm256_mul_ps(X, X),
                   _mm256_add_ps(_mm256_mul_ps(Y, Y),
                                 _mm256_mul_ps(Z, Z)));
        // R <- 1/sqrt(R)
               R = _mm256_rsqrt_ps(R);

        // normalize vectors
        X = _mm256_mul_ps(X, R);
        Y = _mm256_mul_ps(Y, R);
        Z = _mm256_mul_ps(Z, R);

        /////////////////////////////////////////////////////////////////////
        // SOA2AOS: XXXXXXX YYYYYYY ZZZZZZZZ -> XYZXYZXY ZXYZXYZX YZXYZXYZ
        /////////////////////////////////////////////////////////////////////

        // everyday I am shuffeling...
        __m256 RXY = _mm256_shuffle_ps(X,Y, _MM_SHUFFLE(2,0,2,0));
        __m256 RYZ = _mm256_shuffle_ps(Y,Z, _MM_SHUFFLE(3,1,3,1));
        __m256 RZX = _mm256_shuffle_ps(Z,X, _MM_SHUFFLE(3,1,2,0));
        __m256 R03 = _mm256_shuffle_ps(RXY, RZX, _MM_SHUFFLE(2,0,2,0));
        __m256 R14 = _mm256_shuffle_ps(RYZ, RXY, _MM_SHUFFLE(3,1,2,0));
        __m256 R25 = _mm256_shuffle_ps(RZX, RYZ, _MM_SHUFFLE(3,1,3,1));

        // store in AOS (6*4=24)
        M[0] = _mm256_castps256_ps128(R03);
        M[1] = _mm256_castps256_ps128(R14);
        M[2] = _mm256_castps256_ps128(R25);
        M[3] = _mm256_extractf128_ps(R03, 1);
        M[4] = _mm256_extractf128_ps(R14, 1);
        M[5] = _mm256_extractf128_ps(R25, 1);
    }
}

void aos_check(float * xyz, uint64_t length) {

    for (uint64_t i = 0; i < 3*length; i += 3) {

        const float x = xyz[i+0];
        const float y = xyz[i+1];
        const float z = xyz[i+2];

        float rho = x*x+y*y+z*z;

        if ((rho-1)*(rho-1) > 1E-6)
            std::cout << "error too big at position "
                      << i << std::endl;
    }
}
int main () 
{

    //num_vectors 是 1UL << 28，表示 2 的 28 次方，等于 268435456（约 2.68 亿个向量）。
    const uint64_t num_vectors = 1UL << 28;
    //const uint64_t num_vectors = 48;
    const uint64_t num_bytes = 3*num_vectors*sizeof(float);

    TIMERSTART(alloc_memory)
    // _mm_malloc 用于按照指定的对齐方式（这里是 32 字节对齐）分配内存。num_bytes 指定了所需的内存大小，32 是对齐的字节数。
    // 32 字节对齐意味着内存地址是 32 的倍数，这对于 AVX 指令来说是必需的，因为 AVX 指令处理 256 位（32 字节）数据，且需要满足对齐要求。
    auto xyz = static_cast<float*>(_mm_malloc(num_bytes , 32));
    TIMERSTOP(alloc_memory)

    TIMERSTART(init)
    aos_init(xyz, num_vectors);
    TIMERSTOP(init)

    TIMERSTART(avx_aos_normalize)
    avx_aos_norm(xyz, num_vectors);
    TIMERSTOP(avx_aos_normalize)

    TIMERSTART(check)
    aos_check(xyz, num_vectors);
    TIMERSTOP(check)

    TIMERSTART(free_memory)
    _mm_free(xyz);
    TIMERSTOP(free_memory)
}