#include <immintrin.h>
#include <cstdint> // uint32_t
#include <cstdlib> // 包含 rand() 和 RAND_MAX
#include <iostream>
#include <chrono> // 用于计时

using namespace std;
using namespace std::chrono;

void plain__tmm(float *A, float *B, float *C, uint64_t M, uint64_t L, uint64_t N) {
    for (uint64_t i = 0; i < M; i++) {
        for (uint64_t j = 0; j < N; j++) {
            float accum = 0.0f;
            for (uint64_t k = 0; k < L; k++) {
                accum += A[i * L + k] * B[k * N + j]; // 修正了索引
            }
            C[i * N + j] = accum;
        }
    }
}

//AVX是256位的
//SSE是128位的
//_mm256_castps256_ps128(sums) 这是一个SSE指令
//_mm_cvtss_f32 提取第一个的浮点数
inline float hsum_avx(__m256 x) 
{
    __m256 shuf = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1)); // 交换相邻元素
    __m256 sums = _mm256_add_ps(x, shuf); // 部分求和

    shuf = _mm256_shuffle_ps(sums, sums, _MM_SHUFFLE(1, 0, 3, 2)); // 交换相邻对
    sums = _mm256_add_ps(sums, shuf);

    shuf = _mm256_permute2f128_ps(sums, sums, 1); // 高 128-bit 移到低位
    sums = _mm256_add_ps(sums, shuf);

    return _mm_cvtss_f32(_mm256_castps256_ps128(sums)); // 提取最低元素
}

void avx2_tmm(float *A, float *B, float *C, uint64_t M, uint64_t L, uint64_t N) 
{
    for (uint64_t i = 0; i < M; i++) {
        for (uint64_t j = 0; j < N; j++) 
        {
            __m256 X = _mm256_setzero_ps();
            for (uint64_t k = 0; k < L; k++) 
            {
                const __m256 AV = _mm256_loadu_ps(A + i * L + k); // 使用 _mm256_loadu_ps 处理不对齐的内存
                const __m256 BV = _mm256_loadu_ps(B + j * L + k); // 使用 _mm256_loadu_ps 处理不对齐的内存
                X = _mm256_fmadd_ps(AV, BV, X);
            }
            C[i * N + j] = hsum_avx(X);
        }                                                                                                                                                       
                                                                   
    }
}

void initialize_matrices(float *A, float *B, uint64_t M, uint64_t L, uint64_t N) 
{
    for (uint64_t i = 0; i < M * L; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX; // 随机填充 A
    }
    for (uint64_t i = 0; i < L * N; i++) {
        B[i] = static_cast<float>(rand()) / RAND_MAX; // 随机填充 B
    }
}

int main() 
{
    uint64_t M = 1024, L = 1024, N = 1024;
    float *A = new float[M * L];
    float *B = new float[L * N];
    float *C1 = new float[M * N];
    float *C2 = new float[M * N];

    // 计时普通矩阵乘法
    auto start1 = high_resolution_clock::now();
    plain__tmm(A, B, C1, M, L, N);
    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(stop1 - start1);
    cout << "Plain TMM time: " << duration1.count() << " ms" << endl;

    // 计时 AVX2 加速的矩阵乘法
    auto start2 = high_resolution_clock::now();
    avx2_tmm(A, B, C2, M, L, N);
    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(stop2 - start2);
    cout << "AVX2 TMM time: " << duration2.count() << " ms" << endl;

    // 释放内存
    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;

    return 0;
}

//Plain TMM time: 2485 ms
//AVX2 TMM time: 982 ms
//2.53倍
