#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <cmath>
#include "cuda_runtime.h"
#include <functional>
#include <vector>
#include "device_launch_parameters.h"

// Время выполнения на CPU: 23 ms
// Время выполнения на CUDA простого алгоритма euler_simple: 16.9289 ms
// Время выполнения на CUDA более сложного алгоритма euler_shared: 25.2861 ms

#define N 5120          // Количество строк (размер массива параметра a)
#define M 5000          // Количество шагов метода Эйлера
#define BLOCK_SIZE 1024

__global__ void euler_simple(float* a, float* result, float y0, float h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = 0.0f;
    float y = y0;
    for (int j = 0; j < M; j++) {
        y += h * sinf((x + y) / a[i]);
        x += h;
        result[i * M + j] = y; // Сохраняем весь процесс вычислений
    }
}

__global__ void euler_shared(float* a, float* result, float y0, float h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float a_ = a[i];                             // +1
    float x = 0.0f;                              // +1
    float y = y0;                                // +1
    for (int j = 0; j < M; j++) {                // +2 * M
        y += h * sinf((x + y) / a_);             // +~30 * M       
        x += h;                                  // +M
        result[i * M + j] = y;                   // +3*M + M * (запись в память)
    }
                                                 // Итого: 3 + M * (36 + запись в память)
                                                 // euler_shared бе взываема 5120-жды. 
                                                 // Абие же сотворим число общее : 5120 * (3 + M * (36 + запись в память)) ~ 2^10 * 5 * 2^10 * 5 * 36 = 943 718 400
                                                 // Имамы бо число другое терафлопсовое для RTX 2060: 6.5 * 10^12
                                                 // Время бе: 943 718 400 / (6.5 * 10^12) = 1.45 * 10^-4 = 0.000145 секунд = 0.15 ms 
                                                 // Помянухом к тому, братие, такожде и память: 5120 * M * (запись в память) = 25.6 * 10^6 * (запись в память)
                                                 // Егда имамы запись в память ~100 флопс, дадеся нам число велие: (25.6 * 10^6 * 100) / (6.5 * 10^12) = 3.93 * 10^-4 = 0.000393 секунд = 0.393 ms 
                                                    
                                                 // Купно же: 0.393 ms + 0.15 ms ~ 0.55ms
        
}

void checkCorrectness(const std::vector<float>& a, const std::vector<float>& result_cpu, const std::vector<float>& result_gpu) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int idx = i * M + j;
            if (fabs(result_cpu[idx] - result_gpu[idx]) > 1e-5f) {
                std::cout << "Mismatch at " << i << ", step " << j << ": " << result_cpu[idx] << " vs " << result_gpu[idx] << std::endl;
                return;
            }
        }
    }
    std::cout << "Results are correct!" << std::endl;
}

void euler_cpu(const std::vector<float>& a, std::vector<float>& result, float y0, float h) {
    for (int i = 0; i < N; i++) {
        float x = 0.0f;
        float y = y0;
        for (int j = 0; j < M; j++) {
            y += h * sinf((x + y) / a[i]);
            x += h;
            result[i * M + j] = y;
        }
    }
}

int main() {
    float* d_a;
    float* d_result;
    std::vector<float> a(N), result_cpu(N * M), result_gpu(N * M);
    float y0 = 1.0f, h = 1.0f / M;

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f + static_cast<float>(rand()) / RAND_MAX;
    }


#pragma region CPU
    clock_t startCPU, stopCPU;
    float elapsedTimeCPU;
    startCPU = clock();
    euler_cpu(a, result_cpu, y0, h);
    stopCPU = clock();
    elapsedTimeCPU = (double)(stopCPU - startCPU) / CLOCKS_PER_SEC;
    std::cout << "CPU algorithm time = " << elapsedTimeCPU * 1000 << " ms\n";

#pragma endregion

#pragma region GPU
    cudaEvent_t startCUDA, stopCUDA;
    float elapsedTimeCUDA;
    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_result, N * M * sizeof(float));
    cudaMemcpy(d_a, a.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEventRecord(startCUDA, 0);
    euler_simple << <grid, block >> > (d_a, d_result, y0, h);
    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);
    std::cout << "CUDA simple algorithm time = " << elapsedTimeCUDA << " ms\n";

    cudaMemcpy(result_gpu.data(), d_result, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    checkCorrectness(a, result_cpu, result_gpu);

    cudaEventRecord(startCUDA, 0);
    euler_shared << <grid, block >> > (d_a, d_result, y0, h);
    cudaMemcpy(result_gpu.data(), d_result, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);
    std::cout << "CUDA algorithm with shared memory usage time = " << elapsedTimeCUDA << " ms\n";

    checkCorrectness(a, result_cpu, result_gpu);
    cudaFree(d_a);
    cudaFree(d_result);

#pragma endregion
    return 0;
}
