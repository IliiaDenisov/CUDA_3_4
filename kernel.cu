﻿#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <cmath>
#include "cuda_runtime.h"
#include <functional>
#include <vector>
#include "device_launch_parameters.h"

// Время выполнения на CPU: 23 ms
// Время выполнения на CUDA простого алгоритма euler_simple: 4.6121 ms
// Время выполнения на CUDA более сложного алгоритма euler_shared: 1.16752 ms

#define N 1024          // Количество строк (размер массива параметра a)
#define M 1000          // Количество шагов метода Эйлера
#define BLOCK_SIZE 256

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
    __shared__ float shared_a[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    shared_a[threadIdx.x] = a[i];
    __syncthreads();

    float x = 0.0f;
    float y = y0;
    for (int j = 0; j < M; j++) {
        y += h * sinf((x + y) / shared_a[threadIdx.x]);
        x += h;
        result[i * M + j] = y; // Сохраняем весь процесс вычислений
    }
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
