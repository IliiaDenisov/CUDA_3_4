#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <cmath>
#include "cuda_runtime.h"
#include <functional>
#include <vector>
#include "device_launch_parameters.h"

// CPU time                             : 624 ms
// CUDA euler_simple time               : 3.4304 ms
// CUDA euler_shared_memory_usage time  : 0.9532 ms

#define N 5120                          // Количество строк (размер массива параметра a)
#define M 5000                          // Количество шагов метода Эйлера
#define BLOCK_SIZE 10

// COMPLEXITY ANALYSIS
// euler_shared is called N (5120) times.
// Time_res = max(Time_mem, Time_operations)
// Time_operations = N * (3 + 36 * M) / FLOPS = 1.4178e-4 sec = 0.000014178 sec = 0.14178 ms
// Time_memory = (N * M * 4) / MemoryBandwidth = (102 * 10^6) / ((2 * clockRate * BusWidth) / 8) =
// =  (102 * 10^6) / ((2 * 7 * 10^3 * 10^6 * 192) / 8) = 3.047e-4 = 0.3047 ms
// Time_res = max(0.14178 ms, 0.3047 m) = 0.3047 ms
// therefore, ideal time is 0.3047 ms
// test time is: 0.9532 ms

__global__ void euler_simple(float* a, float* result, float y0, float h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = 0.0f;
    float y = y0;
    for (int j = 0; j < M; j++) {
        y += h * sinf((x + y) / a[i]);
        x += h;
        result[i * M + j] = y;
    }
}

__global__ void euler_shared_memory_usage(float* a, float* result, float y0, float h)
{
    // column-major implementation
    __shared__ float a_s[512];
    __shared__ float y_s[512];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a_s[threadIdx.x] = a[idx];
    y_s[threadIdx.x] = y0;
    __syncthreads();

    float x = 0.0f;
    for (size_t i = 0; i < M; i++)
    {
        float y_registry = y_s[threadIdx.x];
        y_s[threadIdx.x] = y_registry + h * __sinf((x + y_registry) / a_s[threadIdx.x]);

        result[i * N + idx] = y_s[threadIdx.x];
        __syncthreads();

        x += h;
    }
}

void checkCorrectnessOld(const std::vector<float>& a, const std::vector<float>& result_cpu, const std::vector<float>& result_gpu) {
    // column-major implementation
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

void checkCorrectness(const std::vector<float>& a, const std::vector<float>& result_cpu, const std::vector<float>& result_gpu) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int idx = i * M + j;
            int idx_2 = j * N + i;
            if (fabs(result_cpu[idx] - result_gpu[idx_2]) > 1e-5f) {
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
    cudaMemcpy(result_gpu.data(), d_result, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "CUDA simple algorithm time = " << elapsedTimeCUDA << " ms\n";
    checkCorrectnessOld(a, result_cpu, result_gpu);

    cudaEventRecord(startCUDA, 0);
    euler_shared_memory_usage << < BLOCK_SIZE, N / BLOCK_SIZE >> > (d_a, d_result, y0, h);
    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);
    cudaMemcpy(result_gpu.data(), d_result, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "CUDA algorithm with shared memory usage time = " << elapsedTimeCUDA << " ms\n";
    checkCorrectness(a, result_cpu, result_gpu);

    cudaFree(d_a);
    cudaFree(d_result);

#pragma endregion
    return 0;
}
