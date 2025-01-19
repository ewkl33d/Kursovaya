#include <algorithm>
#include "cuda_functions.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include <ctime>

// Атомарный максимум для double
__device__ double atomicMax(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ double calculateFitnessCUDA(const img* mainImage, const img* partImage, int offsetX, int offsetY) {
    double match = 0;
    int count = 0;

    for (int j = 0; j < partImage->height; ++j) {
        for (int i = 0; i < partImage->width; ++i) {
            int mainX = i + offsetX;
            int mainY = j + offsetY;

            if (mainX >= 0 && mainX < mainImage->width && mainY >= 0 && mainY < mainImage->height) {
                unsigned mainPixel = mainImage->image[mainY * mainImage->width + mainX];
                unsigned partPixel = partImage->image[j * partImage->width + i];
                if (mainPixel == partPixel) {
                    match++;
                }
                count++;
            }
        }
    }
    return count > 0 ? match / count : 0;
}

__global__ void initializePopulationCUDA(Offset* population, int populationSize, const img* mainImage, const img* partImage, int maxOffsetX, int maxOffsetY, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Инициализация генератора случайных чисел для каждого потока
    curandState state;
    curand_init(seed, idx, 0, &state);

    if (idx < populationSize) {
        // Генерация случайных чисел с помощью curand_uniform
        population[idx].x = (int)(curand_uniform(&state) * maxOffsetX);
        population[idx].y = (int)(curand_uniform(&state) * maxOffsetY);
        population[idx].fitness = 0.0;
    }
}

__global__ void calculateFitnessCUDA(Offset* population, int populationSize, const img* mainImage, const img* partImage, Offset* bestOverall) {
    __shared__ Offset localBest;
    localBest.fitness = 0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize) {
        population[idx].fitness = calculateFitnessCUDA(mainImage, partImage, population[idx].x, population[idx].y);

        if (population[idx].fitness > localBest.fitness) {
            localBest = population[idx];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicMax(&bestOverall->fitness, localBest.fitness);
    }
}

__global__ void mutateCUDA(Offset* population, int populationSize, int maxMutation, int maxOffsetX, int maxOffsetY, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Инициализация генератора случайных чисел для каждого потока
    curandState state;
    curand_init(seed, idx, 0, &state);

    if (idx < populationSize) {
        // Генерация случайных чисел с помощью curand_uniform
        population[idx].x += (int)(curand_uniform(&state) * (2 * maxMutation + 1)) - maxMutation;
        population[idx].y += (int)(curand_uniform(&state) * (2 * maxMutation + 1)) - maxMutation;

        population[idx].x = max(0, min(population[idx].x, maxOffsetX));
        population[idx].y = max(0, min(population[idx].y, maxOffsetY));
    }
}

__global__ void crossoverCUDA(Offset* population, Offset* parents, int populationSize, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Инициализация генератора случайных чисел для каждого потока
    curandState state;
    curand_init(seed, idx, 0, &state);

    if (idx < populationSize) {
        int parent1_idx = (int)(curand_uniform(&state) * populationSize);
        int parent2_idx = (int)(curand_uniform(&state) * populationSize);

        Offset parent1 = parents[parent1_idx];
        Offset parent2 = parents[parent2_idx];

        population[idx].x = (parent1.x + parent2.x) / 2;
        population[idx].y = (parent1.y + parent2.y) / 2;
        population[idx].fitness = 0.0;
    }
}

Offset geneticAlgorithmCUDA(const img* mainImage, const img* partImage, int generations, int populationSize, int numBest, int maxMutation) {
    Offset* d_population;
    Offset* d_bestOverall;
    img* d_mainImage;
    img* d_partImage;

    int maxOffsetX = mainImage->width - partImage->width;
    int maxOffsetY = mainImage->height - partImage->height;

    // Выделение памяти на GPU
    cudaMalloc(&d_population, sizeof(Offset) * populationSize);
    cudaMalloc(&d_bestOverall, sizeof(Offset));
    cudaMalloc(&d_mainImage, sizeof(img));
    cudaMalloc(&d_partImage, sizeof(img));

    cudaMemcpy(d_mainImage, mainImage, sizeof(img), cudaMemcpyHostToDevice);
    cudaMemcpy(d_partImage, partImage, sizeof(img), cudaMemcpyHostToDevice);

    // Инициализация популяции
    int threadsPerBlock = 256;
    int blocksPerGrid = (populationSize + threadsPerBlock - 1) / threadsPerBlock;

    // Генерация seed для случайных чисел
    unsigned long long seed = time(0);

    initializePopulationCUDA << <blocksPerGrid, threadsPerBlock >> > (d_population, populationSize, d_mainImage, d_partImage, maxOffsetX, maxOffsetY, seed);

    Offset bestOverall = { 0, 0, 0.0 };
    cudaMemcpy(d_bestOverall, &bestOverall, sizeof(Offset), cudaMemcpyHostToDevice);

    for (int generation = 0; generation < generations; ++generation) {
        calculateFitnessCUDA << <blocksPerGrid, threadsPerBlock >> > (d_population, populationSize, d_mainImage, d_partImage, d_bestOverall);
        mutateCUDA << <blocksPerGrid, threadsPerBlock >> > (d_population, populationSize, maxMutation, maxOffsetX, maxOffsetY, seed);
        crossoverCUDA << <blocksPerGrid, threadsPerBlock >> > (d_population, d_population, populationSize, seed);
    }

    // Копирование результата обратно на CPU
    cudaMemcpy(&bestOverall, d_bestOverall, sizeof(Offset), cudaMemcpyDeviceToHost);

    // Освобождение памяти
    cudaFree(d_population);
    cudaFree(d_bestOverall);
    cudaFree(d_mainImage);
    cudaFree(d_partImage);

    return bestOverall;
}