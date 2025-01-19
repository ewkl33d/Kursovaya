#include <algorithm>
#include "cuda_functions.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cuda.h>
#include <ctime>

// Инициализация популяции
__global__ void initializePopulationCUDA(Offset* population, int populationSize, int maxOffsetX, int maxOffsetY, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Инициализация генератора случайных чисел
    curandState state;
    curand_init(seed, idx, 0, &state);

    if (idx < populationSize) {
        population[idx].x = (int)(curand_uniform(&state) * maxOffsetX);
        population[idx].y = (int)(curand_uniform(&state) * maxOffsetY);
        population[idx].fitness = 0.0;
    }
}

// Вычисление fitness
__global__ void calculateFitnessCUDA(Offset* population, int populationSize, const unsigned* mainImage, int mainWidth, int mainHeight, const unsigned* partImage, int partWidth, int partHeight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < populationSize) {
        double match = 0;
        double count = 0;

        // Координаты смещения
        int offsetX = population[idx].x;
        int offsetY = population[idx].y;

        // Вычисление fitness
        for (int j = 0; j < partHeight; ++j) {
            for (int i = 0; i < partWidth; ++i) {
                int mainX = i + offsetX;
                int mainY = j + offsetY;

                // Проверка границ
                if (mainX >= 0 && mainX < mainWidth && mainY >= 0 && mainY < mainHeight) {
                    unsigned mainPixel = mainImage[mainY * mainWidth + mainX];
                    unsigned partPixel = partImage[j * partWidth + i];
                    if (mainPixel == partPixel) {
                        match++;
                    }
                    count++;
                }
            }
        }

        // Рассчитываем fitness
        population[idx].fitness = (count > 0) ? (match / count) : 0.0;
    }
}

// Мутация
__global__ void mutateCUDA(Offset* population, int populationSize, int maxMutation, int maxOffsetX, int maxOffsetY, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Инициализация генератора случайных чисел
    curandState state;
    curand_init(seed, idx, 0, &state);

    if (idx < populationSize) {
        population[idx].x += (int)(curand_uniform(&state) * (2 * maxMutation + 1)) - maxMutation;
        population[idx].y += (int)(curand_uniform(&state) * (2 * maxMutation + 1)) - maxMutation;

        // Ограничение в пределах изображения
        population[idx].x = max(0, min(population[idx].x, maxOffsetX));
        population[idx].y = max(0, min(population[idx].y, maxOffsetY));
    }
}

// Селекция лучших особей на GPU
__global__ void selectBestCUDA(Offset* population, Offset* bestIndividuals, int populationSize, int numBest) {
    extern __shared__ Offset sharedPopulation[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < populationSize) {
        // Копируем популяцию в shared memory
        sharedPopulation[threadIdx.x] = population[idx];
    }
    __syncthreads();

    // Упорядочивание по fitness внутри блока
    for (int i = threadIdx.x; i < numBest; i += blockDim.x) {
        for (int j = i + 1; j < populationSize; ++j) {
            if (sharedPopulation[j].fitness > sharedPopulation[i].fitness) {
                Offset temp = sharedPopulation[i];
                sharedPopulation[i] = sharedPopulation[j];
                sharedPopulation[j] = temp;
            }
        }
    }
    __syncthreads();

    // Записываем лучших в global memory
    if (threadIdx.x < numBest) {
        bestIndividuals[threadIdx.x] = sharedPopulation[threadIdx.x];
    }
}

// Генетический алгоритм
Offset geneticAlgorithmCUDA(const img* mainImage, const img* partImage, int generations, int populationSize, int numBest, int maxMutation) {
    Offset* d_population;
    Offset* d_bestIndividuals;
    unsigned* d_mainPixels;
    unsigned* d_partPixels;

    // Размеры изображения
    int maxOffsetX = mainImage->width - partImage->width;
    int maxOffsetY = mainImage->height - partImage->height;

    // Выделение памяти на GPU
    cudaMalloc(&d_population, sizeof(Offset) * populationSize);
    cudaMalloc(&d_bestIndividuals, sizeof(Offset) * numBest);
    cudaMalloc(&d_mainPixels, sizeof(unsigned) * mainImage->width * mainImage->height);
    cudaMalloc(&d_partPixels, sizeof(unsigned) * partImage->width * partImage->height);

    // Копирование изображения
    cudaMemcpy(d_mainPixels, mainImage->image, sizeof(unsigned) * mainImage->width * mainImage->height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_partPixels, partImage->image, sizeof(unsigned) * partImage->width * partImage->height, cudaMemcpyHostToDevice);

    // Инициализация популяции
    int threadsPerBlock = 256;
    int blocksPerGrid = (populationSize + threadsPerBlock - 1) / threadsPerBlock;
    unsigned long long seed = time(0);

    initializePopulationCUDA << <blocksPerGrid, threadsPerBlock >> > (d_population, populationSize, maxOffsetX, maxOffsetY, seed);

    Offset bestOverall = { 0, 0, 0.0 };

    for (int generation = 0; generation < generations; ++generation) {
        // Вычисление fitness
        calculateFitnessCUDA << <blocksPerGrid, threadsPerBlock >> > (
            d_population, populationSize, d_mainPixels, mainImage->width, mainImage->height, d_partPixels, partImage->width, partImage->height);

        // Селекция лучших особей
        selectBestCUDA << <1, threadsPerBlock, sizeof(Offset)* populationSize >> > (
            d_population, d_bestIndividuals, populationSize, numBest);

        // Мутация
        mutateCUDA << <blocksPerGrid, threadsPerBlock >> > (d_population, populationSize, maxMutation, maxOffsetX, maxOffsetY, seed);

        // Копирование лучшей особи
        Offset* bestIndividuals = new Offset[numBest];
        cudaMemcpy(bestIndividuals, d_bestIndividuals, sizeof(Offset) * numBest, cudaMemcpyDeviceToHost);

        for (int i = 0; i < numBest; ++i) {
            if (bestIndividuals[i].fitness > bestOverall.fitness) {
                bestOverall = bestIndividuals[i];
            }
        }

        delete[] bestIndividuals;
    }

    // Освобождение памяти
    cudaFree(d_population);
    cudaFree(d_bestIndividuals);
    cudaFree(d_mainPixels);
    cudaFree(d_partPixels);

    std::cout << "Best: x = " << bestOverall.x << ", y = " << bestOverall.y << ", fitness = " << bestOverall.fitness << "\n";
    return bestOverall;
}
