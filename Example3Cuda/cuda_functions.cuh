#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct Offset {
    int x, y;
    double fitness;
};

struct img {
    int width;
    int height;
    unsigned* image;
};

// Объявления функций и кернелов
__global__ void initializePopulationCUDA(Offset* population, int populationSize, const img* mainImage, const img* partImage, int maxOffsetX, int maxOffsetY);
__global__ void calculateFitnessCUDA(Offset* population, int populationSize, const img* mainImage, const img* partImage, Offset* bestOverall);
__global__ void mutateCUDA(Offset* population, int populationSize, int maxMutation, int maxOffsetX, int maxOffsetY);
__global__ void crossoverCUDA(Offset* population, Offset* parents, int populationSize);
Offset geneticAlgorithmCUDA(const img* mainImage, const img* partImage, int generations, int populationSize, int numBest, int maxMutation);