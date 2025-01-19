#include "Example3Cuda.h"
#include "cuda_functions.cuh"  // Подключаем ваш заголовочный файл с CUDA

// Реализация экспортируемой функции
extern "C" EXAMPLE3CUDA_API Offset RunGeneticAlgorithm(const img* mainImage, const img* partImage, int generations, int populationSize, int numBest, int maxMutation) {
    // Вызываем ваш генетический алгоритм на CUDA
    return geneticAlgorithmCUDA(mainImage, partImage, generations, populationSize, numBest, maxMutation);
}