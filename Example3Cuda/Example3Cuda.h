#ifndef EXAMPLE3CUDA_H
#define EXAMPLE3CUDA_H

#ifdef EXAMPLE3CUDA_EXPORTS
#define EXAMPLE3CUDA_API __declspec(dllexport)
#else
#define EXAMPLE3CUDA_API __declspec(dllimport)
#endif

#include "cuda_functions.cuh"  // Подключаем ваш заголовочный файл с CUDA

// Объявляем функцию, которая будет экспортироваться
extern "C" EXAMPLE3CUDA_API Offset RunGeneticAlgorithm(const img* mainImage, const img* partImage, int generations, int populationSize, int numBest, int maxMutation);

#endif // EXAMPLE3CUDA_H