#include "Example3Cuda.h"
#include "cuda_functions.cuh"  // ���������� ��� ������������ ���� � CUDA

// ���������� �������������� �������
extern "C" EXAMPLE3CUDA_API Offset RunGeneticAlgorithm(const img* mainImage, const img* partImage, int generations, int populationSize, int numBest, int maxMutation) {
    // �������� ��� ������������ �������� �� CUDA
    return geneticAlgorithmCUDA(mainImage, partImage, generations, populationSize, numBest, maxMutation);
}