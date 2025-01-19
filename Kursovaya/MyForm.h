#pragma once
#include<windows.h> 
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <unordered_set> 
#include <string> 
#include <numeric>
#include <random>
#include <omp.h>
#include <chrono>

struct img
{
	int width;
	int height;
	unsigned * image;
};

struct Offset {
	int x, y;
	double fitness; 
};

namespace Kursovaya {

	double sequentialTime = 0.0;
	double openMPTime = 0.0;
	double cudaTime = 0.0;
	int numThreads = omp_get_max_threads();
void UpdateMetrics(System::Windows::Forms::TextBox^ sequentialTextBox,
	System::Windows::Forms::TextBox^ openMPTextBox,
	System::Windows::Forms::TextBox^ cudaTextBox) {
	if (sequentialTime > 0) {
		sequentialTextBox->Text = "Время: " + sequentialTime.ToString("F2") + " мс";
	}

	if (openMPTime > 0) {
		double speedupOpenMP = sequentialTime / openMPTime;
		double efficiencyOpenMP = speedupOpenMP / numThreads;

		openMPTextBox->Text = "Время: " + openMPTime.ToString("F2") + " мс \n" +
			"Ускорение: " + speedupOpenMP.ToString("F2") + "\n" +
			"Эффективность: " + (efficiencyOpenMP * 100).ToString("F2") + "%";
	}

	if (cudaTime > 0) {
		double speedupCUDA = sequentialTime / cudaTime;

		cudaTextBox->Text = "Время: " + cudaTime.ToString("F2") + " мс \n" +
			"Ускорение: " + speedupCUDA.ToString("F2");
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	double calculateFitness(const img* mainImage, const img* partImage, int offsetX, int offsetY) {
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

	std::vector<Offset> initializePopulation(int populationSize, const img* mainImage, const img* partImage) {
		std::vector<Offset> population;
		int maxOffsetX = mainImage->width - partImage->width;   
		int maxOffsetY = mainImage->height - partImage->height; 

		for (int i = 0; i < populationSize; ++i) {
			Offset individual;

			// Генерируем x и y в допустимых пределах
			individual.x = rand() % (maxOffsetX + 1);
			individual.y = rand() % (maxOffsetY + 1); 

			individual.fitness = 0.0;
			population.push_back(individual);
		}

		return population;
	}

	std::vector<Offset> selectBestWithSigmaScaling(const std::vector<Offset>& population, int numBest) {
		int N = population.size();

		double f_avg = std::accumulate(population.begin(), population.end(), 0.0,
			[](double sum, const Offset& individual) {
			return sum + individual.fitness;
		}) / N;

		double variance = std::accumulate(population.begin(), population.end(), 0.0,
			[f_avg](double sum, const Offset& individual) {
			return sum + (individual.fitness - f_avg) * (individual.fitness - f_avg);
		}) / N;
		double sigma = std::sqrt(variance);

		std::vector<double> scaledFitness(N);
		for (int i = 0; i < N; ++i) {
			scaledFitness[i] = 1 + (population[i].fitness - f_avg) / (2 * sigma);
			if (scaledFitness[i] < 0) {
				scaledFitness[i] = 0; 
			}
		}

		double totalFitness = std::accumulate(scaledFitness.begin(), scaledFitness.end(), 0.0);

		std::vector<double> probabilities(N);
		for (int i = 0; i < N; ++i) {
			probabilities[i] = scaledFitness[i] / totalFitness;
		}

		std::vector<Offset> selected;
		std::default_random_engine generator;
		std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());

		for (int i = 0; i < numBest; ++i) {
			int index = distribution(generator);
			selected.push_back(population[index]);
		}

		return selected;
	}

	Offset crossover(const Offset& parent1, const Offset& parent2) {
		Offset child;
		child.x = (parent1.x + parent2.x) / 2;
		child.y = (parent1.y + parent2.y) / 2;
		child.fitness = 0.0; 
		return child;
	}

	// Мутация
void mutate(Offset& individual, int maxMutation, const img* mainImage, const img* partImage) {
		individual.x += rand() % (2 * maxMutation + 1) - maxMutation;
		individual.y += rand() % (2 * maxMutation + 1) - maxMutation;

		individual.x = max(0, min(individual.x, mainImage->width - partImage->width));
		individual.y = max(0, min(individual.y, mainImage->height - partImage->height));
	}

Offset geneticAlgorithm(const img* mainImage, const img* partImage, int generations, int populationSize, int numBest, int maxMutation) {

	std::vector<Offset> population = initializePopulation(populationSize, mainImage, partImage);

	Offset bestOverall = { 0, 0, 0.0 };

	for (int generation = 0; generation < generations; ++generation) {

		std::cout << "Генерация " << generation << "\n";
		for (auto& individual : population) {
			individual.fitness = calculateFitness(mainImage, partImage, individual.x, individual.y);
			if (individual.fitness > bestOverall.fitness) {
				bestOverall = individual; 
			}

			if (individual.fitness == 1) {
				return individual;
			}
		}

		std::vector<Offset> best = selectBestWithSigmaScaling(population, numBest);
		std::vector<Offset> newPopulation;
		for (int i = 0; i < populationSize; ++i) {
			const Offset& parent1 = best[rand() % numBest];
			const Offset& parent2 = best[rand() % numBest];
			Offset child = crossover(parent1, parent2);
			mutate(child, maxMutation, mainImage, partImage);
			newPopulation.push_back(child);
		}

		if (generation != generations - 1) {
			population = newPopulation;
		}
	}

	std::cout << "Best: x = " << bestOverall.x << ", y = " << bestOverall.y << ", fitness = " << bestOverall.fitness << "\n";
	return bestOverall;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<Offset> initializePopulationOpenMP(int populationSize, const img* mainImage, const img* partImage) {
	std::vector<Offset> population(populationSize);
	int maxOffsetX = mainImage->width - partImage->width;
	int maxOffsetY = mainImage->height - partImage->height;

#pragma omp parallel for
	for (int i = 0; i < populationSize; ++i) {
		population[i].x = rand() % (maxOffsetX + 1);
		population[i].y = rand() % (maxOffsetY + 1);
		population[i].fitness = 0.0; 
	}

	return population;
}

void calculateFitnessOpenMP(std::vector<Offset>& population, const img* mainImage, const img* partImage, Offset& bestOverall) {
#pragma omp parallel for schedule (dynamic)
	for (int i = 0; i < population.size(); ++i) {
		double match = 0;
		int count = 0;

		for (int j = 0; j < partImage->height; ++j) {
			for (int k = 0; k < partImage->width; ++k) {
				int mainX = population[i].x + k;
				int mainY = population[i].y + j;

				if (mainX >= 0 && mainX < mainImage->width && mainY >= 0 && mainY < mainImage->height) {
					unsigned mainPixel = mainImage->image[mainY * mainImage->width + mainX];
					unsigned partPixel = partImage->image[j * partImage->width + k];
					if (mainPixel == partPixel) {
						match++;
					}
					count++;
				}
			}
		}

		population[i].fitness = count > 0 ? match / count : 0;

#pragma omp critical
		{
			if (population[i].fitness > bestOverall.fitness) {
				bestOverall = population[i];
			}
		}
	}
}

std::vector<Offset> selectBestWithSigmaScalingOpenMP(const std::vector<Offset>& population, int numBest) {
	int N = population.size();

	double f_avg = std::accumulate(population.begin(), population.end(), 0.0,
		[](double sum, const Offset& individual) {
		return sum + individual.fitness;
	}) / N;

	double variance = std::accumulate(population.begin(), population.end(), 0.0,
		[f_avg](double sum, const Offset& individual) {
		return sum + (individual.fitness - f_avg) * (individual.fitness - f_avg);
	}) / N;
	double sigma = std::sqrt(variance);

	std::vector<double> scaledFitness(N);
#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		scaledFitness[i] = 1 + (population[i].fitness - f_avg) / (2 * sigma);
		if (scaledFitness[i] < 0) {
			scaledFitness[i] = 0; 
		}
	}

	double totalFitness = std::accumulate(scaledFitness.begin(), scaledFitness.end(), 0.0);

	std::vector<double> probabilities(N);
#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		probabilities[i] = scaledFitness[i] / totalFitness;
	}

	std::vector<Offset> selected(numBest);
	std::default_random_engine generator;
	std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());

#pragma omp parallel for
	for (int i = 0; i < numBest; ++i) {
		int index = distribution(generator);
		selected[i] = population[index];
	}

	return selected;
}

Offset crossoverOpenMP(const Offset& parent1, const Offset& parent2) {
	Offset child;
	child.x = (parent1.x + parent2.x) / 2;
	child.y = (parent1.y + parent2.y) / 2;
	child.fitness = 0.0; 
	return child;
}

// Мутация (OpenMP)
void mutateOpenMP(Offset& individual, int maxMutation, const img* mainImage, const img* partImage) {
	individual.x += rand() % (2 * maxMutation + 1) - maxMutation;
	individual.y += rand() % (2 * maxMutation + 1) - maxMutation;

	individual.x = max(0, min(individual.x, mainImage->width - partImage->width));
	individual.y = max(0, min(individual.y, mainImage->height - partImage->height));
}

Offset geneticAlgorithmOpenMP(const img* mainImage, const img* partImage, int generations, int populationSize, int numBest, int maxMutation) {
	
	std::vector<Offset> population = initializePopulationOpenMP(populationSize, mainImage, partImage);

	Offset bestOverall = { 0, 0, 0.0 };

	for (int generation = 0; generation < generations; ++generation) {
	
		calculateFitnessOpenMP(population, mainImage, partImage, bestOverall);

		std::vector<Offset> best = selectBestWithSigmaScalingOpenMP(population, numBest);

		std::vector<Offset> newPopulation(populationSize);
#pragma omp parallel for schedule (dynamic)
		for (int i = 0; i < populationSize; ++i) {
			const Offset& parent1 = best[rand() % numBest];
			const Offset& parent2 = best[rand() % numBest];
			newPopulation[i] = crossoverOpenMP(parent1, parent2);
			mutateOpenMP(newPopulation[i], maxMutation, mainImage, partImage);
		}

		if (generation != generations - 1) {
			population = newPopulation;
		}
	}

	std::cout << "Best: x = " << bestOverall.x << ", y = " << bestOverall.y << ", fitness = " << bestOverall.fitness << "\n";
	return bestOverall;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	int computeOtsuThreshold(struct img* image) {
	
		std::vector<int> histogram(256, 0);
		int totalPixels = image->width * image->height;

		for (int j = 0; j < image->height; j++) {
			for (int i = 0; i < image->width; i++) {
				unsigned pixel = image->image[j * image->width + i];
				unsigned char r = (pixel >> 16) & 0xFF;
				unsigned char g = (pixel >> 8) & 0xFF;
				unsigned char b = pixel & 0xFF;
				unsigned char gray = static_cast<unsigned char>(0.333 * r + 0.333 * g + 0.333 * b); // Преобразование в серый
				histogram[gray]++;
			}
		}

		double sum = 0;
		for (int t = 0; t < 256; ++t) {
			sum += t * histogram[t];
		}

		double sumBackground = 0;
		int weightBackground = 0;
		int weightForeground = 0;

		double maxVariance = 0;
		int threshold = 0;

		for (int t = 0; t < 256; ++t) {
			weightBackground += histogram[t];
			if (weightBackground == 0) continue;

			weightForeground = totalPixels - weightBackground;
			if (weightForeground == 0) break;

			sumBackground += t * histogram[t];
			double meanBackground = sumBackground / weightBackground;
			double meanForeground = (sum - sumBackground) / weightForeground;

			double varianceBetween = weightBackground * weightForeground *
				(meanBackground - meanForeground) * (meanBackground - meanForeground);

			if (varianceBetween > maxVariance) {
				maxVariance = varianceBetween;
				threshold = t;
			}
		}

		return threshold;
	}

	void applyThreshold(struct img* image, int threshold) {
		for (int j = 0; j < image->height; j++) {
			for (int i = 0; i < image->width; i++) {
				unsigned pixel = image->image[j * image->width + i];
				unsigned char r = (pixel >> 16) & 0xFF;
				unsigned char g = (pixel >> 8) & 0xFF;
				unsigned char b = pixel & 0xFF;
				unsigned char gray = static_cast<unsigned char>(0.333 * r + 0.333 * g + 0.333 * b); 
				if (gray > threshold) {
					image->image[j * image->width + i] = 0xFFFFFFFF; 
				}
				else {
					image->image[j * image->width + i] = 0xFF000000; 
				}
			}
		}
	}

	void processImageWithOtsu(struct img* image) {
		int threshold = computeOtsuThreshold(image);
		applyThreshold(image, threshold);
	}

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Сводка для MyForm
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			InitializeComponent();
			//
			//TODO: добавьте код конструктора
			//
		}

	protected:
		/// <summary>
		/// Освободить все используемые ресурсы.
		/// </summary>
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::MenuStrip^  menuStrip1;
	protected:
	private: System::Windows::Forms::ToolStripMenuItem^  fileToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  openMainFileToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  openToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  calculateToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  doPosledovatToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  doOpenMPToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  doCUDAToolStripMenuItem;
	private: System::Windows::Forms::PictureBox^  pictureBox1;
	private: System::Windows::Forms::PictureBox^  pictureBox2;
	private: System::Windows::Forms::PictureBox^  pictureBox3;
	private: System::Windows::Forms::PictureBox^  pictureBox4;
	private: System::Windows::Forms::PictureBox^  pictureBox5;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::Label^  label2;
	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::Label^  label4;
	private: System::Windows::Forms::Label^  label5;
	private: System::Windows::Forms::OpenFileDialog^  openFileDialog1;
	private: System::Windows::Forms::Label^  label6;
	private: System::Windows::Forms::Label^  label7;
	private: System::Windows::Forms::PictureBox^  pictureBox6;
	private: System::Windows::Forms::PictureBox^  pictureBox7;
	private: System::Windows::Forms::ToolStripMenuItem^  doPosledovatToolStripMenuItem1;
	private: System::Windows::Forms::TextBox^ openMPTextBox;
	private: System::Windows::Forms::TextBox^ sequentialTextBox;
	private: System::Windows::Forms::TextBox^ cudaTextBox;


	private:
		/// <summary>
		/// Обязательная переменная конструктора.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Требуемый метод для поддержки конструктора — не изменяйте 
		/// содержимое этого метода с помощью редактора кода.
		/// </summary>
		void InitializeComponent(void)
		{
			this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			this->fileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openMainFileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->calculateToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->doPosledovatToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->doPosledovatToolStripMenuItem1 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->doOpenMPToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->doCUDAToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox2 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox3 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox4 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox5 = (gcnew System::Windows::Forms::PictureBox());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->pictureBox6 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox7 = (gcnew System::Windows::Forms::PictureBox());
			this->openMPTextBox = (gcnew System::Windows::Forms::TextBox());
			this->sequentialTextBox = (gcnew System::Windows::Forms::TextBox());
			this->cudaTextBox = (gcnew System::Windows::Forms::TextBox());
			this->menuStrip1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox2))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox3))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox4))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox5))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox6))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox7))->BeginInit();
			this->SuspendLayout();
			// 
			// menuStrip1
			// 
			this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
				this->fileToolStripMenuItem,
					this->calculateToolStripMenuItem
			});
			this->menuStrip1->Location = System::Drawing::Point(0, 0);
			this->menuStrip1->Name = L"menuStrip1";
			this->menuStrip1->Size = System::Drawing::Size(1764, 24);
			this->menuStrip1->TabIndex = 0;
			this->menuStrip1->Text = L"menuStrip1";
			// 
			// fileToolStripMenuItem
			// 
			this->fileToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
				this->openMainFileToolStripMenuItem,
					this->openToolStripMenuItem
			});
			this->fileToolStripMenuItem->Name = L"fileToolStripMenuItem";
			this->fileToolStripMenuItem->Size = System::Drawing::Size(37, 20);
			this->fileToolStripMenuItem->Text = L"File";
			// 
			// openMainFileToolStripMenuItem
			// 
			this->openMainFileToolStripMenuItem->Name = L"openMainFileToolStripMenuItem";
			this->openMainFileToolStripMenuItem->Size = System::Drawing::Size(180, 22);
			this->openMainFileToolStripMenuItem->Text = L"Open main file";
			this->openMainFileToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::openMainFileToolStripMenuItem_Click);
			// 
			// openToolStripMenuItem
			// 
			this->openToolStripMenuItem->Name = L"openToolStripMenuItem";
			this->openToolStripMenuItem->Size = System::Drawing::Size(180, 22);
			this->openToolStripMenuItem->Text = L"Open part file";
			this->openToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::openToolStripMenuItem_Click);
			// 
			// calculateToolStripMenuItem
			// 
			this->calculateToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(4) {
				this->doPosledovatToolStripMenuItem,
					this->doPosledovatToolStripMenuItem1, this->doOpenMPToolStripMenuItem, this->doCUDAToolStripMenuItem
			});
			this->calculateToolStripMenuItem->Name = L"calculateToolStripMenuItem";
			this->calculateToolStripMenuItem->Size = System::Drawing::Size(68, 20);
			this->calculateToolStripMenuItem->Text = L"Calculate";
			// 
			// doPosledovatToolStripMenuItem
			// 
			this->doPosledovatToolStripMenuItem->Name = L"doPosledovatToolStripMenuItem";
			this->doPosledovatToolStripMenuItem->Size = System::Drawing::Size(223, 22);
			this->doPosledovatToolStripMenuItem->Text = L"Бинаризация изображения";
			this->doPosledovatToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::doPosledovatToolStripMenuItem_Click);
			// 
			// doPosledovatToolStripMenuItem1
			// 
			this->doPosledovatToolStripMenuItem1->Enabled = false;
			this->doPosledovatToolStripMenuItem1->Name = L"doPosledovatToolStripMenuItem1";
			this->doPosledovatToolStripMenuItem1->Size = System::Drawing::Size(223, 22);
			this->doPosledovatToolStripMenuItem1->Text = L"Do posledovat";
			this->doPosledovatToolStripMenuItem1->Click += gcnew System::EventHandler(this, &MyForm::doPosledovatToolStripMenuItem1_Click);
			// 
			// doOpenMPToolStripMenuItem
			// 
			this->doOpenMPToolStripMenuItem->Enabled = false;
			this->doOpenMPToolStripMenuItem->Name = L"doOpenMPToolStripMenuItem";
			this->doOpenMPToolStripMenuItem->Size = System::Drawing::Size(223, 22);
			this->doOpenMPToolStripMenuItem->Text = L"Do OpenMP";
			this->doOpenMPToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::doOpenMPToolStripMenuItem_Click);
			// 
			// doCUDAToolStripMenuItem
			// 
			this->doCUDAToolStripMenuItem->Enabled = false;
			this->doCUDAToolStripMenuItem->Name = L"doCUDAToolStripMenuItem";
			this->doCUDAToolStripMenuItem->Size = System::Drawing::Size(223, 22);
			this->doCUDAToolStripMenuItem->Text = L"Do CUDA";
			this->doCUDAToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::doCUDAToolStripMenuItem_Click);
			// 
			// pictureBox1
			// 
			this->pictureBox1->Location = System::Drawing::Point(12, 54);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(284, 246);
			this->pictureBox1->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBox1->TabIndex = 1;
			this->pictureBox1->TabStop = false;
			// 
			// pictureBox2
			// 
			this->pictureBox2->Location = System::Drawing::Point(12, 349);
			this->pictureBox2->Name = L"pictureBox2";
			this->pictureBox2->Size = System::Drawing::Size(284, 246);
			this->pictureBox2->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBox2->TabIndex = 2;
			this->pictureBox2->TabStop = false;
			// 
			// pictureBox3
			// 
			this->pictureBox3->Location = System::Drawing::Point(727, 54);
			this->pictureBox3->Name = L"pictureBox3";
			this->pictureBox3->Size = System::Drawing::Size(286, 246);
			this->pictureBox3->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBox3->TabIndex = 3;
			this->pictureBox3->TabStop = false;
			// 
			// pictureBox4
			// 
			this->pictureBox4->Location = System::Drawing::Point(1066, 54);
			this->pictureBox4->Name = L"pictureBox4";
			this->pictureBox4->Size = System::Drawing::Size(284, 246);
			this->pictureBox4->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBox4->TabIndex = 4;
			this->pictureBox4->TabStop = false;
			// 
			// pictureBox5
			// 
			this->pictureBox5->Location = System::Drawing::Point(1424, 54);
			this->pictureBox5->Name = L"pictureBox5";
			this->pictureBox5->Size = System::Drawing::Size(284, 246);
			this->pictureBox5->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBox5->TabIndex = 5;
			this->pictureBox5->TabStop = false;
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(1555, 24);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(37, 13);
			this->label1->TabIndex = 6;
			this->label1->Text = L"CUDA";
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(1196, 24);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(49, 13);
			this->label2->TabIndex = 7;
			this->label2->Text = L"OpenMP";
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(820, 24);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(98, 13);
			this->label3->TabIndex = 8;
			this->label3->Text = L"Последовательно";
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(103, 24);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(55, 13);
			this->label4->TabIndex = 9;
			this->label4->Text = L"основная";
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Location = System::Drawing::Point(74, 320);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(130, 13);
			this->label5->TabIndex = 10;
			this->label5->Text = L"для поиска на основной";
			// 
			// openFileDialog1
			// 
			this->openFileDialog1->FileName = L"openFileDialog1";
			// 
			// label6
			// 
			this->label6->AutoSize = true;
			this->label6->Location = System::Drawing::Point(391, 320);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(223, 13);
			this->label6->TabIndex = 14;
			this->label6->Text = L"для поиска на основной отфильтрованная";
			// 
			// label7
			// 
			this->label7->AutoSize = true;
			this->label7->Location = System::Drawing::Point(428, 24);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(148, 13);
			this->label7->TabIndex = 13;
			this->label7->Text = L"основная отфильтрованная";
			// 
			// pictureBox6
			// 
			this->pictureBox6->Location = System::Drawing::Point(359, 349);
			this->pictureBox6->Name = L"pictureBox6";
			this->pictureBox6->Size = System::Drawing::Size(284, 246);
			this->pictureBox6->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBox6->TabIndex = 12;
			this->pictureBox6->TabStop = false;
			// 
			// pictureBox7
			// 
			this->pictureBox7->Location = System::Drawing::Point(359, 54);
			this->pictureBox7->Name = L"pictureBox7";
			this->pictureBox7->Size = System::Drawing::Size(284, 246);
			this->pictureBox7->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBox7->TabIndex = 11;
			this->pictureBox7->TabStop = false;
			// 
			// openMPTextBox
			// 
			this->openMPTextBox->Location = System::Drawing::Point(1066, 349);
			this->openMPTextBox->Multiline = true;
			this->openMPTextBox->Name = L"openMPTextBox";
			this->openMPTextBox->Size = System::Drawing::Size(284, 92);
			this->openMPTextBox->TabIndex = 16;
			// 
			// sequentialTextBox
			// 
			this->sequentialTextBox->Location = System::Drawing::Point(727, 349);
			this->sequentialTextBox->Multiline = true;
			this->sequentialTextBox->Name = L"sequentialTextBox";
			this->sequentialTextBox->Size = System::Drawing::Size(286, 92);
			this->sequentialTextBox->TabIndex = 17;
			// 
			// cudaTextBox
			// 
			this->cudaTextBox->Location = System::Drawing::Point(1424, 349);
			this->cudaTextBox->Multiline = true;
			this->cudaTextBox->Name = L"cudaTextBox";
			this->cudaTextBox->Size = System::Drawing::Size(284, 92);
			this->cudaTextBox->TabIndex = 18;
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1764, 740);
			this->Controls->Add(this->cudaTextBox);
			this->Controls->Add(this->sequentialTextBox);
			this->Controls->Add(this->openMPTextBox);
			this->Controls->Add(this->label6);
			this->Controls->Add(this->label7);
			this->Controls->Add(this->pictureBox6);
			this->Controls->Add(this->pictureBox7);
			this->Controls->Add(this->label5);
			this->Controls->Add(this->label4);
			this->Controls->Add(this->label3);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->pictureBox5);
			this->Controls->Add(this->pictureBox4);
			this->Controls->Add(this->pictureBox3);
			this->Controls->Add(this->pictureBox2);
			this->Controls->Add(this->pictureBox1);
			this->Controls->Add(this->menuStrip1);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedSingle;
			this->MainMenuStrip = this->menuStrip1;
			this->Name = L"MyForm";
			this->Text = L"MyForm";
			this->Load += gcnew System::EventHandler(this, &MyForm::MyForm_Load);
			this->menuStrip1->ResumeLayout(false);
			this->menuStrip1->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox2))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox3))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox4))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox5))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox6))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox7))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private: System::Void MyForm_Load(System::Object^  sender, System::EventArgs^  e) {
	}
	


private: System::Void openMainFileToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
	openFileDialog1->Filter = "Images (bmp, jpg, jpeg) | *.jpg; *.bmp; *.jpeg"; 
		openFileDialog1->Title = "Select image file";
	if (openFileDialog1->ShowDialog() ==
		System::Windows::Forms::DialogResult::OK) {
		pictureBox1->Image =
			Bitmap::FromFile(openFileDialog1->FileName);
		doCUDAToolStripMenuItem->Enabled = false;
		doOpenMPToolStripMenuItem->Enabled = false;
		doPosledovatToolStripMenuItem1->Enabled = false;
	}
}
private: System::Void openToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
	openFileDialog1->Filter = "Images (bmp, jpg, jpeg) | *.jpg; *.bmp; *.jpeg"; 
		openFileDialog1->Title = "Select image file";
	if (openFileDialog1->ShowDialog() ==
		System::Windows::Forms::DialogResult::OK) {
		pictureBox2->Image =
			Bitmap::FromFile(openFileDialog1->FileName);
		doCUDAToolStripMenuItem->Enabled = false;
		doOpenMPToolStripMenuItem->Enabled = false;
		doPosledovatToolStripMenuItem1->Enabled = false;
	}
}
private: System::Void doPosledovatToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
	struct img * image = new img();
	image->width = pictureBox1->Image->Width;
	image->height = pictureBox1->Image->Height;
	image->image = new unsigned[image->width * image->height];
	Bitmap ^ bitmap = gcnew Bitmap(pictureBox1->Image);
	for (int j = 0; j < image->height; j++)
	{
		for (int i = 0; i < image->width; i++)
		{
			image->image[j * image->width + i] = bitmap->GetPixel(i,j).ToArgb();
		}
	}
	int threshold = computeOtsuThreshold(image);
	applyThreshold(image, threshold);
	bitmap = gcnew Bitmap(image->width, image->height);
	for (int j = 0; j < image->height; j++)
	{
		for (int i = 0; i < image->width; i++)
		{
			bitmap->SetPixel(i, j,	Color::FromArgb(image->image[j * image->width +	i]));
		}
	}
	pictureBox7->Image = bitmap;
	image = new img();
	image->width = pictureBox2->Image->Width;
	image->height = pictureBox2->Image->Height;
	image->image = new unsigned[image->width * image->height];
	Bitmap ^ bitmap2 = gcnew Bitmap(pictureBox2->Image);
	for (int j = 0; j < image->height; j++)
	{
		for (int i = 0; i < image->width; i++)
		{
			image->image[j * image->width + i] = bitmap2->GetPixel(i, j).ToArgb();
		}
	}
	applyThreshold(image, threshold);
	bitmap2 = gcnew Bitmap(image->width, image->height);
	for (int j = 0; j < image->height; j++)
	{
		for (int i = 0; i < image->width; i++)
		{
			bitmap2->SetPixel(i, j, Color::FromArgb(image->image[j * image->width + i]));
		}
	}
	pictureBox6->Image = bitmap2;
	doCUDAToolStripMenuItem->Enabled = true;
	doOpenMPToolStripMenuItem->Enabled = true;
	doPosledovatToolStripMenuItem1->Enabled = true;
}
private: System::Void doPosledovatToolStripMenuItem1_Click(System::Object^  sender, System::EventArgs^  e) {
	struct img * MainImage = new img();
	MainImage->width = pictureBox7->Image->Width;
	MainImage->height = pictureBox7->Image->Height;
	MainImage->image = new unsigned[MainImage->width * MainImage->height];
	Bitmap ^ bitmapMain = gcnew Bitmap(pictureBox7->Image);
	for (int j = 0; j < MainImage->height; j++)
	{
		for (int i = 0; i < MainImage->width; i++)
		{
			MainImage->image[j * MainImage->width + i] = bitmapMain->GetPixel(i, j).ToArgb();
		}
	}

	struct img * PartImage = new img();
	PartImage->width = pictureBox6->Image->Width;
	PartImage->height = pictureBox6->Image->Height;
	PartImage->image = new unsigned[PartImage->width * PartImage->height];
	Bitmap ^ bitmapPart = gcnew Bitmap(pictureBox6->Image);
	for (int j = 0; j < PartImage->height; j++)
	{
		for (int i = 0; i < PartImage->width; i++)
		{
			PartImage->image[j * PartImage->width + i] = bitmapPart->GetPixel(i, j).ToArgb();
		}
	}

	int generations = 50;      // Число поколений
	int populationSize = 250;    // Размер популяции
	int numBest = 175;           // Количество лучших для селекции
	int maxMutation = 10;        // Максимальная мутация
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
								// Выполнение генетического алгоритма
	auto start = std::chrono::high_resolution_clock::now();
	Offset bestOffset = geneticAlgorithm(MainImage, PartImage, generations, populationSize, numBest, maxMutation);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Sequential GA execution time: " << duration.count() << " ms" << std::endl;
	sequentialTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	// Обновляем TextBox
	UpdateMetrics(sequentialTextBox, openMPTextBox, cudaTextBox);
	MainImage = new img();
	MainImage->width = pictureBox1->Image->Width;
	MainImage->height = pictureBox1->Image->Height;
	MainImage->image = new unsigned[MainImage->width * MainImage->height];
	bitmapMain = gcnew Bitmap(pictureBox1->Image);
	for (int j = 0; j < MainImage->height; j++)
	{
		for (int i = 0; i < MainImage->width; i++)
		{
			MainImage->image[j * MainImage->width + i] = bitmapMain->GetPixel(i, j).ToArgb();
		}
	}
	PartImage = new img();
	PartImage->width = pictureBox2->Image->Width;
	PartImage->height = pictureBox2->Image->Height;
	PartImage->image = new unsigned[PartImage->width * PartImage->height];
	bitmapPart = gcnew Bitmap(pictureBox2->Image);
	for (int j = 0; j < PartImage->height; j++)
	{
		for (int i = 0; i < PartImage->width; i++)
		{
			PartImage->image[j * PartImage->width + i] = bitmapPart->GetPixel(i, j).ToArgb();
		}
	}
	Bitmap^ resultBitmap = gcnew Bitmap(MainImage->width, MainImage->height);

	for (int j = 0; j < MainImage->height; ++j) {
		for (int i = 0; i < MainImage->width; ++i) {
			unsigned mainColor = MainImage->image[j * MainImage->width + i];

			// Извлекаем компоненты цвета
			unsigned char r = (mainColor >> 16) & 0xFF;
			unsigned char g = (mainColor >> 8) & 0xFF;
			unsigned char b = mainColor & 0xFF;

			// Устанавливаем пиксель в результирующий Bitmap
			resultBitmap->SetPixel(i, j, Color::FromArgb(128, r, g, b)); // Прозрачность
		}
	}

	// Накладываем PartImage с учетом смещения
	for (int j = 0; j < PartImage->height; ++j) {
		for (int i = 0; i < PartImage->width; ++i) {
			// Координаты в MainImage, куда накладывается PartImage
			int mainX = bestOffset.x + i;
			int mainY = bestOffset.y + j;

			// Проверяем, что координаты не выходят за границы MainImage
			if (mainX >= 0 && mainX < MainImage->width && mainY >= 0 && mainY < MainImage->height) {
				// Получаем цвет пикселя из PartImage
				unsigned color = PartImage->image[j * PartImage->width + i];
				resultBitmap->SetPixel(mainX, mainY, Color::FromArgb(color)); // Полная непрозрачность PartImage
			}
		}
	}

	// Отображаем результат в pictureBox3
	pictureBox3->Image = resultBitmap;
	//последовательно
}
private: System::Void doOpenMPToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
	struct img * MainImage = new img();
	MainImage->width = pictureBox7->Image->Width;
	MainImage->height = pictureBox7->Image->Height;
	MainImage->image = new unsigned[MainImage->width * MainImage->height];
	Bitmap ^ bitmapMain = gcnew Bitmap(pictureBox7->Image);
	for (int j = 0; j < MainImage->height; j++)
	{
		for (int i = 0; i < MainImage->width; i++)
		{
			MainImage->image[j * MainImage->width + i] = bitmapMain->GetPixel(i, j).ToArgb();
		}
	}

	struct img * PartImage = new img();
	PartImage->width = pictureBox6->Image->Width;
	PartImage->height = pictureBox6->Image->Height;
	PartImage->image = new unsigned[PartImage->width * PartImage->height];
	Bitmap ^ bitmapPart = gcnew Bitmap(pictureBox6->Image);
	for (int j = 0; j < PartImage->height; j++)
	{
		for (int i = 0; i < PartImage->width; i++)
		{
			PartImage->image[j * PartImage->width + i] = bitmapPart->GetPixel(i, j).ToArgb();
		}
	}

	int generations = 50;      // Число поколений
	int populationSize = 250;    // Размер популяции
	int numBest = 175;           // Количество лучших для селекции
	int maxMutation = 10;        // Максимальная мутация
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	auto start = std::chrono::high_resolution_clock::now();

	// Выполнение генетического алгоритма
	Offset bestOffset = geneticAlgorithmOpenMP(MainImage, PartImage, generations, populationSize, numBest, maxMutation);

	// Останавливаем таймер
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "OpenMP GA execution time: " << duration.count() << " ms" << std::endl;
	openMPTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	// Обновляем TextBox
	UpdateMetrics(sequentialTextBox, openMPTextBox, cudaTextBox);
	MainImage = new img();
	MainImage->width = pictureBox1->Image->Width;
	MainImage->height = pictureBox1->Image->Height;
	MainImage->image = new unsigned[MainImage->width * MainImage->height];
	bitmapMain = gcnew Bitmap(pictureBox1->Image);
	for (int j = 0; j < MainImage->height; j++)
	{
		for (int i = 0; i < MainImage->width; i++)
		{
			MainImage->image[j * MainImage->width + i] = bitmapMain->GetPixel(i, j).ToArgb();
		}
	}
	PartImage = new img();
	PartImage->width = pictureBox2->Image->Width;
	PartImage->height = pictureBox2->Image->Height;
	PartImage->image = new unsigned[PartImage->width * PartImage->height];
	bitmapPart = gcnew Bitmap(pictureBox2->Image);
	for (int j = 0; j < PartImage->height; j++)
	{
		for (int i = 0; i < PartImage->width; i++)
		{
			PartImage->image[j * PartImage->width + i] = bitmapPart->GetPixel(i, j).ToArgb();
		}
	}
	Bitmap^ resultBitmap = gcnew Bitmap(MainImage->width, MainImage->height);

	for (int j = 0; j < MainImage->height; ++j) {
		for (int i = 0; i < MainImage->width; ++i) {
			unsigned mainColor = MainImage->image[j * MainImage->width + i];

			// Извлекаем компоненты цвета
			unsigned char r = (mainColor >> 16) & 0xFF;
			unsigned char g = (mainColor >> 8) & 0xFF;
			unsigned char b = mainColor & 0xFF;

			// Устанавливаем пиксель в результирующий Bitmap
			resultBitmap->SetPixel(i, j, Color::FromArgb(128, r, g, b)); // Прозрачность
		}
	}

	// Накладываем PartImage с учетом смещения
	for (int j = 0; j < PartImage->height; ++j) {
		for (int i = 0; i < PartImage->width; ++i) {
			// Координаты в MainImage, куда накладывается PartImage
			int mainX = bestOffset.x + i;
			int mainY = bestOffset.y + j;

			// Проверяем, что координаты не выходят за границы MainImage
			if (mainX >= 0 && mainX < MainImage->width && mainY >= 0 && mainY < MainImage->height) {
				// Получаем цвет пикселя из PartImage
				unsigned color = PartImage->image[j * PartImage->width + i];
				resultBitmap->SetPixel(mainX, mainY, Color::FromArgb(color)); // Полная непрозрачность PartImage
			}
		}
	}

	// Отображаем результат в pictureBox3
	pictureBox4->Image = resultBitmap;
	//OpenMP
}
private: System::Void doCUDAToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
	// Создание и заполнение структур MainImage и PartImage
	struct img* MainImage = new img();
	MainImage->width = pictureBox7->Image->Width;
	MainImage->height = pictureBox7->Image->Height;
	MainImage->image = new unsigned[MainImage->width * MainImage->height];
	Bitmap^ bitmapMain = gcnew Bitmap(pictureBox7->Image);
	for (int j = 0; j < MainImage->height; j++) {
		for (int i = 0; i < MainImage->width; i++) {
			MainImage->image[j * MainImage->width + i] = bitmapMain->GetPixel(i, j).ToArgb();
		}
	}

	struct img* PartImage = new img();
	PartImage->width = pictureBox6->Image->Width;
	PartImage->height = pictureBox6->Image->Height;
	PartImage->image = new unsigned[PartImage->width * PartImage->height];
	Bitmap^ bitmapPart = gcnew Bitmap(pictureBox6->Image);
	for (int j = 0; j < PartImage->height; j++) {
		for (int i = 0; i < PartImage->width; i++) {
			PartImage->image[j * PartImage->width + i] = bitmapPart->GetPixel(i, j).ToArgb();
		}
	}

	// Параметры генетического алгоритма
	int generations = 50;      // Число поколений
	int populationSize = 250;  // Размер популяции
	int numBest = 175;         // Количество лучших для селекции
	int maxMutation = 10;      // Максимальная мутация
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	HINSTANCE hGetProcIDDLL = LoadLibrary(L"Example3Cuda.dll");
	if (!hGetProcIDDLL) {
		throw gcnew Exception("Failed to load DLL!");
	}

	// Определение типа функции
	typedef Offset(__stdcall* RunGeneticAlgorithmFunc)(const img*, const img*, int, int, int, int);

	// Получение указателя на функцию
	RunGeneticAlgorithmFunc RunGeneticAlgorithm = (RunGeneticAlgorithmFunc)GetProcAddress(hGetProcIDDLL, "RunGeneticAlgorithm");
	if (!RunGeneticAlgorithm) {
		FreeLibrary(hGetProcIDDLL);  // Освобождаем библиотеку перед выходом
		throw gcnew Exception("Failed to find function in DLL!");
	}
	// Выполняем функцию из DLL библиотеки 
	auto start = std::chrono::high_resolution_clock::now();
	Offset bestOffset = RunGeneticAlgorithm(MainImage, PartImage, generations, populationSize, numBest, maxMutation);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "CUDA GA execution time: " << duration.count() << " ms" << std::endl;
	cudaTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	// Обновляем TextBox
	UpdateMetrics(sequentialTextBox, openMPTextBox, cudaTextBox);
	// Освобождение библиотеки
	FreeLibrary(hGetProcIDDLL);

	// Освобождение памяти для MainImage и PartImage
	delete[] MainImage->image;
	delete MainImage;
	delete[] PartImage->image;
	delete PartImage;

	// Создание нового MainImage для результата
	MainImage = new img();
	MainImage->width = pictureBox1->Image->Width;
	MainImage->height = pictureBox1->Image->Height;
	MainImage->image = new unsigned[MainImage->width * MainImage->height];
	bitmapMain = gcnew Bitmap(pictureBox1->Image);
	for (int j = 0; j < MainImage->height; j++) {
		for (int i = 0; i < MainImage->width; i++) {
			MainImage->image[j * MainImage->width + i] = bitmapMain->GetPixel(i, j).ToArgb();
		}
	}

	// Создание нового PartImage для результата
	PartImage = new img();
	PartImage->width = pictureBox2->Image->Width;
	PartImage->height = pictureBox2->Image->Height;
	PartImage->image = new unsigned[PartImage->width * PartImage->height];
	bitmapPart = gcnew Bitmap(pictureBox2->Image);
	for (int j = 0; j < PartImage->height; j++) {
		for (int i = 0; i < PartImage->width; i++) {
			PartImage->image[j * PartImage->width + i] = bitmapPart->GetPixel(i, j).ToArgb();
		}
	}

	// Создание результирующего изображения
	Bitmap^ resultBitmap = gcnew Bitmap(MainImage->width, MainImage->height);

	// Заполнение результирующего изображения с учетом прозрачности
	for (int j = 0; j < MainImage->height; ++j) {
		for (int i = 0; i < MainImage->width; ++i) {
			unsigned mainColor = MainImage->image[j * MainImage->width + i];

			// Извлекаем компоненты цвета
			unsigned char r = (mainColor >> 16) & 0xFF;
			unsigned char g = (mainColor >> 8) & 0xFF;
			unsigned char b = mainColor & 0xFF;

			// Устанавливаем пиксель в результирующий Bitmap с прозрачностью
			resultBitmap->SetPixel(i, j, Color::FromArgb(128, r, g, b));
		}
	}

	// Наложение PartImage с учетом смещения
	for (int j = 0; j < PartImage->height; ++j) {
		for (int i = 0; i < PartImage->width; ++i) {
			// Координаты в MainImage, куда накладывается PartImage
			int mainX = bestOffset.x + i;
			int mainY = bestOffset.y + j;

			// Проверяем, что координаты не выходят за границы MainImage
			if (mainX >= 0 && mainX < MainImage->width && mainY >= 0 && mainY < MainImage->height) {
				// Получаем цвет пикселя из PartImage
				unsigned color = PartImage->image[j * PartImage->width + i];
				resultBitmap->SetPixel(mainX, mainY, Color::FromArgb(color)); // Полная непрозрачность PartImage
			}
		}
	}

	// Отображение результата в pictureBox4
	pictureBox5->Image = resultBitmap;

	// Освобождение памяти
	delete[] MainImage->image;
	delete MainImage;
	delete[] PartImage->image;
	delete PartImage;
	//CUDA
}
};
}
