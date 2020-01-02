#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <device_functions.h>
#include <sm_20_atomic_functions.h>
#include <iostream>
#include <stdio.h>
#include <time.h>
using namespace std;

void StartTimer(StopWatchInterface** timer);
void StopTimer(StopWatchInterface** timer, int id);
void PrintMat(float* mat, int size);

void MatMulti(float* a, float* b, float* c, int size);
__global__ void MatMulti1(float* a, float* b, float* c, int size);
__global__ void MatMulti2(float* a, float* b, float* c, int size);

#define BLOCK_T  32
#define MAT_SIZE 1024
#define BLOCK_SIZE 1024 
#define GRID_SIZE  (MAT_SIZE * MAT_SIZE / BLOCK_SIZE)
#define ELE(a,b,c,d) (*(a + b * c + d))

int main()
{
	float* cpu_a;
	float* cpu_b;
	float* cpu_c;
	cpu_a = new float[MAT_SIZE * MAT_SIZE];
	cpu_b = new float[MAT_SIZE * MAT_SIZE];
	cpu_c = new float[MAT_SIZE * MAT_SIZE];

	StopWatchInterface * timer;
	StartTimer(&timer);
	StopTimer(&timer, 0);

	for (int i = 0; i < MAT_SIZE; i++)
	{
		for (int j = 0; j < MAT_SIZE; j++)
		{
			ELE(cpu_a, MAT_SIZE, i, j) = i;
			ELE(cpu_b, MAT_SIZE, i, j) = i + j;
		}
	}
	//PrintMat(cpu_a, MAT_SIZE);
	//PrintMat(cpu_b, MAT_SIZE);

	//1.cpu serial
	StartTimer(&timer);
	//MatMulti(cpu_a, cpu_b, cpu_c, MAT_SIZE);
	StopTimer(&timer, 1);

	//2.primitive parallel
	float* gpu_a;
	float* gpu_b;
	float* gpu_c;
	cudaMalloc(&gpu_a, MAT_SIZE * MAT_SIZE * sizeof(float));
	cudaMalloc(&gpu_b, MAT_SIZE * MAT_SIZE * sizeof(float));
	cudaMalloc(&gpu_c, MAT_SIZE * MAT_SIZE * sizeof(float));
	cudaMemcpy(gpu_a, cpu_a, MAT_SIZE * MAT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, cpu_b, MAT_SIZE * MAT_SIZE * sizeof(float), cudaMemcpyHostToDevice);


	StartTimer(&timer);
	MatMulti1 << <1024, 1024 >> >(gpu_a, gpu_b, gpu_c, MAT_SIZE);
	StopTimer(&timer, 2);

	StartTimer(&timer);
	MatMulti2 <<<dim3(32, 32), dim3(32, 32) >>>(gpu_a, gpu_b, gpu_c, MAT_SIZE);
	StopTimer(&timer, 3);

	cudaMemcpy(cpu_c, gpu_c, MAT_SIZE * MAT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);
	//PrintMat(cpu_c, MAT_SIZE);

	delete[] cpu_a;
	cpu_a = nullptr;
	delete[] cpu_b;
	cpu_b = nullptr;
	delete[] cpu_c;
	cpu_c = nullptr;

	system("pause");
}

void PrintMat(float* mat, int size)
{
	printf("-------------------------------------------------\n");
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			printf("%10.2f", ELE(mat, size, i, j));
		}
		printf("\n");
	}
	printf("-------------------------------------------------\n");
}

void StartTimer(StopWatchInterface** timer)
{
	sdkCreateTimer(timer);
	sdkStartTimer(timer);
}

void StopTimer(StopWatchInterface** timer, int id)
{
	cudaDeviceSynchronize();
	sdkStopTimer(timer);
	printf("\n%d.gpu_done %.3f(ms)\n", id, sdkGetTimerValue(timer));
}


void MatMulti(float* a, float* b, float* c, int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			float sum = 0;
			for (int k = 0; k < size; k++)
			{
				sum += *(a + i * size + k) * *(b + k * size + j);
			}
			*(c + i * size + j) = sum;
		}
	}
}

__global__ void MatMulti1(float* a, float* b, float* c, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int i = idx / size;
	int j = idx % size;
	float sum = 0;
	for (int k = 0; k < size; k++)
	{
		sum += (*(a + i * size + k)) * (*(b + k * size + j));
	}
	c[idx] = sum;
}

__global__ void MatMulti2(float* a, float* b, float* c, int size)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0;
	for (int i = 0; i < size / blockDim.x; i++)
	{
		__shared__ float cache_a[32][32];
		__shared__ float cache_b[32][32];
		cache_a[threadIdx.y][threadIdx.x] = a[idx_y * gridDim.x * blockDim.x + blockDim.x * i + threadIdx.x];
		cache_b[threadIdx.y][threadIdx.x] = b[(i * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + idx_x];
		__syncthreads();
		for (int k = 0; k < blockDim.x; k++)
		{
			sum += cache_a[threadIdx.y][k] * cache_b[k][threadIdx.x];
			//sum += idx_x + idx_y;
		}
		__syncthreads();
	}

	c[idx_y * gridDim.x * blockDim.x + idx_x] = sum;
}