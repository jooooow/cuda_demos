#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <device_functions.h>
#include <sm_20_atomic_functions.h>
#include <iostream>
#include <stdio.h>
using namespace std;

#define SIZE 20

__global__ void CUDA_KoggeStone(int* arr, int size, int* output);
__global__ void CUDA_Sklansky(int* arr, int size, int* output);
__global__ void CUDA_BrentKung(int* arr, int size, int* output);

int main()
{
	cout << "prefix_scan" << endl;
	int size = SIZE;
	int* cpu_arr = new int[size];
	int* cpu_res = new int[size];

	for (int i = 0; i < size; i++)
	{
		cpu_arr[i] = i + 1;
		printf("%4d", cpu_arr[i]);
	}
	printf("\n");

	int* gpu_arr;
	cudaMalloc(&gpu_arr, size * sizeof(int));
	cudaMemcpy(gpu_arr, cpu_arr, size * sizeof(int), cudaMemcpyHostToDevice);
	int* gpu_res;
	cudaMalloc(&gpu_res, size * sizeof(int));

	/*kogge-stone*/
	//CUDA_KoggeStone << <1, size >> > (gpu_arr, size, gpu_res);

	/*sklansky*/
	//CUDA_Sklansky << <1, size >> > (gpu_arr, size, gpu_res);

	/*brent-kung*/
	CUDA_BrentKung << <1, size >> > (gpu_arr, size, gpu_res);

	cudaMemcpy(cpu_res, gpu_res, size * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++)
	{
		printf("%4d", cpu_res[i]);
	}

	cudaFree(gpu_arr);
	cudaFree(gpu_res);
	free(cpu_arr);
	cpu_arr = nullptr;
	free(cpu_res);
	cpu_res = nullptr;

	system("pause");
	return 0;
}

__global__ void CUDA_KoggeStone(int* arr, int size, int* output)
{
	__shared__ int cache[SIZE];
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	output[id] = arr[id];
	cache[id] = arr[id];
	int cnt = sqrtf(size);
	cnt = cnt * cnt < size ? cnt + 1 : cnt;
	for (int i = 0, step = 1; i < cnt; i++, step *= 2)
	{
		if (id >= step)
		{
			output[id] += cache[id - step];
		}
		__syncthreads();
		cache[id] = output[id];
		__syncthreads();
	}
}

__global__ void CUDA_Sklansky(int* arr, int size, int* output)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int cnt = sqrtf(size);
	cnt = cnt * cnt < size ? cnt + 1 : cnt;
	output[id] = arr[id];
	for (int i = 0, step = 1; i < cnt; i++, step *= 2)
	{
		if ((id + step) % (step * 2) < step)
		{
			output[id] += output[(id / step) * step - 1];
		}
		__syncthreads();
	}
}

__global__ void CUDA_BrentKung(int* arr, int size, int* output)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int cnt = sqrtf(size);
	cnt = cnt * cnt < size ? cnt + 1 : cnt;
	int step = 1;
	output[id] = arr[id];
	/*upsweep*/
	for (int i = 0; i < cnt; i++)
	{
		if ((id + 1) % (step * 2) == 0)
		{
			output[id] += output[id - step];
		}
		step *= 2;
		__syncthreads();
	}
	step /= 2;

	/*downsweep*/
	for (int i = 0; i < cnt - 1; i++)
	{
		step /= 2;
		if ((id - step + 1) % (step * 2) == 0)
		{
			output[id] += output[id - step];
		}
		__syncthreads();
	}
}
