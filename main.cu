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

#define SIZE 320
#define BLOCK_SIZE 4
#define GRID_SIZE  (SIZE / BLOCK_SIZE)
#define A    0
#define P    1
#define X    2

typedef struct
{
	int aggregate;
	int inclusive_prefix;
	int exclusive_prefix;
	unsigned char flag;
}descriptor;

__global__ void CUDA_KoggeStone(int* arr, int size, int* output);
__global__ void CUDA_Sklansky(int* arr, int size, int* output);
__global__ void CUDA_BrentKung(int* arr, int size, int* output);
__global__ void CUDA_ReduceThenScan(int* arr, int size, descriptor* desc, int* output);
__global__ void CUDA_ReduceThenScan_Reduce(int* arr, int size, int* output);
__global__ void CUDA_ReduceThenScan_Chain(int* arr, int size, int block_size, int* output);
__global__ void CUDA_ReduceThenScan_Scan(int* arr, int size, int* output);

int main()
{
	cout << "prefix_scan" << endl;
	int size = SIZE;
	int* cpu_arr = new int[size];
	int* cpu_res = new int[size];

	for (int i = 0; i < size; i++)
	{
		cpu_arr[i] = i + 1;
		printf("%8d", cpu_arr[i]);
		if ((i + 1) % BLOCK_SIZE == 0)
			printf("\n");
	}
	printf("\n\n");

	int* gpu_arr;
	cudaMalloc(&gpu_arr, size * sizeof(int));
	cudaMemcpy(gpu_arr, cpu_arr, size * sizeof(int), cudaMemcpyHostToDevice);
	int* gpu_res;
	cudaMalloc(&gpu_res, size * sizeof(int));
	int* gpu_res2;
	cudaMalloc(&gpu_res2, size * sizeof(int));
	StopWatchInterface * timer_cublas;

	/*kogge-stone*/
	//CUDA_KoggeStone << <1, size >> > (gpu_arr, size, gpu_res);

	/*sklansky*/
	//CUDA_Sklansky << <1, size >> > (gpu_arr, size, gpu_res);

	/*brent-kung*/
	//CUDA_BrentKung << <1, size >> > (gpu_arr, size, gpu_res);

	/*reduce-then-scan*/
	descriptor cpu_block_descriptor[GRID_SIZE];
	for (int i = 0; i < GRID_SIZE; i++)
	{
		cpu_block_descriptor[i].aggregate = 0;
		cpu_block_descriptor[i].inclusive_prefix = 0;
		cpu_block_descriptor[i].exclusive_prefix = 0;
		cpu_block_descriptor[i].flag = A;
	}
	descriptor* gpu_block_descriptor;
	cudaMalloc(&gpu_block_descriptor, GRID_SIZE * sizeof(descriptor));
	cudaMemcpy(gpu_block_descriptor, cpu_block_descriptor, GRID_SIZE * sizeof(descriptor), cudaMemcpyHostToDevice);

	float time_sum = 0.0f;
	for (int i = 0; i < 1; i++)
	{
		sdkCreateTimer(&timer_cublas);
		sdkStartTimer(&timer_cublas);

		CUDA_ReduceThenScan << <GRID_SIZE, BLOCK_SIZE >> > (gpu_arr, size, gpu_block_descriptor, gpu_res2);

		/*reduce-then-scan*/
		//CUDA_ReduceThenScan_Reduce << <GRID_SIZE, BLOCK_SIZE >> > (gpu_arr, size, gpu_res);
		//CUDA_ReduceThenScan_Chain << <1, 1 >> > (gpu_res, size, BLOCK_SIZE, gpu_res);
		//CUDA_ReduceThenScan_Scan << <GRID_SIZE, BLOCK_SIZE >> > (gpu_res, size, gpu_res2);

		cudaDeviceSynchronize();
		sdkStopTimer(&timer_cublas);
		time_sum += sdkGetTimerValue(&timer_cublas);
		printf("\ngpu_done %.3f(ms)\n\n", sdkGetTimerValue(&timer_cublas));
	}
	printf("\nAVERAGE_TIME : %3f\n\n", (time_sum / 1.0));

	cudaMemcpy(cpu_res, gpu_res2, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_block_descriptor, gpu_block_descriptor, GRID_SIZE * sizeof(descriptor), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++)
	{
		printf("%8d", cpu_res[i]);
		if ((i + 1) % BLOCK_SIZE == 0)
			printf("\n");
	}
	printf("\n");

	for (int i = 0; i < GRID_SIZE; i++)
	{
		printf("[%d]%d,%d,%d,%d\n", i, cpu_block_descriptor[i].flag,
			                           cpu_block_descriptor[i].aggregate,
									   cpu_block_descriptor[i].exclusive_prefix,
									   cpu_block_descriptor[i].inclusive_prefix);
	}

	cudaFree(gpu_arr);
	cudaFree(gpu_res);
	cudaFree(gpu_res2);
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

__global__ void CUDA_ReduceThenScan(int* arr, int size, descriptor* desc, int* output)
{
	__shared__ int cache[BLOCK_SIZE];
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	cache[threadIdx.x] = arr[id];
	output[id] = cache[threadIdx.x];

	if (id > size - blockDim.x - 1)
		return;

	for (int i = 1; i < blockDim.x; i *= 2)
	{
		if ((id + 1) % (i * 2) == 0)
		{
			cache[threadIdx.x] += cache[threadIdx.x - i];
		}
		__syncthreads();
	}

	if (threadIdx.x == blockDim.x - 1)
	{
		/*desc[blockIdx.x].aggregate = cache[threadIdx.x];
		__threadfence();
		desc[blockIdx.x].flag = A;

		if (blockIdx.x == 0)
		{
			desc[blockIdx.x].inclusive_prefix = 10;
			__threadfence();
			desc[blockIdx.x].flag = P;
		}
		else if(blockIdx.x < GRID_SIZE - 1)
		{
			int index = blockIdx.x - 1;
			int is_continue_inspect = 1;
			while (is_continue_inspect == 1)
			{
				switch (desc[index].flag)
				{
				case X:while (1); break;
				case A:
					desc[blockIdx.x].exclusive_prefix += desc[index].aggregate;
					index--;
					if (index < 0)
						is_continue_inspect = 0;
					break;
				case P:
					desc[blockIdx.x].exclusive_prefix += desc[index].inclusive_prefix;
					is_continue_inspect = 0;
					break;
				}
				
			}
			desc[blockIdx.x].aggregate += desc[blockIdx.x].exclusive_prefix;
			desc[blockIdx.x].inclusive_prefix = desc[blockIdx.x].aggregate;
			cache[threadIdx.x] = desc[blockIdx.x].inclusive_prefix;
			__threadfence();
			desc[blockIdx.x].flag = P;
		}*/
		
		desc[blockIdx.x].aggregate = cache[threadIdx.x];
		desc[blockIdx.x].inclusive_prefix = desc[blockIdx.x].aggregate;
		//__threadfence();
		desc[blockIdx.x].flag = A;

		if (blockIdx.x == 0)
		{
			desc[blockIdx.x].flag = P;
		}
		else if (blockIdx.x < GRID_SIZE - 1)
		{
			int index = blockIdx.x - 1;
			int is_continue_inspect = 1;
			while (is_continue_inspect == 1)
			{
				switch (desc[index].flag)
				{
				case X:while (1); break;
				case A:
					desc[blockIdx.x].inclusive_prefix += desc[index].aggregate;
					index--;
					if (index < 0)
						is_continue_inspect = 0;
					break;
				case P:
					desc[blockIdx.x].inclusive_prefix += desc[index].inclusive_prefix;
					is_continue_inspect = 0;
					desc[blockIdx.x].flag = P;
					break;
				}

			}
			cache[threadIdx.x] = desc[blockIdx.x].inclusive_prefix;
		}
	}

	__syncthreads();

	for (int i = 1; i < blockDim.x; i *= 2)
	{
		int step = blockDim.x / i / 2;
		if ((id - step + 1) % (step * 2) == 0 && (id - step + 1) != 0)
		{
			int offset_index = threadIdx.x - step;
			if (offset_index < 0)
				cache[threadIdx.x] += desc[blockIdx.x].inclusive_prefix - desc[blockIdx.x].aggregate;
			else
				cache[threadIdx.x] += cache[offset_index];
		}
		__syncthreads();
	}

	output[id] = cache[threadIdx.x];
}

__global__ void CUDA_ReduceThenScan_Reduce(int* arr, int size, int* output)
{
	__shared__ int cache[4];
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	cache[threadIdx.x] = arr[id];
	output[id] = cache[threadIdx.x];

	if (id > size - blockDim.x - 1)
		return;

	for (int i = 1; i < blockDim.x; i *= 2)
	{
		if ((id + 1) % (i * 2) == 0)
		{
			cache[threadIdx.x] += cache[threadIdx.x - i];
		}
		__syncthreads();
	}

	output[id] = cache[threadIdx.x];
}

__global__ void CUDA_ReduceThenScan_Chain(int* arr, int size, int block_size, int* output)
{
	for (int i = block_size * 2 - 1; i < size - block_size; i += block_size)
	{
		output[i] += arr[i - block_size];
	}
}

__global__ void CUDA_ReduceThenScan_Scan(int* arr, int size, int* output)
{
	__shared__ int cache[4];
	__shared__ int pre_prefix[1];
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	cache[threadIdx.x] = arr[id];
	int pre_prefix_index = blockDim.x * blockIdx.x - 1;
	if(pre_prefix_index >= 0)
		pre_prefix[0] = arr[pre_prefix_index];
	output[id] = cache[threadIdx.x];
	
	for (int i = 1; i < blockDim.x; i *= 2)
	{
		int step = blockDim.x / i / 2;
		if ((id - step + 1) % (step * 2) == 0 && (id - step + 1) != 0)
		{
			int offset_index = threadIdx.x - step;
			if (offset_index < 0)
				cache[threadIdx.x] += pre_prefix[0];
			else
				cache[threadIdx.x] += cache[offset_index];
		}
		__syncthreads();
	}

	output[id] = cache[threadIdx.x];
}
