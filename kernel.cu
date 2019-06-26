#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <ctime>


// Helper function for using CUDA to call kernel functions
cudaError_t cuda_code(float* a, float* x, int N, int dim);
__device__ float sum = 0;
__global__ void elimination(float *a, int N, int m);
__global__ void substitution(int i, int N, float *row, float *a, float*x);
__global__ void kernel_func(float* a, float* x, int N);
void readm(float* m, int n);
void printm(float *m, float*x, int n);

int main() {
	float* m;
	float* x;
	int dim;
	int N = 1000;
	dim = (N + 1) * N;

	m = (float*)malloc(dim * sizeof(float));
	x = (float*)malloc(N * sizeof(float));

	readm(m, N);
	clock_t begin = clock();
	cudaError_t cudaStatus = cuda_code(m, x, N, dim);
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin);// / CLOCKS_PER_SEC;
	printm(m, x, N);
	std::cout << "Elapsed time: " << elapsed_secs << std::endl;
	return 0;
}

cudaError_t cuda_code(float* a, float* x, int N, int dim)
{
	cudaError_t cudaStatus;
	size_t size1 = dim * sizeof(float);
	size_t size2 = N * sizeof(float);
	float* dev_a;
	float* dev_x;

	// Allocate GPU buffers for two vectors (one input, one output).
	cudaStatus = cudaMalloc((void**)&dev_a, size1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_x, size2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, dim * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	// Launch a kernel on the GPU with one thread for each element.
	kernel_func << <1, 1 >> > (dev_a, dev_x, N);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(a, dev_a, size1, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMemcpy(x, dev_x, size2, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	cudaStatus = cudaFree(dev_a);
	cudaStatus = cudaFree(dev_x);

	return cudaStatus;
}
__global__ void elimination(float *a, int N, int m)
{
	//m is added to indexes i & j to assign the threads to the elements from m to N of row m
	int i = m + threadIdx.x + blockIdx.x * blockDim.x;
	int j = m + blockIdx.y * blockDim.y + threadIdx.y;
	int ij;		//element i,j of the matrix
	int ebmri;	//element below m, row i
	int mm;		//pivot previous row, diagonal element
	int prcj;	//element of previous row at column j

	if (i > m && i < N)
	{
		if (j > m && j < (N + 1))
		{
			ij = j + (N + 1)*i;
			ebmri = m + (N + 1)*i;
			mm = m + (N + 1)*m;
			prcj = j + (N + 1)*m;
			a[ij] = a[ij] - ((a[ebmri] / a[mm])*a[prcj]);
		}
	}
	__syncthreads();
}
__global__ void substitution(int i, int N, float *row, float *a, float*x) {
	int j = i + blockIdx.x * blockDim.x + threadIdx.x;
	int ij;		//element i,j of the matrix
	if (j > i && j < N)
	{
		ij = j + (N + 1)*i;
		row[j] = a[ij] * x[j];
		atomicAdd(&sum, row[j]);
	}
	__syncthreads();
}
__global__ void kernel_func(float* a, float* x, int N)
{
	//Blocks:	Number of blocks per axis
	//Blocksize: 	Number of threads per Block per axis
	//elimination step
	for (int m = 0; m < N; m++)
	{
		//32 is the warp size
		int Dx = ceilf((float)(N - m) / 32.0);		//Number of blocks in x
		int Dy = ceilf((float)(N + 1 - m) / 32.0);	//Number of blocks in y
		dim3 Blocks(Dx, Dy);
		dim3 Blocksize(32, 32);		//Maximum amount of threads permitted per block, as per GTX 1050 ti
		elimination << <Blocks, Blocksize >> > (a, N, m);
		cudaDeviceSynchronize();
	}
	//The last element of x, can be solved directly as follows
	x[N - 1] = a[N*(N + 1) - 1] / a[N*(N + 1) - 2];
	//*row array stores temporarily at location i, the multiplication a[ij] * x[j]
	//In order to add each of of them atomically into "sum"
	float *row;
	row = (float*)malloc((N + 1) * sizeof(float));
	//backwards substitution
	int eltb;
	int eltij;
	for (int i = N - 2; 0 <= i; i--)
	{
		sum = 0;
		int Dy = ceilf((float)(N + 1 - i) / 32.0);

		substitution << <Dy, 32 >> > (i, N, row, a, x);
		cudaDeviceSynchronize();
		eltb = N + (N + 1)*i;
		eltij = i + (N + 1)*i;
		x[i] = (a[eltb] - sum) / a[eltij];
	}
}
void readm(float* m, int n) {
	using namespace std;
	ifstream myfile;
	myfile.open("matrix.txt");

	if (myfile.is_open()) {
		for (int j = 0; j < n; j++)
		{
			for (int i = 0; i < (n + 1); i++)
			{
				int ij = i + ((n + 1) * j);
				myfile >> m[ij];
			}
		}

	}


	myfile.close();
}
void printm(float* m, float* x, int n) {
	std::ofstream myfile;
	myfile.open("matrix2.txt");

	if (myfile.is_open()) {
		for (int j = 0; j < n; j++)
		{
			for (int i = 0; i < n + 1; i++)
			{
				myfile << m[i + (n + 1) * j] << "\t";
			}
			myfile << "\n";
		}
		for (int j = 0; j < n; j++)
		{
			myfile << x[j] << "\n";
		}

	}
	myfile.close();
}
