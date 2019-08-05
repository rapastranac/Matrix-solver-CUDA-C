#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <ctime>

// Helper function for using CUDA to call kernel functions
cudaError_t cuda_code(float* , float*, int , int );
__device__ float sum = 0;
__global__ void elimination(float *, int , int );
__global__ void substitution(int , int , float *, float *, float*);
__global__ void kernel_func(float* , float* , int );
void readMatrix(float* , int );
void printMatrix(float *, float*, int );

int main() {

	/*
	this is the size of matrix A this constant need modification if the size of the matrix
	in the file inputMatrix.txt was modified
	*/
	int N = 1000; //number of column/line in matrix A


	float* matrix;//matrix (A|B)
	float* resultVector;
	int dim; //total number of term in matrix (A|B)
	dim = (N + 1) * N;

	matrix = (float*)malloc(dim * sizeof(float));
	resultVector = (float*)malloc(N * sizeof(float));

	readMatrix(matrix, N);
	clock_t begin = clock();
	cudaError_t cudaStatus = cuda_code(matrix, resultVector, N, dim);
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	printMatrix(matrix, resultVector, N);
	std::cout << "Elapsed time: " << elapsed_secs << std::endl;
	return 0;
}

cudaError_t cuda_code(float* matrix, float* resultVector, int N, int dim)
{
	cudaError_t cudaStatus;
	size_t size1 = dim * sizeof(float);
	size_t size2 = N * sizeof(float);
	float* dev_matrix;
	float* dev_resultVector;

	// Allocate GPU buffers for two vectors (one input, one output).
	cudaStatus = cudaMalloc((void**)&dev_matrix, size1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_resultVector, size2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_matrix, matrix, dim * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	// Launch a kernel on the GPU with one thread for each element.
	kernel_func << <1, 1 >> > (dev_matrix, dev_resultVector, N);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(matrix, dev_matrix, size1, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMemcpy(resultVector, dev_resultVector, size2, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	cudaStatus = cudaFree(dev_matrix);
	cudaStatus = cudaFree(dev_resultVector);

	return cudaStatus;
}
__global__ void elimination(float *matrix, int N, int m)
{
	int i = m + threadIdx.x + blockIdx.x * blockDim.x;
	int j = m + blockIdx.y * blockDim.y + threadIdx.y;
	//From previous lines, "m" assigns the initial thread index, so threads are not 
	//created for indexes that have been already done
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
			matrix[ij] = matrix[ij] - ((matrix[ebmri] / matrix[mm])*matrix[prcj]);
		}
	}
	__syncthreads(); //Barrier to wait all threads to finish their tasks
}
__global__ void substitution(int i, int N, float *row, float *matrix, float*resultVector) {
	int j = i + blockIdx.x * blockDim.x + threadIdx.x;
	//From previous line, "i" assigns the initial thread index, so threads are not 
	//created for indexes that will not affect the results
	int ij;		//element i,j of the matrix
	if (j > i && j < N)
	{
		ij = j + (N + 1)*i;
		row[j] = matrix[ij] * resultVector[j];
		atomicAdd(&sum, row[j]);
	}
	__syncthreads();//Barrier to wait all threads to finish their tasks
}
__global__ void kernel_func(float* matrix, float* resultVector, int N)
{
	
	//elimination
	for (int currentrow = 0; currentrow < N; currentrow++)
	{
		int Dx = ceilf((float)(N - currentrow) / 32.0);
		int Dy = ceilf((float)(N + 1 - currentrow) / 32.0);
		//Previous lines optimize threads creation such that if it is required fewer than 1024 threads
		//then, fewer resources will be used
		dim3 Blocks(Dx, Dy); //Number of blocks per axis
		dim3 Blocksize(32, 32); //Number of threads per Block per axis
		elimination << <Blocks, Blocksize >> > (matrix, N, currentrow);
		cudaDeviceSynchronize();//Barrier to let function to finish before to continue
	}
	//The last element of resultVector, can be solved directly as follows
	resultVector[N - 1] = matrix[N*(N + 1) - 1] / matrix[N*(N + 1) - 2];
	//This array stores temporarily at location i, the multiplication a[ij] * resultVector[j]
	//In order to add each of of them atomically into "sum"
	float *row;
	row = (float*)malloc((N + 1) * sizeof(float));
	//backwards substitution
	int eltb;	//element b @ i
	int eltij;	//element i,j @ i
	for (int i = N - 2; 0 <= i; i--)
	{
		sum = 0;
		int Dy = ceilf((float)(N + 1 - i) / 32.0);	// Dy*32 = Threads required, optimized
		substitution << <Dy, 32 >> > (i, N, row, matrix, resultVector);	
		cudaDeviceSynchronize();	//Barrier to let function to finish before to continue
		eltb = N + (N + 1)*i;	//last element of row i
		eltij = i + (N + 1)*i;	//element of diagonal
		resultVector[i] = (matrix[eltb] - sum) / matrix[eltij];
	}
}

void readMatrix(float* matrix, int n) {
	using namespace std;
	ifstream myfile;
	myfile.open("InputMatrix.txt");

	if (myfile.is_open()) {
		for (int j = 0; j < n; j++)
		{
			for (int i = 0; i < (n + 1); i++)
			{
				int ij = i + ((n + 1) * j);
				myfile >> matrix[ij];
			}
		}

	}
	myfile.close();
}

void printMatrix(float* matrix, float* resultVector, int n) {
	std::ofstream myfile;
	myfile.open("outputVector.txt");

	if (myfile.is_open()) {
		for (int j = 0; j < n; j++)
		{
			myfile << resultVector[j] << "\n";
		}
	}
	myfile.close();
}
