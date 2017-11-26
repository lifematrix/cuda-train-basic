/*
 *  A tutorial program for cuda programming. It implement algorithm of matrix multipling.
 *  Steven Liu
 *  
 */


#include <stdio.h>
#include <cuda.h>
#include <iostream>

#define checkCuda(ret)  checkCuda_func( (cudaError_t)(ret), __FILE__, __LINE__ )


int pid = 0;

float * init_matrix(int n_rows, int n_cols, float default_val)
{
    float *p;
    int n_elems = n_rows*n_cols;

    p = (float*)malloc(n_elems*sizeof(float));
    for(int i=0; i < n_elems; i ++) 
       p[i] = default_val;

    return p;
}

float * init_array(int n_len, float default_val)
{
    p = (float*)malloc(n_len*sizeof(float));
    for(int i=0; i < n_len; i ++)
        p[i] = default_val;

    return p;
}

__global__ void matrix_add_kernel(float* d_mA, float* d_mB, float *d_mP, int n_rows, int n_cols)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;	
    int ty = blockIdx.y * blockDim.y + threadIdx.y;	

    int idx = tx*n_rows + ty; 
    d_mP[idx] = d_mA[idx] + d_mB[idx];
}

__global__ void matrix_mul_kernel(float* d_mA, float* d_mB, float *d_mP, int n_rows, int n_cols)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;	
    int ty = blockIdx.y * blockDim.y + threadIdx.y;	

    float p_val = 0.0;

    int idx = tx*n_rows + ty; 
    for(int k=0; k < n_rows; k++) {
        p_val += d_mA[k*n_cols+tx] * d_mB[ty*n_cols+k];
    }

    d_mP[tx*n_rows+ty] = p_val;
}

void matrix_mul_on_device(float *mA, float *mB, float *mP, int n_rows, int n_cols)
{
    int n_elems = n_rows*n_cols;
    int size = n_elems*sizeof(float);

    float *d_mA, *d_mB, *d_mP;

    checkCuda( cudaMalloc(&d_mA, size) );
    checkCuda( cudaMalloc(&d_mB, size) );
    checkCuda( cudaMalloc(&d_mP, size) );
    checkCuda( cudaMemcpy(d_mA, mA, size, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_mB, mB, size, cudaMemcpyHostToDevice) );
    
    int block_size=32;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(n_rows/block_size, n_cols/block_size);
    matrix_mul_kernel<<<dimGrid, dimBlock>>>(d_mA, d_mB, d_mP, n_rows, n_cols);


    checkCuda( cudaMemcpy(mP, d_mP, size, cudaMemcpyDeviceToHost) );
    checkCuda( cudaFree(d_mA) );
    checkCuda( cudaFree(d_mB) );
    checkCuda( cudaFree(d_mP) );
    
}

void reduction_min_max(float *mR, float *mA, int n_len)
{
    for(int i=0; i<n_len; i++)
        for(int j=0; j<n_len; j++) {
            mA[i*n_len+j] = mR[i] * mR[j];
        }

}


int main(int argc, char *argv[])
{
    int n_len=8192;

    float *R = init_array(n_len, 1.0);
    float *mA = init_matrix(n_len, n_len, 0.0);


    float *mB = init_matrix(n_rows, n_cols, 3.0);
    float *mP = init_matrix(n_rows, n_cols, 0.0);

    matrix_mul_on_device(mA, mB, mP, n_rows, n_cols);
	display_array(mA, 100);
	display_array(mB, 100);
	display_array(mP, 100);

    free(mA);
    free(mB);
    free(mP);
}


