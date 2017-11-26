/*
 *  A tutorial program for cuda programming. It implement algorithm of tensordot operation.
 *
 *  by Steven Liu <stevenliucx@gmail.com>
 *  Nov 26, 2017
 *  
 */


#include <stdio.h>
#include <cuda.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include "algutils.h"




/*
 * A macro for check cuda function return status. It will call checkCuda_func.
 */
#define checkCuda(ret)  checkCuda_func( (cudaError_t)(ret), __FILE__, __LINE__ )

/*
 * Check cuda return status. exit program when error occur.
 */
inline cudaError_t checkCuda_func(cudaError_t ret, const char * file, const int line)
{
    if(ret != cudaSuccess) {
        fprintf(stderr, "cuda operation returned: %s (code %d), in file: %s(%d), the program (pid: %d) exit.\n",
               cudaGetErrorString(ret), ret, file, line, getpid());
        fflush(stderr);
        exit(-1);
    }

    return ret;
}

typedef Shape std::vector<size_t>;

size_t get_shape_size(Shape shape)
{
    size_t n_elems = 0;
    for(size_t i:shape)
        n_elems *= i;
    return n_elems;
}

float *init_tensor(Shape shape, float default_val)
{
    float *p;
    size_t n_elems = get_shape_size(shape);

    p = (float*)malloc(n_elems*sizeof(float));
    for(int i=0; i < n_elems; i ++) 
       p[i] = default_val;

    return p;
}

/*
 *  Kernel function for 3D tensor dot. Just for 3D dot.
 *  Shape is: A[n0, n1, n2] * B[n1, n2, n3] = C[n0, n3]
 */
__global__ void kernel_tensor3D_dot(float* d_TA, float* d_TB, float *d_TC,
                                    size_t n0, size_t n1, size_t n2, size_t n3)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;	
    int ty = blockIdx.y * blockDim.y + threadIdx.y;	

    float p_val = 0.0;

    int idx = tx*n_rows + ty; 
    for(int j=0; j<n1; j++)
        for(int k=0; k<n2; k++)
            p_val += d_TA[tx*n1*n2+j*n2+k] * d_TB[j*n2*n3+k*n3+ty];

    d_TC[tx*n1+ty] = p_val;
}

/*
 * Cuda version of tensor dot operation.
 */
void tensor_dot_cu(float *TA, Shape shapeA, float *TB,
                   Shape shapeB, float *TC, Shape shapeC)
{


    float *d_TA, *d_TB, *d_TC;    // corresponding device memory pointer.

    checkCuda( cudaMalloc(&d_TA, get_shape_size(shapeA))*sizeof(float) );
    checkCuda( cudaMalloc(&d_TB, get_shape_size(shapeB))*sizeof(float) );
    checkCuda( cudaMalloc(&d_TC, get_shape_size(shapeC))*sizeof(float) );

    checkCuda( cudaMemcpy(d_TA, TA, get_shape_size(shapeA)*sizeof(float),
                          cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_TB, TB, get_shape_size(shapeB)*sizeof(float),
                          cudaMemcpyHostToDevice) );

    dim3 dimBlock(16, 16);
    dim3 dimGrid(shapeC[0]/dimBlock.x, shapeC[1]/dimBlock.y);
    matrix_mul_kernel<<<dimGrid, dimBlock>>>(d_TA, d_TB, d_TC,
            shapeA[0], shapeA[1], shapeA[2], shapeB[2]);


    checkCuda( cudaMemcpy(TC, d_TC, get_shape_size(shapeC)*sizeof(float),
                          cudaMemcpyDeviceToHost) );

    checkCuda( cudaFree(d_TA) );
    checkCuda( cudaFree(d_TB) );
    checkCuda( cudaFree(d_TC) );
}

void display_array(float *p, size_t n, size_t offset=0)
{
    std::cout << "first 100 results: [ ";
    p += offset;
    for(int i=0; i < n; i++, p++)
       std::cout << *p << " ";
    std::cout << " ] " << std::endl;
}


int main(int argc, char *argv[])
{

    Shape shapeA{64,32,128};
    Shape shapeB{32,128,16};
    Shape shapeC{shapeA[0], shapeB[2]};

    float *TA = init_tensor(shapeA, 1.0);
    float *TB = init_tensor(shapeB, 2.0);
    float *TC = init_tensor(shapeC, 0.0);

    tensor_dot_cu(TA, shapeA, TB, shapeB, TC, shapeC);

	display_array(TA, 100);
	display_array(TB, 100);
	display_array(TC, 100);

    free(TA);
    free(TB);
    free(TC);
}



