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
#include <time.h>
#include <stdarg.h>

#include <iostream>
#include <vector>


clock_t time_start;
void log_info(const char* format, ...)
{
    clock_t t = clock() - time_start;
    static clock_t last = 0;
    if(last == 0)
        last = t;

    printf("------- [%12.3fms][%12.3fms] ", (float)t/(CLOCKS_PER_SEC/1000.0), (float)(t-last)/(CLOCKS_PER_SEC/1000.0));
    va_list ap;
    va_start(ap, format);
    vprintf(format, ap);
    printf("\n");
    va_end(ap);

	fflush(stdout);
    last = t;
}

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

/*
 * Define a type for shape of tensor.
 */
typedef std::vector<size_t> Shape;

/*
 * Get num of elements of shape.
 */
size_t get_shape_size(Shape shape)
{
    size_t n_elems = 1;
    for(size_t i:shape)
        n_elems *= i;
    return n_elems;
}

float *init_tensor(Shape shape, float default_val)
{
    float *p;
    size_t n_elems = get_shape_size(shape);
	// log_info("n_elems: %d", n_elems);

    p = (float *)malloc(n_elems*sizeof(float));
    for(int i=0; i<n_elems; i++) 
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
    float tmp = 0.0;

	// Check if out of bound of TC
	if (tx >= n0||ty >= n3)
		return;

    for(int j=0; j<n1; j++)
        for(int k=0; k<n2; k++)
            tmp += d_TA[tx*n1*n2+j*n2+k] * d_TB[j*n2*n3+k*n3+ty];

    d_TC[tx*n3+ty] = tmp;
}

/*
 * Cuda version of tensor dot operation.
 */
void tensor3D_dot_cu(float *TA, Shape shapeA, float *TB,
                   Shape shapeB, float *TC, Shape shapeC)
{
    float *d_TA, *d_TB, *d_TC;    // corresponding device memory pointer.

	log_info("cudaMalloc");
    checkCuda( cudaMalloc(&d_TA, get_shape_size(shapeA)*sizeof(float)) );
    checkCuda( cudaMalloc(&d_TB, get_shape_size(shapeB)*sizeof(float)) );
    checkCuda( cudaMalloc(&d_TC, get_shape_size(shapeC)*sizeof(float)) );

	log_info("copy data to device");
    checkCuda( cudaMemcpy(d_TA, TA, get_shape_size(shapeA)*sizeof(float),
                          cudaMemcpyHostToDevice) );
	log_info("size: %d", get_shape_size(shapeB)*sizeof(float));
    checkCuda( cudaMemcpy(d_TB, TB, get_shape_size(shapeB)*sizeof(float),
                          cudaMemcpyHostToDevice) );

    dim3 dimBlock(16, 16);
	// It is a algorithm trick. For get ceiling(M/N), you can caclulate: (M-1)/N + 1
	// Ref: <https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c>
    dim3 dimGrid((shapeC[0]-1)/dimBlock.x+1, (shapeC[1]-1)/dimBlock.y+1);

	log_info("launch kernel");
    kernel_tensor3D_dot<<<dimGrid, dimBlock>>>(d_TA, d_TB, d_TC,
            shapeA[0], shapeA[1], shapeA[2], shapeB[2]);
	cudaDeviceSynchronize();
	checkCuda( cudaGetLastError() );
	log_info("kernel execution successfully");

	log_info("copy back result in device to host memory");
    checkCuda( cudaMemcpy(TC, d_TC, get_shape_size(shapeC)*sizeof(float),
                          cudaMemcpyDeviceToHost) );

    checkCuda( cudaFree(d_TA) );
    checkCuda( cudaFree(d_TB) );
    checkCuda( cudaFree(d_TC) );
}

/*
 * CPU version of tensor dot operation to compare performance with cuda version.
 */
void tensor3D_dot_cpu(float *TA, Shape shapeA, float *TB,
                   Shape shapeB, float *TC, Shape shapeC)
{
	float tmp;
	size_t n0 = shapeA[0], n1 = shapeA[1], n2 = shapeA[2], n3 = shapeC[1];

	for(size_t i=0; i<n0; i++)
		for(size_t l=0; l<n3; l++) {
			tmp = 0.0;
			for(size_t j=0; j<n1; j++)
				for(size_t k=0; k<n2; k++)
					tmp += TA[i*n1*n2+j*n2+k] * TB[j*n2*n3+k*n3+l];
			TC[i*n3+l] = tmp;
		}	
}

void display_array(float *p, size_t n, size_t offset=0)
{
    std::cout << "**first " << n << "(+" << offset << ") results** : [ ";
    p += offset;
    for(int i=0; i < n; i++)
       std::cout << (int)p[i] << " ";
    std::cout << " ] " << std::endl;
}

/*
 * get sum of all elements of an array.
 * return double precision to prevent overflow.
 */
double sum_array(float *p, size_t n)
{
	double s = 0.0;
	for(int i=0; i<n; i++, p++)
		s += *p;

	return s;
}



/*
 *  MAIN FUNCTION
 */
int main(int argc, char *argv[])
{
	const float init_val_A = 2.0; 
	const float init_val_B = 1.5; 

	time_start = clock();
    log_info("program start");

    Shape shapeA = {64,32,128};
    Shape shapeB = {32,128,16};
    Shape shapeC = {shapeA[0], shapeB[2]}; 

	log_info("initialize tensor");
    float *TA = init_tensor(shapeA, init_val_A);
    float *TB = init_tensor(shapeB, init_val_B);
    float *TC_cu = init_tensor(shapeC, 0.0);
    float *TC_cpu = init_tensor(shapeC, 0.0);

	log_info("tensor3D_dot_cu");
    tensor3D_dot_cu(TA, shapeA, TB, shapeB, TC_cu, shapeC);
	log_info("tensor3D_dot_cu OK");

	log_info("tensor3D_dot_cpu");
    tensor3D_dot_cpu(TA, shapeA, TB, shapeB, TC_cpu, shapeC);
	log_info("tensor3D_dot_cpu OK");

	// Check correctness of calculation.
    double t1, t2;
    t1 = sum_array(TC_cu, get_shape_size(shapeC));
    // t2 is correct answer in theroy.
    t2 = init_val_A*init_val_B*shapeA[1]*shapeA[2]*get_shape_size(shapeC);

	log_info("sum of TC_cu: %f, correct answer: %f, equal: %s",
		t1, t2, t1==t2?"TRUE":"FLASE");

    t1 = sum_array(TC_cpu, get_shape_size(shapeC));
	log_info("sum of TC_cpu: %f, correct answer: %f, equal: %s",
             t1, t2, t1== t2?"TRUE":"FLASE");

	// display first n elements for check.
	display_array(TA, 100);
	display_array(TB, 100);
	display_array(TC_cu, 100);
	display_array(TC_cpu, 100);

    free(TA);
    free(TB);
    free(TC_cu);
    free(TC_cpu);

    log_info("program end");
}



