//
// Created by Steven on 17/11/25.
//

#include "algutils.h"


inline cudaError_t checkCuda_func(cudaError_t ret, const char * file, const int line)
{
    if(ret != cudaSuccess) {
        printf("cuda operation returned: %s (code %d), in file: %s(%d), the program (pid: %d) exit.\n",
               cudaGetErrorString(ret), ret, file, line, pid);
        fflush(stdout);
        exit(-1);
    }

    return ret;
}


void display_array(float *p, int n)
{
    int n_display = 1024;
    std::cout << "first 100 results: [ ";
    for(int i=0; i < n_display; i++)
        std::cout << p[i] << " ";
    std::cout << " ] " << std::endl;
}