//
// Created by Steven on 17/11/25.
//

#ifndef CUDA_TRAIN_BASIC_UTILS_H
#define CUDA_TRAIN_BASIC_UTILS_H

#define checkCuda(ret)  checkCuda_func( (cudaError_t)(ret), __FILE__, __LINE__ )

inline cudaError_t checkCuda_func(cudaError_t ret, const char * file, const int line);

#endif //CUDA_TRAIN_BASIC_UTILS_H
