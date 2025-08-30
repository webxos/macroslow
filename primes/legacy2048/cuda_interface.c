#ifndef CUDA_INTERFACE_H
#define CUDA_INTERFACE_H

// Team Instruction: Define C/C++ header for CUDA integration with CUDASieve.
// Enable interop with CPython and Fortran, inspired by Emeagwali’s parallel architecture.
#ifdef __cplusplus
extern "C" {
#endif

void* sieve_cuda(unsigned int* input, size_t size, size_t* prime_count);
void free_cuda(void* ptr);

#ifdef __cplusplus
}
#endif

#endif