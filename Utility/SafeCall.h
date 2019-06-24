#ifndef SAFECALL_H__
#define SAFECALL_H__

#include <iostream>
#include <cuda_runtime.h>

#if defined(__GNUC__)
    #define SafeCall(expr)  ___SafeCall(expr, __FILE__, __LINE__, __func__)
#else
    #define SafeCall(expr)  ___SafeCall(expr, __FILE__, __LINE__)
#endif

static inline void error(const char *error_string, const char *file, const int line, const char *func) {
    std::cout << "Error: " << error_string << "\t" << file << ":" << line << std::endl;
    exit(0);
}

static inline void ___SafeCall(cudaError_t err, const char *file, const int line, const char *func = "") {
    if (cudaSuccess != err)
        error(cudaGetErrorString(err), file, line, func);
}

static inline int DivUp(int a, unsigned int b) {
	return (a + b - 1) / b;
}

static inline int DivUp(int a, int b) {
	return (a + b - 1) / b;
}

static inline int DivUp(unsigned int a, int b) {
	return (a + b - 1) / b;
}

static inline int DivUp(unsigned int a, unsigned int b) {
	return (a + b - 1) / b;
}

static inline int DivUp(size_t a, unsigned int b) {
	return (a + b - 1) / b;
}

#endif
