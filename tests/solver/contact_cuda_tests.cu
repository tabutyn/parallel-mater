// SPDX-License-Identifier: MIT
#include "contact_cases.hpp"
#include <cuda_runtime.h>
#include <iostream>

__global__ void evaluate_contacts(contact_cases::Result *results) {
    if (threadIdx.x < contact_cases::count)
        results[threadIdx.x] = contact_cases::evaluate(threadIdx.x);
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    contact_cases::Result *results = nullptr;
    if (cudaMallocManaged(&results, contact_cases::count * sizeof(*results)) != cudaSuccess) return 1;
    evaluate_contacts<<<1, 32>>>(results);
    bool passed = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    for (unsigned index = 0; passed && index < contact_cases::count; ++index) {
        passed = contact_cases::close(results[index], contact_cases::expected(index)) &&
                 contact_cases::close(results[index], contact_cases::evaluate(index));
        if (!passed) std::cerr << "CUDA/host contact fixture failed: " << index << '\n';
    }
    cudaFree(results);
    return passed ? 0 : 1;
}
