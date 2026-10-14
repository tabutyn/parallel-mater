// SPDX-License-Identifier: MIT
#include "contact_friction_cases.hpp"

#include <cuda_runtime.h>

#include <iostream>

__global__ void evaluate_contact_friction(bool *results) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < contact_friction_cases::count)
        results[index] = contact_friction_cases::evaluate(index);
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    bool *results = nullptr;
    if (cudaMallocManaged(&results,
                          contact_friction_cases::count * sizeof(*results)) !=
        cudaSuccess)
        return 1;
    evaluate_contact_friction<<<1U, 32U>>>(results);
    bool passed = cudaGetLastError() == cudaSuccess &&
        cudaDeviceSynchronize() == cudaSuccess;
    for (unsigned index = 0U;
         passed && index < contact_friction_cases::count; ++index) {
        passed = results[index] && contact_friction_cases::evaluate(index);
        if (!passed)
            std::cerr << "CUDA/host contact-friction fixture failed: "
                      << index << '\n';
    }
    cudaFree(results);
    return passed ? 0 : 1;
}
