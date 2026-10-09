// SPDX-License-Identifier: MIT
#include "avbd_cases.hpp"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void evaluate_avbd(avbd_cases::Result *results) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < avbd_cases::count) results[index] = avbd_cases::evaluate(index);
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    avbd_cases::Result *results = nullptr;
    if (cudaMallocManaged(&results, avbd_cases::count * sizeof(*results)) != cudaSuccess) return 1;
    evaluate_avbd<<<(avbd_cases::count + 63)/64, 64>>>(results);
    const bool completed = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    bool passed = completed;
    for (unsigned index = 0; completed && index < avbd_cases::count; ++index) {
        const auto host = avbd_cases::evaluate(index);
        const bool device_valid = avbd_cases::validate(index, results[index]);
        const bool host_valid = avbd_cases::validate(index, host);
        const bool equivalent = avbd_cases::equivalent(index, results[index], host);
        if (!device_valid || !host_valid || !equivalent) {
            passed = false;
            std::fprintf(stderr, "AVBD CUDA/host fixture %u failed (%s): device=%d host=%d parity=%d\n",
                         index, avbd_cases::name(index), device_valid, host_valid, equivalent);
            for (unsigned component = 0; component < 8; ++component)
                std::fprintf(stderr, "  [%u] device=%.9g host=%.9g\n", component,
                             results[index].value[component], host.value[component]);
        }
    }
    cudaFree(results);
    if (passed) std::printf("%u portable AVBD CUDA/host fixtures passed\n", avbd_cases::count);
    return passed ? 0 : 1;
}
