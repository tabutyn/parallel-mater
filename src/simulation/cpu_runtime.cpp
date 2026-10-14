// SPDX-License-Identifier: MIT

// The host adapter deliberately reuses the one shared command scheduler.  Its
// CUDA-shaped service calls resolve to cpu_compat/cuda_runtime_api.h, while all
// physics execution still enters generated Slang dispatch modules.
#include "runtime.cu"
