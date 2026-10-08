// SPDX-License-Identifier: MIT

// The CUDA and Metal producers intentionally share one implementation so the
// result schema, aggregation, stable sampling, commands, and checkpoint order
// cannot drift between backends. Backend-specific uploads/readbacks and
// provenance are selected inside the implementation.
#define PARALLEL_MATER_CONFORMANCE_METAL 1
#include "cuda_runner.cpp"
