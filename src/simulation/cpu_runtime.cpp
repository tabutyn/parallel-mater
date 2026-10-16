// SPDX-License-Identifier: MIT

// Resource setup and completion use the host compatibility adapter. The frame
// algorithm lives in reference_schedule.hpp and binds through the same native
// operation adapter as CUDA; physics still enters generated Slang modules.
#include "runtime.cu"
