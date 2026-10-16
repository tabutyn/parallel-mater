// SPDX-License-Identifier: MIT
#pragma once

#include "cuda_runtime_api.h"

#include <cmath>

inline float rsqrtf(float value) noexcept { return 1.0F / std::sqrt(value); }
