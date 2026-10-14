// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/scene.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <ostream>
#include <vector>

namespace parallel_mater::benchmark {

[[nodiscard]] inline StepOptions standard_step_options(bool collect_timings) {
    return {.timestep = 1.0F / 60.0F, .substeps = 4U,
            .gravity = {0.0F, -9.81F, 0.0F},
            .collect_kernel_timings = collect_timings};
}

enum class PercentileMode { nearest_sample, nearest_rank };

struct Samples {
    std::vector<double> values{};

    void add(double value) { values.push_back(value); }
    void clear() { values.clear(); }

    [[nodiscard]] double mean() const {
        assert(!values.empty());
        return std::accumulate(values.begin(), values.end(), 0.0) /
               static_cast<double>(values.size());
    }

    [[nodiscard]] double percentile(
        double fraction,
        PercentileMode mode = PercentileMode::nearest_sample) {
        assert(!values.empty() && fraction >= 0.0 && fraction <= 1.0);
        assert(mode != PercentileMode::nearest_rank || fraction > 0.0);
        const auto rank = static_cast<std::size_t>(
            mode == PercentileMode::nearest_rank
                ? std::ceil(fraction * static_cast<double>(values.size())) - 1.0
                : std::round(fraction * static_cast<double>(values.size() - 1U)));
        std::sort(values.begin(), values.end());
        return values[rank];
    }

    void print_summary(std::ostream &output, const char *name) {
        output << name << " median=" << percentile(0.5)
               << " p5=" << percentile(0.05)
               << " p95=" << percentile(0.95) << " ms\n";
    }
};

} // namespace parallel_mater::benchmark
