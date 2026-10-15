// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

namespace parallel_mater::simulation {
enum class AvbdStage : std::uint32_t {
#define PM_AVBD_STAGE(name, value) name = value,
#include "../slang/avbd_stages.inc"
#undef PM_AVBD_STAGE
};

// Each invocation is a separate dispatch with a device-memory dependency on
// the preceding one. Bounds must cover the prepared schedule (at most one
// color per dynamic body; iterations include enabled joint policy). The GPU
// guards unused colors/iterations and absent impact phases. No host readback
// is needed. Adapters may optimize empty dispatches with indirect work later.
template<class Emit>
void encode_avbd_schedule(std::uint32_t iteration_bound,
                          std::uint32_t color_bound, Emit &&emit) {
    for (std::uint32_t phase = 0; phase < 2; ++phase) {
        const auto stage = [&](AvbdStage operation) {
            emit(operation, phase, 0U, 0U);
        };
        if (phase != 0) {
            stage(AvbdStage::impact_gaps);
            stage(AvbdStage::impact_bodies);
            stage(AvbdStage::impact_constraints);
        }
        for (std::uint32_t iteration = 0; iteration < iteration_bound; ++iteration) {
            for (std::uint32_t color = 0; color < color_bound; ++color)
                emit(AvbdStage::solve_color, phase, iteration, color);
            emit(AvbdStage::advance_duals, phase, iteration, 0U);
        }
        stage(AvbdStage::free_mean);
        stage(AvbdStage::free_subtract);
        stage(AvbdStage::free_clear);
        stage(AvbdStage::apply_states);
        if (phase == 0) stage(AvbdStage::project_hinges);
        stage(AvbdStage::finish_phase);
    }
}
} // namespace parallel_mater::simulation
