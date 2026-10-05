// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

namespace parallel_mater::gallery {

// Always permit the first due fixed step. Catch up only while another step,
// estimated from the last one, fits the display frame's physics work budget.
// This bounds extra work, not a single step's duration or its simulated dt.
class PhysicsFrameBudget {
  public:
    explicit PhysicsFrameBudget(std::uint32_t maximum_steps = 4U,
                                double seconds = 0.008) noexcept
        : maximum_steps_(maximum_steps), seconds_(seconds) {}

    [[nodiscard]] bool can_step() const noexcept {
        return steps_ < maximum_steps_ &&
               (steps_ == 0U || elapsed_ + last_step_ <= seconds_);
    }

    void record_step(double seconds) noexcept {
        ++steps_;
        elapsed_ += seconds;
        last_step_ = seconds;
    }

  private:
    std::uint32_t maximum_steps_{};
    std::uint32_t steps_{};
    double seconds_{};
    double elapsed_{};
    double last_step_{};
};

} // namespace parallel_mater::gallery
