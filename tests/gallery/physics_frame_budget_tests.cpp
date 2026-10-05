// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/physics_frame_budget.hpp>

#include <iostream>

int main() {
    using parallel_mater::gallery::PhysicsFrameBudget;
    auto check = [](bool ok, const char *message) {
        if (!ok) std::cerr << message << '\n';
        return ok;
    };
    bool ok = true;
    PhysicsFrameBudget slow;
    ok &= check(slow.can_step(), "A due first step must always be allowed");
    slow.record_step(0.044);
    ok &= check(!slow.can_step(), "A slow frame must not multiply stalls by catching up");
    PhysicsFrameBudget predicted;
    predicted.record_step(0.005);
    ok &= check(!predicted.can_step(), "Reserve room for rendering before another expensive step");
    PhysicsFrameBudget fast;
    for (unsigned i = 0; i < 4; ++i) {
        ok &= check(fast.can_step(), "Fast steps may catch up");
        fast.record_step(0.001);
    }
    ok &= check(!fast.can_step(), "The four-step cap still applies");
    PhysicsFrameBudget exact(4, 0.008);
    exact.record_step(0.004);
    ok &= check(exact.can_step(), "A second step may fit exactly");
    exact.record_step(0.004);
    ok &= check(!exact.can_step(), "Exhausted work budget must stop catch-up");
    PhysicsFrameBudget next_frame;
    ok &= check(next_frame.can_step(), "A fresh display frame permits physics again");
    return ok ? 0 : 1;
}
