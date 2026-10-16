// SPDX-License-Identifier: MIT
#include <parallel_mater/pm_layer_args.h>
#include <parallel_mater_scene/command_scene.hpp>
#include <parallel_mater_scene/preparation.hpp>
#include <cmath>
#include <vector>

int main() {
    const parallel_mater::Vec3 line[]{{0, 0, 0}, {0, 1, 0}};
    std::vector<parallel_mater::Vec3> samples;
    const auto status = parallel_mater::sample_rope_centerline({line, 2}, .25F, samples);
    if (!status.ok() || samples.size() != 5 ||
        std::abs(samples.back().y - 1.0F) > 1e-6F) return 1;
    // Instantiate the public command client, including its out-of-line lifetime.
    parallel_mater::gallery::CommandScene scene;
    return 0;
}
