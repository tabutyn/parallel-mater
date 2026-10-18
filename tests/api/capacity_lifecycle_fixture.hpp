// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace parallel_mater::test {

struct LifecycleResourceObservation {
    std::string id;
    bool active{};
    std::uint64_t generation{};
    std::string state_sha256;
};

struct LifecycleCheckpointObservation {
    std::string id;
    std::string outcome;
    std::vector<LifecycleResourceObservation> resources;
};

struct CapacityLifecycleResult {
    int status{2};
    std::string device;
    std::vector<LifecycleCheckpointObservation> observations;
};

} // namespace parallel_mater::test

int parallel_mater_capacity_recovery_fixture(
    parallel_mater::test::CapacityLifecycleResult *result);
