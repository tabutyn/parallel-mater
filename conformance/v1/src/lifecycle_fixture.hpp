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
    std::string active_graph_sha256;
};

struct LifecycleFixtureResult {
    int status{2};
    std::string device;
    std::string reason;
    std::vector<LifecycleCheckpointObservation> observations;
};

using CapacityLifecycleResult = LifecycleFixtureResult;
using GraphLifecycleResult = LifecycleFixtureResult;
using SmokeLifecycleResult = LifecycleFixtureResult;

} // namespace parallel_mater::test

int parallel_mater_capacity_recovery_fixture(
    parallel_mater::test::CapacityLifecycleResult *result);
int parallel_mater_interaction_graph_lifecycle_fixture(
    parallel_mater::test::GraphLifecycleResult *result);
int parallel_mater_smoke_lifecycle_fixture(
    bool grid, parallel_mater::test::SmokeLifecycleResult *result);
