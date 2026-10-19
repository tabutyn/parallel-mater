// SPDX-License-Identifier: MIT
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace parallel_mater::conformance {

enum class ExecutionStatus { completed, unsupported, failed };

struct CaseExecution {
    std::string case_id{};
    std::string json{};
    ExecutionStatus status{ExecutionStatus::failed};
};

[[nodiscard]] std::vector<std::string> conformance_case_ids();
[[nodiscard]] CaseExecution execute_conformance_case(
    const std::string &case_id, const std::filesystem::path &root,
    bool every_frame = false);
[[nodiscard]] std::string conformance_device_provenance(
    const std::filesystem::path &root);
[[nodiscard]] std::string conformance_advertised_capabilities(
    const std::filesystem::path &root);
[[nodiscard]] std::string conformance_rigid_brick_quality(
    const std::filesystem::path &root);
[[nodiscard]] std::string conformance_rigid_attachment_quality(
    const std::filesystem::path &root);
[[nodiscard]] std::string conformance_rope_self_contact_quality(
    const std::filesystem::path &root);
[[nodiscard]] std::string conformance_fluid_cloth_containment_quality(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_fluid_soft_contact_quality(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_fluid_rope_contact_quality(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_smoke_self_viscosity_quality(
    const std::filesystem::path &root);
[[nodiscard]] std::string conformance_smoke_pressure_quality(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_fluid_cloth_contact_quality(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_smoke_fluid_drag_quality(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_fluid_smoke_phase_quality(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_externally_heated_fluid_quality(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_fluid_pair_quality(
    const std::filesystem::path &temporary_output);
[[nodiscard]] std::string conformance_fluid_self_quality(
    const std::filesystem::path &temporary_output,
    const std::string &profile_id,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_adverse_lifecycle(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_fluid_cloth_containment_lifecycle(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_smoke_lifecycle(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_smoke_pressure_lifecycle(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_fluid_cloth_contact_lifecycle(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_smoke_fluid_drag_lifecycle(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_fluid_smoke_phase_lifecycle(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);
[[nodiscard]] std::string conformance_externally_heated_fluid_lifecycle(
    const std::filesystem::path &root,
    ExecutionStatus *status = nullptr);

int conformance_command_main(int argc, char **argv);

} // namespace parallel_mater::conformance
