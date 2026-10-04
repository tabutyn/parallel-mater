// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace parallel_mater::conformance {

enum class CaseKind { analytic, integrated_glb };

struct Command {
    std::uint32_t frame{};
    std::string operation{};
    std::string target{};
    std::array<double, 4U> value{};
    std::uint32_t value_count{};
};

struct CaseDefinition {
    std::string id{};
    std::string title{};
    CaseKind kind{CaseKind::analytic};
    std::string glb_path{};
    std::string glb_sha256{};
    std::uint32_t frames{60U};
    std::uint32_t substeps{4U};
    double timestep{1.0 / 60.0};
    std::array<double, 3U> gravity{0.0, -9.81, 0.0};
    std::string tolerance_profile{"constraint_contact"};
    std::string resources_json{"[]"};
    std::vector<Command> commands{};
    std::vector<std::uint32_t> checkpoints{0U, 60U};
    std::vector<std::string> invariants{};
    std::vector<std::string> coverage{};
    bool chaotic_envelope{};
};

[[nodiscard]] const std::vector<CaseDefinition> &case_registry();
[[nodiscard]] const CaseDefinition *find_case(std::string_view id);
[[nodiscard]] std::string serialize_case(const CaseDefinition &definition);
[[nodiscard]] std::filesystem::path case_file_path(
    const std::filesystem::path &directory, const CaseDefinition &definition);

} // namespace parallel_mater::conformance
