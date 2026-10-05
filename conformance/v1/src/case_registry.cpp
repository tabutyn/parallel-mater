// SPDX-License-Identifier: MIT
#include <parallel_mater_conformance/case_registry.hpp>

#include <algorithm>
#include <charconv>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace parallel_mater::conformance {
namespace {

std::string quote(std::string_view value) {
    std::string result{"\""};
    for (const unsigned char byte : value) {
        switch (byte) {
        case '\"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\b': result += "\\b"; break;
        case '\f': result += "\\f"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (byte < 0x20U) {
                std::ostringstream escaped;
                escaped << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<unsigned>(byte);
                result += escaped.str();
            } else {
                result.push_back(static_cast<char>(byte));
            }
        }
    }
    result.push_back('\"');
    return result;
}

std::string number(double value) {
    std::array<char, 64U> buffer{};
    const auto [end, error] = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value,
        std::chars_format::general, std::numeric_limits<double>::max_digits10);
    if (error != std::errc{}) throw std::runtime_error("float serialization failed");
    return {buffer.data(), end};
}

std::string strings(const std::vector<std::string> &values) {
    std::string result{"["};
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) result.push_back(',');
        result += quote(values[index]);
    }
    result.push_back(']');
    return result;
}

std::string checkpoints(const std::vector<std::uint32_t> &values) {
    std::string result{"["};
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) result.push_back(',');
        result += std::to_string(values[index]);
    }
    result.push_back(']');
    return result;
}

std::string commands(const std::vector<Command> &values) {
    std::string result{"["};
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) result.push_back(',');
        const Command &command = values[index];
        result += "{\"frame\":" + std::to_string(command.frame) +
                  ",\"operation\":" + quote(command.operation);
        if (!command.target.empty())
            result += ",\"target\":" + quote(command.target);
        if (command.value_count != 0U) {
            result += ",\"value\":[";
            for (std::uint32_t component = 0U;
                 component < command.value_count; ++component) {
                if (component != 0U) result.push_back(',');
                result += number(command.value[component]);
            }
            result.push_back(']');
        }
        result.push_back('}');
    }
    result.push_back(']');
    return result;
}

CaseDefinition integrated(
    std::string id, std::string title, std::string glb, std::string sha,
    std::vector<std::string> coverage, std::uint32_t frames = 60U,
    std::string tolerance = "constraint_contact",
    std::vector<Command> case_commands = {}, bool chaotic = false) {
    CaseDefinition result{};
    result.id = std::move(id);
    result.title = std::move(title);
    result.kind = CaseKind::integrated_glb;
    result.glb_path = "examples/assets/" + glb + ".glb";
    result.glb_sha256 = std::move(sha);
    result.frames = frames;
    result.checkpoints = {0U, frames / 2U, frames};
    result.tolerance_profile = std::move(tolerance);
    result.resources_json = "[{\"definition\":{\"path\":" +
        quote(result.glb_path) + "},\"name\":\"scene\",\"type\":\"glb\"}]";
    result.commands = std::move(case_commands);
    result.invariants = {"finite_state", "stable_ids", "topology",
                         "bounded_energy", "clearance"};
    result.coverage = std::move(coverage);
    if (std::find(result.coverage.begin(), result.coverage.end(),
                  "equal_opposite_transfer") != result.coverage.end())
        result.invariants.push_back("equal_and_opposite_transfer");
    result.chaotic_envelope = chaotic;
    return result;
}

CaseDefinition analytic_rigid_direct() {
    CaseDefinition result{};
    result.id = "rigid-direct";
    result.title = "Rigid integration, impulse, contact, kinematic target, and lifecycle";
    result.frames = 60U;
    result.tolerance_profile = "direct";
    result.resources_json = R"json([{"definition":{"half_extents":[0.25,0.25,0.25],"inertia_diagonal":[0.083333333333333329,0.083333333333333329,0.083333333333333329],"mass":2,"motion":"dynamic","position":[0,2,0]},"name":"dynamic-box","type":"rigid_body"},{"definition":{"half_extents":[3,0.050000000000000003,3],"motion":"static","position":[0,-0.050000000000000003,0]},"name":"ground","type":"rigid_body"},{"definition":{"half_extents":[0.20000000000000001,0.20000000000000001,0.20000000000000001],"motion":"kinematic","position":[-1,1,0]},"name":"kinematic-box","type":"rigid_body"}])json";
    result.commands = {
        {0U, "apply_impulse", "dynamic-box", {1.5, 0.0, 0.0, 0.0}, 3U},
        {20U, "set_kinematic_target", "kinematic-box", {1.0, 1.5, 0.0, 0.0}, 3U},
        {40U, "replace_body", "dynamic-box", {0.0, 2.0, 0.0, 0.0}, 3U}};
    result.checkpoints = {0U, 1U, 20U, 40U, 60U};
    result.invariants = {"finite_state", "semi_implicit_integration",
                         "stable_resource_names", "generation_changes_on_replace",
                         "nonpenetration"};
    result.coverage = {"rigid_integration", "impulse", "contact", "static_body",
                       "kinematic_body", "lifecycle_edit"};
    return result;
}

CaseDefinition analytic_weld() {
    CaseDefinition result{};
    result.id = "compound-weld-lifecycle";
    result.title = "Compound weld pickup, release, and rebuild";
    result.frames = 72U;
    result.tolerance_profile = "constraint_contact";
    result.resources_json = R"json([{"definition":{"count":3,"half_extents":[0.20000000000000001,0.20000000000000001,0.20000000000000001],"mass":1,"motion":"dynamic","spacing":0.40000000000000002},"name":"weld-members","type":"rigid_body_group"},{"definition":{"breaking_impulse_threshold":0,"disable_collisions":true,"enabled":true,"solver_iterations":8,"type":"fixed"},"name":"weld-links","type":"rigid_constraint_group"}])json";
    result.commands = {
        {0U, "apply_impulse", "member-0", {2.0, 0.5, 0.0, 0.0}, 3U},
        {24U, "remove_constraints", "weld-links", {}, 0U},
        {48U, "add_constraints", "weld-links", {}, 0U}};
    result.checkpoints = {0U, 1U, 24U, 25U, 48U, 49U, 72U};
    result.invariants = {"finite_state", "fixed_members_move_together",
                         "release_changes_enabled_state", "rebuild_preserves_handles"};
    result.coverage = {"fixed_joint", "compound_weld", "pickup", "release",
                       "lifecycle_edit"};
    return result;
}

CaseDefinition analytic_breaking() {
    CaseDefinition result{};
    result.id = "constraint-breaking";
    result.title = "Breakable fixed constraint state transition";
    result.frames = 24U;
    result.tolerance_profile = "constraint_contact";
    result.resources_json = R"json([{"definition":{"count":2,"half_extents":[0.20000000000000001,0.20000000000000001,0.20000000000000001],"mass":1,"motion":"dynamic"},"name":"break-members","type":"rigid_body_group"},{"definition":{"breaking_impulse_threshold":0.050000000000000003,"disable_collisions":true,"enabled":true,"solver_iterations":8,"type":"fixed"},"name":"break-link","type":"rigid_constraint"}])json";
    result.commands = {{0U, "apply_impulse", "break-b", {4.0, 0.0, 0.0, 0.0}, 3U}};
    result.checkpoints = {0U, 1U, 2U, 24U};
    result.invariants = {"finite_state", "broken_state_is_sticky",
                         "broken_constraint_is_disabled"};
    result.coverage = {"fixed_joint", "breaking", "impulse", "enabled_state"};
    return result;
}

} // namespace

const std::vector<CaseDefinition> &case_registry() {
    static const std::vector<CaseDefinition> registry = [] {
        std::vector<CaseDefinition> result;
        result.push_back(analytic_rigid_direct());
        result.push_back(analytic_weld());
        result.push_back(analytic_breaking());
        result.push_back(integrated(
            "passive-active", "Static, kinematic, and dynamic authored bodies",
            "PassiveActive", "38082fc757cd767eedb079f2256440199e4a41457d25e944bda3fedfa9427051",
            {"rigid_integration", "static_body", "kinematic_body", "contact"}, 48U,
            "constraint_contact",
            {{16U, "set_first_kinematic_target", "scene", {0.5, 1.0, 0.0, 0.0}, 3U}}));
        result.push_back(integrated(
            "constraint-fixed", "Authored fixed collector and welded assembly",
            "ConstraintFixed", "0c9e0bfc4f2629808e3c458967039776d64459b08fdccd9a1c234e9ba8c6e1ad",
            {"fixed_joint", "compound_weld", "contact"}));
        result.push_back(integrated(
            "constraint-point", "Four authored point constraints",
            "ConstraintPoint", "9cb2893d41defed5def2ae97448f1a5db33172333beb6bf4bcc3e3faeb504f08",
            {"point_joint", "enabled_state"}, 60U, "constraint_contact",
            {{20U, "set_constraints_enabled", "scene", {0.0, 0.0, 0.0, 0.0}, 1U},
             {40U, "set_constraints_enabled", "scene", {1.0, 0.0, 0.0, 0.0}, 1U}}));
        result.push_back(integrated(
            "constraint-hinge", "Three-gear authored hinge train",
            "ConstraintHinge", "9ec3cb5f5a82c83ae9e3532823ef2ec3865f3a0cc3541169ed46a698270393c6",
            {"hinge_joint", "contact", "alternating_rotation", "energy"}, 90U));
        result.push_back(integrated(
            "constraint-slider", "Authored slider constraint",
            "ConstraintSlider", "d5a550dda612db759de680db480ada6c40a00f71e0621165f285671453c3a0a1",
            {"slider_joint", "limits", "contact"}));
        result.push_back(integrated(
            "constraint-piston", "Authored piston constraint",
            "ConstraintPiston", "d5228ddac1ed86ca947342653ce2fba9bc15cba4d1d602dd6b00ec607e1ec201",
            {"piston_joint", "limits", "contact"}));
        result.push_back(integrated(
            "constraint-generic", "Authored generic constraint",
            "ConstraintGeneric", "373331fb7022047a29d2e5031d0fcd0625e1b0337ce23758ce4822ff27f8dab1",
            {"generic_joint", "linear_limits", "angular_limits"}));
        result.push_back(integrated(
            "constraint-generic-spring", "Authored generic spring constraint",
            "ConstraintGenericSpring", "79d2ee43e3d2e879a1190328d2e93d8493c1747465079c61efcafae69362404d",
            {"generic_spring_joint", "springs", "damping"}));
        result.push_back(integrated(
            "constraint-motor", "Authored motor constraint",
            "ConstraintMotor", "1e2a313ab4d3e31a689b99a68fa62462d6d4bc65e13f77b9097380d13538b5b7",
            {"motor_joint", "linear_motor", "angular_motor"}));
        result.push_back(integrated(
            "fluid-lifecycle", "Fluid source, outflow, foam, and stable IDs",
            "Fluid", "d638064df8f044d65c0579eb006138c048480a8bc4fdaa5ddf6ce49107ee74d7",
            {"fluid", "particle_source", "outflow", "foam", "stable_particle_ids"},
            48U, "deformable", {}, true));
        result.push_back(integrated(
            "fluid-rigid", "Fluid and rigid two-way contact",
            "FluidRigid", "1bc734ad306a7e604af076e2f2b4ebc4110f95f130cc31a823f5ae4a63e08f57",
            {"fluid", "rigid_body", "fluid_rigid_coupling", "equal_opposite_transfer"},
            48U, "deformable", {}, true));
        result.push_back(integrated(
            "cloth-core", "Cloth pinning, pressure, and rigid contact",
            "Cloth", "52f904972c0757833e672004aa130a648d8a8b1d05b7625306b1df7267ebec29",
            {"cloth", "pinning", "pressure", "rigid_cloth_contact", "topology"},
            48U, "deformable"));
        {
            auto tear = integrated(
                "cloth-tear", "Cloth tearing and topology",
                "ClothTear", "f173a1982f08b99179dabf65e71c870193763dfa290167caa02bf70d9ab6ef3a",
                {"cloth", "tearing", "topology", "contact"}, 420U,
                "deformable",
                {{120U, "set_gravity", "world",
                  {0.0, -6.93671752, -6.93671752, 0.0}, 3U}});
            tear.checkpoints = {0U, 120U, 240U, 420U};
            result.push_back(std::move(tear));
        }
        result.push_back(integrated(
            "cloth-water", "Fluid-cloth containment and reaction",
            "ClothWater", "6267ffeddeaebff558025e75593172fef1f6045a253d999e5a43e98959bda33e",
            {"fluid", "cloth", "fluid_cloth_coupling", "equal_opposite_transfer",
             "pressure", "topology"}, 48U, "deformable", {}, true));
        result.push_back(integrated(
            "soft-body-core", "Soft-body springs, shape matching, and skinning",
            "Softbody", "65ff2ed4d0981c005ccd891ec6a809b2a93076337c0a36aada9d08260bb44c6e",
            {"soft_body", "springs", "shape_matching", "surface_skinning"},
            48U, "deformable"));
        result.push_back(integrated(
            "soft-body-rigid", "Soft-body and rigid contact",
            "SoftbodyRigidBody", "3e75b4224a740bf182407369839717cb6ebd723a730e09a821e2f2d650786219",
            {"soft_body", "rigid_body", "soft_body_rigid_contact",
             "equal_opposite_transfer"}, 48U, "deformable"));
        result.push_back(integrated(
            "soft-body-fluid", "Fluid-soft-body two-way contact",
            "SoftbodyFluid", "6e832ba1ab91d170a14c307b152ed1655a012179eebe364e18df60325ad77cfb",
            {"soft_body", "fluid", "fluid_soft_body_coupling",
             "equal_opposite_transfer"}, 24U, "deformable", {}, true));
        result.push_back(integrated(
            "soft-body-cloth", "Soft-body-cloth two-way contact",
            "SoftbodyCloth", "13644a3e690ad204db5d529c955138b5b3d5e3dd2a6a99f7521e3d375c592b7a",
            {"soft_body", "cloth", "soft_body_cloth_coupling",
             "equal_opposite_transfer", "topology"}, 48U, "deformable"));
        result.push_back(integrated(
            "rope-core", "Rope settling, winding drive, release, and contact",
            "Rope", "58bddccc7510e3203cfa6e931eb23bc7c7c12dee44eeea81a33760ef8d95241b",
            {"rope", "settling", "winding", "release", "rigid_rope_contact"},
            72U, "deformable",
            {{20U, "set_gravity", "world", {6.9367, -6.9367, 0.0, 0.0}, 3U},
             {48U, "set_gravity", "world", {0.0, -9.81, 0.0, 0.0}, 3U}}));
        result.push_back(integrated(
            "rope-fluid", "Fluid-rope two-way contact",
            "RopeFluid", "d4b175c1b9a1d1386b448a7896cc814adfce4a0916defa9cf7af65169915cb16",
            {"rope", "fluid", "fluid_rope_coupling", "equal_opposite_transfer"},
            48U, "deformable", {}, true));
        result.push_back(integrated(
            "rope-soft-body", "Rope-soft-body attachment and contact",
            "RopeSoftbody", "ef65a94c6a6d259bd5588959274deb499dc69cb01d24ebc6e62c108c45f68dd6",
            {"rope", "soft_body", "rope_soft_body_coupling",
             "equal_opposite_transfer", "winding"}, 48U, "deformable"));
        result.push_back(integrated(
            "rope-cloth", "Rope-cloth attachment and contact",
            "RopeCloth", "055057d00a4c38f689a8eb4327c3eb24f827b85de49c96ef9d427c9bacdb25ab",
            {"rope", "cloth", "rope_cloth_coupling", "equal_opposite_transfer",
             "topology"}, 48U, "deformable"));
        result.push_back(integrated(
            "smoke-grid", "Smoke tracers, projected Eulerian grid, emission, and transport",
            "Smoke", "ce42aa5c06ed28758fc19b5fb854791aba984b6b18dbcde0feb69913d095820a",
            {"smoke", "tracers", "eulerian_grid", "projection", "emission",
             "transport", "smoke_rigid_coupling"}, 36U, "deformable", {}, true));
        result.push_back(integrated(
            "smoke-water", "Fluid-smoke heat and momentum exchange",
            "SmokeWater", "f5aa6a6dd96b90d971adb70b2cd469f47dea1e2c86254ccce444b31328f1d3dd",
            {"smoke", "fluid", "fluid_smoke_coupling", "equal_opposite_transfer",
             "emission", "transport"}, 36U, "deformable", {}, true));
        result.push_back(integrated(
            "smoke-soft-body", "Smoke-soft-body interaction",
            "SmokeSoftbody", "04b46169a52d8e993b07537efcc1841121cb7ce4763e0509828497a426179c0b",
            {"smoke", "soft_body", "smoke_soft_body_coupling",
             "equal_opposite_transfer"}, 36U, "deformable", {}, true));
        result.push_back(integrated(
            "smoke-cloth", "Smoke-cloth interaction",
            "SmokeCloth", "228aa2074b63cbd8403c5debfa31ac70f0a0b3379c805d3fe9d527bdfa5ee636",
            {"smoke", "cloth", "smoke_cloth_coupling", "equal_opposite_transfer",
             "topology"}, 36U, "deformable", {}, true));
        result.push_back(integrated(
            "smoke-rope", "Smoke-rope interaction",
            "SmokeRope", "faa8a9dce23dd99b1bb2ac39782ad0407056691e78c7e017843aff0ac5e254b1",
            {"smoke", "rope", "smoke_rope_coupling", "equal_opposite_transfer"},
            36U, "deformable", {}, true));
        return result;
    }();
    return registry;
}

const CaseDefinition *find_case(std::string_view id) {
    const auto &registry = case_registry();
    const auto found = std::find_if(
        registry.begin(), registry.end(),
        [id](const CaseDefinition &definition) { return definition.id == id; });
    return found == registry.end() ? nullptr : &*found;
}

std::string serialize_case(const CaseDefinition &definition) {
    std::string result;
    result += "{\"checkpoints\":" + checkpoints(definition.checkpoints);
    result += ",\"commands\":" + commands(definition.commands);
    result += ",\"comparison\":{\"complete_state_limit\":256,";
    result += "\"contact_order\":\"canonical\",\"large_state\":\"stable_id_samples_and_invariants\",";
    result += "\"quaternion_sign_equivalent\":true}";
    result += ",\"coverage\":" + strings(definition.coverage);
    result += ",\"id\":" + quote(definition.id);
    result += ",\"invariants\":" + strings(definition.invariants);
    result += ",\"kind\":" + quote(
        definition.kind == CaseKind::analytic ? "analytic" : "integrated_glb");
    result += ",\"resources\":" + definition.resources_json;
    result += ",\"schema\":\"parallel-mater-conformance-case/v1\"";
    result += ",\"title\":" + quote(definition.title);
    result += ",\"tolerances\":{\"profile\":" +
              quote(definition.tolerance_profile) +
              ",\"values\":{\"chaotic_energy\":{\"abs\":25,\"rel\":0.25},\"chaotic_momentum\":{\"abs\":25,\"rel\":0.25},\"chaotic_position\":{\"abs\":0.10000000000000001,\"rel\":0.050000000000000003},\"chaotic_scalar\":{\"abs\":0.050000000000000003,\"rel\":0.10000000000000001},\"constraint_contact\":{\"position\":{\"abs\":0.002,\"rel\":0.002},\"velocity\":{\"abs\":0.02,\"rel\":0.02}},\"deformable\":{\"position\":{\"abs\":0.005,\"rel\":0.005},\"velocity\":{\"abs\":0.05,\"rel\":0.05}},\"direct\":{\"position\":{\"abs\":1e-05,\"rel\":1e-05},\"velocity\":{\"abs\":1e-05,\"rel\":1e-05}},\"quaternion_angular\":{\"abs\":0.002,\"rel\":0}}}";
    result += ",\"world\":{\"chaotic_envelope\":";
    result += definition.chaotic_envelope ? "true" : "false";
    result += ",\"deterministic\":true,\"frames\":" +
              std::to_string(definition.frames) + ",\"gravity\":[" +
              number(definition.gravity[0]) + ',' +
              number(definition.gravity[1]) + ',' +
              number(definition.gravity[2]) + "]";
    result += ",\"substeps\":" + std::to_string(definition.substeps);
    result += ",\"timestep\":" + number(definition.timestep);
    result += ",\"units\":{\"angle\":\"rad\",\"length\":\"m\",\"mass\":\"kg\",\"time\":\"s\"}}}\n";
    return result;
}

std::filesystem::path case_file_path(
    const std::filesystem::path &directory, const CaseDefinition &definition) {
    return directory / (definition.id + ".json");
}

} // namespace parallel_mater::conformance
