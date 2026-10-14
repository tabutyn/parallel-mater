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
            "PassiveActive", "fafa0682fdf58d083d60e7eec30991341c11bc5515247835d0396f14b20f57f0",
            {"rigid_integration", "static_body", "kinematic_body", "contact"}, 48U,
            "constraint_contact",
            {{16U, "set_first_kinematic_target", "scene", {0.5, 1.0, 0.0, 0.0}, 3U}}));
        result.push_back(integrated(
            "constraint-fixed", "Authored fixed collector and welded assembly",
            "ConstraintFixed", "c4aa05878338908ac420a059f4de25b767c4a1f3b573c5f41aee6841e5852bdc",
            {"fixed_joint", "compound_weld", "contact"}));
        result.push_back(integrated(
            "constraint-point", "Four authored point constraints",
            "ConstraintPoint", "f307d291cf451a20cf9c65e88a7dad2afb2cf31275dfc4ec925671695c3e5d0c",
            {"point_joint", "enabled_state"}, 60U, "constraint_contact",
            {{20U, "set_constraints_enabled", "scene", {0.0, 0.0, 0.0, 0.0}, 1U},
             {40U, "set_constraints_enabled", "scene", {1.0, 0.0, 0.0, 0.0}, 1U}}));
        result.push_back(integrated(
            "constraint-hinge", "Three-gear authored hinge train",
            "ConstraintHinge", "db5447b00a33d29990e2f0d441d154c0509dee99689fc940261c1c92c2fe2a20",
            {"hinge_joint", "contact", "alternating_rotation", "energy"}, 90U));
        result.push_back(integrated(
            "constraint-slider", "Authored slider constraint",
            "ConstraintSlider", "79b6e4f4073c803bd3cf38d33b0ebf72d1924375af54810465458283ac5edb9a",
            {"slider_joint", "limits", "contact"}));
        result.push_back(integrated(
            "constraint-piston", "Authored piston constraint",
            "ConstraintPiston", "48dcc6b5e0241592f0cc8ba46aaf34314d083047b41025525c3897ffd279f50a",
            {"piston_joint", "limits", "contact"}));
        result.push_back(integrated(
            "constraint-generic", "Authored generic constraint",
            "ConstraintGeneric", "1f66ea70d80db6febdde1b3265842d6ccae1d07e88741b2db3d750f1434069be",
            {"generic_joint", "linear_limits", "angular_limits"}));
        result.push_back(integrated(
            "constraint-generic-spring", "Authored generic spring constraint",
            "ConstraintGenericSpring", "4e476ca6c8070e9d60b3995ca1f458afcd8c174a77327c9c9dd2b2540fc2df0a",
            {"generic_spring_joint", "springs", "damping"}));
        result.push_back(integrated(
            "constraint-motor", "Authored motor constraint",
            "ConstraintMotor", "92bc99edca04b7c16890d2a0938d1b0ba9d727c3170498f0fd5e1bd7114136df",
            {"motor_joint", "linear_motor", "angular_motor"}));
        result.push_back(integrated(
            "fluid-lifecycle", "Fluid source, outflow, foam, and stable IDs",
            "Fluid", "740af8ab070eccb8117bfc2a51d00ee0a8227b27c7801e44bcf38a166d72e817",
            {"fluid", "particle_source", "outflow", "foam", "stable_particle_ids"},
            48U, "deformable", {}, true));
        result.push_back(integrated(
            "fluid-rigid", "Fluid and rigid two-way contact",
            "FluidRigid", "a73e2b800569ffc038329d4f57624d766dd2ecf9eaea02bf3f58e268f2c224a0",
            {"fluid", "rigid_body", "fluid_rigid_coupling", "equal_opposite_transfer"},
            48U, "deformable", {}, true));
        result.push_back(integrated(
            "cloth-core", "Cloth pinning, pressure, and rigid contact",
            "Cloth", "f4dc3dd53aa6f2459ef3032bbbd4711406a049f31bb7b2a7ea9bf3006b503033",
            {"cloth", "pinning", "pressure", "rigid_cloth_contact", "topology"},
            48U, "deformable"));
        {
            auto tear = integrated(
                "cloth-tear", "Cloth tearing and topology",
                "ClothTear", "70d83b88395001ff074174cdc9247889796c6a21d4a1a9f552f71cb591828405",
                {"cloth", "tearing", "topology", "contact"}, 420U,
                "deformable",
                {{120U, "set_gravity", "world",
                  {0.0, -6.93671752, -6.93671752, 0.0}, 3U}});
            tear.checkpoints = {0U, 120U, 240U, 420U};
            result.push_back(std::move(tear));
        }
        result.push_back(integrated(
            "cloth-water", "Fluid-cloth containment and reaction",
            "ClothWater", "2d27f380ef01c8a20998d4a4f0e05574d111a14faaf0e199780be415a02c6782",
            {"fluid", "cloth", "fluid_cloth_coupling", "equal_opposite_transfer",
             "pressure", "topology"}, 48U, "deformable", {}, true));
        result.push_back(integrated(
            "soft-body-core", "Soft-body springs, shape matching, and skinning",
            "Softbody", "c13ae5df9651d687f1233481dd094e2e06ad3b51fe4cd58be851400a63b954af",
            {"soft_body", "springs", "shape_matching", "surface_skinning"},
            48U, "deformable"));
        result.push_back(integrated(
            "soft-body-rigid", "Soft-body and rigid contact",
            "SoftbodyRigidBody", "5d40b70591f6ef771a78d238571571b8d1c09ab8af953d447d80361e90d2c5b9",
            {"soft_body", "rigid_body", "soft_body_rigid_contact",
             "equal_opposite_transfer"}, 48U, "deformable"));
        result.push_back(integrated(
            "soft-body-fluid", "Fluid-soft-body two-way contact",
            "SoftbodyFluid", "d2e3738810af12d7075e8ae45040bd981dad49abd8daf5ff39e23f8be7b1f2ee",
            {"soft_body", "fluid", "fluid_soft_body_coupling",
             "equal_opposite_transfer"}, 24U, "deformable", {}, true));
        result.push_back(integrated(
            "soft-body-cloth", "Soft-body-cloth two-way contact",
            "SoftbodyCloth", "e773f33595f17a2940d5e920a14c5e4914cb8704ee3cb2a53d23663b173ea110",
            {"soft_body", "cloth", "soft_body_cloth_coupling",
             "equal_opposite_transfer", "topology"}, 48U, "deformable"));
        result.push_back(integrated(
            "rope-core", "Rope settling, winding drive, release, and contact",
            "Rope", "7e081c0e0b8606dda1edd10e3fb6e1c143209f86204d1d48af34298146cbf6ba",
            {"rope", "settling", "winding", "release", "rigid_rope_contact"},
            72U, "deformable",
            {{20U, "set_gravity", "world", {6.9367, -6.9367, 0.0, 0.0}, 3U},
             {48U, "set_gravity", "world", {0.0, -9.81, 0.0, 0.0}, 3U}}));
        result.push_back(integrated(
            "rope-fluid", "Fluid-rope two-way contact",
            "RopeFluid", "b4ee8129d9a660ba004075cc0edaf03f1f2ceec79f07ebcf98037b9da90a476d",
            {"rope", "fluid", "fluid_rope_coupling", "equal_opposite_transfer"},
            48U, "deformable", {}, true));
        result.push_back(integrated(
            "rope-soft-body", "Rope-soft-body attachment and contact",
            "RopeSoftbody", "9d625cd29ae031f9e62d149db1f3aea9d274ae0d5ec4d1a3c96f5d1a39e0d0fe",
            {"rope", "soft_body", "rope_soft_body_coupling",
             "equal_opposite_transfer", "winding"}, 48U, "deformable"));
        result.push_back(integrated(
            "rope-cloth", "Rope-cloth attachment and contact",
            "RopeCloth", "2f4218e89236187421fd8c35c6d53088ee0d0e0ecbd83cb8afde6350dfeae645",
            {"rope", "cloth", "rope_cloth_coupling", "equal_opposite_transfer",
             "topology"}, 48U, "deformable"));
        result.push_back(integrated(
            "smoke-grid", "Smoke tracers, projected Eulerian grid, emission, and transport",
            "Smoke", "0b2d2a482c2c34f219e9656fea907fd522821b37feb1960a49f670a285aa8ad8",
            {"smoke", "tracers", "eulerian_grid", "projection", "emission",
             "transport", "smoke_rigid_coupling"}, 36U, "deformable", {}, true));
        result.push_back(integrated(
            "smoke-water", "Fluid-smoke heat and momentum exchange",
            "SmokeWater", "cf1cddb0a7f6e73f767125f6e15f80bb35c32a9d696ffcb9cd987035d1102bdd",
            {"smoke", "fluid", "fluid_smoke_coupling", "equal_opposite_transfer",
             "emission", "transport"}, 36U, "deformable", {}, true));
        result.push_back(integrated(
            "smoke-soft-body", "Smoke-soft-body interaction",
            "SmokeSoftbody", "b89f07fb4381c4e9c59eb03911402bd6ae71268f0697a52927815ed1a9738516",
            {"smoke", "soft_body", "smoke_soft_body_coupling",
             "equal_opposite_transfer"}, 36U, "deformable", {}, true));
        result.push_back(integrated(
            "smoke-cloth", "Smoke-cloth interaction",
            "SmokeCloth", "6256953f47d53504bc3c90dcef2f63447439e989b9eb23a0259f7c35dee8c79d",
            {"smoke", "cloth", "smoke_cloth_coupling", "equal_opposite_transfer",
             "topology"}, 36U, "deformable", {}, true));
        result.push_back(integrated(
            "smoke-rope", "Smoke-rope interaction",
            "SmokeRope", "09687092f7e5abf234f4c041fd2f5e02a452f4db7ce531c590f7ff7c539d750e",
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
