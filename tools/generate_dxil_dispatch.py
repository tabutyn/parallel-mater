#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bind existing schedule calls to DXIL using Slang's reflected uniform ABI.

CUDA adapters remain the inventory of host argument order. Only their launch
arguments are consumed: no CUDA types, generated shader code, or physics enter
the output. Unsupported syntax fails generation rather than emitting a stub.
Reflection (not sizeof a host struct) determines every uniform offset.
"""
import argparse
import json
import pathlib
import re

from slang_abi import ReflectedABI


def split_arguments(text):
    parts, start, depth = [], 0, 0
    for index, char in enumerate(text):
        if char in '(<[{':
            depth += 1
        elif char in ')>]}':
            depth -= 1
        elif char == ',' and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def unwrap(expression):
    expression = expression.strip()
    cast = re.fullmatch(r'(?:as|const_cast|reinterpret_cast)<[^<>]+>\s*\((.*)\)',
                        expression, re.S)
    if cast:
        return unwrap(cast[1])
    expression = re.sub(r'\s+', ' ', expression)
    dereference = re.fullmatch(r'\*as<(\w+)_0>\((.*)\)', expression, re.S)
    if dereference:
        return (f'*static_cast<const {dereference[1]} *>('
                f'{unwrap(dereference[2])})')
    if expression.startswith('*as<Vec3_0>('):
        return expression.replace('*as<Vec3_0>(', '*static_cast<const Vec3 *>(', 1)
    return re.sub(r'\bVec3_0\s*\{', 'Vec3{', expression)


def pack(abi, type_info, expression, offset):
    type_info = abi.resolve(type_info)
    name = type_info.get('name')
    kind = type_info['kind']
    if name in ('PMGraphicsBuffer', 'PMByteBuffer'):
        return [f'    put(bytes, {offset}U, binding(*execution, {expression}));']
    if kind == 'scalar':
        cpp_type = {'float32': 'float', 'uint32': 'std::uint32_t',
                    'int32': 'std::int32_t', 'uint64': 'std::uint64_t',
                    'int64': 'std::int64_t', 'bool': 'std::uint32_t'}[
                        type_info['scalarType']]
        return [f'    put(bytes, {offset}U, static_cast<{cpp_type}>({expression}));']
    if kind == 'struct':
        result = []
        for field in type_info['fields']:
            result += pack(abi, field['type'], f'{expression}.{field["name"]}',
                           offset + field['binding']['offset'])
        return result
    if kind == 'array':
        result = []
        stride = type_info['uniformStride']
        for index in range(type_info['elementCount']):
            result += pack(abi, type_info['elementType'], f'{expression}[{index}]',
                           offset + index * stride)
        return result
    raise ValueError(f'Unsupported uniform: {type_info}')


def wrappers(source):
    signature = re.compile(r'\b(void|PmResult)\s+(\w+)\s*\((.*?)\)\s*noexcept\s*\{', re.S)
    for match in signature.finditer(source):
        depth, end = 1, match.end()
        while depth:
            if source[end] == '{':
                depth += 1
            elif source[end] == '}':
                depth -= 1
            end += 1
        body = source[match.end():end - 1]
        launches = list(re.finditer(r'(\w+)<<<(.*?)>>>\s*\((.*?)\);', body, re.S))
        if not launches:
            continue
        yield match, body, launches


def launch_selector(body, launches):
    if len(launches) == 1:
        return None
    if len(launches) != 2:
        raise ValueError('dispatch wrappers support at most two selected launches')
    prefix = body[:launches[0].start()]
    selected = re.search(
        r'const bool\s+(\w+)\s*=\s*(.*?);\s*if\s*\(\s*\1\s*\)\s*$',
        prefix,
        re.S,
    )
    between = body[launches[0].end():launches[1].start()]
    if selected is None or not re.fullmatch(r'\s*else\s*', between):
        raise ValueError('multiple launches require a bool-selected if/else')
    expression = re.sub(r'\s+', ' ', selected[2]).strip()
    if not re.fullmatch(r'[\w\s.!=|&()<>]+', expression):
        raise ValueError(f'unsupported launch selector {expression}')
    return selected[1], expression


PREAMBLE = '''// SPDX-License-Identifier: MIT
// Generated from common host launches and Slang reflection. Do not edit.
#include "driver/d3d12/schedule_context.hpp"
#include "simulation/geometry_storage.hpp"
#include "simulation/types.hpp"
#include <array>
#include <cstring>
#include <cassert>
#include <type_traits>
namespace parallel_mater::simulation {
namespace {
template <std::size_t N, typename T>
void put(std::array<std::byte, N> &bytes, std::size_t offset, const T &value) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    assert(offset + sizeof(T) <= N);
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}
d3d12::GraphicsBufferAddress binding(d3d12::ScheduleContext &execution, const void *pointer) noexcept {
    return execution.address(pointer);
}
d3d12::GraphicsBufferAddress binding(d3d12::ScheduleContext &, d3d12::GraphicsBufferAddress address) noexcept {
    return address;
}
Vec3 vec3(const float *value) noexcept { return {value[0], value[1], value[2]}; }
Vec3 vector(const float *value) noexcept { return vec3(value); }
struct ShapeMatrix { Vec3 columns[3]; };
template <typename... Pointers> bool all_present(Pointers... pointers) noexcept {
    return ((pointers != nullptr) && ...);
}
FluidOptions constants(const FluidDispatchConstants &value) noexcept {
    FluidOptions result{};
    result.capacity = value.count;
    result.particle_radius = value.particle_radius;
    result.support_radius = value.support_radius;
    result.repulsion = value.repulsion;
    result.viscosity = value.viscosity;
    result.normal_damping = value.normal_damping;
    result.velocity_damping = value.velocity_damping;
    result.maximum_speed = value.maximum_speed;
    result.maximum_pair_acceleration = value.maximum_pair_acceleration;
    result.rest_density = value.rest_density;
    result.rest_particle_volume = value.rest_particle_volume;
    result.maximum_neighbors = value.maximum_neighbors;
    return result;
}
FluidClothCouplingOptions converted(const FluidClothDispatchOptions &value) noexcept {
    FluidClothCouplingOptions result{};
    result.contact_distance = value.contact_distance;
    result.interaction_radius = value.interaction_radius;
    result.stiffness = value.stiffness;
    result.damping = value.damping;
    result.tangential_drag = value.tangential_drag;
    result.maximum_force = value.maximum_force;
    return result;
}
struct SmokeOptionsBinding {
    std::uint32_t capacity{};
    Vec3 emitter_center{};
    Vec2 emitter_half_extents{};
    Vec3 initial_velocity{};
    Vec3 wind{};
    float particles_per_second{}, lifetime{}, particle_radius{}, buoyancy{};
    float response{}, rest_number_density{}, pressure_stiffness{}, viscosity{};
    float vorticity_confinement{}, maximum_speed{};
    std::uint32_t grid_resolution{}, grid_vertical_resolution{};
    std::uint32_t grid_pressure_iterations{};
    float grid_kinematic_viscosity{}, grid_les_coefficient{};
    float grid_pressure_tolerance{};
    Vec3 grid_minimum{};
    float grid_edge_length{};
};
SmokeOptionsBinding smoke_options(const SmokeDispatchOptions &value) noexcept {
    SmokeOptionsBinding result{};
    result.capacity = value.capacity;
    result.particle_radius = value.particle_radius;
    result.lifetime = value.lifetime;
    result.rest_number_density = value.rest_number_density;
    result.pressure_stiffness = value.pressure_stiffness;
    result.maximum_speed = value.maximum_speed;
    return result;
}
SmokeOptionsBinding smoke_options(const SmokeDomainDispatchOptions &value) noexcept {
    SmokeOptionsBinding result{};
    result.capacity = value.capacity;
    result.emitter_center = vec3(value.emitter_center);
    result.emitter_half_extents = {
        value.emitter_half_extents[0], value.emitter_half_extents[1]};
    result.initial_velocity = vec3(value.initial_velocity);
    result.wind = vec3(value.wind);
    result.particles_per_second = value.particles_per_second;
    result.lifetime = value.lifetime;
    result.particle_radius = value.particle_radius;
    result.buoyancy = value.buoyancy;
    result.response = value.response;
    result.rest_number_density = value.rest_number_density;
    result.pressure_stiffness = value.pressure_stiffness;
    result.viscosity = value.viscosity;
    result.vorticity_confinement = value.vorticity_confinement;
    result.maximum_speed = value.maximum_speed;
    result.grid_resolution = value.grid_resolution;
    result.grid_vertical_resolution = value.grid_vertical_resolution;
    result.grid_pressure_iterations = value.grid_pressure_iterations;
    result.grid_kinematic_viscosity = value.grid_kinematic_viscosity;
    result.grid_les_coefficient = value.grid_les_coefficient;
    result.grid_pressure_tolerance = value.grid_pressure_tolerance;
    result.grid_minimum = vec3(value.grid_minimum);
    result.grid_edge_length = value.grid_edge_length;
    return result;
}
SmokeOptionsBinding smoke_options(const FluidSmokeDispatchOptions &value) noexcept {
    return smoke_options(value.smoke);
}
struct IdentifierBinding { std::uint32_t index{}, generation{}; };
struct QuaternionBinding { float x{}, y{}, z{}, w{}; };
struct ParticlePlaneBinding {
    Vec3 center{};
    QuaternionBinding orientation{};
    Vec2 half_extents{};
};
struct FluidSmokeCouplingBinding {
    IdentifierBinding fluid{}, smoke{};
    ParticlePlaneBinding heater{};
    float heater_temperature{}, boiling_temperature{}, heat_transfer_rate{};
    float wind_drag{}, steam_rise_speed{};
};
FluidSmokeCouplingBinding coupling_options(
    const FluidSmokeDispatchOptions &value) noexcept {
    FluidSmokeCouplingBinding result{};
    result.heater.center = vec3(value.heater_center);
    result.heater.orientation = {value.heater_orientation[0],
        value.heater_orientation[1], value.heater_orientation[2],
        value.heater_orientation[3]};
    result.heater.half_extents = {
        value.heater_half_extents[0], value.heater_half_extents[1]};
    result.heater_temperature = value.heater_temperature;
    result.boiling_temperature = value.boiling_temperature;
    result.heat_transfer_rate = value.heat_transfer_rate;
    result.wind_drag = value.wind_drag;
    result.steam_rise_speed = value.steam_rise_speed;
    return result;
}
struct SmokeGridFieldBinding {
    std::uint32_t n{}, height{};
    Vec3 minimum{};
    float spacing{};
    const void *velocity{}, *density{}, *pressure{};
    const void *face[3]{};
    const void *vorticity{}, *strain{};
};
SmokeGridFieldBinding grid_field(const SmokeGridDispatchField &value) noexcept {
    SmokeGridFieldBinding result{};
    result.n = value.resolution;
    result.height = value.height;
    result.minimum = vec3(value.minimum);
    result.spacing = value.spacing;
    result.velocity = value.velocity;
    result.density = value.density;
    result.pressure = value.pressure;
    for (std::size_t axis = 0; axis != 3; ++axis)
        result.face[axis] = value.face_velocity[axis];
    result.vorticity = value.vorticity;
    result.strain = value.strain;
    return result;
}
SmokeOptionsBinding converted(const SmokeDispatchOptions &value) noexcept {
    return smoke_options(value);
}
SmokeGridFieldBinding converted(const SmokeGridDispatchField &value) noexcept {
    return grid_field(value);
}
} // namespace
'''


def generate(root, reflection, sources):
    includes, functions = [], []
    for source_name in sources:
        path = root / source_name
        source = path.read_text()
        header = re.search(r'#include\s+"([^"\n]+_dispatch\.hpp)"', source)
        includes.append(
            f'#include "simulation/{header[1] if header else path.with_suffix(".hpp").name}"')
        module = path.stem.removesuffix('_dispatch')
        if module == 'exports':
            module = 'queries'
        for signature, body, launches in wrappers(source):
            selector = launch_selector(body, launches)
            variants = []
            for launch in launches:
                variant_entry = launch[1]
                reflected = json.loads(
                    (reflection / module / f'{variant_entry}.json').read_text())
                variant_abi = ReflectedABI(reflected)
                entries = reflected.get('entryPoints', [])
                if len(entries) != 1 or entries[0].get('name') != variant_entry:
                    raise ValueError(
                        f'{variant_entry}: expected isolated matching reflection')
                variant_params = variant_abi.validate_entry(entries[0])
                variant_args = [
                    unwrap(arg) for arg in split_arguments(launch[3])]
                if len(variant_args) != len(variant_params):
                    raise ValueError(
                        f'{variant_entry}: {len(variant_args)} args != '
                        f'{len(variant_params)} parameters')
                variant_config = split_arguments(launch[2])
                if len(variant_config) != 4 or variant_config[3] not in (
                        'stream', 'native',
                        'static_cast<cudaStream_t>(stream.native_handle)'):
                    raise ValueError(
                        f'Unsupported launch configuration {variant_entry}: '
                        f'{variant_config}')
                variants.append((variant_entry, variant_abi, entries[0],
                                 variant_params, variant_args, variant_config))
            entry, abi, reflected_entry, params, args, config = variants[0]
            constant_size = abi.constant_size(reflected_entry)
            for (variant_entry, variant_abi, variant_reflected_entry,
                 variant_params, variant_args, variant_config) in variants[1:]:
                if (variant_params != params or variant_args != args or
                        variant_config != config or
                        variant_abi.constant_size(variant_reflected_entry) !=
                        constant_size):
                    raise ValueError(
                        f'{entry}/{variant_entry}: selected launches must share ABI')
            is_export = signature[1] == 'PmResult'
            out = [signature[0], '    auto *execution = d3d12::context(' +
                   ('queue.native_handle' if is_export else 'stream.native_handle') + ');',
                   '    if (execution == nullptr) return' +
                   (' PM_ERROR_INVALID_ARGUMENT;' if is_export else ';')]
            if is_export:
                checks = body.split('const cudaStream_t stream =')[0]
                checks = re.sub(r'static_assert\(.*?\);', '', checks, flags=re.S)
                out.append(checks)
                out.append('    constexpr std::uint32_t block_size = 256U;')
            local_values = {
                'id': '*static_cast<const FluidId *>(fluid_id)',
                'fluid_id': 'FluidId{fluid_index, fluid_generation}',
                'plane': '*static_cast<const ParticleDestroyPlaneOptions *>(plane_options)',
                'matrix': '*static_cast<const ShapeMatrix *>(inverse_rest)',
                'data': 'rope',
                'first_target': 'soft_a',
                'second_target': 'soft_b',
                'shader_gravity': 'gravity',
            }
            if is_export:
                local_values['converted'] = 'box'
            out.append(f'    std::array<std::byte, {constant_size}> bytes{{}};')
            for index, (param, arg) in enumerate(zip(params, args)):
                if arg == 'meshes':
                    # Resolve-only passes reuse the preceding body-bounds pass's
                    # mesh table; other passes provide the exact body range.
                    count = ('body_count' if re.search(r'\bbody_count\b', signature[3])
                             else 'body_index + 1U' if re.search(r'\bbody_index\b', signature[3])
                             else '0U')
                    paint_arguments = (
                        ', paint_fields, paint_field_capacity'
                        if re.search(r'\bpaint_fields\b', signature[3]) else '')
                    arg = (f'execution->translated_meshes('
                           f'meshes, body_parameters, {count}, 0U'
                           f'{paint_arguments})')
                elif arg == 'paint_fields':
                    arg = ('execution->translated_paint_fields('
                           'paint_fields, paint_field_capacity)')
                elif arg == 'paint_rules':
                    arg = ('execution->translated_paint_rules('
                           'paint_rules, paint_rule_capacity)')
                arg = local_values.get(arg, arg)
                if not re.fullmatch(r'[\w\s.*+<>,{}()\[\]:?/-]+', arg):
                    raise ValueError(f'Unsupported argument {entry}: {arg}')
                out.append(f'    const auto &arg{index} = {arg};')
                out += pack(abi, param['type'], f'arg{index}', param['binding']['offset'])
            def dispatch_expression(variant):
                variant_entry, _, _, _, _, variant_config = variant
                converted = [
                    item.replace('dim3(', 'device::DispatchSize(')
                    for item in variant_config]
                return (f'execution->dispatch("{module}", "{variant_entry}", '
                        f'device::DispatchSize({converted[0]}), '
                        f'device::DispatchSize({converted[1]}), bytes)')
            if selector is None:
                expression = dispatch_expression(variants[0])
                out.append(
                    ('    return device::from_driver_error(' + expression + ');')
                    if is_export else '    (void)' + expression + ';')
            else:
                if is_export:
                    raise ValueError('selected export launches are unsupported')
                name, condition = selector
                out.append(f'    const bool {name} = {condition};')
                out.append(f'    if ({name})')
                out.append(f'        (void){dispatch_expression(variants[0])};')
                out.append('    else')
                out.append(f'        (void){dispatch_expression(variants[1])};')
            out.append('}\n')
            functions += out
    return '\n'.join(includes) + '\n' + PREAMBLE + '\n'.join(functions) + '\n} // namespace\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--reflection', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('sources', nargs='+')
    options = parser.parse_args()
    output = generate(options.root, options.reflection, options.sources)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(output, encoding='utf-8')


if __name__ == '__main__':
    main()
