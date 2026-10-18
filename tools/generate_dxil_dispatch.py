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
    if expression.startswith('*as<Vec3_0>('):
        return expression.replace('*as<Vec3_0>(', '*static_cast<const Vec3 *>(', 1)
    return expression


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
        if len(launches) != 1:
            raise ValueError(f'{match[2]} must have exactly one launch')
        yield match, body, launches[0]


PREAMBLE = '''// SPDX-License-Identifier: MIT
// Generated from common host launches and Slang reflection. Do not edit.
#include "driver/d3d12/schedule_context.hpp"
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
} // namespace
'''


def generate(root, reflection, sources):
    includes, functions = [], []
    for source_name in sources:
        path = root / source_name
        includes.append(f'#include "simulation/{path.with_suffix(".hpp").name}"')
        module = path.stem.removesuffix('_dispatch')
        if module == 'exports':
            module = 'queries'
        for signature, body, launch in wrappers(path.read_text()):
            entry = launch[1]
            reflected = json.loads((reflection / module / f'{entry}.json').read_text())
            abi = ReflectedABI(reflected)
            entries = reflected.get('entryPoints', [])
            if len(entries) != 1 or entries[0].get('name') != entry:
                raise ValueError(f'{entry}: expected isolated matching reflection')
            params = abi.validate_entry(entries[0])
            args = [unwrap(arg) for arg in split_arguments(launch[3])]
            if len(args) != len(params):
                raise ValueError(f'{entry}: {len(args)} args != {len(params)} parameters')
            config = split_arguments(launch[2])
            if len(config) != 4 or config[3] not in (
                    'stream', 'static_cast<cudaStream_t>(stream.native_handle)'):
                raise ValueError(f'Unsupported launch configuration {entry}: {config}')
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
            }
            if is_export:
                local_values['converted'] = 'box'
            size = abi.constant_size(entries[0])
            out.append(f'    std::array<std::byte, {size}> bytes{{}};')
            for index, (param, arg) in enumerate(zip(params, args)):
                if arg == 'meshes':
                    # Resolve-only passes reuse the preceding body-bounds pass's
                    # mesh table; other passes provide the exact body range.
                    count = ('body_count' if re.search(r'\bbody_count\b', signature[3])
                             else 'body_index + 1U' if re.search(r'\bbody_index\b', signature[3])
                             else '0U')
                    arg = f'execution->translated_meshes(meshes, body_parameters, {count})'
                arg = local_values.get(arg, arg)
                if not re.fullmatch(r'[\w\s.*+<>,{}()\[\]:?/-]+', arg):
                    raise ValueError(f'Unsupported argument {entry}: {arg}')
                out.append(f'    const auto &arg{index} = {arg};')
                out += pack(abi, param['type'], f'arg{index}', param['binding']['offset'])
            config = [c.replace('dim3(', 'device::DispatchSize(') for c in config]
            expression = (f'execution->dispatch("{module}", "{entry}", '
                          f'device::DispatchSize({config[0]}), device::DispatchSize({config[1]}), bytes)')
            out.append(('    return device::from_driver_error(' + expression + ');')
                       if is_export else '    (void)' + expression + ';')
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
