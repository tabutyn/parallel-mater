#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Embed DXIL and emit its only host-visible layout from Slang reflection."""
import argparse
import json
import pathlib

from slang_abi import ReflectedABI, identifier


def manifest_records(path):
    records, identities = [], set()
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line:
            continue
        fields = line.split('|')
        if len(fields) != 3:
            raise ValueError(f'{path}:{line_number}: malformed DXIL manifest record')
        module, entry, input_name = fields
        identifier(module)
        identifier(entry)
        identity = (module, entry)
        if identity in identities:
            raise ValueError(f'{module}.{entry}: duplicate DXIL manifest entry')
        identities.add(identity)
        input_path = pathlib.Path(input_name)
        if not input_path.is_file():
            raise ValueError(f'{module}.{entry}: missing DXIL schedule: {input_path}')
        records.append((module, entry, input_path))
    if not records:
        raise ValueError(f'DXIL manifest is empty: {path}')
    return records


def byte_lines(data):
    values = [f'0x{value:02x},' for value in data]
    return ['    ' + ''.join(values[index:index + 16])
            for index in range(0, len(values), 16)]


def reflected_layout(module, entry_name, input_path):
    reflection_path = input_path.with_suffix('.json')
    if not reflection_path.is_file():
        raise ValueError(
            f'{module}.{entry_name}: missing Slang reflection: {reflection_path}')
    reflection = json.loads(reflection_path.read_text(encoding='utf-8'))
    entries = reflection.get('entryPoints', [])
    if len(entries) != 1 or entries[0].get('name') != entry_name:
        raise ValueError(
            f'{module}.{entry_name}: expected isolated matching reflection')
    entry = entries[0]
    group_size = entry.get('threadGroupSize')
    if not isinstance(group_size, list) or len(group_size) != 3 or any(
            not isinstance(value, int) or value < 1 for value in group_size):
        raise ValueError(f'{module}.{entry_name}: invalid reflected thread-group size')
    abi = ReflectedABI(reflection)
    parameters = abi.validate_entry(entry)
    return parameters, abi.constant_size(entry), group_size


def generate(records):
    declarations = []
    table = []
    for index, (module, entry, input_path) in enumerate(records):
        parameters, constant_size, group_size = reflected_layout(
            module, entry, input_path)
        layout = [f'namespace layout::{module} {{', f'struct Kernel_{entry} {{',
                  f'    inline static constexpr std::size_t constant_size = {constant_size}U;']
        for parameter in parameters:
            binding = parameter['binding']
            layout.append(
                f'    inline static constexpr SlangUniformLayout arg_{identifier(parameter["name"])}'
                f'{{"{parameter["name"]}", {binding["offset"]}U, {binding["size"]}U}};')
        layout.append(
            f'    inline static constexpr std::array<SlangUniformLayout, {len(parameters)}U> uniforms{{{{')
        layout += [f'        arg_{parameter["name"]},' for parameter in parameters]
        layout += ['    }};', '};', f'}} // namespace layout::{module}']
        data = input_path.read_bytes()
        if not data:
            raise ValueError(f'{module}.{entry}: empty DXIL schedule')
        layout.append(f'inline constexpr std::uint8_t kernel_{index}[] = {{')
        layout += byte_lines(data)
        layout.append('};')
        declarations += layout
        x, y, z = group_size
        kernel = f'layout::{module}::Kernel_{entry}'
        table.append(
            f'    {{"{module}", "{entry}", kernel_{index}, sizeof(kernel_{index}), '
            f'{x}U, {y}U, {z}U, {kernel}::uniforms.data(), '
            f'{kernel}::uniforms.size(), {kernel}::constant_size}},')
    return '\n'.join([
        '// Generated from DXIL and common Slang reflection. Do not edit.',
        '#pragma once',
        '#include <array>',
        '#include <cstddef>',
        '#include <cstdint>',
        'namespace parallel_mater::d3d12::generated {',
        'struct SlangUniformLayout {',
        '    const char* name;',
        '    std::size_t offset;',
        '    std::size_t size;',
        '};',
        'struct SlangKernelBytecode {',
        '    const char* module;',
        '    const char* entry;',
        '    const std::uint8_t* code;',
        '    std::size_t size;',
        '    std::uint32_t threads_x, threads_y, threads_z;',
        '    const SlangUniformLayout* uniforms;',
        '    std::size_t uniform_count;',
        '    std::size_t constant_size;',
        '};',
        *declarations,
        'inline constexpr SlangKernelBytecode slang_kernels[] = {',
        *table,
        '};',
        'inline constexpr std::size_t slang_kernel_count =',
        '    sizeof(slang_kernels) / sizeof(slang_kernels[0]);',
        '} // namespace parallel_mater::d3d12::generated',
        '',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    options = parser.parse_args()
    output = generate(manifest_records(options.manifest))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(output, encoding='utf-8')


if __name__ == '__main__':
    main()
