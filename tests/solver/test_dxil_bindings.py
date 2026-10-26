#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DXIL adapters consume only validated Slang reflection for uniform layout."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import embed_dxil_library
import generate_dxil_dispatch


def scalar(name, size):
    return {'kind': 'scalar', 'scalarType': name,
            'sizes': [{'kind': 'uniform', 'value': size, 'alignment': size}]}


def parameter(name, typ, offset, size):
    return {'name': name, 'type': typ,
            'binding': {'kind': 'uniform', 'offset': offset, 'size': size}}


def resource():
    return {'kind': 'struct', 'name': 'PMGraphicsBuffer', 'fields': [
        parameter('descriptor_index', scalar('uint32', 4), 0, 4),
        parameter('byte_offset', scalar('uint32', 4), 4, 4),
        parameter('available', scalar('uint32', 4), 8, 4)],
        'sizes': [{'kind': 'uniform', 'value': 12, 'alignment': 16}]}


def reflection(entry='execute'):
    return {'entryPoints': [{
        'name': entry,
        'stage': 'compute',
        'threadGroupSize': [64, 1, 1],
        'scope': {'kind': 'constantBuffer', 'parameters': [
            parameter('values', resource(), 0, 12),
            parameter('enabled', scalar('bool', 4), 12, 4)]},
    }]}


class DxilBindingGeneratorTest(unittest.TestCase):
    def test_generated_struct_dereference_uses_host_descriptor(self):
        self.assertEqual(
            generate_dxil_dispatch.unwrap('*as<TriangleMeshResource_0>(mesh)'),
            '*static_cast<const TriangleMeshResource *>(mesh)')

    def test_generated_vec3_value_uses_host_descriptor(self):
        self.assertEqual(
            generate_dxil_dispatch.unwrap(
                'Vec3_0{gravity[0], gravity[1], gravity[2]}'),
            'Vec3{gravity[0], gravity[1], gravity[2]}')

    def test_cuda_mutable_pointer_wrapper_is_removed(self):
        self.assertEqual(
            generate_dxil_dispatch.unwrap(
                'mutable_pointer(as<Vec3_0>(first.previous_positions))'),
            'first.previous_positions')
        self.assertEqual(
            generate_dxil_dispatch.unwrap(
                'mutable_pointer(constraints.first_endpoints)'),
            'constraints.first_endpoints')

    def test_embedded_library_emits_reflected_symbolic_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shader = root / 'execute.dxil'
            shader.write_bytes(b'\x01\x02\x03\x04')
            shader.with_suffix('.json').write_text(
                json.dumps(reflection()), encoding='utf-8')
            manifest = root / 'manifest.txt'
            manifest.write_text(
                f'fixture|execute|{shader}\n', encoding='utf-8')
            text = embed_dxil_library.generate(
                embed_dxil_library.manifest_records(manifest))
            self.assertIn('struct Kernel_execute', text)
            self.assertIn('constant_size = 16U', text)
            self.assertIn('arg_values{"values", 0U, 12U}', text)
            self.assertIn('arg_enabled{"enabled", 12U, 4U}', text)
            self.assertIn('64U, 1U, 1U', text)
            self.assertIn('0x01,0x02,0x03,0x04,', text)

    def test_embedded_library_rejects_duplicate_and_mismatched_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shader = root / 'execute.dxil'
            shader.write_bytes(b'code')
            shader.with_suffix('.json').write_text(
                json.dumps(reflection('other')), encoding='utf-8')
            manifest = root / 'manifest.txt'
            manifest.write_text(
                f'fixture|execute|{shader}\nfixture|execute|{shader}\n',
                encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate DXIL'):
                embed_dxil_library.manifest_records(manifest)
            manifest.write_text(f'fixture|execute|{shader}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'isolated matching reflection'):
                embed_dxil_library.generate(
                    embed_dxil_library.manifest_records(manifest))

    def test_domain_adapter_uses_reflected_offsets_and_checks_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'fixture_dispatch.cu'
            source.write_text('''
void run_fixture(device::DispatchSize grid, device::DispatchSize block,
                 device::ExecutionQueue stream, const void *values,
                 bool enabled) noexcept {
    execute<<<dim3(grid.x, grid.y, grid.z),
              dim3(block.x, block.y, block.z), 0U,
              static_cast<cudaStream_t>(stream.native_handle)>>>(values, enabled);
}
''', encoding='utf-8')
            reflected = root / 'reflection' / 'fixture'
            reflected.mkdir(parents=True)
            (reflected / 'execute.json').write_text(
                json.dumps(reflection()), encoding='utf-8')
            text = generate_dxil_dispatch.generate(
                root, root / 'reflection', [Path('fixture_dispatch.cu')])
            self.assertIn('std::array<std::byte, 16> bytes{};', text)
            self.assertIn('put(bytes, 0U, binding(*execution, arg0));', text)
            self.assertIn(
                'put(bytes, 12U, static_cast<std::uint32_t>(arg1));', text)

            changed = reflection()
            changed['entryPoints'][0]['scope']['parameters'].append(
                parameter('extra', scalar('uint32', 4), 16, 4))
            (reflected / 'execute.json').write_text(
                json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'args !='):
                generate_dxil_dispatch.generate(
                    root, root / 'reflection', [Path('fixture_dispatch.cu')])

    def test_accepts_prebound_native_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'fixture_dispatch.cu').write_text('''
void fixture(device::ExecutionQueue stream, void *values,
             bool enabled) noexcept {
    const auto native = static_cast<cudaStream_t>(stream.native_handle);
    execute<<<1U, 64U, 0U, native>>>(values, enabled);
}
''', encoding='utf-8')
            reflected = root / 'reflection' / 'fixture'
            reflected.mkdir(parents=True)
            (reflected / 'execute.json').write_text(
                json.dumps(reflection()), encoding='utf-8')
            text = generate_dxil_dispatch.generate(
                root, root / 'reflection', [Path('fixture_dispatch.cu')])
            self.assertIn('execution->dispatch("fixture", "execute"', text)

    def test_preserves_bool_selected_dual_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'fixture_dispatch.cu').write_text('''
void fixture(device::ExecutionQueue stream, void *values,
             bool enabled) noexcept {
    const auto native = static_cast<cudaStream_t>(stream.native_handle);
    const bool coupled = enabled || values != nullptr;
    if (coupled)
        execute_coupled<<<1U, 64U, 0U, native>>>(values, enabled);
    else
        execute<<<1U, 64U, 0U, native>>>(values, enabled);
}
''', encoding='utf-8')
            reflected = root / 'reflection' / 'fixture'
            reflected.mkdir(parents=True)
            for entry in ('execute_coupled', 'execute'):
                data = reflection(entry)
                (reflected / f'{entry}.json').write_text(
                    json.dumps(data), encoding='utf-8')
            text = generate_dxil_dispatch.generate(
                root, root / 'reflection', [Path('fixture_dispatch.cu')])
            self.assertIn(
                'const bool coupled = enabled || values != nullptr;', text)
            self.assertIn(
                'execution->dispatch("fixture", "execute_coupled"', text)
            self.assertIn('execution->dispatch("fixture", "execute"', text)

    def test_translates_pointer_bearing_paint_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'fixture_dispatch.cu').write_text('''
void fixture(device::ExecutionQueue stream, const void *paint_fields,
             std::uint32_t paint_field_capacity, const void *paint_rules,
             std::uint32_t paint_rule_capacity) noexcept {
    execute<<<1U, 64U, 0U, stream>>>(
        as<PaintFieldResource_0>(paint_fields), paint_field_capacity,
        as<PaintRuleResource_0>(paint_rules), paint_rule_capacity);
}
''', encoding='utf-8')
            data = reflection()
            data['entryPoints'][0]['scope']['parameters'] = [
                parameter('paint_fields', resource(), 0, 12),
                parameter('field_capacity', scalar('uint32', 4), 12, 4),
                parameter('paint_rules', resource(), 16, 12),
                parameter('rule_capacity', scalar('uint32', 4), 28, 4)]
            reflected = root / 'reflection' / 'fixture'
            reflected.mkdir(parents=True)
            (reflected / 'execute.json').write_text(
                json.dumps(data), encoding='utf-8')
            text = generate_dxil_dispatch.generate(
                root, root / 'reflection', [Path('fixture_dispatch.cu')])
            self.assertIn(
                'translated_paint_fields(paint_fields, paint_field_capacity)',
                text)
            self.assertIn(
                'translated_paint_rules(paint_rules, paint_rule_capacity)',
                text)

    def test_unsupported_uniform_shape_fails_both_dxil_generators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = reflection()
            bad = {'kind': 'matrix',
                   'sizes': [{'kind': 'uniform', 'value': 16, 'alignment': 4}]}
            data['entryPoints'][0]['scope']['parameters'][1] = \
                parameter('enabled', bad, 12, 16)
            shader = root / 'execute.dxil'
            shader.write_bytes(b'code')
            shader.with_suffix('.json').write_text(
                json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'unsupported uniform ABI type'):
                embed_dxil_library.reflected_layout('fixture', 'execute', shader)
            reflected = root / 'reflection' / 'fixture'
            reflected.mkdir(parents=True)
            (reflected / 'execute.json').write_text(
                json.dumps(data), encoding='utf-8')
            source = root / 'fixture_dispatch.cu'
            source.write_text('''
void run_fixture(device::DispatchSize grid, device::DispatchSize block,
                 device::ExecutionQueue stream, const void *values,
                 bool enabled) noexcept {
    execute<<<dim3(grid.x, grid.y, grid.z),
              dim3(block.x, block.y, block.z), 0U,
              static_cast<cudaStream_t>(stream.native_handle)>>>(values, enabled);
}
''', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'unsupported uniform ABI type'):
                generate_dxil_dispatch.generate(
                    root, root / 'reflection', [Path('fixture_dispatch.cu')])


if __name__ == '__main__':
    unittest.main()
