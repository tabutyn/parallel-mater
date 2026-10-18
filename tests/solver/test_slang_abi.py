#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Neutral ABI generation rejects layout ambiguity before adapter compilation."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from slang_abi import ReflectedABI


def scalar(name, size):
    return {'kind': 'scalar', 'scalarType': name,
            'sizes': [{'kind': 'uniform', 'value': size, 'alignment': size}]}

def field(name, typ, offset, size):
    return {'name': name, 'type': typ, 'binding': {'kind': 'uniform', 'offset': offset, 'size': size}}

def fixture():
    record = {'kind': 'struct', 'name': 'Record', 'fields': [
        field('enabled', scalar('bool', 4), 0, 4),
        field('count', scalar('uint32', 4), 4, 4)],
        'sizes': [{'kind': 'uniform', 'value': 8, 'alignment': 4}]}
    pointer = {'kind': 'pointer', 'valueType': record,
               'sizes': [{'kind': 'uniform', 'value': 8, 'alignment': 8}]}
    return {'entryPoints': [{'name': 'execute', 'parameters': [field('records', pointer, 0, 8)]}]}


class ReflectedABITest(unittest.TestCase):
    def test_names_and_layout_come_from_reflection(self):
        abi = ReflectedABI(fixture())
        text = '\n'.join(abi.declarations())
        self.assertIn('bool enabled;', text)
        self.assertIn('unsigned char pm_padding_1[3];', text)
        self.assertIn('offsetof(PmShader_Record, count) == 4U', text)
        self.assertEqual(abi.signature(abi.entries['execute']), 'PmShader_Record* arg_records')
        self.assertEqual(abi.normalize_legacy_wrapper('Record_17 r; r.count_4 = 1; keep_9();'),
                         'PmShader_Record r; r.count = 1; keep_9();')

    def test_unknown_type_fails(self):
        abi = ReflectedABI(fixture())
        with self.assertRaisesRegex(ValueError, 'unknown reflected type'):
            abi.cpp_type('Mystery')
        with self.assertRaisesRegex(ValueError, 'unsupported ABI type'):
            abi.cpp_type({'kind': 'matrix'})

    def test_overlapping_and_out_of_bounds_records_fail(self):
        for offset in (0, 9):
            data = fixture()
            data['entryPoints'][0]['parameters'][0]['type']['valueType']['fields'][1]['binding']['offset'] = offset
            with self.assertRaisesRegex(ValueError, 'overlapping/out-of-bounds'):
                ReflectedABI(data).declarations()

    def test_conflicting_reflected_definitions_fail(self):
        data = fixture()
        other = copy.deepcopy(data['entryPoints'][0])
        other['name'] = 'other'
        other['parameters'][0]['type']['valueType']['sizes'][0]['value'] = 16
        data['entryPoints'].append(other)
        with self.assertRaisesRegex(ValueError, 'inconsistent reflected layout'):
            ReflectedABI(data)
        other['parameters'][0]['type']['valueType']['sizes'][0]['value'] = 8
        other['parameters'][0]['type']['valueType']['fields'][0]['name'] = 'renamed'
        with self.assertRaisesRegex(ValueError, 'inconsistent reflected layout'):
            ReflectedABI(data)

    def test_padded_array_cannot_silently_change_element_stride(self):
        abi = ReflectedABI(fixture())
        array = {'kind': 'array', 'elementType': scalar('uint32', 4),
                 'elementCount': 3, 'uniformStride': 16,
                 'sizes': [{'kind': 'uniform', 'value': 48, 'alignment': 16}]}
        with self.assertRaisesRegex(ValueError, 'padded array stride'):
            abi.host_size(array)

    def test_recursive_pointer_is_allowed_but_by_value_cycle_is_not(self):
        data = fixture()
        record = data['entryPoints'][0]['parameters'][0]['type']['valueType']
        record['fields'] = [field('next', {'kind': 'pointer', 'valueType': 'Record'}, 0, 8)]
        record['sizes'][0]['alignment'] = 8
        self.assertIn('PmShader_Record* next;', '\n'.join(ReflectedABI(data).declarations()))
        record['fields'][0]['type'] = 'Record'
        with self.assertRaisesRegex(ValueError, 'recursive by-value'):
            ReflectedABI(data).declarations()

    def test_cpp_storage_rounds_reflected_record_size_to_alignment(self):
        record = {'kind': 'struct', 'name': 'Record', 'fields': [
            field('address', {'kind': 'pointer', 'valueType': scalar('float32', 4),
                              'sizes': [{'kind': 'uniform', 'value': 8, 'alignment': 8}]},
                  0, 8),
            field('enabled', scalar('bool', 4), 8, 4)],
            'sizes': [{'kind': 'uniform', 'value': 12, 'alignment': 8}]}
        data = {'entryPoints': [{'name': 'execute', 'parameters': [
            field('record', record, 0, 12)]}]}
        text = '\n'.join(ReflectedABI(data).declarations())
        self.assertIn('static_assert(sizeof(PmShader_Record) == 16U);', text)
        self.assertIn('offsetof(PmShader_Record, enabled) == 8U', text)


if __name__ == '__main__':
    unittest.main()
