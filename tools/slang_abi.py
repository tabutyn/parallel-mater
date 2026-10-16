# SPDX-License-Identifier: MIT
"""Target-neutral C++ storage declarations from Slang JSON reflection.

No compiler-emitted CUDA/C++ text is consumed. Names come from the Slang source;
all reflected sizes, offsets and alignments become compile-time ABI checks.
Unsupported types and inconsistent reflection fail generation.
"""
import re

SCALARS = {
    'uint8': ('unsigned char', 1), 'int8': ('signed char', 1),
    'uint16': ('unsigned short', 2), 'int16': ('short', 2),
    'uint32': ('unsigned int', 4), 'int32': ('int', 4),
    'uint64': ('unsigned long long', 8), 'int64': ('long long', 8),
    'float32': ('float', 4), 'float64': ('double', 8), 'bool': ('bool', 1),
}
ALIASES = {'uint': 'uint32', 'int': 'int32', 'float': 'float32', 'double': 'float64',
           'uint8_t': 'uint8', 'int8_t': 'int8', 'uint16_t': 'uint16',
           'int16_t': 'int16', 'uint32_t': 'uint32', 'int32_t': 'int32',
           'uint64_t': 'uint64', 'int64_t': 'int64', 'bool': 'bool'}


def identifier(name):
    if not re.fullmatch(r'[A-Za-z_]\w*', name):
        raise ValueError(f'invalid reflected identifier: {name}')
    return name


def layout(typ):
    entries = [s for s in typ.get('sizes', []) if s['kind'] == 'uniform']
    if len(entries) != 1:
        raise ValueError(f'missing/ambiguous storage layout: {typ.get("name", typ.get("kind"))}')
    size, alignment = entries[0]['value'], entries[0]['alignment']
    if size < 1 or alignment < 1 or alignment & (alignment - 1):
        raise ValueError('invalid reflected size/alignment')
    return size, alignment


def type_identity(typ):
    if isinstance(typ, str):
        return ALIASES.get(typ, typ)
    kind = typ['kind']
    if kind == 'struct':
        return typ['name']
    if kind == 'scalar':
        return typ['scalarType']
    if kind == 'pointer':
        return ('pointer', type_identity(typ['valueType']))
    if kind == 'array':
        return ('array', type_identity(typ['elementType']), typ['elementCount'],
                typ.get('uniformStride'))
    raise ValueError(f'unsupported ABI type: {kind}')


def record_identity(typ):
    return (layout(typ), [(f['name'], f['binding']['offset'], f['binding']['size'],
                          type_identity(f['type'])) for f in typ['fields']])


class ReflectedABI:
    def __init__(self, reflection):
        self.types = {}
        self.entries = {e['name']: e for e in reflection['entryPoints']}
        def collect(value):
            if isinstance(value, dict):
                if value.get('kind') == 'struct':
                    name = identifier(value['name'])
                    previous = self.types.get(name)
                    if previous and record_identity(previous) != record_identity(value):
                        raise ValueError(f'inconsistent reflected layout: {name}')
                    self.types[name] = value
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)
        collect(reflection)

    def resolve(self, typ):
        if not isinstance(typ, str):
            return typ
        if typ in self.types:
            return self.types[typ]
        if typ in ALIASES:
            return {'kind': 'scalar', 'scalarType': ALIASES[typ]}
        raise ValueError(f'unknown reflected type: {typ}')

    def cpp_type(self, typ):
        typ = self.resolve(typ)
        kind = typ['kind']
        if kind == 'struct':
            return 'PmShader_' + identifier(typ['name'])
        if kind == 'scalar':
            if typ['scalarType'] not in SCALARS:
                raise ValueError(f'unsupported scalar: {typ["scalarType"]}')
            return SCALARS[typ['scalarType']][0]
        if kind == 'pointer':
            return self.cpp_type(typ['valueType']) + '*'
        if kind == 'array':
            return f'PmShaderArray<{self.cpp_type(typ["elementType"])}, {typ["elementCount"]}>'
        raise ValueError(f'unsupported ABI type: {kind}')

    def host_size(self, typ):
        typ = self.resolve(typ)
        if typ['kind'] == 'scalar':
            return SCALARS[typ['scalarType']][1]
        if typ['kind'] == 'pointer':
            return 8
        if typ['kind'] == 'array':
            stride = self.host_size(typ['elementType'])
            size = stride * typ['elementCount']
            if typ.get('uniformStride', stride) != stride or layout(typ)[0] != size:
                raise ValueError('padded array stride requires an explicit storage adapter')
            return size
        return layout(typ)[0]

    def declarations(self):
        lines = ['template<class T, unsigned N> struct PmShaderArray { T data[N]; };',
                 'static_assert(sizeof(void*) == 8, "reflected pointer ABI requires 64-bit host");']
        lines += [f'struct PmShader_{name};' for name in sorted(self.types)]
        emitted, visiting = set(), set()
        def dependency(typ):
            typ = self.resolve(typ)
            if typ['kind'] == 'struct':
                emit(typ['name'])
            elif typ['kind'] == 'array':
                dependency(typ['elementType'])
        def emit(name):
            if name in emitted:
                return
            if name in visiting:
                raise ValueError(f'recursive by-value record: {name}')
            visiting.add(name)
            typ = self.types[name]
            for field in typ['fields']:
                dependency(field['type'])
            size, alignment = layout(typ)
            lines.append(f'struct alignas({alignment}) PmShader_{name} {{')
            cursor = 0
            for n, field in enumerate(typ['fields']):
                field_name = identifier(field['name'])
                offset, field_size = field['binding']['offset'], self.host_size(field['type'])
                if offset < cursor or offset + field_size > size:
                    raise ValueError(f'{name}.{field_name}: overlapping/out-of-bounds layout')
                if offset > cursor:
                    lines.append(f'    unsigned char pm_padding_{n}[{offset - cursor}];')
                lines.append(f'    {self.cpp_type(field["type"])} {field_name};')
                cursor = offset + field_size
            if size > cursor:
                lines.append(f'    unsigned char pm_padding_tail[{size - cursor}];')
            lines.extend(['};', f'static_assert(sizeof(PmShader_{name}) == {size}U);',
                      f'static_assert(alignof(PmShader_{name}) == {alignment}U);'])
            for field in typ['fields']:
                lines.append(f'static_assert(offsetof(PmShader_{name}, {field["name"]}) == {field["binding"]["offset"]}U);')
            visiting.remove(name)
            emitted.add(name)
        for name in sorted(self.types):
            emit(name)
        return lines

    def parameters(self, entry):
        return [p for p in entry['parameters'] if p.get('binding', {}).get('kind') == 'uniform']

    def signature(self, entry):
        return ', '.join(f'{self.cpp_type(p["type"])} arg_{identifier(p["name"])}'
                         for p in self.parameters(entry))

    def normalize_legacy_wrapper(self, source):
        # Transitional handwritten wrappers refer to compiler-mangled names.
        # Match only known source types/fields; never infer a declaration/layout
        # from that spelling. These wrappers can migrate independently to C++.
        fields = {f['name'] for t in self.types.values() for f in t['fields']}
        def replace(match):
            name = match[1]
            if name in self.types:
                return 'PmShader_' + name
            if name in fields:
                return name
            return match[0]
        return re.sub(r'\b(\w+)_\d+\b', replace, source)
