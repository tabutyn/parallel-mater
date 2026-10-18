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
        size, alignment = layout(typ)
        return (size + alignment - 1) & ~(alignment - 1)

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
            reflected_size, alignment = layout(typ)
            size = self.host_size(typ)
            lines.append(f'struct alignas({alignment}) PmShader_{name} {{')
            cursor = 0
            for n, field in enumerate(typ['fields']):
                field_name = identifier(field['name'])
                offset, field_size = field['binding']['offset'], self.host_size(field['type'])
                if offset < cursor or offset + field_size > reflected_size:
                    raise ValueError(f'{name}.{field_name}: overlapping/out-of-bounds layout')
                if offset > cursor:
                    lines.append(f'    unsigned char pm_padding_{n}[{offset - cursor}];')
                lines.append(f'    {self.cpp_type(field["type"])} {field_name};')
                cursor = offset + field_size
            if reflected_size > cursor:
                lines.append(f'    unsigned char pm_padding_tail[{reflected_size - cursor}];')
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
        scope = entry.get('scope', entry)
        return [p for p in scope['parameters']
                if p.get('binding', {}).get('kind') == 'uniform']

    def storage_size(self, typ):
        typ = self.resolve(typ)
        kind = typ['kind']
        if kind == 'scalar':
            scalar = typ.get('scalarType')
            if scalar not in SCALARS:
                raise ValueError(f'unsupported scalar: {scalar}')
            if typ.get('sizes'):
                return layout(typ)[0]
            return SCALARS[scalar][1]
        if kind == 'pointer':
            size = layout(typ)[0] if typ.get('sizes') else 8
            if size != 8:
                raise ValueError(f'unsupported pointer size: {size}')
            return size
        if kind in ('struct', 'array'):
            return layout(typ)[0]
        raise ValueError(f'unsupported uniform ABI type: {kind}')

    def validate_storage(self, typ, context='uniform'):
        typ = self.resolve(typ)
        kind = typ['kind']
        size = self.storage_size(typ)
        if kind == 'scalar':
            scalar = typ['scalarType']
            native = SCALARS[scalar][1]
            if size != native and not (scalar == 'bool' and size == 4):
                raise ValueError(
                    f'{context}: unsupported {scalar} storage size {size}')
            return
        if kind == 'pointer':
            # The pointee layout does not occupy constant storage, but it must
            # still be a type the common reflection model can name.
            self.cpp_type(typ['valueType'])
            return
        if kind == 'array':
            count = typ.get('elementCount')
            stride = typ.get('uniformStride')
            if not isinstance(count, int) or count < 1 or \
                    not isinstance(stride, int) or stride < 1:
                raise ValueError(f'{context}: invalid reflected array')
            element_size = self.storage_size(typ['elementType'])
            if stride < element_size or size != stride * (count - 1) + element_size:
                raise ValueError(f'{context}: inconsistent reflected array stride')
            self.validate_storage(typ['elementType'], context + '[]')
            return
        if kind != 'struct':
            raise ValueError(f'{context}: unsupported uniform ABI type: {kind}')
        fields = typ.get('fields')
        if not isinstance(fields, list):
            raise ValueError(f'{context}: reflected record fields missing')
        names, cursor = set(), 0
        for field in fields:
            name = identifier(field['name'])
            if name in names:
                raise ValueError(f'{context}: duplicate reflected field {name}')
            names.add(name)
            binding = field.get('binding', {})
            if binding.get('kind') != 'uniform':
                raise ValueError(f'{context}.{name}: non-uniform record field')
            offset, field_size = binding.get('offset'), binding.get('size')
            expected = self.storage_size(field['type'])
            if not isinstance(offset, int) or not isinstance(field_size, int) or \
                    field_size != expected or offset < cursor or offset + field_size > size:
                raise ValueError(
                    f'{context}.{name}: inconsistent reflected field layout')
            self.validate_storage(field['type'], context + '.' + name)
            cursor = offset + field_size

    def validate_entry(self, entry):
        name = identifier(entry.get('name', 'entry'))
        names, cursor = set(), 0
        parameters = self.parameters(entry)
        for parameter in parameters:
            parameter_name = identifier(parameter['name'])
            if parameter_name in names:
                raise ValueError(f'{name}: duplicate uniform {parameter_name}')
            names.add(parameter_name)
            binding = parameter.get('binding', {})
            offset, size = binding.get('offset'), binding.get('size')
            expected = self.storage_size(parameter['type'])
            if not isinstance(offset, int) or not isinstance(size, int) or \
                    size != expected or offset < cursor:
                raise ValueError(
                    f'{name}.{parameter_name}: inconsistent reflected parameter layout')
            self.validate_storage(parameter['type'], name + '.' + parameter_name)
            cursor = offset + size
        return parameters

    def constant_size(self, entry):
        parameters = self.validate_entry(entry)
        return max((p['binding']['offset'] + p['binding']['size']
                    for p in parameters), default=4)

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
        source = re.sub(r'\b(\w+)_\d+\b', replace, source)
        # Slang's CUDA emitter wraps fixed-size arrays in a helper exposing
        # m_data. The neutral reflected declaration uses PmShaderArray::data.
        source = source.replace('.m_data', '.data')
        return re.sub(r'\bulonglong\b', 'unsigned long long', source)
