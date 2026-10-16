// SPDX-License-Identifier: MIT
// Pointer-shaped storage shared by CUDA and graphics-resource entry points.
// Physics modules use PMBuffer<T>; the selected target decides whether that
// is a native device pointer or a descriptor-backed structured buffer.

#ifndef PM_TARGET_D3D12
#define PM_TARGET_D3D12 0
#endif
#define PM_TARGET_GRAPHICS_RESOURCE PM_TARGET_D3D12

#if PM_TARGET_GRAPHICS_RESOURCE
public typealias PMResourceUint8 = uint;

public struct PMGraphicsBuffer<T>
{
    // Plain integers keep PMBuffer legal inside other resource records. The
    // native adapter owns the shader-visible heap and supplies the descriptor
    // index; physics code only sees pointer-shaped storage.
    public uint descriptor_index;
    public uint byte_offset;
    public uint available;

    public __subscript(uint index) -> T
    {
        [nonmutating]
        get
        {
            RWByteAddressBuffer resource =
                ResourceDescriptorHeap[descriptor_index];
            return resource.Load<T>(byte_offset + index * sizeof(T));
        }
        [mutating]
        set
        {
            RWByteAddressBuffer resource =
                ResourceDescriptorHeap[descriptor_index];
            resource.Store<T>(byte_offset + index * sizeof(T), newValue);
        }
    }
};

public RWByteAddressBuffer pmGraphicsResource<T>(PMGraphicsBuffer<T> value)
{
    return ResourceDescriptorHeap[value.descriptor_index];
}

public typealias PMBuffer<T> = PMGraphicsBuffer<T>;

// DXIL has no byte-sized storage operations. Keep byte flags in their common
// packed ABI instead of allocating widened copies on the CPU. Compare/exchange
// preserves neighboring flags written concurrently by other lanes.
public struct PMByteBuffer
{
    public uint descriptor_index;
    public uint byte_offset;
    public uint available;

    public __subscript(uint index) -> uint
    {
        [nonmutating]
        get
        {
            RWByteAddressBuffer resource = ResourceDescriptorHeap[descriptor_index];
            uint address = byte_offset + index;
            return (resource.Load<uint>(address & ~3u) >> ((address & 3u) * 8u)) & 255u;
        }
        [mutating]
        set
        {
            RWByteAddressBuffer resource = ResourceDescriptorHeap[descriptor_index];
            uint address = byte_offset + index;
            uint shift = (address & 3u) * 8u;
            uint mask = 255u << shift;
            uint expected = resource.Load<uint>(address & ~3u);
            for (;;)
            {
                uint observed;
                uint desired = (expected & ~mask) | ((newValue & 255u) << shift);
                resource.InterlockedCompareExchange(address & ~3u, expected, desired, observed);
                if (observed == expected) break;
                expected = observed;
            }
        }
    }
};

public bool pmBufferPresent(PMByteBuffer value) { return value.available != 0u; }
public PMByteBuffer pmBufferOffset(PMByteBuffer value, uint elements)
{
    value.byte_offset += elements;
    return value;
}
public PMByteBuffer operator+(PMByteBuffer value, uint elements)
{
    return pmBufferOffset(value, elements);
}
public PMByteBuffer operator+(PMByteBuffer value, int elements)
{
    value.byte_offset = uint(int(value.byte_offset) + elements);
    return value;
}
public uint operator*(PMByteBuffer value) { return value[0]; }

public __prefix T operator*<T>(PMGraphicsBuffer<T> value)
{
    return value[0];
}

public PMGraphicsBuffer<T> operator+<T>(PMGraphicsBuffer<T> value,
    uint elements)
{
    value.byte_offset += elements * sizeof(T);
    return value;
}

public PMGraphicsBuffer<T> operator+<T>(PMGraphicsBuffer<T> value,
    int elements)
{
    value.byte_offset = uint(int(value.byte_offset) + elements * int(sizeof(T)));
    return value;
}

public bool pmBufferPresent<T>(PMBuffer<T> value)
{
    return value.available != 0u;
}

public PMBuffer<T> pmBufferOffset<T>(PMBuffer<T> value, uint elements)
{
    value.byte_offset += elements * sizeof(T);
    return value;
}
#else
public typealias PMResourceUint8 = uint8_t;
public typealias PMByteBuffer = uint8_t*;

public typealias PMBuffer<T> = T*;

public bool pmBufferPresent<T>(PMBuffer<T> value)
{
    return value != nullptr;
}

public PMBuffer<T> pmBufferOffset<T>(PMBuffer<T> value, uint elements)
{
    return value + elements;
}
#endif
