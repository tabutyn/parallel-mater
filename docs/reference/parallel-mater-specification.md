# Parallel Mater 0.2 — API Specification

This specification defines the Parallel Mater 0.2 command API: object ownership, device discovery, explicit storage, physics pipelines, command execution, and synchronization. It is the behavioral contract for implementations and applications. The companion reference guide lists declarations and points here by section number.

Requirements stated as must, must not, required, or an imperative define valid usage and required implementation behavior. Examples illustrate those requirements and do not override them. The core domains and optional extension boundaries are specified in section 16.6; the API-family overview is not a claim that every device supports every physics domain.

## Contents

- [1 Parallel Mater](#spec-overview)

- [2 Return codes](#spec-return_codes)

- [3 Function Pointers and Instancing](#spec-instancing)

- [4 Devices & queues](#spec-discovery)

- [5 Explicit device memory](#spec-memory)

- [6 Buffers & triangle geometry](#spec-resources)

- [7 Simulation state & rigid bodies](#spec-simulation)

- [8 Rigid constraints](#spec-constraints)

- [9 Physics pipelines & materials](#spec-pipelines)

- [10 Simulation domains & events](#spec-domains)

- [11 Command recording & lifecycle](#spec-commands)

- [12 Data movement & output](#spec-transfer)

- [13 Submission & synchronization](#spec-sync)

- [14 Native API interoperability](#spec-interop)

- [15 Timing, work & memory queries](#spec-queries)

- [16 Features, limits & formats](#spec-capabilities)

- [17 A frame, recorded explicitly](#spec-example)

- [18 Fundamental declarations](#spec-fundamentals)

- [A Symbol index](#spec-symbol_index)

<a id="spec-overview"></a>

## 1 Parallel Mater

API 0.2

<a id="spec-intent"></a>

### 1.1 Overview

Parallel Mater is a physics API offering functions and structures that allow you to define physics simulations. Parallel Mater supports Rigid Bodies, Fluid, Cloth, Rope, Smoke, and Soft Bodies. Parallel Mater uses a configured fixed-function pipeline to simulate the specified physics simulations and their specified couplings.

<a id="spec-legend"></a>

### 1.2 Color key

Function names: pmCreateInstance
Structure and type names: PmInstanceCreateInfo
Constants and return codes: PM_SUCCESS
PFN_pmName: pointer type for pmName
[n.n] Specification section

<a id="spec-conventions"></a>

### 1.3 Calling conventions

pmCreate*/pmDestroy* own objects; pmGet* queries; pmCmd* records work. Counts precede arrays. Input pointers are const. A null array requests an enumeration count; a short array returns PM_INCOMPLETE and the number written. Unless stated otherwise, input data is copied before a call returns.

<a id="spec-validity"></a>

### 1.4 Validity & availability

Set sType to its matching PM_STRUCTURE_TYPE_* value; pNext = NULL in core 0.2. Unknown chains and reserved flags are errors. Initialize required nonzero fields. Null handles are allowed only where specified. Physics domains and couplings require their corresponding features or extensions. This reference specifies the behavior required of conforming implementations.

<a id="spec-units"></a>

### 1.5 Units & lifetime

Right-handed; +Y up; meters, seconds, kilograms, radians, kelvin. Quaternions: normalized x,y,z,w, local-to-world. No implicit coordinate conversion. Device children share one device. Destroy in reverse dependency order after GPU completion; destruction does not wait. Command buffers borrow resources. Externally synchronize host access to queues, pools, and mutable objects.

<a id="spec-return_codes"></a>

## 2 Return codes

Results and failures

<a id="spec-results"></a>

### 2.1 Success & error codes

```c
typedef enum PmResult {
    PM_SUCCESS = 0,
    PM_NOT_READY = 1,
    PM_TIMEOUT = 2,
    PM_INCOMPLETE = 3,
    PM_ERROR_INVALID_ARGUMENT = -1,
    PM_ERROR_INVALID_STATE = -2,
    PM_ERROR_OUT_OF_HOST_MEMORY = -3,
    PM_ERROR_OUT_OF_DEVICE_MEMORY = -4,
    PM_ERROR_FEATURE_NOT_PRESENT = -5,
    PM_ERROR_FORMAT_NOT_SUPPORTED = -6,
    PM_ERROR_CAPACITY_EXCEEDED = -7,
    PM_ERROR_DEVICE_LOST = -8,
    PM_ERROR_INCOMPATIBLE_VERSION = -9
} PmResult;
```

Nonnegative codes report success or status; negative codes report failure. Creation clears the output handle before attempting work. Failed creation owns nothing. A failed recording call latches an error; End fails and the buffer must be reset. Runtime capacity overflow is reported in the step statistics; it is not silently successful physics.

<a id="spec-instancing"></a>

## 3 Function Pointers and Instancing

Resolve entry points and create an instance

<a id="spec-function_pointers"></a>

### 3.1 Function pointers

```c
typedef void (PMAPI_CALL *PFN_pmVoidFunction)(
    void);
```

```c
PFN_pmVoidFunction pmGetInstanceProcAddr(
    PmInstance instance, const char* pName);
PFN_pmVoidFunction pmGetDeviceProcAddr(
    PmDevice device, const char* pName);
```

Every command has a matching PFN_pmCommandName typedef in the header. Cast a returned pointer to that command's exact type before calling it; never call through PFN_pmVoidFunction. pName is an exact, case-sensitive, null-terminated command name. Unknown or unavailable names return NULL.

With instance = NULL, pmGetInstanceProcAddr resolves only itself, pmEnumerateInstanceVersion, and pmCreateInstance. With a live instance it also resolves core instance, physical-device and device commands. pmGetDeviceProcAddr requires a live device and returns only device commands supported by its enabled features and extensions. A non-NULL pointer does not enable a feature.

Obtain pmGetInstanceProcAddr by linking or the platform's library-symbol lookup. Instance-resolved pointers are valid only for that instance and its children; device-resolved pointers only for that device and its children. Keep their owner and the library alive while using them.

Related sections: [3.2](#spec-instance), [4.5](#spec-device).

<a id="spec-instance"></a>

### 3.2 Instance creation

```c
typedef struct PmInstanceCreateInfo {
    PmStructureType sType;
    const void* pNext;
    uint32_t apiVersion;
    const char* pApplicationName;
    PmBool32 enableValidation;
} PmInstanceCreateInfo;
```

```c
PmResult pmEnumerateInstanceVersion(
    uint32_t* pApiVersion);
PmResult pmCreateInstance(
    const PmInstanceCreateInfo* pInfo,
    const PmAllocationCallbacks* pAllocator,
    PmInstance* pInstance);
void pmDestroyInstance(PmInstance instance);
```

pmEnumerateInstanceVersion reports the highest supported API version; pApiVersion must be non-NULL. Set apiVersion = PM_API_VERSION_0_2 to use this reference. Unsupported versions return PM_ERROR_INCOMPATIBLE_VERSION. Destroy every logical device before its instance. Validation is a host-side diagnostic request, not permission to omit required synchronization.

Related sections: [3.3](#spec-allocator), [4](#spec-discovery).

<a id="spec-allocator"></a>

### 3.3 Host allocation callbacks

```c
typedef void* (PMAPI_CALL *PmAllocateFn)(
    void* user, size_t size, size_t alignment);
typedef void (PMAPI_CALL *PmFreeFn)(
    void* user, void* allocation);
typedef struct PmAllocationCallbacks {
    void* pUserData;
    PmAllocateFn pfnAllocate;
    PmFreeFn pfnFree;
} PmAllocationCallbacks;
```

Callbacks allocate CPU bookkeeping only, with power-of-two alignment. Both functions are required if callbacks are supplied. NULL selects the library host allocator. The instance copies callbacks; user data must survive the instance and all children. This does not allocate device physics storage.

Related sections: [5](#spec-memory).

<a id="spec-discovery"></a>

## 4 Devices & queues

Discover support before creating resources

<a id="spec-physical_devices"></a>

### 4.1 Physical devices

```c
PmResult pmEnumeratePhysicalDevices(
    PmInstance instance, uint32_t* pCount,
    PmPhysicalDevice* pDevices);
```

Physical-device handles are borrowed from the instance. Pass pDevices = NULL to query the count, then provide storage and enumerate. If the list grows, PM_INCOMPLETE returns the number written; query the count again before retrying.

<a id="spec-properties"></a>

### 4.2 Physical-device identity

```c
typedef struct PmPhysicalDeviceProperties {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorId;
    uint32_t deviceId;
    uint8_t deviceUuid[16];
    char deviceName[256];
} PmPhysicalDeviceProperties;
```

```c
void pmGetPhysicalDeviceProperties(
    PmPhysicalDevice physicalDevice,
    PmPhysicalDeviceProperties* pProperties);
```

Use the UUID to match a renderer's adapter. Names and enumeration indices are not stable identities. driverVersion is implementation-defined; do not decode it as the API version.

Related sections: [16](#spec-capabilities), [14](#spec-interop).

<a id="spec-queues"></a>

### 4.3 Queue families

```c
enum {
    PM_QUEUE_PHYSICS_BIT = 0x1,
    PM_QUEUE_TRANSFER_BIT = 0x2
};
typedef struct PmQueueFamilyProperties {
    PmFlags queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    float timestampPeriodNanoseconds;
} PmQueueFamilyProperties;
```

```c
PmResult pmGetPhysicalDeviceQueueFamilies(
    PmPhysicalDevice physicalDevice,
    uint32_t* pCount,
    PmQueueFamilyProperties* pProperties);
```

PHYSICS queues also support transfer. A zero timestampValidBits means timestamps are unavailable on that family. Queues execute asynchronously; submission order alone is not a memory dependency.

Related sections: [13](#spec-sync), [15](#spec-queries).

<a id="spec-features"></a>

### 4.4 Supported vs enabled features

```c
typedef struct PmPhysicalDeviceFeatures {
    PmBool32 rigidBodies;
    PmBool32 rigidConstraints;
    PmBool32 particleLiquids;
    PmBool32 liquidRigidCoupling;
    PmBool32 liquidLiquidCoupling;
    PmBool32 cloth;
    PmBool32 rope;
    PmBool32 softBodies;
    PmBool32 smoke;
    PmBool32 domainCouplings;
    PmBool32 particleEmitters;
    PmBool32 paint;
    PmBool32 contactEvents;
    PmBool32 debugData;
    PmBool32 timelineSemaphores;
    PmBool32 timestampQueries;
    PmBool32 statisticsQueries;
    PmBool32 memoryBudget;
    PmBool32 vulkanInterop;
} PmPhysicalDeviceFeatures;
```

```c
void pmGetPhysicalDeviceFeatures(
    PmPhysicalDevice physicalDevice,
    PmPhysicalDeviceFeatures* pFeatures);
```

Supported features are not automatically enabled. Request only needed features at device creation. Constraints require rigidBodies; each coupling requires its participating domains. Timeline semaphores are required for a conforming command-API device; other features may be false.

<a id="spec-device"></a>

### 4.5 Logical device

```c
typedef struct PmDeviceQueueCreateInfo {
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float* pPriorities;
} PmDeviceQueueCreateInfo;
typedef struct PmDeviceCreateInfo {
    PmStructureType sType;
    const void* pNext;
    uint32_t queueCreateInfoCount;
    const PmDeviceQueueCreateInfo* pQueueInfos;
    PmPhysicalDeviceFeatures enabledFeatures;
} PmDeviceCreateInfo;
```

```c
PmResult pmCreateDevice(
    PmPhysicalDevice physicalDevice,
    const PmDeviceCreateInfo* pInfo,
    PmDevice* pDevice);
void pmDestroyDevice(PmDevice device);
void pmGetDeviceQueue(PmDevice device,
    uint32_t family, uint32_t index,
    PmQueue* pQueue);
```

Request each family at most once; priorities are finite in [0,1]. Unsupported features fail creation. Queues are borrowed, not destroyed separately. Wait for work and destroy children before destroying the device.

<a id="spec-memory"></a>

## 5 Explicit device memory

Create → requirements → allocate → bind → use

<a id="spec-memory_types"></a>

### 5.1 Heaps & types

```c
enum {
    PM_MEMORY_DEVICE_LOCAL_BIT = 0x1,
    PM_MEMORY_HOST_VISIBLE_BIT = 0x2,
    PM_MEMORY_HOST_COHERENT_BIT = 0x4,
    PM_MEMORY_HOST_CACHED_BIT = 0x8
};
typedef struct PmMemoryHeap {
    PmDeviceSize size;
    PmFlags flags;
} PmMemoryHeap;
typedef struct PmMemoryType {
    PmFlags propertyFlags;
    uint32_t heapIndex;
} PmMemoryType;
typedef struct PmMemoryProperties {
    uint32_t heapCount;
    PmMemoryHeap heaps[16];
    uint32_t typeCount;
    PmMemoryType types[32];
} PmMemoryProperties;
```

```c
void pmGetPhysicalDeviceMemoryProperties(
    PmPhysicalDevice physicalDevice,
    PmMemoryProperties* pProperties);
```

Heap flags are reserved zero. DEVICE_LOCAL describes location, not mappability. A type can be both device-local and host-visible. Heap size is not currently available budget.

Related sections: [15.5](#spec-memory_budget).

<a id="spec-requirements"></a>

### 5.2 Requirements & suballocation

```c
typedef struct PmMemoryRequirements {
    PmDeviceSize size;
    PmDeviceSize alignment;
    uint32_t memoryTypeBits;
    PmBool32 requiresDedicatedAllocation;
} PmMemoryRequirements;
```

Choose a type t for which memoryTypeBits & (1u << t) is nonzero. Bind at an offset divisible by alignment. The allocation must contain the full required range. Dedicated resources occupy a whole allocation at offset zero.

Suballocate distinct, non-overlapping ranges from larger blocks. Live bindings must not overlap, including scratch. Free a block only after destroying all bound resources and observing GPU completion. Requirements include resource-internal GPU data; bindings are immutable until destruction.

Related sections: [6.2](#spec-buffer_binding), [7.2](#spec-simulation_memory).

<a id="spec-allocation"></a>

### 5.3 Allocate & free

```c
typedef struct PmMemoryAllocateInfo {
    PmStructureType sType;
    const void* pNext;
    PmDeviceSize allocationSize;
    uint32_t memoryTypeIndex;
} PmMemoryAllocateInfo;
```

```c
PmResult pmAllocateMemory(PmDevice device,
    const PmMemoryAllocateInfo* pInfo,
    PmDeviceMemory* pMemory);
void pmFreeMemory(PmDevice device,
    PmDeviceMemory memory);
```

allocationSize is nonzero. Allocation failure is explicit; no fallback silently changes the memory type. Pipeline code storage and driver bookkeeping are implementation-owned and reported separately; simulation, geometry acceleration and buffer storage are application-bound.

<a id="spec-mapping"></a>

### 5.4 Map & host visibility

```c
typedef struct PmMappedMemoryRange {
    PmDeviceMemory memory;
    PmDeviceSize offset;
    PmDeviceSize size;
} PmMappedMemoryRange;
```

```c
PmResult pmMapMemory(PmDevice device,
    PmDeviceMemory memory,
    PmDeviceSize offset, PmDeviceSize size,
    void** ppData);
void pmUnmapMemory(PmDevice device,
    PmDeviceMemory memory);
PmResult pmFlushMappedMemoryRanges(
    PmDevice device, uint32_t count,
    const PmMappedMemoryRange* pRanges);
PmResult pmInvalidateMappedMemoryRanges(
    PmDevice device, uint32_t count,
    const PmMappedMemoryRange* pRanges);
```

Map HOST_VISIBLE memory once per allocation; the returned pointer addresses offset. PM_WHOLE_SIZE extends to allocation end. Flush host writes before GPU consumption. Wait for GPU writes before invalidating and reading. Noncoherent ranges start on atomSize boundaries and end on a boundary or allocation end; they must lie in the mapped range. Unmap does not flush. COHERENT removes cache-maintenance calls, not race-prevention duties.

Related sections: [16.1](#spec-limits), [13.2](#spec-barriers).

<a id="spec-resources"></a>

## 6 Buffers & triangle geometry

One rigid geometry model • explicit storage and build work

<a id="spec-buffer"></a>

### 6.1 Buffer creation

```c
enum {
    PM_BUFFER_TRANSFER_SRC_BIT = 0x1,
    PM_BUFFER_TRANSFER_DST_BIT = 0x2,
    PM_BUFFER_GEOMETRY_INPUT_BIT = 0x4,
    PM_BUFFER_PARTICLE_INPUT_BIT = 0x8,
    PM_BUFFER_SIMULATION_OUTPUT_BIT = 0x10
};
typedef struct PmBufferCreateInfo {
    PmStructureType sType;
    const void* pNext;
    PmDeviceSize size;
    PmFlags usage;
    uint32_t initialQueueFamilyIndex;
} PmBufferCreateInfo;
```

```c
PmResult pmCreateBuffer(PmDevice device,
    const PmBufferCreateInfo* pInfo,
    PmBuffer* pBuffer);
void pmDestroyBuffer(PmDevice device,
    PmBuffer buffer);
```

Creation declares a resource; it does not allocate its storage. All buffers have exclusive queue-family ownership. size is nonzero. Declare every usage required for the resource's lifetime.

<a id="spec-buffer_binding"></a>

### 6.2 Buffer requirements & binding

```c
typedef struct PmBufferRange {
    PmBuffer buffer;
    PmDeviceSize offset;
    PmDeviceSize size;
} PmBufferRange;
```

```c
void pmGetBufferMemoryRequirements(
    PmDevice device, PmBuffer buffer,
    PmMemoryRequirements* pRequirements);
PmResult pmBindBufferMemory(PmDevice device,
    PmBuffer buffer, PmDeviceMemory memory,
    PmDeviceSize offset);
```

Bind exactly once before use. Buffer ranges are bounds-checked, nonempty, and measured in bytes; PM_WHOLE_SIZE is allowed only for APIs that explicitly say so. Imported Vulkan buffers are already bound and must not be rebound.

Related sections: [5.2](#spec-requirements), [14](#spec-interop).

<a id="spec-format_enum"></a>

### 6.3 Element formats

```c
typedef enum PmFormat {
    PM_FORMAT_UNDEFINED = 0,
    PM_FORMAT_R32_UINT = 1,
    PM_FORMAT_R32_SFLOAT = 2,
    PM_FORMAT_R32G32B32_SFLOAT = 3,
    PM_FORMAT_R32G32B32A32_SFLOAT = 4,
    PM_FORMAT_PARTICLE_STATE = 5,
    PM_FORMAT_RIGID_STATE = 6,
    PM_FORMAT_FLUID_PARTICLE_OUTPUT = 7
} PmFormat;
typedef struct PmBufferView {
    PmBufferRange range;
    PmFormat format;
    uint32_t count;
    uint32_t stride;
} PmBufferView;
```

A format defines byte interpretation, not a collider or solver type. stride is in bytes, at least the element size and a multiple of its alignment. count is nonzero; the final element must fit in range.size. Query usage support before choosing a format.

Related sections: [16.2](#spec-formats).

<a id="spec-geometry"></a>

### 6.4 Triangle geometry

```c
typedef struct PmGeometryCreateInfo {
    PmStructureType sType;
    const void* pNext;
    PmBufferView vertices;
    PmBufferView indices;
    uint32_t queueFamilyIndex;
} PmGeometryCreateInfo;
```

```c
PmResult pmCreateGeometry(PmDevice device,
    const PmGeometryCreateInfo* pInfo,
    PmGeometry* pGeometry);
void pmDestroyGeometry(PmDevice device,
    PmGeometry geometry);
void pmGetGeometryMemoryRequirements(
    PmDevice device, PmGeometry geometry,
    PmMemoryRequirements* pRequirements);
PmResult pmBindGeometryMemory(
    PmDevice device, PmGeometry geometry,
    PmDeviceMemory memory,
    PmDeviceSize offset);
```

Vertices use R32G32B32_SFLOAT; indices use R32_UINT and a count divisible by three. Indices must be in bounds; coordinates finite; triangles nondegenerate. Dynamic bodies require closed, convex, consistently wound meshes. Nonconvex static geometry is outside the core specification. Spheres and boxes are authored as triangle meshes.

Related sections: [7.4](#spec-bodies).

<a id="spec-geometry_build"></a>

### 6.5 Recorded geometry build

```c
PmResult pmCmdBuildGeometry(
    PmCommandBuffer commandBuffer,
    PmGeometry geometry);
```

Build after uploading and making input writes visible. Build creates acceleration data in bound storage and includes its internal dependencies. Barrier GEOMETRY_BUILD/WRITE → PHYSICS/READ before the first solve. Rebuild only when no dependent simulation is using the geometry. Inputs and their bindings remain alive for the entire geometry lifetime, not just the build. One geometry can be instanced by many bodies.

Related sections: [13.2](#spec-barriers), [9.6](#spec-pipeline_flow).

<a id="spec-resource_recipe"></a>

### 6.6 Resource creation checklist

| Item | Definition |
| --- | --- |
| 1 · Describe | Size, usage, format, capacity, owner family |
| 2 · Query | Exact size, alignment and type bits |
| 3 · Allocate | Choose a compatible memory type |
| 4 · Bind | Reserve a non-overlapping aligned range |
| 5 · Initialize | Record upload / reset / build |
| 6 · Depend | Make producer writes visible to users |
| 7 · Retire | Complete GPU work, destroy, then free |

<a id="spec-simulation"></a>

## 7 Simulation state & rigid bodies

Mutable state lives separately from immutable solver configuration

<a id="spec-simulation_create"></a>

### 7.1 Capacity-based simulation

```c
typedef struct PmFluidCapacity {
    uint32_t particleCapacity;
    uint32_t neighborCapacityPerParticle;
} PmFluidCapacity;
typedef struct PmClothCapacity {
    uint32_t vertexCapacity;
    uint32_t triangleIndexCapacity;
    uint32_t bondCapacity;
} PmClothCapacity;
typedef struct PmRopeCapacity {
    uint32_t nodeCapacity;
} PmRopeCapacity;
typedef struct PmSoftBodyCapacity {
    uint32_t nodeCapacity;
    uint32_t bondCapacity;
    uint32_t surfaceVertexCapacity;
    uint32_t surfaceTriangleIndexCapacity;
} PmSoftBodyCapacity;
typedef struct PmSmokeCapacity {
    uint32_t particleCapacity;
    uint32_t gridCellCapacity;
} PmSmokeCapacity;
typedef struct PmPaintFieldCapacity {
    uint32_t vertexCapacity;
    uint32_t width;
    uint32_t height;
} PmPaintFieldCapacity;
typedef struct PmSimulationCreateInfo {
    PmStructureType sType;
    const void* pNext;
    uint32_t queueFamilyIndex;
    uint32_t rigidBodyCapacity;
    uint32_t constraintCapacity;
    uint32_t contactCapacity;
    uint32_t fluidCount;
    const PmFluidCapacity* pFluidCapacities;
    uint32_t clothCount;
    const PmClothCapacity* pClothCapacities;
    uint32_t ropeCount;
    const PmRopeCapacity* pRopeCapacities;
    uint32_t softBodyCount;
    const PmSoftBodyCapacity* pSoftBodyCapacities;
    uint32_t smokeCount;
    const PmSmokeCapacity* pSmokeCapacities;
    uint32_t couplingCapacity;
    uint32_t emitterCapacity;
    uint32_t destroyerCapacity;
    uint32_t paintFieldCount;
    const PmPaintFieldCapacity* pPaintFieldCapacities;
    uint32_t paintRuleCapacity;
    uint32_t debugFrameCapacity;
} PmSimulationCreateInfo;
```

```c
PmResult pmCreateSimulation(PmDevice device,
    const PmSimulationCreateInfo* pInfo,
    PmSimulation* pSimulation);
void pmDestroySimulation(PmDevice device,
    PmSimulation simulation);
```

At least one body slot or fluid is required. Zero capacity disables a domain. Body and constraint IDs are stable, zero-based slots, not pointers. Each fluid is a separate particle state range with its own material. No hidden growth is permitted; change capacity by creating and migrating to a new simulation.

<a id="spec-simulation_memory"></a>

### 7.2 Persistent & scratch bindings

```c
typedef struct PmSimulationMemoryRequirements {
    PmMemoryRequirements persistent;
    PmMemoryRequirements scratch;
} PmSimulationMemoryRequirements;
typedef struct PmSimulationMemoryBinding {
    PmDeviceMemory persistentMemory;
    PmDeviceSize persistentOffset;
    PmDeviceMemory scratchMemory;
    PmDeviceSize scratchOffset;
} PmSimulationMemoryBinding;
```

```c
void pmGetSimulationMemoryRequirements(
    PmDevice device, PmSimulation simulation,
    PmSimulationMemoryRequirements* pInfo);
PmResult pmBindSimulationMemory(
    PmDevice device, PmSimulation simulation,
    const PmSimulationMemoryBinding* pBinding);
```

Requirements depend on declared capacities and include bounded contact/neighbor workspaces. Bind before reset or execution. Persistent and scratch ranges cannot overlap. Neither is a renderer-readable ABI. Use explicit output commands for portable state. Simulation state is exclusive to its creation queue family in core 0.2; share through output buffers.

Related sections: [5.3](#spec-allocation), [12.2](#spec-outputs).

<a id="spec-rigid_state"></a>

### 7.3 Portable rigid state record

```c
typedef struct PmRigidState {
    float position[3];
    float orientation[4];
    float linearVelocity[3];
    float angularVelocity[3];
} PmRigidState;
typedef enum PmMotionType {
    PM_MOTION_STATIC = 0,
    PM_MOTION_KINEMATIC = 1,
    PM_MOTION_DYNAMIC = 2
} PmMotionType;
```

Quaternion identity is (0,0,0,1). Angular velocity is radians/second in world space. Static bodies do not integrate; kinematic states are application-driven and do not respond to forces.

Related sections: [16.2](#spec-formats).

<a id="spec-bodies"></a>

### 7.4 Body descriptions

```c
typedef struct PmRigidBodyInfo {
    PmBool32 active;
    PmMotionType motionType;
    PmGeometry geometry;
    PmRigidState state;
    float mass;
    float inertiaDiagonal[3];
    float friction;
    float restitution;
    float linearDamping;
    float angularDamping;
    float maximumLinearSpeed;
    float maximumAngularSpeed;
    float collisionMargin;
    uint64_t userData;
} PmRigidBodyInfo;
```

```c
PmResult pmCmdWriteRigidBodies(
    PmCommandBuffer commandBuffer,
    PmSimulation simulation,
    uint32_t firstBody, uint32_t count,
    const PmRigidBodyInfo* pBodies);
```

Host descriptions are copied during recording, including state; handles remain borrowed. Writing replaces each addressed slot. Dynamic bodies require positive mass and principal inertia about their center of mass, in kg·m². No implicit inertia approximation. Restitution is [0,1]; friction, damping and margin are nonnegative. Inactive entries require only active = PM_FALSE; all other fields are ignored. Remove or disable referencing constraints first.

<a id="spec-body_inputs"></a>

### 7.5 Forces, impulses & targets

```c
typedef struct PmRigidInput {
    uint32_t bodyIndex;
    float force[3];
    float torque[3];
    float linearImpulse[3];
    float angularImpulse[3];
} PmRigidInput;
```

```c
PmResult pmCmdApplyRigidInputs(
    PmCommandBuffer commandBuffer,
    PmSimulation simulation,
    uint32_t count, const PmRigidInput* pInputs);
PmResult pmCmdSetRigidStates(
    PmCommandBuffer commandBuffer,
    PmSimulation simulation,
    uint32_t firstBody, uint32_t count,
    const PmRigidState* pStates);
PmResult pmCmdSetKinematicTargets(
    PmCommandBuffer commandBuffer,
    PmSimulation simulation,
    uint32_t firstBody, uint32_t count,
    const PmRigidState* pTargets);
```

Inputs accumulate for the next step, then clear. Forces apply across that step's substeps; impulses apply once, not once per substep. Units: N, N·m, N·s, N·m·s. Dynamic state edits and inputs wake bodies. SetRigidStates replaces pose and velocity without changing material or geometry. SetKinematicTargets drives kinematic bodies through the next step.

<a id="spec-hit_boxes"></a>

### 7.6 Hit-box queries

```c
typedef struct PmHitBox {
    float center[3];
    float orientation[4];
    float halfExtents[3];
} PmHitBox;
typedef struct PmHitBoxParticle {
    uint32_t fluidIndex;
    uint32_t stableParticleId;
} PmHitBoxParticle;
typedef struct PmHitBoxOutputHeader {
    uint32_t rigidBodyCount;
    uint32_t particleCount;
    uint64_t revision;
} PmHitBoxOutputHeader;
```

```c
PmResult pmCmdQueryHitBox(
    PmCommandBuffer commandBuffer,
    PmSimulation simulation,
    const PmHitBox* pBox,
    const PmBufferRange* pDestination);
```

Output begins with PmHitBoxOutputHeader, followed by rigidBodyCount uint32_t body slots and particleCount PmHitBoxParticle records. Results are sorted by slot and stable particle ID.

<a id="spec-constraints"></a>

## 8 Rigid constraints

Contacts and joints enter the same AVBD solve

<a id="spec-constraint_types"></a>

### 8.1 Joint families

```c
typedef enum PmConstraintType {
    PM_CONSTRAINT_FIXED = 0,
    PM_CONSTRAINT_POINT = 1,
    PM_CONSTRAINT_HINGE = 2,
    PM_CONSTRAINT_SLIDER = 3,
    PM_CONSTRAINT_PISTON = 4,
    PM_CONSTRAINT_GENERIC = 5,
    PM_CONSTRAINT_GENERIC_SPRING = 6,
    PM_CONSTRAINT_MOTOR = 7
} PmConstraintType;
typedef struct PmConstraintFrame {
    float position[3];
    float orientation[4];
} PmConstraintFrame;
```

Frames are local to their body centers of mass. If bodyB is PM_INVALID_INDEX, frameB is in world space. Hinge, slider and piston axes are local +X of frameA. Both quaternions are normalized. A body cannot constrain itself.

Related sections: [9.1](#spec-rigid_solver).

<a id="spec-axis_control"></a>

### 8.2 Per-axis controls

```c
typedef struct PmAxisLimit {
    PmBool32 enabled;
    float lower;
    float upper;
} PmAxisLimit;
typedef struct PmAxisSpring {
    PmBool32 enabled;
    float stiffness;
    float damping;
    float equilibrium;
} PmAxisSpring;
typedef struct PmAxisMotor {
    PmBool32 enabled;
    float targetVelocity;
    float maximumForce;
} PmAxisMotor;
```

Linear limits/equilibria are meters; angular ones radians. lower ≤ upper; equality locks an axis. Disabled limits leave eligible axes free. Springs use nonnegative N/m and N·s/m, or rotational N·m/rad and N·m·s/rad. Motors use m/s and N, or rad/s and N·m. These are force/torque limits, not frame-dependent impulse limits.

<a id="spec-constraint_info"></a>

### 8.3 Constraint description

```c
typedef struct PmRigidConstraintInfo {
    PmBool32 active;
    PmConstraintType type;
    uint32_t bodyA;
    uint32_t bodyB;
    PmConstraintFrame frameA;
    PmConstraintFrame frameB;
    PmAxisLimit linearLimits[3];
    PmAxisLimit angularLimits[3];
    PmAxisSpring linearSprings[3];
    PmAxisSpring angularSprings[3];
    PmAxisMotor linearMotors[3];
    PmAxisMotor angularMotors[3];
    PmBool32 collideConnected;
    float breakingImpulseThreshold;
    uint32_t solverIterations;
} PmRigidConstraintInfo;
```

```c
PmResult pmCmdWriteRigidConstraints(
    PmCommandBuffer commandBuffer,
    PmSimulation simulation,
    uint32_t firstConstraint, uint32_t count,
    const PmRigidConstraintInfo* pConstraints);
```

Writing replaces stable slots. Inactive slots ignore other fields. Each active endpoint must refer to an active body or the world sentinel, and at least one endpoint must be dynamic. collideConnected = false suppresses that body's pair contact while the joint is active. No per-joint private solver or iteration loop exists.

<a id="spec-constraint_dofs"></a>

### 8.4 Degrees of freedom

| Item | Definition |
| --- | --- |
| FIXED | All translation and rotation locked |
| POINT | Three translations locked; rotation free |
| HINGE | Only local X rotation eligible |
| SLIDER | Only local X translation eligible |
| PISTON | Local X translation and rotation eligible |
| GENERIC | All six axes eligible for limits/motors |
| GENERIC_SPRING | All six axes; limits, springs and motors |
| MOTOR | All axes free except driven/limited axes |

Controls on intrinsically locked axes must be disabled. Springs are allowed only for GENERIC_SPRING; limit/motor controls apply on eligible axes for all other families. Incompatible combinations fail validation instead of being silently ignored.

<a id="spec-constraint_example"></a>

### 8.5 Point joint to the world

A body-local anchor one meter above the body's origin is held at a world anchor three meters high. The body can rotate freely. Its geometry, body slot and simulation storage must already exist. Submit the write before the next solve with the required update-to-physics dependency.

Example (informative):

```c
PmRigidConstraintInfo joint = {0};
joint.active = PM_TRUE;
joint.type = PM_CONSTRAINT_POINT;
joint.bodyA = 0;
joint.bodyB = PM_INVALID_INDEX;
joint.frameA.position[1] = 1.0f;
joint.frameA.orientation[3] = 1.0f;
joint.frameB.position[1] = 3.0f;
joint.frameB.orientation[3] = 1.0f;
PmResult result = pmCmdWriteRigidConstraints(
    cmd, sim, 0, 1, &joint);
```

Related sections: [13.2](#spec-barriers).

<a id="spec-single_solve"></a>

### 8.6 Iteration accounting

Pipeline rigidIterations is the scene-wide AVBD budget. When no explicit runtime override is recorded, the greatest solverIterations value authored by any enabled constraint raises that single budget for the solve. The pose phase runs for each substep; impacted bodies can also run a velocity-impact phase. Statistics report the executed budget. One iteration is valid, but does not guarantee convergence.

Related sections: [15.3](#spec-step_stats), [9.1](#spec-rigid_solver).

<a id="spec-pipelines"></a>

## 9 Physics pipelines & materials

Immutable configuration • explicit inter-domain coupling

<a id="spec-rigid_solver"></a>

### 9.1 Rigid solver state

```c
typedef struct PmRigidSolverState {
    uint32_t rigidIterations;
    PmBool32 enableSleeping;
    float sleepLinearThreshold;
    float sleepAngularThreshold;
    float sleepTimeSeconds;
} PmRigidSolverState;
```

One AVBD path solves rigid geometry contacts and joints. rigidIterations is the nonzero base body budget and is ≤ maxRigidIterations; constraint solverIterations can raise only their endpoint budgets. Sleeping thresholds are nonnegative m/s and rad/s; sleep time is positive when enabled.

<a id="spec-liquid_material"></a>

### 9.2 Particle-liquid material

```c
typedef struct PmLiquidMaterialState {
    float restDensity;
    float particleRadius;
    float supportRadius;
    float restParticleVolume;
    float pressureStiffness;
    float kinematicViscosity;
    float normalDamping;
    float velocityDamping;
    float maximumSpeed;
    float maximumPairAcceleration;
    uint32_t densityIterations;
} PmLiquidMaterialState;
```

Density: kg/m³; radii: m; rest volume: m³; pressure stiffness: Pa; kinematic viscosity: m²/s; damping: s⁻¹; speed: m/s. Density, radii, volume, maximumSpeed and iterations are positive; supportRadius ≥ 2 × particleRadius. Pressure stiffness, viscosity and damping are nonnegative. Material index equals the simulation's fluid index.

Materials configure a particle-liquid domain. Smoke, gas grids and non-Newtonian rheology are outside this material model.

<a id="spec-coupling"></a>

### 9.3 Coupling state

```c
typedef enum PmCouplingType {
    PM_COUPLING_LIQUID_RIGID = 0,
    PM_COUPLING_LIQUID_LIQUID = 1,
    PM_COUPLING_LIQUID_CLOTH = 2,
    PM_COUPLING_LIQUID_ROPE = 3,
    PM_COUPLING_LIQUID_SOFT_BODY = 4,
    PM_COUPLING_LIQUID_SMOKE = 5,
    PM_COUPLING_SOFT_BODY_CLOTH = 6,
    PM_COUPLING_ROPE_SOFT_BODY = 7,
    PM_COUPLING_ROPE_CLOTH = 8,
    PM_COUPLING_SMOKE_RIGID = 9,
    PM_COUPLING_SMOKE_CLOTH = 10,
    PM_COUPLING_SMOKE_ROPE = 11,
    PM_COUPLING_SMOKE_SOFT_BODY = 12
} PmCouplingType;
typedef struct PmMechanicalCouplingState {
    float contactDistance;
    float stiffness;
    float damping;
    float friction;
    float tangentialDrag;
    float maximumForce;
    float maximumAcceleration;
    PmBool32 twoWay;
} PmMechanicalCouplingState;
typedef struct PmSmokeCouplingState {
    float windDrag;
    float maximumWindAcceleration;
    float airDensity;
    float dragCoefficient;
    float contactDistance;
    PmBool32 tracerContact;
} PmSmokeCouplingState;
typedef struct PmThermalCouplingState {
    float heaterCenter[3];
    float heaterOrientation[4];
    float heaterHalfExtents[2];
    float heaterTemperature;
    float boilingTemperature;
    float heatTransferRate;
    float windDrag;
    float steamRiseSpeed;
} PmThermalCouplingState;
typedef struct PmAttachmentCouplingState {
    uint32_t firstVertex;
    uint32_t lastVertex;
    float effectiveMass;
    float supportRadiusScale;
    float contactSupportRadiusScale;
    float maximumAcceleration;
    PmBool32 attachFirst;
    PmBool32 attachLast;
} PmAttachmentCouplingState;
typedef struct PmCouplingParameters {
    PmMechanicalCouplingState mechanical;
    PmSmokeCouplingState smoke;
    PmAttachmentCouplingState attachment;
    PmThermalCouplingState thermal;
} PmCouplingParameters;
typedef struct PmCouplingState {
    PmCouplingType type;
    uint32_t first;
    uint32_t second;
    PmCouplingParameters parameters;
} PmCouplingState;
```

first and second are stable slots in the domains named by type. PM_INVALID_INDEX denotes the all-rigid endpoint for LIQUID_RIGID. Each coupling type reads its applicable parameter groups. Duplicate endpoint pairs are invalid.

<a id="spec-domain_materials"></a>

### 9.4 Deformable, rope & smoke materials

```c
typedef struct PmClothMaterialState {
    float vertexMass;
    float thickness;
    float stretchCompliance;
    float bendingCompliance;
    float velocityDamping;
    float contactFriction;
    uint32_t solverIterations;
    float breakStrain;
    uint32_t fracturePersistenceSubsteps;
    float impactBreakImpulse;
    PmBool32 preserveVolume;
    float targetVolume;
    float volumeCompliance;
} PmClothMaterialState;
typedef struct PmRopeMaterialState {
    float nodeSpacing;
    float radius;
    float mass;
    float stretchCompliance;
    float velocityDamping;
    float maximumSubstepTimestep;
    float friction;
    float maximumSpeed;
    uint32_t solverIterations;
    PmBool32 selfCollision;
} PmRopeMaterialState;
typedef struct PmSoftBodyMaterialState {
    float nodeMass;
    float nodeRadius;
    float stretchCompliance;
    float velocityDamping;
    float springDamping;
    float contactFriction;
    float shapeMatchingStiffness;
    float maximumProjectionFraction;
    float constraintVelocityResponse;
    float maximumSpeed;
    uint32_t solverIterations;
} PmSoftBodyMaterialState;
typedef struct PmSmokeMaterialState {
    float particleRadius;
    float buoyancy;
    float response;
    float restNumberDensity;
    float pressureStiffness;
    float viscosity;
    float vorticityConfinement;
    float maximumSpeed;
    uint32_t gridResolution;
    uint32_t gridVerticalResolution;
    uint32_t gridPressureIterations;
    float gridKinematicViscosity;
    float gridLesCoefficient;
    float gridPressureTolerance;
} PmSmokeMaterialState;
```

<a id="spec-pipeline_create"></a>

### 9.5 Pipeline creation

```c
typedef struct PmPhysicsPipelineCreateInfo {
    PmStructureType sType;
    const void* pNext;
    const PmRigidSolverState* pRigidState;
    uint32_t fluidMaterialCount;
    const PmLiquidMaterialState* pFluidMaterials;
    uint32_t couplingCount;
    const PmCouplingState* pCouplings;
    uint32_t clothMaterialCount;
    const PmClothMaterialState* pClothMaterials;
    uint32_t ropeMaterialCount;
    const PmRopeMaterialState* pRopeMaterials;
    uint32_t softBodyMaterialCount;
    const PmSoftBodyMaterialState* pSoftBodyMaterials;
    uint32_t smokeMaterialCount;
    const PmSmokeMaterialState* pSmokeMaterials;
} PmPhysicsPipelineCreateInfo;
```

```c
PmResult pmCreatePhysicsPipeline(
    PmDevice device,
    const PmPhysicsPipelineCreateInfo* pInfo,
    PmPhysicsPipeline* pPipeline);
void pmDestroyPhysicsPipeline(
    PmDevice device,
    PmPhysicsPipeline pipeline);
```

Creation copies all settings. pRigidState may be NULL only when no rigid domain is needed. The material count must equal a bound simulation's fluid count. Pipelines are immutable and shareable across compatible simulations. Changing materials creates a new pipeline, without reallocating state if its capacity requirements remain satisfied.

<a id="spec-pipeline_flow"></a>

### 9.6 What a step records

pmCmdStepSimulation records a complete operation, not one CPU call per internal kernel. Its implementation owns dependencies among integration, broad phase, contact/neighbor generation, solve/coupling, velocity reconstruction and sleep. Applications supply dependencies between uploads, steps, exports, and other APIs.

A substep is a slice of physical time. An iteration refines constraints within that slice. Neither is a renderer frame. The API exposes the work budget explicitly and the queries report work actually executed.

Related sections: [11.5](#spec-step), [15.3](#spec-step_stats).

<a id="spec-pipeline_validity"></a>

### 9.7 Compatibility & bounds

A rigid simulation requires pRigidState and enabled rigidBodies. Live constraints additionally require rigidConstraints. Each active fluid requires particleLiquids. Each coupling requires its matching enabled feature. A disabled domain cannot allocate hidden capacity.

All numerical inputs must be finite. Neighbors, contacts and output ranges are bounded. If runtime generation exceeds capacity, the step reports CAPACITY_EXCEEDED and its output is invalid; the application must reset/reinitialize before reuse. There is no silent truncation advertised as correct physics.

Related sections: [7.1](#spec-simulation_create), [15.3](#spec-step_stats).

<a id="spec-domains"></a>

## 10 Simulation domains & events

Recorded creation, updates, output and diagnostics

<a id="spec-domain_records"></a>

### 10.1 Shared records

```c
typedef struct PmFloat2 { float x; float y; } PmFloat2;
typedef struct PmFloat3 { float x; float y; float z; } PmFloat3;
typedef struct PmSoftBodyBond {
    uint32_t first;
    uint32_t second;
    float restLength;
} PmSoftBodyBond;
typedef struct PmSoftBodySurfaceBinding {
    uint32_t nodes[4];
    float weights[4];
} PmSoftBodySurfaceBinding;
typedef struct PmDomainOutputHeader {
    uint32_t activeCount;
    uint32_t capacity;
    uint64_t revision;
} PmDomainOutputHeader;
```

<a id="spec-cloth_resources"></a>

### 10.2 Cloth resources

```c
typedef struct PmClothInfo {
    PmBool32 active;
    uint32_t materialIndex;
    uint32_t vertexCount;
    const PmFloat3* pVertices;
    uint32_t triangleIndexCount;
    const uint32_t* pTriangleIndices;
    const float* pInverseMasses;
} PmClothInfo;
typedef struct PmClothVertexState {
    PmFloat3 position;
    PmFloat3 velocity;
    PmFloat3 rigidContactForce;
    PmFloat3 liquidContactForce;
    PmFloat3 softBodyContactForce;
    PmFloat3 ropeContactForce;
} PmClothVertexState;
```

```c
PmResult pmCmdWriteCloths(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t firstCloth,
    uint32_t count, const PmClothInfo* pCloths);
PmResult pmCmdCopyClothState(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t clothIndex,
    const PmBufferRange* pDestination);
PmResult pmCmdCopyClothSurface(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t clothIndex,
    const PmBufferRange* pDestination);
```

<a id="spec-rope_resources"></a>

### 10.3 Rope resources

```c
typedef struct PmRopeAttachment {
    uint32_t rigidBodyIndex;
    PmFloat3 localAnchor;
    PmBool32 enabled;
} PmRopeAttachment;
typedef struct PmRopeInfo {
    PmBool32 active;
    uint32_t materialIndex;
    uint32_t pointCount;
    const PmFloat3* pCenterline;
    PmRopeAttachment first;
    PmRopeAttachment last;
} PmRopeInfo;
typedef struct PmRopeNodeState {
    PmFloat3 position;
    PmFloat3 velocity;
    PmFloat3 constraintForce;
    PmFloat3 contactForce;
    PmFloat3 liquidContactForce;
    PmFloat3 softBodyContactForce;
} PmRopeNodeState;
```

```c
PmResult pmCmdWriteRopes(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t firstRope,
    uint32_t count, const PmRopeInfo* pRopes);
PmResult pmCmdCopyRopeState(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t ropeIndex,
    const PmBufferRange* pDestination);
```

<a id="spec-soft_body_resources"></a>

### 10.4 Soft-body resources

```c
typedef struct PmSoftBodyInfo {
    PmBool32 active;
    uint32_t materialIndex;
    uint32_t nodeCount;
    const PmFloat3* pNodes;
    uint32_t bondCount;
    const PmSoftBodyBond* pBonds;
    const float* pInverseMasses;
    uint32_t surfaceVertexCount;
    const PmFloat3* pSurfaceVertices;
    uint32_t surfaceTriangleIndexCount;
    const uint32_t* pSurfaceTriangleIndices;
    const PmSoftBodySurfaceBinding* pSurfaceBindings;
} PmSoftBodyInfo;
typedef struct PmSoftBodyNodeState {
    PmFloat3 position;
    PmFloat3 velocity;
    PmFloat3 rigidContactForce;
    PmFloat3 clothContactForce;
    PmFloat3 liquidContactForce;
    PmFloat3 ropeContactForce;
} PmSoftBodyNodeState;
```

```c
PmResult pmCmdWriteSoftBodies(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t firstSoftBody,
    uint32_t count, const PmSoftBodyInfo* pSoftBodies);
PmResult pmCmdCopySoftBodyState(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t softBodyIndex,
    const PmBufferRange* pDestination);
PmResult pmCmdCopySoftBodySurface(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t softBodyIndex,
    const PmBufferRange* pDestination);
```

<a id="spec-smoke_resources"></a>

### 10.5 Smoke resources

```c
typedef struct PmSmokeInfo {
    PmBool32 active;
    uint32_t materialIndex;
    uint32_t particleCapacity;
    PmFloat3 emitterCenter;
    PmFloat2 emitterHalfExtents;
    PmFloat3 initialVelocity;
    PmFloat3 wind;
    float particlesPerSecond;
    float lifetime;
    PmFloat3 gridMinimum;
    float gridEdgeLength;
} PmSmokeInfo;
typedef struct PmSmokeParticleState {
    PmFloat3 position;
    PmFloat3 velocity;
    float age;
    float numberDensity;
    float pressure;
    PmFloat3 vorticity;
} PmSmokeParticleState;
```

```c
PmResult pmCmdWriteSmoke(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t firstSmoke,
    uint32_t count, const PmSmokeInfo* pSmoke);
PmResult pmCmdCopySmokeState(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t smokeIndex,
    const PmBufferRange* pDestination);
```

<a id="spec-emitters"></a>

### 10.6 Emitters & destruction

```c
typedef struct PmParticleEmitterInfo {
    PmBool32 active;
    uint32_t fluidIndex;
    uint32_t vertexCount;
    const PmFloat3* pVertices;
    uint32_t triangleIndexCount;
    const uint32_t* pTriangleIndices;
    float spacing;
    PmFloat3 initialVelocity;
    float initialTemperatureCelsius;
} PmParticleEmitterInfo;
typedef enum PmCrossingDirection {
    PM_CROSSING_ALONG_NORMAL = 0,
    PM_CROSSING_AGAINST_NORMAL = 1,
    PM_CROSSING_EITHER = 2
} PmCrossingDirection;
typedef struct PmParticleDestroyerInfo {
    PmBool32 active;
    uint32_t fluidIndex;
    PmFloat3 center;
    float orientation[4];
    PmFloat2 halfExtents;
    PmCrossingDirection crossing;
} PmParticleDestroyerInfo;
```

```c
PmResult pmCmdWriteParticleEmitters(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t firstEmitter,
    uint32_t count, const PmParticleEmitterInfo* pEmitters);
PmResult pmCmdWriteParticleDestroyers(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t firstDestroyer,
    uint32_t count, const PmParticleDestroyerInfo* pDestroyers);
```

<a id="spec-paint_resources"></a>

### 10.7 Paint fields & rules

```c
typedef enum PmPaintTargetType {
    PM_PAINT_TARGET_RIGID = 0,
    PM_PAINT_TARGET_CLOTH = 1
} PmPaintTargetType;
typedef struct PmPaintFieldInfo {
    PmBool32 active;
    PmPaintTargetType targetType;
    uint32_t targetIndex;
    PmGeometry geometry;
    uint32_t vertexCount;
    const PmFloat2* pVertexUvs;
    uint32_t width;
    uint32_t height;
} PmPaintFieldInfo;
typedef enum PmPaintSourceType {
    PM_PAINT_SOURCE_LIQUID = 0,
    PM_PAINT_SOURCE_RIGID = 1
} PmPaintSourceType;
typedef struct PmPaintRuleInfo {
    PmBool32 active;
    PmPaintSourceType sourceType;
    uint32_t sourceIndex;
    uint32_t targetFieldIndex;
    float reach;
    float brushRadius;
} PmPaintRuleInfo;
```

```c
PmResult pmCmdWritePaintFields(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t firstField,
    uint32_t count, const PmPaintFieldInfo* pFields);
PmResult pmCmdWritePaintRules(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t firstRule,
    uint32_t count, const PmPaintRuleInfo* pRules);
PmResult pmCmdClearPaintFields(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t firstField,
    uint32_t count);
PmResult pmCmdCopyPaintField(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t fieldIndex,
    const PmBufferRange* pDestination);
```

<a id="spec-event_outputs"></a>

### 10.8 Constraint, contact & debug output

```c
typedef struct PmConstraintState {
    PmBool32 enabled;
    PmBool32 broken;
    float appliedImpulse;
} PmConstraintState;
typedef enum PmContactEventType {
    PM_CONTACT_RIGID_RIGID = 0,
    PM_CONTACT_LIQUID_RIGID = 1
} PmContactEventType;
typedef struct PmContactEvent {
    PmContactEventType type;
    uint32_t first;
    uint32_t second;
    PmFloat3 position;
    PmFloat3 normal;
    float penetration;
    float normalImpulse;
    PmFloat3 frictionImpulse;
} PmContactEvent;
typedef enum PmDebugRecordType {
    PM_DEBUG_RIGID = 0,
    PM_DEBUG_LIQUID = 1,
    PM_DEBUG_CLOTH = 2,
    PM_DEBUG_ROPE = 3,
    PM_DEBUG_SOFT_BODY = 4
} PmDebugRecordType;
typedef struct PmDebugRecord {
    PmDebugRecordType type;
    uint32_t objectIndex;
    uint32_t elementIndex;
    PmFloat3 position;
    PmFloat3 velocity;
    PmFloat3 force;
} PmDebugRecord;
```

```c
PmResult pmCmdCopyConstraintStates(PmCommandBuffer commandBuffer,
    PmSimulation simulation, uint32_t firstConstraint,
    uint32_t count, const PmBufferRange* pDestination);
PmResult pmCmdCopyContactEvents(PmCommandBuffer commandBuffer,
    PmSimulation simulation, const PmBufferRange* pDestination);
PmResult pmCmdCopyDebugData(PmCommandBuffer commandBuffer,
    PmSimulation simulation, const PmBufferRange* pDestination);
```

<a id="spec-commands"></a>

## 11 Command recording & lifecycle

Record now; execute when a queue receives the buffer

<a id="spec-command_pool"></a>

### 11.1 Command pools

```c
enum {
    PM_COMMAND_POOL_RESET_BUFFER_BIT = 0x1,
    PM_COMMAND_BUFFER_ONE_TIME_BIT = 0x1
};
typedef struct PmCommandPoolCreateInfo {
    PmStructureType sType;
    const void* pNext;
    PmFlags flags;
    uint32_t queueFamilyIndex;
} PmCommandPoolCreateInfo;
```

```c
PmResult pmCreateCommandPool(PmDevice device,
    const PmCommandPoolCreateInfo* pInfo,
    PmCommandPool* pPool);
void pmDestroyCommandPool(PmDevice device,
    PmCommandPool pool);
PmResult pmResetCommandPool(PmDevice device,
    PmCommandPool pool);
```

A pool belongs to one family and is externally synchronized, including all recording operations on its buffers. Use one pool per recording thread. Reset returns every buffer to Initial; no buffer may be Pending. Destroying a pool frees its command buffers, after completion.

<a id="spec-command_buffers"></a>

### 11.2 Allocate, record & reset

```c
typedef struct PmCommandBufferBeginInfo {
    PmStructureType sType;
    const void* pNext;
    PmFlags flags;
} PmCommandBufferBeginInfo;
```

```c
PmResult pmAllocateCommandBuffers(
    PmDevice device, PmCommandPool pool,
    uint32_t count, PmCommandBuffer* pBuffers);
void pmFreeCommandBuffers(PmDevice device,
    PmCommandPool pool, uint32_t count,
    const PmCommandBuffer* pBuffers);
PmResult pmBeginCommandBuffer(
    PmCommandBuffer commandBuffer,
    const PmCommandBufferBeginInfo* pInfo);
PmResult pmEndCommandBuffer(
    PmCommandBuffer commandBuffer);
PmResult pmResetCommandBuffer(
    PmCommandBuffer commandBuffer);
```

Primary buffers only. Begin requires Initial. End requires Recording and no latched error or open query. Individual reset requires RESET_BUFFER on the pool. No simultaneous submission of one command buffer. Host input arrays are copied at record time; no device work occurs while recording.

<a id="spec-lifecycle"></a>

### 11.3 Legal states & resource lifetime

| Item | Definition |
| --- | --- |
| Initial | Begin → Recording; may free/reset |
| Recording | pmCmd* only; End → Executable |
| Executable | QueueSubmit → Pending; may reset/free |
| Pending | No edits, reset, free or resubmit |
| Completion | Reusable → Executable; one-time → Invalid |
| Invalid | Reset → Initial, or free |

A recording failure makes End fail and leaves the buffer Invalid. Destroying a referenced nonpending resource invalidates dependent buffers. Destroying a pending resource is invalid usage. Application code must preserve pipeline, geometry, buffers, query pools and memory until completion. Pipeline bindings do not carry across buffers.

Related sections: [13.4](#spec-submission), [13.5](#spec-fences).

<a id="spec-simulation_init"></a>

### 11.4 Reset & particle initialization

```c
typedef struct PmParticleState {
    float position[3];
    float velocity[3];
    float temperatureCelsius;
} PmParticleState;
typedef struct PmFluidParticleOutput {
    float position[3];
    float velocity[3];
    float acceleration[3];
    float temperatureCelsius;
    float foam;
    uint32_t stableId;
} PmFluidParticleOutput;
```

```c
PmResult pmCmdResetSimulation(
    PmCommandBuffer commandBuffer,
    PmSimulation simulation);
PmResult pmCmdSetFluidParticles(
    PmCommandBuffer commandBuffer,
    PmSimulation simulation,
    uint32_t fluidIndex,
    const PmBufferView* pParticles);
```

Reset is required after binding memory. It clears active bodies, joints, particles, loads, sleep history and step status; it preserves capacities. SetFluidParticles replaces one fluid population from PARTICLE_STATE input. count ≤ capacity; count zero clears the population and ignores range/format/stride. Nonzero input requires PARTICLE_INPUT usage. Temperature is carried state; the liquid model does not solve heat transfer.

<a id="spec-step"></a>

### 11.5 Bind & step

```c
enum {
    PM_STEP_COLLECT_CONTACT_EVENTS_BIT = 0x1
};
typedef struct PmStepInfo {
    float timestep;
    uint32_t substeps;
    float gravity[3];
    PmFlags flags;
} PmStepInfo;
```

```c
PmResult pmCmdBindPhysicsPipeline(
    PmCommandBuffer commandBuffer,
    PmPhysicsPipeline pipeline);
PmResult pmCmdStepSimulation(
    PmCommandBuffer commandBuffer,
    PmSimulation simulation,
    const PmStepInfo* pStep);
```

timestep > 0 seconds; 1 ≤ substeps ≤ maxSubsteps. Each substep advances timestep/substeps. Time advances only by the supplied timestep. PM_STEP_COLLECT_CONTACT_EVENTS_BIT records rigid and liquid contact events for later copy; without it the step avoids event-generation work and the copied event count is zero. A compatible pipeline must be bound. The simulation must be initialized, geometry built, and prior writes visible. Two submissions must not mutate one simulation concurrently.

Related sections: [9](#spec-pipelines), [13.2](#spec-barriers).

<a id="spec-transfer"></a>

## 12 Data movement & output

Renderer-visible buffers without exposing private solver layouts

<a id="spec-buffer_copy"></a>

### 12.1 Buffer transfers

```c
typedef struct PmBufferCopy {
    PmDeviceSize srcOffset;
    PmDeviceSize dstOffset;
    PmDeviceSize size;
} PmBufferCopy;
```

```c
PmResult pmCmdCopyBuffer(
    PmCommandBuffer commandBuffer,
    PmBuffer src, PmBuffer dst,
    uint32_t count,
    const PmBufferCopy* pRegions);
```

Use TRANSFER_SRC and TRANSFER_DST buffer usage. Regions must be in bounds, nonzero, four-byte aligned, and must not overlap through aliases. Host staging writes require a flush if noncoherent and must complete before submission. Insert a transfer-write dependency before a subsequent build or input read.

<a id="spec-outputs"></a>

### 12.2 Copy simulation output

```c
PmResult pmCmdCopyRigidStates(
    PmCommandBuffer commandBuffer,
    PmSimulation simulation,
    uint32_t firstBody, uint32_t count,
    const PmBufferRange* pDestination);
PmResult pmCmdCopyFluidParticles(
    PmCommandBuffer commandBuffer,
    PmSimulation simulation,
    uint32_t fluidIndex,
    const PmBufferRange* pDestination);
```

Destination requires SIMULATION_OUTPUT and TRANSFER_DST usage. Rigid records are tightly packed PmRigidState values in slot order; inactive slots produce zero position/velocities and identity orientation. Fluid output begins with PmDomainOutputHeader followed by active PmFluidParticleOutput records in population order. The destination must cover the addressed rigid count, or the fluid header plus declared particle capacity. Bytes beyond active records are unchanged.

These commands execute in TRANSFER and read simulation state after a PHYSICS_WRITE → TRANSFER_READ dependency. A subsequent renderer needs a TRANSFER_WRITE dependency. Output copies make the portable contract independent of private SoA solver storage.

Related sections: [16.2](#spec-formats), [13.2](#spec-barriers), [14.9](#spec-handoff).

<a id="spec-frame_recipe"></a>

### 12.3 Frame sequence (pseudocode)

Updates and state changes are recorded commands. The next frame needs a dependency on this frame's simulation reads/writes, not only on the renderer buffer. Reusing one output buffer requires waiting for its renderer consumer.

Example (informative):

```c
begin(cmd);
  resetQueries();
  writeTimestamp(start);
  applyInputs(sim);
  barrier(UPDATE_WRITE, PHYSICS_READ_WRITE);
  bindPhysicsPipeline(pipeline);
  beginStatistics(stats);
  stepSimulation(sim, dt, substeps);
  endStatistics(stats);
  barrier(PHYSICS_WRITE, TRANSFER_READ);
  copyRigidStates(sim, renderBuffer);
  writeTimestamp(end);
end(cmd);
submit(waitRender, cmd, signalPhysics, fence);
```

Related sections: [17](#spec-example), [13.4](#spec-submission).

<a id="spec-reusable_work"></a>

### 12.4 Reusable lists & frames in flight

Recorded commands capture scalar settings and host records by value. Editing the original C struct does not change an executable command buffer. Re-record changing inputs or use GPU input buffers with explicit ownership and dependencies.

Reusable buffers can return to Executable after completion. Multiple frames in flight need separate command buffers and output slots, plus timeline dependencies that serialize mutation of one simulation. More command buffers do not make one physics state safely concurrent. Use distinct simulations for independent workloads.

<a id="spec-no_hidden_work"></a>

### 12.5 Allocation and execution boundaries

The implementation may allocate CPU command-recording storage in a pool and GPU code/driver bookkeeping at pipeline creation. All physics persistent/scratch storage comes from the declared bindings. Record and submit must not grow physics device allocations. Geometry rebuild, reset, state uploads and result copies are visible commands; the application can put them on its own frame graph.

<a id="spec-sync"></a>

## 13 Submission & synchronization

Execution order, memory visibility and ownership are separate obligations

<a id="spec-stage_access"></a>

### 13.1 Stage and access scopes

```c
#define PM_STAGE_HOST_BIT UINT64_C(0x1)
#define PM_STAGE_TRANSFER_BIT UINT64_C(0x2)
#define PM_STAGE_UPDATE_BIT UINT64_C(0x4)
#define PM_STAGE_GEOMETRY_BUILD_BIT UINT64_C(0x8)
#define PM_STAGE_PHYSICS_BIT UINT64_C(0x10)
#define PM_STAGE_ALL_COMMANDS_BIT UINT64_C(0x1e)
#define PM_ACCESS_HOST_READ_BIT UINT64_C(0x1)
#define PM_ACCESS_HOST_WRITE_BIT UINT64_C(0x2)
#define PM_ACCESS_READ_BIT UINT64_C(0x4)
#define PM_ACCESS_WRITE_BIT UINT64_C(0x8)
```

READ/WRITE are generic device accesses narrowed by the stage mask. UPDATE covers reset, body/constraint/state writes and particle initialization. TRANSFER covers buffer and output copies. ALL_COMMANDS covers device stages, not HOST. A host flush is not a GPU completion wait.

<a id="spec-barriers"></a>

### 13.2 Memory & buffer barriers

```c
typedef struct PmMemoryBarrier {
    PmStageFlags srcStages;
    PmAccessFlags srcAccess;
    PmStageFlags dstStages;
    PmAccessFlags dstAccess;
} PmMemoryBarrier;
typedef struct PmBufferMemoryBarrier {
    PmMemoryBarrier memory;
    PmBufferRange range;
    uint32_t srcQueueFamilyIndex;
    uint32_t dstQueueFamilyIndex;
} PmBufferMemoryBarrier;
```

```c
PmResult pmCmdPipelineBarrier(
    PmCommandBuffer commandBuffer,
    uint32_t memoryBarrierCount,
    const PmMemoryBarrier* pMemoryBarriers,
    uint32_t bufferBarrierCount,
    const PmBufferMemoryBarrier* pBufferBarriers);
```

Memory barriers cover all matching device accesses, including opaque simulation storage. Buffer barriers narrow scope and can transfer ownership. Use IGNORED for both family indices when ownership is unchanged. For a family transfer, record a release on the source and a matching acquire on the destination, with a semaphore between queues. Release uses zero destination stage/access; acquire uses zero source stage/access. Matching ranges and family pairs are required.

Related sections: [14.9](#spec-handoff).

<a id="spec-semaphores"></a>

### 13.3 Timeline semaphores

```c
typedef struct PmSemaphoreCreateInfo {
    PmStructureType sType;
    const void* pNext;
    uint64_t initialValue;
} PmSemaphoreCreateInfo;
typedef struct PmSemaphoreSubmitInfo {
    PmSemaphore semaphore;
    uint64_t value;
    PmStageFlags stages;
} PmSemaphoreSubmitInfo;
```

```c
PmResult pmCreateSemaphore(PmDevice device,
    const PmSemaphoreCreateInfo* pInfo,
    PmSemaphore* pSemaphore);
void pmDestroySemaphore(PmDevice device,
    PmSemaphore semaphore);
PmResult pmGetSemaphoreCounterValue(
    PmDevice device, PmSemaphore semaphore,
    uint64_t* pValue);
PmResult pmWaitSemaphore(PmDevice device,
    PmSemaphore semaphore, uint64_t value,
    uint64_t timeoutNanoseconds);
```

Waits complete at counter ≥ value. Signal values increase strictly and fit within the device's maximum pending difference. Core signals occur through submission, not a host signal call. A timeline is not reset. Use separate timelines when producers cannot order their signal values.

<a id="spec-submission"></a>

### 13.4 Queue submission

```c
typedef struct PmSubmitInfo {
    PmStructureType sType;
    const void* pNext;
    uint32_t waitCount;
    const PmSemaphoreSubmitInfo* pWaits;
    uint32_t commandBufferCount;
    const PmCommandBuffer* pCommandBuffers;
    uint32_t signalCount;
    const PmSemaphoreSubmitInfo* pSignals;
} PmSubmitInfo;
```

```c
PmResult pmQueueSubmit(PmQueue queue,
    const PmSubmitInfo* pInfo, PmFence fence);
PmResult pmQueueWaitIdle(PmQueue queue);
PmResult pmDeviceWaitIdle(PmDevice device);
```

Submit accepts Executable buffers from the queue's family. A successful call means accepted, not complete. Wait scopes name the first consuming device stages; signal scopes must be ALL_COMMANDS. Signals publish prior writes; waits make them visible in the chosen stages. Ownership transfers may still be necessary. fence may be NULL. WaitIdle is useful for shutdown, not required between frames.

<a id="spec-fences"></a>

### 13.5 Host completion fences

```c
enum { PM_FENCE_SIGNALED_BIT = 0x1 };
```

```c
PmResult pmCreateFence(PmDevice device,
    PmFlags flags, PmFence* pFence);
void pmDestroyFence(PmDevice device,
    PmFence fence);
PmResult pmGetFenceStatus(PmDevice device,
    PmFence fence);
PmResult pmResetFences(PmDevice device,
    uint32_t count, const PmFence* pFences);
PmResult pmWaitForFences(PmDevice device,
    uint32_t count, const PmFence* pFences,
    PmBool32 waitAll, uint64_t timeoutNanoseconds);
```

Submit requires an unsignaled fence not associated with pending work. Get returns SUCCESS or NOT_READY. Wait accepts zero for polling and PM_TIMEOUT_INFINITE for an unbounded wait; expiration returns TIMEOUT. Only reset a fence after its associated work completes. A fence synchronizes GPU completion with the CPU, not another GPU queue.

<a id="spec-interop"></a>

## 14 Native API interoperability

CUDA, Metal, DirectX 12 and Vulkan

<a id="spec-native_support"></a>

### 14.1 Backend capabilities

```c
typedef enum PmNativeBackend {
    PM_NATIVE_BACKEND_CUDA = 1,
    PM_NATIVE_BACKEND_METAL = 2,
    PM_NATIVE_BACKEND_D3D12 = 3,
    PM_NATIVE_BACKEND_VULKAN = 4
} PmNativeBackend;
typedef uint64_t PmNativeHandle;
typedef struct PmNativeInteropProperties {
    PmBool32 deviceImport;
    PmBool32 bufferImport;
    PmBool32 timelineImport;
} PmNativeInteropProperties;
```

```c
PmResult pmGetPhysicalDeviceInteropProperties(
    PmPhysicalDevice physicalDevice,
    PmNativeBackend backend,
    PmNativeInteropProperties* pProperties);
```

Query each native backend on the selected physical device before importing objects. An unsupported backend returns FEATURE_NOT_PRESENT and zero properties. Each import requires its own capability; support for one backend does not imply support for another. The vulkanInterop feature additionally gates the typed Vulkan extension entry points.

PmNativeHandle is an in-process carrier, not a file descriptor, Win32 shared handle, or serialized object. Encode native pointer handles through uintptr_t, then uint64_t; encode integer handles without changing their bits. Decode using the original native type. Zero means no object, except a CUDA physical-device ordinal may be zero. Imported native objects remain owned by the application. PM wrappers must be destroyed, and their work completed, before native objects are released.

Related sections: [4.4](#spec-features), [4.2](#spec-properties).

<a id="spec-native_device"></a>

### 14.2 Device and queue import

```c
enum {
    PM_NATIVE_TIMELINE_ENABLED_BIT = 0x1,
    PM_NATIVE_SYNCHRONIZATION2_BIT = 0x2
};
typedef struct PmNativeQueueInfo {
    PmNativeHandle queue;
    uint32_t familyIndex;
    uint32_t queueIndex;
} PmNativeQueueInfo;
typedef struct PmNativeDeviceInfo {
    PmNativeBackend backend;
    PmNativeHandle instance;
    PmNativeHandle physicalDevice;
    PmNativeHandle device;
} PmNativeDeviceInfo;
typedef struct PmNativeDeviceImportInfo {
    PmStructureType sType;
    const void* pNext;
    PmNativeDeviceInfo nativeDevice;
    uint32_t queueCount;
    const PmNativeQueueInfo* pQueues;
    PmFlags enabledNativeFeatures;
    PmPhysicalDeviceFeatures enabledFeatures;
} PmNativeDeviceImportInfo;
```

```c
PmResult pmImportNativeDevice(
    PmInstance instance,
    const PmNativeDeviceImportInfo* pInfo,
    PmDevice* pDevice);
PmResult pmGetNativeDevice(PmDevice device,
    PmNativeDeviceInfo* pNativeDevice);
```

Import borrows a device/context and at least one explicit queue/stream. enabledFeatures selects PM features; enabledNativeFeatures records native capabilities already enabled by the caller, not requests to enable them. Unknown bits fail. TIMELINE_ENABLED is required for shared counter synchronization. SYNCHRONIZATION2 is required only for Vulkan 1.3 import, and must be zero for other backends. Get returns borrowed handles, including for PM-created native devices; it does not transfer ownership.

CUDA uses instance = 0, physicalDevice = CUdevice ordinal, device = CUcontext, queue = a non-default CUstream from that context. Metal uses instance = physicalDevice = 0, device = id<MTLDevice>, queue = id<MTLCommandQueue>. DirectX 12 uses instance = physicalDevice = 0, device = ID3D12Device*, queue = ID3D12CommandQueue* of DIRECT or COMPUTE type. Vulkan uses VkInstance, VkPhysicalDevice, VkDevice and VkQueue. Verify adapter identity before combining native objects.

Vulkan retains native queue family/index values. Other backends use familyIndex = 0 and contiguous queueIndex values starting at zero; PM exposes these as its imported family. Native queues must belong to the supplied device. Serialize host-side access when both APIs submit through the same native queue. Imports do not adopt a caller's partially recorded command list or encoder.

Related sections: [13.4](#spec-submission), [4.3](#spec-queues).

<a id="spec-native_buffer"></a>

### 14.3 Buffer import

```c
typedef struct PmNativeBufferInfo {
    PmNativeHandle buffer;
    PmDeviceSize offset;
    PmDeviceSize size;
    PmFlags usage;
    uint64_t nativeUsage;
    uint32_t owningQueueFamilyIndex;
} PmNativeBufferInfo;
```

```c
PmResult pmImportNativeBuffer(PmDevice device,
    const PmNativeBufferInfo* pInfo,
    PmBuffer* pBuffer);
PmResult pmGetNativeBuffer(PmDevice device,
    PmBuffer buffer,
    PmNativeBufferInfo* pNativeBuffer);
```

The imported buffer is already allocated and bound. size is nonzero; offset + size must fit the native allocation without overflow and satisfy PM alignment requirements. All PM offsets are relative to this imported range. No pmBindBufferMemory call is permitted on the wrapper. usage declares PM buffer usage and must be compatible with native storage. Get reports the same borrowed native object and range.

CUDA buffer is a CUdeviceptr, not a host pointer; its context must match. Metal buffer is id<MTLBuffer>; device and storage mode must permit GPU access. DirectX 12 buffer is an ID3D12Resource* with BUFFER dimension and appropriate resource flags; it enters and leaves PM ownership in D3D12_RESOURCE_STATE_COMMON. PM performs its internal state transitions. Set nativeUsage to D3D12_RESOURCE_FLAGS for DirectX 12 and to VkBufferUsageFlags for Vulkan; set it to zero for CUDA and Metal. Vulkan buffers use exclusive sharing and require usages matching the typed import contract.

owningQueueFamilyIndex is a Vulkan family for Vulkan; other backends use zero for PM ownership or PM_QUEUE_FAMILY_EXTERNAL for native ownership. On those backends, a PM buffer barrier with EXTERNAL → 0 acquires native storage; 0 → EXTERNAL releases it. The application brackets native access with the matching timeline wait/signal. Queue execution ordering does not remove resource-state or memory-visibility requirements.

Related sections: [13.2](#spec-barriers), [14.7](#spec-vulkan_buffer).

<a id="spec-native_timeline"></a>

### 14.4 Timeline import

```c
PmResult pmImportNativeSemaphore(PmDevice device,
    PmNativeHandle nativeSemaphore,
    PmSemaphore* pSemaphore);
PmResult pmGetNativeSemaphore(PmDevice device,
    PmSemaphore semaphore,
    PmNativeHandle* pNativeSemaphore);
```

Import borrows the native counter object for the device backend. CUDA accepts CUexternalSemaphore imported as a timeline semaphore or D3D12 fence, not a CUevent or binary semaphore. Metal accepts id<MTLSharedEvent>; DirectX 12 accepts ID3D12Fence*; Vulkan accepts a timeline VkSemaphore. The object must be compatible with the native device/context and its counter values must obey PM's increasing-signal rule. Get returns the borrowed object; if no compatible native representation exists it clears the output and returns FEATURE_NOT_PRESENT. CUDA external-handle import and destruction remain the application's responsibility.

Related sections: [13.3](#spec-semaphores).

<a id="spec-native_mapping"></a>

### 14.5 Native object mapping

| Item | Definition |
| --- | --- |
| CUDA | CUcontext / CUstream; CUdeviceptr; CUexternalSemaphore |
| Metal | MTLDevice / MTLCommandQueue; MTLBuffer; MTLSharedEvent |
| DirectX 12 | ID3D12Device / CommandQueue; Resource; Fence |
| Vulkan | VkDevice / VkQueue; VkBuffer; timeline VkSemaphore |

<a id="spec-vulkan_device"></a>

### 14.6 Typed Vulkan device adapter

```c
typedef struct PmVulkanQueueInfo {
    VkQueue queue;
    uint32_t familyIndex;
    uint32_t queueIndex;
} PmVulkanQueueInfo;
typedef struct PmVulkanDeviceInfo {
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
} PmVulkanDeviceInfo;
typedef struct PmVulkanDeviceImportInfo {
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    uint32_t queueCount;
    const PmVulkanQueueInfo* pQueues;
    VkBool32 timelineSemaphoreEnabled;
    VkBool32 synchronization2Enabled;
    PmPhysicalDeviceFeatures enabledFeatures;
} PmVulkanDeviceImportInfo;
```

```c
PmResult pmImportVulkanDeviceEXT(
    PmInstance instance,
    const PmVulkanDeviceImportInfo* pInfo,
    PmDevice* pDevice);
PmResult pmGetVulkanDeviceEXT(PmDevice device,
    PmVulkanDeviceInfo* pNativeDevice);
```

Declared in interop_vulkan.h. Requires Vulkan 1.3, timelineSemaphore and synchronization2 actually enabled at native device creation. Caller must report enabled state truthfully. Match an enumerated physical-device UUID with vulkanInterop support. Imports borrow every native object; PM destroys only its wrappers and internal children. Get returns borrowed handles for either an imported or PM-owned native device; never destroy PM-owned handles through Vulkan. Queue family/index values retain their Vulkan meanings.

<a id="spec-vulkan_buffer"></a>

### 14.7 Typed Vulkan buffer adapter

```c
typedef struct PmVulkanBufferImportInfo {
    VkBuffer buffer;
    VkDeviceSize size;
    VkBufferUsageFlags nativeUsage;
    PmFlags usage;
    uint32_t owningQueueFamilyIndex;
} PmVulkanBufferImportInfo;
```

```c
PmResult pmImportVulkanBufferEXT(
    PmDevice device,
    const PmVulkanBufferImportInfo* pInfo,
    PmBuffer* pBuffer);
PmResult pmGetVulkanBufferEXT(PmDevice device,
    PmBuffer buffer, VkBuffer* pNativeBuffer);
```

The native buffer must be bound, large enough, exclusive-sharing, on the same VkDevice and created with all declared native usages. PM input buffers need STORAGE_BUFFER; transfer source/destination usages must match. SIMULATION_OUTPUT needs TRANSFER_DST. Native renderer usages such as VERTEX_BUFFER remain the renderer's responsibility. Destroying an imported PM wrapper does not destroy its VkBuffer or memory.

<a id="spec-vulkan_semaphore"></a>

### 14.8 Typed Vulkan timeline adapter

```c
PmResult pmImportVulkanSemaphoreEXT(
    PmDevice device, VkSemaphore semaphore,
    PmSemaphore* pSemaphore);
PmResult pmGetVulkanSemaphoreEXT(
    PmDevice device, PmSemaphore semaphore,
    VkSemaphore* pNativeSemaphore);
```

Import borrows an existing timeline semaphore from the same device. Binary semaphores are invalid. Get returns a borrowed native handle; do not destroy it through Vulkan if PM owns it. The same increasing counter values can be used in PmSubmitInfo and VkSemaphoreSubmitInfo. No CPU wait is needed merely to connect the two queues.

<a id="spec-handoff"></a>

### 14.9 Native work → physics → native work

Before overwriting a shared output slot, physics waits for the native consumer's done value, acquires the buffer, solves, copies state, and releases the buffer. It signals completion only after the writes and release. The native consumer waits, establishes its required visibility and resource state, reads the data, then releases it and signals a later done value. The same pattern applies to rendering, compute, or another physics producer.

CUDA enqueues external-semaphore waits and signals on streams. Metal encodes shared-event waits and signals on command buffers. DirectX 12 uses command-queue Wait/Signal with a fence and the appropriate resource transitions. Vulkan uses timeline semaphore waits/signals plus matching queue-family release/acquire where necessary. A shared queue does not eliminate memory hazards; host calls sharing that queue must be serialized. No CPU idle wait is required merely to connect supported native queues.

Related sections: [13.2](#spec-barriers), [13.4](#spec-submission), [14.4](#spec-native_timeline).

<a id="spec-stage_mapping"></a>

### 14.10 Native synchronization

| Item | Definition |
| --- | --- |
| CUDA | Stream order + external timeline waits/signals |
| Metal | Command buffers + shared events + resource hazards |
| DirectX 12 | Queue fences + resource barriers |
| Vulkan | Timeline semaphores + memory/ownership barriers |

PM stages describe logical dependencies, not a native shader ABI. CUDA and Metal may conservatively synchronize complete earlier operations where a finer PM stage is unavailable. DirectX 12 must also transition shared resources at the ownership boundary. For Vulkan, TRANSFER maps to ALL_TRANSFER, UPDATE to COMPUTE_SHADER | ALL_TRANSFER, GEOMETRY_BUILD and PHYSICS to COMPUTE_SHADER, and ALL_COMMANDS to ALL_COMMANDS; PM read/write accesses map to MEMORY_READ/MEMORY_WRITE. The native consumer selects its actual consuming stage and access.

<a id="spec-interop_boundary"></a>

### 14.11 Sharing boundary

Importing an existing in-process native object is distinct from importing operating-system external-memory handles or sharing across processes. Allocate and import external resources through the native APIs first; pass the resulting compatible device-local objects to PM only when the relevant capability is reported. This contract does not promise arbitrary cross-API or cross-device sharing, handle export, or a host-copy fallback. Unsupported imports return FEATURE_NOT_PRESENT. The typed Vulkan adapters have the same ownership and capability requirements as the backend-neutral calls.

<a id="spec-queries"></a>

## 15 Timing, work & memory queries

Measure GPU execution separately from host overhead

<a id="spec-query_pool"></a>

### 15.1 Query pools

```c
typedef enum PmQueryType {
    PM_QUERY_TIMESTAMP = 0,
    PM_QUERY_STEP_STATISTICS = 1
} PmQueryType;
typedef struct PmQueryPoolCreateInfo {
    PmStructureType sType;
    const void* pNext;
    PmQueryType queryType;
    uint32_t queryCount;
} PmQueryPoolCreateInfo;
```

```c
PmResult pmCreateQueryPool(PmDevice device,
    const PmQueryPoolCreateInfo* pInfo,
    PmQueryPool* pPool);
void pmDestroyQueryPool(PmDevice device,
    PmQueryPool pool);
PmResult pmCmdResetQueryPool(
    PmCommandBuffer commandBuffer,
    PmQueryPool pool, uint32_t first,
    uint32_t count);
```

Query pools own implementation-managed result storage, accounted as library bytes. Reset slots before first use and after consuming prior results. Do not reset results still in flight. timestampQueries/statisticsQueries must be enabled for the selected type. queryCount is positive and bounded by maxQueriesPerPool.

<a id="spec-timestamps"></a>

### 15.2 GPU timestamps

```c
PmResult pmCmdWriteTimestamp(
    PmCommandBuffer commandBuffer,
    PmStageFlags stage,
    PmQueryPool pool, uint32_t query);
```

Use ALL_COMMANDS for a portable before/after measurement. A timestamp becomes available once earlier commands have completed and the sample has been written. Timestamp values are unsigned 64-bit ticks; only timestampValidBits are meaningful. GPU nanoseconds = modular tick delta × timestampPeriodNanoseconds. Samples must be on the same queue, and duration must be less than one wrap period. They do not measure CPU recording, queue-submit overhead or wall-clock time.

<a id="spec-step_stats"></a>

### 15.3 Step statistics & execution status

```c
typedef struct PmStepStatistics {
    uint64_t stepCount;
    uint64_t substepsExecuted;
    uint64_t poseIterations;
    uint64_t impactIterations;
    uint64_t peakContactCount;
    uint64_t peakActiveBodies;
    uint64_t peakAwakeBodies;
    uint64_t peakParticleCount;
    uint64_t scratchHighWaterBytes;
    uint64_t overflowCount;
    int32_t executionResult;
    uint32_t earlyExitCount;
} PmStepStatistics;
```

```c
PmResult pmCmdBeginStatisticsQuery(
    PmCommandBuffer commandBuffer,
    PmQueryPool pool, uint32_t query);
PmResult pmCmdEndStatisticsQuery(
    PmCommandBuffer commandBuffer,
    PmQueryPool pool, uint32_t query);
PmResult pmGetSimulationStatus(
    PmDevice device, PmSimulation simulation,
    PmResult* pExecutionResult);
```

Queries accumulate across enclosed steps: work counters sum, peak fields and high-water bytes take maxima. executionResult is the first failing PmResult, or SUCCESS. Do not nest statistics queries; Begin and End must be in one buffer. After observing completion, GetSimulationStatus exposes the last step status even when statistics are disabled. It returns NOT_READY while simulation work is pending. Overflow invalidates that step's output and requires reinitialization.

<a id="spec-query_results"></a>

### 15.4 Readback & availability

```c
enum {
    PM_QUERY_RESULT_WAIT_BIT = 0x1,
    PM_QUERY_RESULT_WITH_AVAILABILITY_BIT = 0x2
};
```

```c
PmResult pmGetQueryPoolResults(PmDevice device,
    PmQueryPool pool, uint32_t first,
    uint32_t count, size_t dataSize,
    void* pData, PmDeviceSize stride,
    PmFlags flags);
```

A timestamp result is uint64_t; statistics use PmStepStatistics (88 bytes). WITH_AVAILABILITY appends uint64_t, zero/unavailable or one/available, after each result. stride is an eight-byte multiple large enough for a record; pData is eight-byte aligned; dataSize covers all records. Without WAIT, unavailable result bytes remain untouched and NOT_READY is returned; availability fields are still written. WAIT has no timeout: use a fence or semaphore wait first for bounded waiting.

<a id="spec-memory_budget"></a>

### 15.5 Device budget & library accounting

```c
typedef struct PmMemoryBudget {
    uint32_t heapCount;
    PmDeviceSize budget[16];
    PmDeviceSize usage[16];
    PmDeviceSize applicationAllocationBytes;
    PmDeviceSize libraryDeviceBytes;
    PmDeviceSize libraryHostBytes;
    PmDeviceSize borrowedBufferBytes;
} PmMemoryBudget;
```

```c
PmResult pmGetDeviceMemoryBudget(
    PmDevice device, PmMemoryBudget* pBudget);
```

Requires enabled memoryBudget. Budget/usage are live driver estimates per heap, not guaranteed free space or a reservation. Application bytes sum owned pmAllocateMemory blocks. Library bytes cover queries, pipeline code and internal bookkeeping. Borrowed bytes sum imported buffer sizes, not physical residency or unique allocations. These accounting categories are not additive with heap usage. Resource requirements describe reservation; scratchHighWaterBytes describes use.

Related sections: [5.1](#spec-memory_types).

<a id="spec-capabilities"></a>

## 16 Features, limits & formats

Capability decisions are made before allocation and pipeline creation

<a id="spec-limits"></a>

### 16.1 Physical-device limits

```c
typedef struct PmPhysicalDeviceLimits {
    PmDeviceSize maxAllocationSize;
    PmDeviceSize maxBufferSize;
    PmDeviceSize nonCoherentAtomSize;
    uint32_t maxMemoryAllocations;
    uint32_t maxRigidBodies;
    uint32_t maxRigidConstraints;
    uint32_t maxContacts;
    uint32_t maxFluidSystems;
    uint32_t maxParticlesPerFluid;
    uint32_t maxNeighborsPerParticle;
    uint32_t maxRigidIterations;
    uint32_t maxLiquidIterations;
    uint32_t maxSubsteps;
    uint32_t maxQueriesPerPool;
    uint64_t maxTimelineValueDifference;
} PmPhysicalDeviceLimits;
```

```c
void pmGetPhysicalDeviceLimits(
    PmPhysicalDevice physicalDevice,
    PmPhysicalDeviceLimits* pLimits);
```

Limits are maxima for one resource or field, not performance targets. Memory pressure can still make a legal allocation fail. Every count and numerical budget must be checked against the corresponding limit before creation/recording.

<a id="spec-formats"></a>

### 16.2 Format support query

```c
enum {
    PM_FORMAT_GEOMETRY_VERTEX_BIT = 0x1,
    PM_FORMAT_GEOMETRY_INDEX_BIT = 0x2,
    PM_FORMAT_PARTICLE_INPUT_BIT = 0x4,
    PM_FORMAT_SIMULATION_OUTPUT_BIT = 0x8
};
typedef struct PmFormatProperties {
    uint32_t elementSize;
    uint32_t elementAlignment;
    PmFlags supportedUsages;
} PmFormatProperties;
```

```c
void pmGetPhysicalDeviceFormatProperties(
    PmPhysicalDevice physicalDevice,
    PmFormat format,
    PmFormatProperties* pProperties);
```

UNDEFINED or an unsupported format returns zero properties. Supported formats have the fixed layouts below; implementation-specific packed formats are not inferred. Query flags express usage support, not a separate simulation model.

<a id="spec-format_layouts"></a>

### 16.3 Portable buffer layouts

| Item | Definition |
| --- | --- |
| R32_UINT | 4 bytes / alignment 4 / unsigned index |
| R32_SFLOAT | 4 bytes / alignment 4 / IEEE binary32 |
| R32G32B32_SFLOAT | 12 bytes / alignment 4 / x,y,z |
| R32G32B32A32_SFLOAT | 16 bytes / alignment 4 / x,y,z,w |
| PARTICLE_STATE | 28 bytes / alignment 4 / PmParticleState |
| FLUID_PARTICLE_OUTPUT | 48 bytes / alignment 4 / PmFluidParticleOutput |
| RIGID_STATE | 52 bytes / alignment 4 / PmRigidState |

Record arrays are little-endian. These buffer layouts have fixed offsets, regardless of a shader language's native vector packing. PmRigidBodyInfo and other handle-containing host descriptors are not GPU binary layouts. Do not memcpy them into storage buffers. Scalar fields are 32-bit unless explicitly declared otherwise.

<a id="spec-feature_matrix"></a>

### 16.4 Dependency checklist

| Item | Definition |
| --- | --- |
| Rigid contacts | rigidBodies |
| Rigid joints | rigidBodies + rigidConstraints |
| One liquid | particleLiquids |
| Liquid ↔ bodies | both domains + liquidRigidCoupling |
| Liquid ↔ liquid | particleLiquids + liquidLiquidCoupling |
| GPU timing | timestampQueries + queue timestamp bits |
| Solver counters | statisticsQueries |
| Native object sharing | Per-backend import capabilities |
| Typed Vulkan adapters | vulkanInterop + required native features |

Report a feature only when its required behavior is supported. Requesting unsupported features or coupling pairs fails with FEATURE_NOT_PRESENT.

Related sections: [4.4](#spec-features), [14.1](#spec-native_support).

<a id="spec-threading"></a>

### 16.5 Threading & data races

Physical-device queries are safe to call concurrently. Distinct command pools can record independently. Host access to the same queue, pool, fence, mapped range or mutable resource must be externally synchronized. Destroying an object must also be synchronized against queries that use it.

GPU races are a separate concern. Serialize writes to a simulation, coordinate shared input/output buffers, and avoid changing CPU-mapped bytes while a pending command may read them. Callback implementations must support concurrency across independently accessed objects.

<a id="spec-not_capabilities"></a>

### 16.6 Core scope

Core 0.2 defines primary command buffers, explicit memory binding, rigid triangle geometry, rigid constraints, particle liquids and the listed couplings. Secondary buffers, simultaneous reuse, sparse resources, shader injection and cross-device migration are outside this specification.

Cloth, ropes, soft bodies, smoke, emitters and paint require domain-specific extensions. Unknown pNext structures are errors.

Related sections: [4.4](#spec-features).

<a id="spec-example"></a>

## 17 A frame, recorded explicitly

Record, submit and consume a simulation frame

<a id="spec-example_record"></a>

### 17.1 Record a rigid-body frame

TRY returns the first non-success result. On any recording error, reset the buffer before trying again. This example requires initialized resources, completed prior command/query use, and a renderer output buffer in the same queue family. All referenced objects and memory must remain alive until completion.

Example (informative):

```c
PmResult record_frame(PmCommandBuffer cmd,
    PmPhysicsPipeline pipeline, PmSimulation sim,
    PmBuffer output, uint32_t bodyCount,
    PmQueryPool timestamps, const PmStepInfo* step)
{
    PmCommandBufferBeginInfo begin;
    memset(&begin, 0, sizeof(begin));
    begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = PM_COMMAND_BUFFER_ONE_TIME_BIT;
    TRY(pmResetCommandBuffer(cmd));
    TRY(pmBeginCommandBuffer(cmd, &begin));
    TRY(pmCmdResetQueryPool(cmd, timestamps, 0, 2));
    TRY(pmCmdWriteTimestamp(cmd,
        PM_STAGE_ALL_COMMANDS_BIT, timestamps, 0));

    PmMemoryBarrier before = {
        PM_STAGE_ALL_COMMANDS_BIT, PM_ACCESS_WRITE_BIT,
        PM_STAGE_PHYSICS_BIT,
        PM_ACCESS_READ_BIT | PM_ACCESS_WRITE_BIT
    };
    TRY(pmCmdPipelineBarrier(cmd, 1, &before, 0, NULL));
    TRY(pmCmdBindPhysicsPipeline(cmd, pipeline));
    TRY(pmCmdStepSimulation(cmd, sim, step));

    PmMemoryBarrier after = {
        PM_STAGE_PHYSICS_BIT, PM_ACCESS_WRITE_BIT,
        PM_STAGE_TRANSFER_BIT, PM_ACCESS_READ_BIT
    };
    TRY(pmCmdPipelineBarrier(cmd, 1, &after, 0, NULL));
    PmBufferRange destination = {
        output, 0, (PmDeviceSize)bodyCount * 64u
    };
    TRY(pmCmdCopyRigidStates(
        cmd, sim, 0, bodyCount, &destination));
    TRY(pmCmdWriteTimestamp(cmd,
        PM_STAGE_ALL_COMMANDS_BIT, timestamps, 1));
    return pmEndCommandBuffer(cmd);
}
```

Related sections: [11.3](#spec-lifecycle), [14.9](#spec-handoff).

<a id="spec-example_submit"></a>

### 17.2 Submit a frame between timeline values

The renderer signals renderDone after its previous read. Physics signals physicsDone after copying output. The renderer waits for that new physics value at its actual consuming stage. completion is an unsignaled fence, or NULL. Semaphore signal values must be globally monotonic.

Example (informative):

```c
PmResult submit_frame(PmQueue queue, PmCommandBuffer cmd,
    PmSemaphore renderDone, uint64_t renderValue,
    PmSemaphore physicsDone, uint64_t physicsValue,
    PmFence completion)
{
    PmSemaphoreSubmitInfo wait = {
        renderDone, renderValue, PM_STAGE_ALL_COMMANDS_BIT
    };
    PmSemaphoreSubmitInfo signal = {
        physicsDone, physicsValue, PM_STAGE_ALL_COMMANDS_BIT
    };
    PmSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitCount = 1;
    submit.pWaits = &wait;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    submit.signalCount = 1;
    submit.pSignals = &signal;
    return pmQueueSubmit(queue, &submit, completion);
}
```

Related sections: [13.4](#spec-submission).

<a id="spec-example_prepare"></a>

### 17.3 Before the first frame

| Item | Definition |
| --- | --- |
| Instance / device | Enumerate; query support; enable features |
| Geometry / buffers | Create, allocate, bind, upload and build |
| Simulation | Declare capacities; bind both memory ranges |
| Pipeline | Set rigid state and optional liquid materials |
| Initialization list | Reset state; write bodies and constraints |
| Frame objects | Pool, buffers, timelines, queries and fence |

Initialize geometry and simulation state before recording a frame. Make all initialization writes visible to the first physics step.

<a id="spec-example_after"></a>

### 17.4 After submission

Wait or poll completion before reusing a command buffer or query slot. Read simulation status before accepting results after suspected overflow. Read timestamps with availability, using the queue's period and valid-bit count. A CPU readback additionally needs a device-to-host dependency and invalidation for noncoherent memory.

Do not call DeviceWaitIdle every frame to compensate for missing dependencies. Track the resources and completion values that each consumer actually uses.

Related sections: [15.2](#spec-timestamps), [15.4](#spec-query_results), [5.4](#spec-mapping).

<a id="spec-fundamentals"></a>

## 18 Fundamental declarations

Types and identifiers

<a id="spec-base_types"></a>

### 18.1 Types, handles & constants

```c
#define PM_API_VERSION_0_2 ((uint32_t)(2u << 12))
#define PM_WHOLE_SIZE UINT64_MAX
#define PM_TIMEOUT_INFINITE UINT64_MAX
#define PM_INVALID_INDEX UINT32_MAX
#define PM_QUEUE_FAMILY_IGNORED UINT32_MAX
#define PM_QUEUE_FAMILY_EXTERNAL (UINT32_MAX - 1u)
#define PM_FALSE 0u
#define PM_TRUE 1u
typedef uint32_t PmBool32;
typedef uint32_t PmFlags;
typedef uint64_t PmDeviceSize;
typedef uint64_t PmStageFlags;
typedef uint64_t PmAccessFlags;
typedef struct PmInstance_T* PmInstance;
typedef struct PmPhysicalDevice_T* PmPhysicalDevice;
typedef struct PmDevice_T* PmDevice;
typedef struct PmQueue_T* PmQueue;
typedef struct PmDeviceMemory_T* PmDeviceMemory;
typedef struct PmBuffer_T* PmBuffer;
typedef struct PmGeometry_T* PmGeometry;
typedef struct PmSimulation_T* PmSimulation;
typedef struct PmPhysicsPipeline_T* PmPhysicsPipeline;
typedef struct PmCommandPool_T* PmCommandPool;
typedef struct PmCommandBuffer_T* PmCommandBuffer;
typedef struct PmSemaphore_T* PmSemaphore;
typedef struct PmFence_T* PmFence;
typedef struct PmQueryPool_T* PmQueryPool;
```

Handles are opaque pointers, not structures applications may dereference. PMAPI_ATTR and PMAPI_CALL default to empty and may be supplied by the platform ABI. All declarations use C linkage in C++. PmBool32 is PM_FALSE or PM_TRUE. Each command has a PFN_pmCommandName function-pointer typedef with exactly the displayed return type, parameters and PMAPI_CALL convention.

<a id="spec-structure_types"></a>

### 18.2 Structure types

```c
typedef enum PmStructureType {
    PM_STRUCTURE_TYPE_UNDEFINED = 0,
    PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
    PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 2,
    PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 3,
    PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 4,
    PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO = 5,
    PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO = 6,
    PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO = 7,
    PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 8,
    PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 9,
    PM_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO = 10,
    PM_STRUCTURE_TYPE_SUBMIT_INFO = 11,
    PM_STRUCTURE_TYPE_NATIVE_DEVICE_IMPORT_INFO = 12,
    PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO = 13,
} PmStructureType;
```

Every extensible input structure sets sType to the identifier for its exact structure type. pNext is NULL unless a supported extension defines the chained structure. An identifier does not enable a feature or extension.

<a id="spec-symbol_index"></a>

## A Symbol index

| Name | Section |
| --- | --- |
| PFN_pmVoidFunction | [3.1](#spec-function_pointers) |
| PmAccessFlags | [18.1](#spec-base_types) |
| pmAllocateCommandBuffers | [11.2](#spec-command_buffers) |
| PmAllocateFn | [3.3](#spec-allocator) |
| pmAllocateMemory | [5.3](#spec-allocation) |
| PmAllocationCallbacks | [3.3](#spec-allocator) |
| PmAttachmentCouplingState | [9.3](#spec-coupling) |
| PmAxisLimit | [8.2](#spec-axis_control) |
| PmAxisMotor | [8.2](#spec-axis_control) |
| PmAxisSpring | [8.2](#spec-axis_control) |
| pmBeginCommandBuffer | [11.2](#spec-command_buffers) |
| pmBindBufferMemory | [6.2](#spec-buffer_binding) |
| pmBindGeometryMemory | [6.4](#spec-geometry) |
| pmBindSimulationMemory | [7.2](#spec-simulation_memory) |
| PmBool32 | [18.1](#spec-base_types) |
| PmBuffer | [18.1](#spec-base_types) |
| PmBufferCopy | [12.1](#spec-buffer_copy) |
| PmBufferCreateInfo | [6.1](#spec-buffer) |
| PmBufferMemoryBarrier | [13.2](#spec-barriers) |
| PmBufferRange | [6.2](#spec-buffer_binding) |
| PmBufferView | [6.3](#spec-format_enum) |
| PmClothCapacity | [7.1](#spec-simulation_create) |
| PmClothInfo | [10.2](#spec-cloth_resources) |
| PmClothMaterialState | [9.4](#spec-domain_materials) |
| PmClothVertexState | [10.2](#spec-cloth_resources) |
| pmCmdApplyRigidInputs | [7.5](#spec-body_inputs) |
| pmCmdBeginStatisticsQuery | [15.3](#spec-step_stats) |
| pmCmdBindPhysicsPipeline | [11.5](#spec-step) |
| pmCmdBuildGeometry | [6.5](#spec-geometry_build) |
| pmCmdClearPaintFields | [10.7](#spec-paint_resources) |
| pmCmdCopyBuffer | [12.1](#spec-buffer_copy) |
| pmCmdCopyClothState | [10.2](#spec-cloth_resources) |
| pmCmdCopyClothSurface | [10.2](#spec-cloth_resources) |
| pmCmdCopyConstraintStates | [10.8](#spec-event_outputs) |
| pmCmdCopyContactEvents | [10.8](#spec-event_outputs) |
| pmCmdCopyDebugData | [10.8](#spec-event_outputs) |
| pmCmdCopyFluidParticles | [12.2](#spec-outputs) |
| pmCmdCopyPaintField | [10.7](#spec-paint_resources) |
| pmCmdCopyRigidStates | [12.2](#spec-outputs) |
| pmCmdCopyRopeState | [10.3](#spec-rope_resources) |
| pmCmdCopySmokeState | [10.5](#spec-smoke_resources) |
| pmCmdCopySoftBodyState | [10.4](#spec-soft_body_resources) |
| pmCmdCopySoftBodySurface | [10.4](#spec-soft_body_resources) |
| pmCmdEndStatisticsQuery | [15.3](#spec-step_stats) |
| pmCmdPipelineBarrier | [13.2](#spec-barriers) |
| pmCmdQueryHitBox | [7.6](#spec-hit_boxes) |
| pmCmdResetQueryPool | [15.1](#spec-query_pool) |
| pmCmdResetSimulation | [11.4](#spec-simulation_init) |
| pmCmdSetFluidParticles | [11.4](#spec-simulation_init) |
| pmCmdSetKinematicTargets | [7.5](#spec-body_inputs) |
| pmCmdSetRigidStates | [7.5](#spec-body_inputs) |
| pmCmdStepSimulation | [11.5](#spec-step) |
| pmCmdWriteCloths | [10.2](#spec-cloth_resources) |
| pmCmdWritePaintFields | [10.7](#spec-paint_resources) |
| pmCmdWritePaintRules | [10.7](#spec-paint_resources) |
| pmCmdWriteParticleDestroyers | [10.6](#spec-emitters) |
| pmCmdWriteParticleEmitters | [10.6](#spec-emitters) |
| pmCmdWriteRigidBodies | [7.4](#spec-bodies) |
| pmCmdWriteRigidConstraints | [8.3](#spec-constraint_info) |
| pmCmdWriteRopes | [10.3](#spec-rope_resources) |
| pmCmdWriteSmoke | [10.5](#spec-smoke_resources) |
| pmCmdWriteSoftBodies | [10.4](#spec-soft_body_resources) |
| pmCmdWriteTimestamp | [15.2](#spec-timestamps) |
| PmCommandBuffer | [18.1](#spec-base_types) |
| PmCommandBufferBeginInfo | [11.2](#spec-command_buffers) |
| PmCommandPool | [18.1](#spec-base_types) |
| PmCommandPoolCreateInfo | [11.1](#spec-command_pool) |
| PmConstraintFrame | [8.1](#spec-constraint_types) |
| PmConstraintState | [10.8](#spec-event_outputs) |
| PmConstraintType | [8.1](#spec-constraint_types) |
| PmContactEvent | [10.8](#spec-event_outputs) |
| PmContactEventType | [10.8](#spec-event_outputs) |
| PmCouplingParameters | [9.3](#spec-coupling) |
| PmCouplingState | [9.3](#spec-coupling) |
| PmCouplingType | [9.3](#spec-coupling) |
| pmCreateBuffer | [6.1](#spec-buffer) |
| pmCreateCommandPool | [11.1](#spec-command_pool) |
| pmCreateDevice | [4.5](#spec-device) |
| pmCreateFence | [13.5](#spec-fences) |
| pmCreateGeometry | [6.4](#spec-geometry) |
| pmCreateInstance | [3.2](#spec-instance) |
| pmCreatePhysicsPipeline | [9.5](#spec-pipeline_create) |
| pmCreateQueryPool | [15.1](#spec-query_pool) |
| pmCreateSemaphore | [13.3](#spec-semaphores) |
| pmCreateSimulation | [7.1](#spec-simulation_create) |
| PmCrossingDirection | [10.6](#spec-emitters) |
| PmDebugRecord | [10.8](#spec-event_outputs) |
| PmDebugRecordType | [10.8](#spec-event_outputs) |
| pmDestroyBuffer | [6.1](#spec-buffer) |
| pmDestroyCommandPool | [11.1](#spec-command_pool) |
| pmDestroyDevice | [4.5](#spec-device) |
| pmDestroyFence | [13.5](#spec-fences) |
| pmDestroyGeometry | [6.4](#spec-geometry) |
| pmDestroyInstance | [3.2](#spec-instance) |
| pmDestroyPhysicsPipeline | [9.5](#spec-pipeline_create) |
| pmDestroyQueryPool | [15.1](#spec-query_pool) |
| pmDestroySemaphore | [13.3](#spec-semaphores) |
| pmDestroySimulation | [7.1](#spec-simulation_create) |
| PmDevice | [18.1](#spec-base_types) |
| PmDeviceCreateInfo | [4.5](#spec-device) |
| PmDeviceMemory | [18.1](#spec-base_types) |
| PmDeviceQueueCreateInfo | [4.5](#spec-device) |
| PmDeviceSize | [18.1](#spec-base_types) |
| pmDeviceWaitIdle | [13.4](#spec-submission) |
| PmDomainOutputHeader | [10.1](#spec-domain_records) |
| pmEndCommandBuffer | [11.2](#spec-command_buffers) |
| pmEnumerateInstanceVersion | [3.2](#spec-instance) |
| pmEnumeratePhysicalDevices | [4.1](#spec-physical_devices) |
| PmFence | [18.1](#spec-base_types) |
| PmFlags | [18.1](#spec-base_types) |
| PmFloat2 | [10.1](#spec-domain_records) |
| PmFloat3 | [10.1](#spec-domain_records) |
| PmFluidCapacity | [7.1](#spec-simulation_create) |
| PmFluidParticleOutput | [11.4](#spec-simulation_init) |
| pmFlushMappedMemoryRanges | [5.4](#spec-mapping) |
| PmFormat | [6.3](#spec-format_enum) |
| PmFormatProperties | [16.2](#spec-formats) |
| pmFreeCommandBuffers | [11.2](#spec-command_buffers) |
| PmFreeFn | [3.3](#spec-allocator) |
| pmFreeMemory | [5.3](#spec-allocation) |
| PmGeometry | [18.1](#spec-base_types) |
| PmGeometryCreateInfo | [6.4](#spec-geometry) |
| pmGetBufferMemoryRequirements | [6.2](#spec-buffer_binding) |
| pmGetDeviceMemoryBudget | [15.5](#spec-memory_budget) |
| pmGetDeviceProcAddr | [3.1](#spec-function_pointers) |
| pmGetDeviceQueue | [4.5](#spec-device) |
| pmGetFenceStatus | [13.5](#spec-fences) |
| pmGetGeometryMemoryRequirements | [6.4](#spec-geometry) |
| pmGetInstanceProcAddr | [3.1](#spec-function_pointers) |
| pmGetNativeBuffer | [14.3](#spec-native_buffer) |
| pmGetNativeDevice | [14.2](#spec-native_device) |
| pmGetNativeSemaphore | [14.4](#spec-native_timeline) |
| pmGetPhysicalDeviceFeatures | [4.4](#spec-features) |
| pmGetPhysicalDeviceFormatProperties | [16.2](#spec-formats) |
| pmGetPhysicalDeviceInteropProperties | [14.1](#spec-native_support) |
| pmGetPhysicalDeviceLimits | [16.1](#spec-limits) |
| pmGetPhysicalDeviceMemoryProperties | [5.1](#spec-memory_types) |
| pmGetPhysicalDeviceProperties | [4.2](#spec-properties) |
| pmGetPhysicalDeviceQueueFamilies | [4.3](#spec-queues) |
| pmGetQueryPoolResults | [15.4](#spec-query_results) |
| pmGetSemaphoreCounterValue | [13.3](#spec-semaphores) |
| pmGetSimulationMemoryRequirements | [7.2](#spec-simulation_memory) |
| pmGetSimulationStatus | [15.3](#spec-step_stats) |
| pmGetVulkanBufferEXT | [14.7](#spec-vulkan_buffer) |
| pmGetVulkanDeviceEXT | [14.6](#spec-vulkan_device) |
| pmGetVulkanSemaphoreEXT | [14.8](#spec-vulkan_semaphore) |
| PmHitBox | [7.6](#spec-hit_boxes) |
| PmHitBoxOutputHeader | [7.6](#spec-hit_boxes) |
| PmHitBoxParticle | [7.6](#spec-hit_boxes) |
| pmImportNativeBuffer | [14.3](#spec-native_buffer) |
| pmImportNativeDevice | [14.2](#spec-native_device) |
| pmImportNativeSemaphore | [14.4](#spec-native_timeline) |
| pmImportVulkanBufferEXT | [14.7](#spec-vulkan_buffer) |
| pmImportVulkanDeviceEXT | [14.6](#spec-vulkan_device) |
| pmImportVulkanSemaphoreEXT | [14.8](#spec-vulkan_semaphore) |
| PmInstance | [18.1](#spec-base_types) |
| PmInstanceCreateInfo | [3.2](#spec-instance) |
| pmInvalidateMappedMemoryRanges | [5.4](#spec-mapping) |
| PmLiquidMaterialState | [9.2](#spec-liquid_material) |
| pmMapMemory | [5.4](#spec-mapping) |
| PmMappedMemoryRange | [5.4](#spec-mapping) |
| PmMechanicalCouplingState | [9.3](#spec-coupling) |
| PmMemoryAllocateInfo | [5.3](#spec-allocation) |
| PmMemoryBarrier | [13.2](#spec-barriers) |
| PmMemoryBudget | [15.5](#spec-memory_budget) |
| PmMemoryHeap | [5.1](#spec-memory_types) |
| PmMemoryProperties | [5.1](#spec-memory_types) |
| PmMemoryRequirements | [5.2](#spec-requirements) |
| PmMemoryType | [5.1](#spec-memory_types) |
| PmMotionType | [7.3](#spec-rigid_state) |
| PmNativeBackend | [14.1](#spec-native_support) |
| PmNativeBufferInfo | [14.3](#spec-native_buffer) |
| PmNativeDeviceImportInfo | [14.2](#spec-native_device) |
| PmNativeDeviceInfo | [14.2](#spec-native_device) |
| PmNativeHandle | [14.1](#spec-native_support) |
| PmNativeInteropProperties | [14.1](#spec-native_support) |
| PmNativeQueueInfo | [14.2](#spec-native_device) |
| PmPaintFieldCapacity | [7.1](#spec-simulation_create) |
| PmPaintFieldInfo | [10.7](#spec-paint_resources) |
| PmPaintRuleInfo | [10.7](#spec-paint_resources) |
| PmPaintSourceType | [10.7](#spec-paint_resources) |
| PmPaintTargetType | [10.7](#spec-paint_resources) |
| PmParticleDestroyerInfo | [10.6](#spec-emitters) |
| PmParticleEmitterInfo | [10.6](#spec-emitters) |
| PmParticleState | [11.4](#spec-simulation_init) |
| PmPhysicalDevice | [18.1](#spec-base_types) |
| PmPhysicalDeviceFeatures | [4.4](#spec-features) |
| PmPhysicalDeviceLimits | [16.1](#spec-limits) |
| PmPhysicalDeviceProperties | [4.2](#spec-properties) |
| PmPhysicsPipeline | [18.1](#spec-base_types) |
| PmPhysicsPipelineCreateInfo | [9.5](#spec-pipeline_create) |
| PmQueryPool | [18.1](#spec-base_types) |
| PmQueryPoolCreateInfo | [15.1](#spec-query_pool) |
| PmQueryType | [15.1](#spec-query_pool) |
| PmQueue | [18.1](#spec-base_types) |
| PmQueueFamilyProperties | [4.3](#spec-queues) |
| pmQueueSubmit | [13.4](#spec-submission) |
| pmQueueWaitIdle | [13.4](#spec-submission) |
| pmResetCommandBuffer | [11.2](#spec-command_buffers) |
| pmResetCommandPool | [11.1](#spec-command_pool) |
| pmResetFences | [13.5](#spec-fences) |
| PmResult | [2.1](#spec-results) |
| PmRigidBodyInfo | [7.4](#spec-bodies) |
| PmRigidConstraintInfo | [8.3](#spec-constraint_info) |
| PmRigidInput | [7.5](#spec-body_inputs) |
| PmRigidSolverState | [9.1](#spec-rigid_solver) |
| PmRigidState | [7.3](#spec-rigid_state) |
| PmRopeAttachment | [10.3](#spec-rope_resources) |
| PmRopeCapacity | [7.1](#spec-simulation_create) |
| PmRopeInfo | [10.3](#spec-rope_resources) |
| PmRopeMaterialState | [9.4](#spec-domain_materials) |
| PmRopeNodeState | [10.3](#spec-rope_resources) |
| PmSemaphore | [18.1](#spec-base_types) |
| PmSemaphoreCreateInfo | [13.3](#spec-semaphores) |
| PmSemaphoreSubmitInfo | [13.3](#spec-semaphores) |
| PmSimulation | [18.1](#spec-base_types) |
| PmSimulationCreateInfo | [7.1](#spec-simulation_create) |
| PmSimulationMemoryBinding | [7.2](#spec-simulation_memory) |
| PmSimulationMemoryRequirements | [7.2](#spec-simulation_memory) |
| PmSmokeCapacity | [7.1](#spec-simulation_create) |
| PmSmokeCouplingState | [9.3](#spec-coupling) |
| PmSmokeInfo | [10.5](#spec-smoke_resources) |
| PmSmokeMaterialState | [9.4](#spec-domain_materials) |
| PmSmokeParticleState | [10.5](#spec-smoke_resources) |
| PmSoftBodyBond | [10.1](#spec-domain_records) |
| PmSoftBodyCapacity | [7.1](#spec-simulation_create) |
| PmSoftBodyInfo | [10.4](#spec-soft_body_resources) |
| PmSoftBodyMaterialState | [9.4](#spec-domain_materials) |
| PmSoftBodyNodeState | [10.4](#spec-soft_body_resources) |
| PmSoftBodySurfaceBinding | [10.1](#spec-domain_records) |
| PmStageFlags | [18.1](#spec-base_types) |
| PmStepInfo | [11.5](#spec-step) |
| PmStepStatistics | [15.3](#spec-step_stats) |
| PmStructureType | [18.2](#spec-structure_types) |
| PmSubmitInfo | [13.4](#spec-submission) |
| PmThermalCouplingState | [9.3](#spec-coupling) |
| pmUnmapMemory | [5.4](#spec-mapping) |
| PmVulkanBufferImportInfo | [14.7](#spec-vulkan_buffer) |
| PmVulkanDeviceImportInfo | [14.6](#spec-vulkan_device) |
| PmVulkanDeviceInfo | [14.6](#spec-vulkan_device) |
| PmVulkanQueueInfo | [14.6](#spec-vulkan_device) |
| pmWaitForFences | [13.5](#spec-fences) |
| pmWaitSemaphore | [13.3](#spec-semaphores) |
