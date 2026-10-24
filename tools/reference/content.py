"""Consumer-facing reference and specification content."""
from dataclasses import dataclass, field
from textwrap import dedent
from pathlib import Path
import re
import xml.etree.ElementTree as ET


@dataclass
class Card:
    key: str
    title: str
    text: str = ""
    types: str = ""
    functions: str = ""
    code: str = ""  # explanatory code, never emitted as an API declaration
    rows: list = field(default_factory=list)
    links: tuple = ()
    tone: str = "cool"
    extension: bool = False
    emit_header: bool = True


@dataclass
class Section:
    key: str
    title: str
    subtitle: str
    cards: list
    diagram: str = ""


def c(key, title, text="", *, types="", functions="", code="", rows=(), links=(), tone="cool", extension=False, emit_header=True):
    return Card(key, title, text, dedent(types).strip(), dedent(functions).strip(),
                dedent(code).strip(), list(rows), tuple(links), tone, extension, emit_header)


PRELUDE = ""  # Populated from registry/pm.xml by apply_registry().

SECTIONS = [
    Section("overview", "Parallel Mater", "API 0.2", [
        c("intent", "Overview", "Parallel Mater is a physics API offering functions and structures that allow you to define physics simulations. Parallel Mater supports Rigid Bodies, Fluid, Cloth, Rope, Smoke, and Soft Bodies. Parallel Mater uses a configured fixed-function pipeline to simulate the specified physics simulations and their specified couplings."),
        c("legend", "Color key", "Function names: pmCreateInstance<br/>Structure and type names: PmInstanceCreateInfo<br/>Constants and return codes: PM_SUCCESS<br/>PFN_pmName: pointer type for pmName<br/>[n.n] Specification section"),
        c("conventions", "Calling conventions", "pmCreate*/pmDestroy* own objects; pmGet* queries; pmCmd* records work. Counts precede arrays. Input pointers are const. A null array requests an enumeration count; a short array returns PM_INCOMPLETE and the number written. Unless stated otherwise, input data is copied before a call returns."),
        c("validity", "Validity & availability", "Set sType to its matching PM_STRUCTURE_TYPE_* value. pNext is NULL unless a section explicitly permits a chained structure; unknown chains and reserved flags are errors. Initialize required nonzero fields. Null handles are allowed only where specified. Physics domains and couplings require their corresponding features or extensions. This reference specifies the behavior required of conforming implementations."),
        c("units", "Units & lifetime", "Right-handed; +Y up; meters, seconds, kilograms and radians. Particle fields named temperatureCelsius and the API 0.2 coupling fields heaterTemperature and boilingTemperature use degrees Celsius; temperature differences have the same magnitude in kelvin. Future absolute thermodynamic fields must state their scale. Quaternions: normalized x,y,z,w, local-to-world. No implicit coordinate conversion. Device children share one device. Destroy in reverse dependency order. Destruction may wait for pending device work; explicit completion before destruction avoids this implicit wait. Command buffers borrow resources. Externally synchronize host access to queues, pools, and mutable objects."),
    ]),
    Section("return_codes", "Return codes", "Results and failures", [
        c("results", "Success & error codes", types="""
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
        """, text="Nonnegative codes report success or status; negative codes report failure. Creation clears the output handle before attempting work. Failed creation owns nothing. A failed recording call latches an error; End fails and the buffer must be reset. Runtime capacity overflow is reported in the step statistics; it is not silently successful physics."),
    ]),
    Section("instancing", "Function Pointers and Instancing", "Resolve entry points and create an instance", [
        c("function_pointers", "Function pointers", types="""
            typedef void (PMAPI_CALL *PFN_pmVoidFunction)(
                void);
        """, functions="""
            PFN_pmVoidFunction pmGetInstanceProcAddr(
                PmInstance instance, const char* pName);
            PFN_pmVoidFunction pmGetDeviceProcAddr(
                PmDevice device, const char* pName);
        """, text="Every command has a matching PFN_pmCommandName typedef in the header. Cast a returned pointer to that command's exact type before calling it; never call through PFN_pmVoidFunction. pName is an exact, case-sensitive, null-terminated command name. Unknown or unavailable names return NULL.<br/><br/>With instance = NULL, pmGetInstanceProcAddr resolves only itself, pmEnumerateInstanceVersion, and pmCreateInstance. With a live instance it also resolves core instance, physical-device and device commands. pmGetDeviceProcAddr requires a live device and returns only device commands supported by its enabled features and extensions. A non-NULL pointer does not enable a feature.<br/><br/>Obtain pmGetInstanceProcAddr by linking or the platform's library-symbol lookup. Instance-resolved pointers are valid only for that instance and its children; device-resolved pointers only for that device and its children. Keep their owner and the library alive while using them.", links=("instance", "device")),
        c("instance", "Instance creation", types="""
            typedef struct PmInstanceCreateInfo {
                PmStructureType sType;
                const void* pNext;
                uint32_t apiVersion;
                const char* pApplicationName;
                PmBool32 enableValidation;
            } PmInstanceCreateInfo;
        """, functions="""
            PmResult pmEnumerateInstanceVersion(
                uint32_t* pApiVersion);
            PmResult pmCreateInstance(
                const PmInstanceCreateInfo* pInfo,
                const PmAllocationCallbacks* pAllocator,
                PmInstance* pInstance);
            void pmDestroyInstance(PmInstance instance);
        """, text="pmEnumerateInstanceVersion reports the highest supported API version; pApiVersion must be non-NULL. Set apiVersion = PM_API_VERSION_0_2 to use this reference; zero selects 0.2. Other versions are unsupported, including older pre-stable versions. Unsupported versions return PM_ERROR_INCOMPATIBLE_VERSION. Destroy every logical device before its instance. Validation is a host-side diagnostic request, not permission to omit required synchronization.", links=("allocator", "discovery")),
        c("validation_diagnostics", "Structured validation diagnostics", text="Chain exactly one PmValidationDiagnosticsCreateInfo through PmInstanceCreateInfo::pNext to enable validation and receive diagnostics. pfnDiagnostic must be non-NULL; the loader copies the function and user-data pointers before pmCreateInstance returns. Keep the callback target and user data valid until pmDestroyInstance returns.<br/><br/>Rule-ID strings and enum values are stable and only extended. object is the application-visible dispatch object named by objectType. pCommandName, pArguments, and pMessage remain valid only for the callback duration; cast pArguments to the matching registry-generated PmLayerArgs_pmCommand after checking argumentSize. Callbacks may arrive concurrently and must externally synchronize their own state. A callback may make independent Parallel Mater calls, but must not destroy or mutate the originating object while its call is active.<br/><br/>Delivery performs no allocation and occurs outside validation bookkeeping locks. If validation bookkeeping exhausts host memory, PM-VAL-0004 is delivered once for the triggering call, tracked state is discarded, and stateful validation is disabled while stateless argument checks and driver execution continue. Diagnostics never replace the API result. Applications must not throw or long-jump through the callback.", links=("instance", "validity")),
        c("allocator", "Host allocation callbacks", types="""
            typedef void* (PMAPI_CALL *PmAllocateFn)(
                void* user, size_t size, size_t alignment);
            typedef void (PMAPI_CALL *PmFreeFn)(
                void* user, void* allocation);
            typedef struct PmAllocationCallbacks {
                void* pUserData;
                PmAllocateFn pfnAllocate;
                PmFreeFn pfnFree;
            } PmAllocationCallbacks;
        """, text="Callbacks allocate CPU bookkeeping only, with power-of-two alignment. Both functions are required if callbacks are supplied. NULL selects the library host allocator. The instance copies callbacks; user data must survive the instance and all children. This does not allocate device physics storage.", links=("memory",)),
    ]),
    Section("discovery", "Devices & queues", "Discover support before creating resources", [
        c("physical_devices", "Physical devices", functions="""
            PmResult pmEnumeratePhysicalDevices(
                PmInstance instance, uint32_t* pCount,
                PmPhysicalDevice* pDevices);
        """, text="Physical-device handles are borrowed from the instance. Pass pDevices = NULL to query the count, then provide storage and enumerate. If the list grows, PM_INCOMPLETE returns the number written; query the count again before retrying."),
        c("properties", "Physical-device identity", types="""
            typedef struct PmPhysicalDeviceProperties {
                uint32_t apiVersion;
                uint32_t driverVersion;
                uint32_t vendorId;
                uint32_t deviceId;
                uint8_t deviceUuid[16];
                char deviceName[256];
            } PmPhysicalDeviceProperties;
        """, functions="""
            void pmGetPhysicalDeviceProperties(
                PmPhysicalDevice physicalDevice,
                PmPhysicalDeviceProperties* pProperties);
        """, text="Use the UUID to match a renderer's adapter. Names and enumeration indices are not stable identities. driverVersion is implementation-defined; do not decode it as the API version.", links=("capabilities", "interop")),
        c("queues", "Queue families", types="""
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
        """, functions="""
            PmResult pmGetPhysicalDeviceQueueFamilies(
                PmPhysicalDevice physicalDevice,
                uint32_t* pCount,
                PmQueueFamilyProperties* pProperties);
        """, text="PHYSICS queues also support transfer. A zero timestampValidBits means timestamps are unavailable on that family. Queues execute asynchronously; submission order alone is not a memory dependency.", links=("sync", "queries")),
        c("features", "Supported vs enabled features", types="""
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
        """, functions="""
            void pmGetPhysicalDeviceFeatures(
                PmPhysicalDevice physicalDevice,
                PmPhysicalDeviceFeatures* pFeatures);
        """, text="Supported features are not automatically enabled. Request only needed features at device creation. Constraints require rigidBodies; each coupling requires its participating domains. Timeline semaphores are required for a conforming command-API device; other features may be false."),
        c("execution_profiles", "Execution profile negotiation", types="""
            #define PM_EXT_SIMULATION_PROFILE_EXTENSION_NAME "PM_EXT_simulation_profile"
            #define PM_EXT_SIMULATION_PROFILE_SPEC_VERSION 1u
            #define PM_EXECUTION_PROFILE_NAME_MAX_SIZE 64u
            #define PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP "realtime-single-step"
            typedef struct PmExecutionProfileProperties {
                char profileName[PM_EXECUTION_PROFILE_NAME_MAX_SIZE];
                uint32_t versionMajor;
                uint32_t versionMinor;
            } PmExecutionProfileProperties;
            typedef struct PmSimulationProfileCreateInfo {
                PmStructureType sType;
                const void* pNext;
                const char* pProfileName;
                uint32_t versionMajor;
                uint32_t versionMinor;
            } PmSimulationProfileCreateInfo;
        """, functions="""
            PmResult pmEnumerateExecutionProfiles(
                PmPhysicalDevice physicalDevice, uint32_t* pCount,
                PmExecutionProfileProperties* pProperties);
        """, text="Lists executable name/version pairs, not qualification. Enable PM_EXT_simulation_profile and chain PmSimulationProfileCreateInfo at simulation creation; omission keeps the legacy schedule. Unknown profiles and versions fail without fallback. realtime-single-step 1.0 requires one substep and 1/60 second within one binary32 ULP, enforced without validation."),
        c("device", "Logical device", types="""
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
        """, functions="""
            PmResult pmCreateDevice(
                PmPhysicalDevice physicalDevice,
                const PmDeviceCreateInfo* pInfo,
                PmDevice* pDevice);
            void pmDestroyDevice(PmDevice device);
            void pmGetDeviceQueue(PmDevice device,
                uint32_t family, uint32_t index,
                PmQueue* pQueue);
        """, text="Request each family at most once; priorities are finite in [0,1]. Unsupported features fail creation. Queues are borrowed, not destroyed separately. Wait for work and destroy children before destroying the device."),
    ], "ownership"),
    Section("memory", "Explicit device memory", "Create → requirements → allocate → bind → use", [
        c("memory_types", "Heaps & types", types="""
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
        """, functions="""
            void pmGetPhysicalDeviceMemoryProperties(
                PmPhysicalDevice physicalDevice,
                PmMemoryProperties* pProperties);
        """, text="Heap flags are reserved zero. DEVICE_LOCAL describes location, not mappability. A type can be both device-local and host-visible. Heap size is not currently available budget.", links=("memory_budget",)),
        c("requirements", "Requirements & suballocation", types="""
            typedef struct PmMemoryRequirements {
                PmDeviceSize size;
                PmDeviceSize alignment;
                uint32_t memoryTypeBits;
                PmBool32 requiresDedicatedAllocation;
            } PmMemoryRequirements;
        """, text="Choose a type t for which memoryTypeBits &amp; (1u &lt;&lt; t) is nonzero. Bind at an offset divisible by alignment. The allocation must contain the full required range. Dedicated resources occupy a whole allocation at offset zero.<br/><br/>Suballocate distinct, non-overlapping ranges from larger blocks. Live bindings must not overlap, including scratch. Free a block only after destroying all bound resources and observing GPU completion. Requirements include resource-internal GPU data; bindings are immutable until destruction.", links=("buffer_binding", "simulation_memory")),
        c("allocation", "Allocate & free", types="""
            typedef struct PmMemoryAllocateInfo {
                PmStructureType sType;
                const void* pNext;
                PmDeviceSize allocationSize;
                uint32_t memoryTypeIndex;
            } PmMemoryAllocateInfo;
        """, functions="""
            PmResult pmAllocateMemory(PmDevice device,
                const PmMemoryAllocateInfo* pInfo,
                PmDeviceMemory* pMemory);
            void pmFreeMemory(PmDevice device,
                PmDeviceMemory memory);
        """, text="allocationSize is nonzero. Allocation failure is explicit; no fallback silently changes the memory type. Pipeline code storage and driver bookkeeping are implementation-owned and reported separately; simulation, geometry acceleration and buffer storage are application-bound.", tone="warm"),
        c("mapping", "Map & host visibility", types="""
            typedef struct PmMappedMemoryRange {
                PmDeviceMemory memory;
                PmDeviceSize offset;
                PmDeviceSize size;
            } PmMappedMemoryRange;
        """, functions="""
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
        """, text="Map HOST_VISIBLE memory once per allocation; the returned pointer addresses offset. PM_WHOLE_SIZE extends to allocation end. Flush host writes before GPU consumption. Wait for GPU writes before invalidating and reading. Noncoherent ranges start on atomSize boundaries and end on a boundary or allocation end; they must lie in the mapped range. Unmap does not flush. COHERENT removes cache-maintenance calls, not race-prevention duties.", links=("limits", "barriers")),
    ], "allocation"),
    Section("resources", "Buffers & triangle geometry", "One rigid geometry model • explicit storage and build work", [
        c("buffer", "Buffer creation", types="""
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
        """, functions="""
            PmResult pmCreateBuffer(PmDevice device,
                const PmBufferCreateInfo* pInfo,
                PmBuffer* pBuffer);
            void pmDestroyBuffer(PmDevice device,
                PmBuffer buffer);
        """, text="Creation declares a resource; it does not allocate its storage. All buffers have exclusive queue-family ownership. size is nonzero. Declare every usage required for the resource's lifetime."),
        c("buffer_binding", "Buffer requirements & binding", functions="""
            void pmGetBufferMemoryRequirements(
                PmDevice device, PmBuffer buffer,
                PmMemoryRequirements* pRequirements);
            PmResult pmBindBufferMemory(PmDevice device,
                PmBuffer buffer, PmDeviceMemory memory,
                PmDeviceSize offset);
        """, types="""
            typedef struct PmBufferRange {
                PmBuffer buffer;
                PmDeviceSize offset;
                PmDeviceSize size;
            } PmBufferRange;
        """, text="Bind exactly once before use. Buffer ranges are bounds-checked, nonempty, and measured in bytes; PM_WHOLE_SIZE is allowed only for APIs that explicitly say so. Imported Vulkan buffers are already bound and must not be rebound.", links=("requirements", "interop")),
        c("format_enum", "Element formats", types="""
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
        """, text="A format defines byte interpretation, not a collider or solver type. stride is in bytes, at least the element size and a multiple of its alignment. count is nonzero; the final element must fit in range.size. Query usage support before choosing a format.", links=("formats",)),
        c("geometry", "Triangle geometry", types="""
            typedef struct PmGeometryCreateInfo {
                PmStructureType sType;
                const void* pNext;
                PmBufferView vertices;
                PmBufferView indices;
                uint32_t queueFamilyIndex;
            } PmGeometryCreateInfo;
        """, functions="""
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
        """, text="Vertices use R32G32B32_SFLOAT; indices use R32_UINT and a count divisible by three. Indices must be in bounds; coordinates finite; triangles nondegenerate. Dynamic bodies require closed, convex, consistently wound meshes. Nonconvex static geometry is outside the core specification. Spheres and boxes are authored as triangle meshes.", links=("bodies",)),
        c("geometry_build", "Recorded geometry build", functions="""
            PmResult pmCmdBuildGeometry(
                PmCommandBuffer commandBuffer,
                PmGeometry geometry);
        """, text="Build after uploading and making input writes visible. Build creates acceleration data in bound storage and includes its internal dependencies. Barrier GEOMETRY_BUILD/WRITE → PHYSICS/READ before the first solve. Rebuild only when no dependent simulation is using the geometry. Inputs and their bindings remain alive for the entire geometry lifetime, not just the build. One geometry can be instanced by many bodies.<br/><br/>The application keeps the geometry object, its bound memory, and its input buffers alive while a recorded or pending command refers to it and while any live simulation body or paint field uses it. An executed body replacement or removal retires that body's reference; other bodies that instance the same geometry remain valid. Simulation reset retires all of that simulation's geometry references. Destroy geometry and its dependencies only after those commands complete and the references are retired.", links=("barriers", "pipeline_flow")),
        c("resource_recipe", "Resource creation checklist", rows=[("1 · Describe", "Size, usage, format, capacity, owner family"), ("2 · Query", "Exact size, alignment and type bits"), ("3 · Allocate", "Choose a compatible memory type"), ("4 · Bind", "Reserve a non-overlapping aligned range"), ("5 · Initialize", "Record upload / reset / build"), ("6 · Depend", "Make producer writes visible to users"), ("7 · Retire", "Complete GPU work, destroy, then free")], tone="warm"),
    ]),
]

SECTIONS += [
    Section("simulation", "Simulation state & rigid bodies", "Mutable state lives separately from immutable solver configuration", [
        c("simulation_create", "Capacity-based simulation", types="""
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
        """, functions="""
            PmResult pmCreateSimulation(PmDevice device,
                const PmSimulationCreateInfo* pInfo,
                PmSimulation* pSimulation);
            void pmDestroySimulation(PmDevice device,
                PmSimulation simulation);
        """, text="At least one body slot or fluid is required. Zero capacity disables a domain. Body and constraint IDs are stable, zero-based slots, not pointers. Each fluid is a separate particle state range with its own material. No hidden growth is permitted; change capacity by creating and migrating to a new simulation."),
        c("simulation_memory", "Persistent & scratch bindings", types="""
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
        """, functions="""
            void pmGetSimulationMemoryRequirements(
                PmDevice device, PmSimulation simulation,
                PmSimulationMemoryRequirements* pInfo);
            PmResult pmBindSimulationMemory(
                PmDevice device, PmSimulation simulation,
                const PmSimulationMemoryBinding* pBinding);
        """, text="Requirements depend on declared capacities and include bounded contact/neighbor workspaces. Bind before reset or execution. Persistent and scratch ranges cannot overlap. Neither is a renderer-readable ABI. Use explicit output commands for portable state. Simulation state is exclusive to its creation queue family in core 0.2; share through output buffers.", links=("allocation", "outputs")),
        c("rigid_state", "Portable rigid state record", types="""
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
        """, text="Quaternion identity is (0,0,0,1). Angular velocity is radians/second in world space. Static bodies do not integrate; kinematic states are application-driven and do not respond to forces.", links=("formats",)),
        c("bodies", "Body descriptions", types="""
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
        """, functions="""
            PmResult pmCmdWriteRigidBodies(
                PmCommandBuffer commandBuffer,
                PmSimulation simulation,
                uint32_t firstBody, uint32_t count,
                const PmRigidBodyInfo* pBodies);
        """, text="Host descriptions are copied during recording, including state; handles remain borrowed. Writing replaces each addressed slot. Dynamic bodies require positive mass and principal inertia about their center of mass, in kg·m². No implicit inertia approximation. Restitution is [0,1]; friction, damping and margin are nonnegative. Inactive entries require only active = PM_FALSE; all other fields are ignored. Remove or disable referencing constraints first."),
        c("body_inputs", "Forces, impulses & targets", types="""
            typedef struct PmRigidInput {
                uint32_t bodyIndex;
                float force[3];
                float torque[3];
                float linearImpulse[3];
                float angularImpulse[3];
            } PmRigidInput;
        """, functions="""
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
        """, text="Inputs accumulate for the next step, then clear. Forces apply across that step's substeps; impulses apply once, not once per substep. Units: N, N·m, N·s, N·m·s. Dynamic state edits and inputs wake bodies. SetRigidStates replaces pose and velocity without changing material or geometry. SetKinematicTargets drives kinematic bodies through the next step."),
        c("hit_boxes", "Hit-box queries", types="""
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
        """, functions="""
            PmResult pmCmdQueryHitBox(
                PmCommandBuffer commandBuffer,
                PmSimulation simulation,
                const PmHitBox* pBox,
                const PmBufferRange* pDestination);
        """, text="Output begins with PmHitBoxOutputHeader, followed by rigidBodyCount uint32_t body slots and particleCount PmHitBoxParticle records. Results are sorted by slot and stable particle ID."),
    ]),
    Section("constraints", "Rigid constraints", "Contacts and joints enter the same AVBD solve", [
        c("constraint_types", "Joint families", types="""
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
        """, text="Frames are local to their body centers of mass. If bodyB is PM_INVALID_INDEX, frameB is in world space. Hinge, slider and piston axes are local +X of frameA. Both quaternions are normalized. A body cannot constrain itself.", links=("rigid_solver",)),
        c("axis_control", "Per-axis controls", types="""
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
        """, text="Linear limits/equilibria are meters; angular ones radians. lower ≤ upper; equality locks an axis. Disabled limits leave eligible axes free. Springs use nonnegative N/m and N·s/m, or rotational N·m/rad and N·m·s/rad. Motors use m/s and N, or rad/s and N·m. These are force/torque limits, not frame-dependent impulse limits."),
        c("constraint_info", "Constraint description", types="""
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
        """, functions="""
            PmResult pmCmdWriteRigidConstraints(
                PmCommandBuffer commandBuffer,
                PmSimulation simulation,
                uint32_t firstConstraint, uint32_t count,
                const PmRigidConstraintInfo* pConstraints);
        """, text="Writing replaces stable slots. Inactive slots ignore other fields. Each active endpoint must refer to an active body or the world sentinel, and at least one endpoint must be dynamic. collideConnected = false suppresses that body's pair contact while the joint is active. No per-joint private solver or iteration loop exists."),
        c("constraint_dofs", "Degrees of freedom", rows=[("FIXED", "All translation and rotation locked"), ("POINT", "Three translations locked; rotation free"), ("HINGE", "Only local X rotation eligible"), ("SLIDER", "Only local X translation eligible"), ("PISTON", "Local X translation and rotation eligible"), ("GENERIC", "All six axes eligible for limits/motors"), ("GENERIC_SPRING", "All six axes; limits, springs and motors"), ("MOTOR", "All axes free except driven/limited axes")], text="Controls on intrinsically locked axes must be disabled. Springs are allowed only for GENERIC_SPRING; limit/motor controls apply on eligible axes for all other families. Incompatible combinations fail validation instead of being silently ignored.", tone="warm"),
        c("constraint_example", "Point joint to the world", code="""
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
        """, text="A body-local anchor one meter above the body's origin is held at a world anchor three meters high. The body can rotate freely. Its geometry, body slot and simulation storage must already exist. Submit the write before the next solve with the required update-to-physics dependency.", links=("barriers",)),
        c("single_solve", "Iteration accounting", text="Pipeline rigidIterations is the scene-wide AVBD budget. When no explicit runtime override is recorded, the greatest solverIterations value authored by any enabled constraint raises that single budget for the solve. The pose phase runs for each substep; impacted bodies can also run a velocity-impact phase. Statistics report the executed budget. One iteration is valid, but does not guarantee convergence.", links=("step_stats", "rigid_solver")),
    ]),
    Section("pipelines", "Physics pipelines & materials", "Immutable configuration • explicit inter-domain coupling", [
        c("rigid_solver", "Rigid solver state", types="""
            typedef struct PmRigidSolverState {
                uint32_t rigidIterations;
                PmBool32 enableSleeping;
                float sleepLinearThreshold;
                float sleepAngularThreshold;
                float sleepTimeSeconds;
            } PmRigidSolverState;
        """, text="One AVBD path solves rigid geometry contacts and joints. rigidIterations is the nonzero base body budget and is ≤ maxRigidIterations; constraint solverIterations can raise only their endpoint budgets. Sleeping thresholds are nonnegative m/s and rad/s; sleep time is positive when enabled. The current reference drivers require enableSleeping = PM_FALSE and reject enabled sleeping with FEATURE_NOT_PRESENT; omitted rigid state defaults to sleeping disabled."),
        c("liquid_material", "Particle-liquid material", types="""
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
        """, text="Density: kg/m³; radii: m; rest volume: m³; pressure stiffness: Pa; kinematic viscosity: m²/s; damping: s⁻¹; speed: m/s. Density, radii, volume, maximumSpeed and iterations are positive; supportRadius ≥ 2 × particleRadius. Pressure stiffness, viscosity and damping are nonnegative. Material index equals the simulation's fluid index.<br/><br/>Materials configure a particle-liquid domain. Smoke, gas grids and non-Newtonian rheology are outside this material model.", tone="warm"),
        c("coupling", "Coupling state", types="""
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
                PM_COUPLING_SMOKE_SOFT_BODY = 12,
                PM_COUPLING_SMOKE_SMOKE = 13,
                PM_COUPLING_SOFT_BODY_SOFT_BODY = 14,
                PM_COUPLING_CLOTH_CLOTH = 15,
                PM_COUPLING_ROPE_ROPE = 16
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
                uint32_t first, second;
                PmInteractionModeFlags modeFlags;
                PmInteractionMaterialProfile materialProfile;
                PmCouplingParameters parameters;
            } PmCouplingState;
            typedef struct PmCouplingStateInfo {
                PmStructureType sType;
                const void* pNext;
                PmCouplingType type;
                uint32_t first, second;
                PmInteractionModeFlags modeFlags;
                PmInteractionMaterialProfile materialProfile;
                PmCouplingParameters parameters;
            } PmCouplingStateInfo;
            typedef struct PmFluidPairMaterialState {
                PmStructureType sType;
                const void* pNext;
                uint32_t firstMaterialIndex;
                uint32_t secondMaterialIndex;
            } PmFluidPairMaterialState;
            typedef struct PmFluidInterfaceState {
                PmStructureType sType;
                const void* pNext;
                float surfaceTension;
            } PmFluidInterfaceState;
            typedef enum PmRopeEndpoint {
                PM_ROPE_ENDPOINT_FIRST = 0,
                PM_ROPE_ENDPOINT_LAST = 1
            } PmRopeEndpoint;
            typedef struct PmRopeAttachmentState {
                PmStructureType sType;
                const void* pNext;
                PmRopeEndpoint firstEndpoint;
                PmRopeEndpoint secondEndpoint;
                float compliance;
                float damping;
                float captureDistance;
                float breakingImpulseThreshold;
            } PmRopeAttachmentState;
        """, functions="""
            PmResult pmCmdSetCouplings(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t couplingCount,
                const PmCouplingState* pCouplings);
            PmResult pmCmdSetCouplings2(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t couplingCount,
                const PmCouplingStateInfo* pCouplings);
        """, text="first and second are stable domain slots; PM_INVALID_INDEX names all rigid bodies. SMOKE_SMOKE requires first &lt; second: SMOKE_PARTICLE/GRID select PRESSURE|VISCOSITY|THERMAL; SMOKE_PARTICLE_GRID selects DRAG|THERMAL. ROPE_ROPE also requires first &lt; second and the ROPE_CAPSULE profile; CONTACT and ATTACHMENT are independently selectable and remain unadvertised. Other bundles are fixed; invalid bits are errors.<br/><br/>PmCouplingState is the API 0.2 descriptor. PmCouplingStateInfo is an ABI-compatible extensible path: sType identifies that exact structure and pNext is NULL except for separately permitted typed state. A LIQUID_LIQUID edge using MISCIBLE or IMMISCIBLE_INTERFACE must chain exactly one PmFluidPairMaterialState. Its material indices must equal first and second, so the edge consumes the two endpoint PmLiquidMaterialState records without copying, averaging or silently changing ownership. MISCIBLE requests contact, pressure and viscosity as one mechanical bundle. IMMISCIBLE_INTERFACE also chains exactly one PmFluidInterfaceState and requests contact, pressure and interface. surfaceTension is finite and nonnegative in N/m; zero disables capillary traction while retaining the material boundary. The interface normal points from the first fluid endpoint to the second, so positive curvature obeys pressure_first - pressure_second = surfaceTension * (1 / R1 + 1 / R2). Interfacial viscosity remains a separate capability. Both different-material profiles remain unsupported until their operator and qualification leaves complete. A well-formed request therefore returns FEATURE_NOT_PRESENT, while missing, mismatched, duplicate, unknown, profile-inapplicable or nonfinite typed state returns INVALID_ARGUMENT and cannot replace a live graph.<br/><br/>A ROPE_ROPE edge requesting ATTACHMENT must use PmCouplingStateInfo and chain exactly one PmRopeAttachmentState. firstEndpoint and secondEndpoint select FIRST or LAST on their corresponding canonical rope slots. compliance in m/N, damping in N·s/m, captureDistance in m and breakingImpulseThreshold in N·s are finite and nonnegative. Zero compliance is rigid and zero breaking impulse is unbreakable. The typed state is invalid without ATTACHMENT, and ATTACHMENT without it is invalid. Contact-only rope-capsule edges have no typed state. Every well-formed rope-pair request returns FEATURE_NOT_PRESENT until later operator and qualification tasks advertise its exact mode; no rope-capsule capability is implied by the legacy node-only self-contact record.<br/><br/>Each coupling type reads its applicable parameter groups. Duplicate endpoint pairs are invalid. pmCmdSetCouplings and pmCmdSetCouplings2 copy a complete simulation-owned edge list; count must fit couplingCapacity and count zero removes explicit edges. The override persists across pipeline binds and endpoint replacement until reset. Without an override, the bound pipeline supplies the edge list. Endpoint slots are resolved at execution; retired generations are never retained. The complete replacement is validated before live edges are removed. If installation then fails, the previous usable graph is restored. Intrinsic rigid/liquid contact remains part of the reference solver even with an empty explicit graph.<br/><br/>Coupling units are: contact distance and heater geometry in m; stiffness in N/m; damping and tangential drag in N·s/m; maximum force in N; acceleration limits in m/s²; surface tension in N/m; wind drag and heat-transfer rate in s⁻¹; air density in kg/m³; effective mass in kg; steam rise speed in m/s; heater and boiling temperatures in °C. Friction, drag coefficient and radius scales are dimensionless.<br/><br/>The reference mechanical contact pairs currently require twoWay = PM_TRUE; one-way requests return FEATURE_NOT_PRESENT. API 0.2 directions are fixed by coupling type and checked through the exact capability records. Liquid/cloth derives interaction radius from liquid supportRadius; maximumAcceleration must be zero for this pair. Collision, attachment, drag, containment, pressure, viscosity, interface, thermal and phase behavior are independent capabilities; a driver must not infer one from another or from the broad domainCouplings bit."),
        c("domain_materials", "Deformable, rope & smoke materials", types="""
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
        """),
        c("pipeline_create", "Pipeline creation", types="""
            typedef struct PmPhysicsPipelineCouplingStateInfo {
                PmStructureType sType;
                const void* pNext;
                uint32_t couplingCount;
                const PmCouplingStateInfo* pCouplings;
            } PmPhysicsPipelineCouplingStateInfo;
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
        """, functions="""
            PmResult pmCreatePhysicsPipeline(
                PmDevice device,
                const PmPhysicsPipelineCreateInfo* pInfo,
                PmPhysicsPipeline* pPipeline);
            void pmDestroyPhysicsPipeline(
                PmDevice device,
                PmPhysicsPipeline pipeline);
        """, text="Creation copies all settings. pRigidState may be NULL only when no rigid domain is needed. The material count must equal a bound simulation's fluid count. Pipelines are immutable and shareable across compatible simulations. Changing materials creates a new pipeline, without reallocating state if its capacity requirements remain satisfied.<br/><br/>Chain at most one PmPhysicsPipelineCouplingStateInfo to supply extensible coupling descriptors. Its own pNext is NULL. When present, the base couplingCount is zero and pCouplings is NULL; the chained count and array provide the complete pipeline edge list. Descriptor validation and deep-copy rules match pmCmdSetCouplings2. Unknown, nested or conflicting chains return INVALID_ARGUMENT."),
        c("pipeline_flow", "What a step records", text="pmCmdStepSimulation records a complete operation, not one CPU call per internal kernel. Its implementation owns dependencies among integration, broad phase, contact/neighbor generation, solve/coupling, velocity reconstruction and sleep. Applications supply dependencies between uploads, steps, exports, and other APIs.<br/><br/>A substep is a slice of physical time. An iteration refines constraints within that slice. Neither is a renderer frame. The API exposes the work budget explicitly and the queries report work actually executed.", links=("step", "step_stats")),
        c("pipeline_validity", "Compatibility & bounds", text="A rigid simulation requires pRigidState and enabled rigidBodies. Live constraints additionally require rigidConstraints. Each active fluid requires particleLiquids. Each coupling requires its matching enabled feature. A disabled domain cannot allocate hidden capacity.<br/><br/>All numerical inputs must be finite. Neighbors, contacts and output ranges are bounded. If runtime generation exceeds capacity, the step reports CAPACITY_EXCEEDED and its output is invalid; the application must reset/reinitialize before reuse. There is no silent truncation advertised as correct physics.", links=("simulation_create", "step_stats"), tone="warm"),
    ], "pipeline"),
]

SECTIONS += [
    Section("domains", "Simulation domains & events", "Recorded creation, updates, output and diagnostics", [
        c("domain_records", "Shared records", types="""
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
        """),
        c("cloth_resources", "Cloth resources", types="""
            typedef struct PmClothInfo {
                PmBool32 active;
                uint32_t materialIndex;
                uint32_t vertexCount;
                const PmFloat3* pVertices;
                uint32_t triangleIndexCount;
                const uint32_t* pTriangleIndices;
                const float* pInverseMasses;
            } PmClothInfo;
            typedef struct PmClothStateInfo {
                PmStructureType sType;
                const void* pNext;
                PmBool32 active;
                uint32_t materialIndex;
                uint32_t vertexCount;
                const PmFloat3* pVertices;
                uint32_t triangleIndexCount;
                const uint32_t* pTriangleIndices;
                const float* pInverseMasses;
            } PmClothStateInfo;
            typedef struct PmClothSelfContactState {
                PmStructureType sType;
                const void* pNext;
                PmBool32 enabled;
            } PmClothSelfContactState;
            typedef struct PmClothVertexState {
                PmFloat3 position;
                PmFloat3 velocity;
                PmFloat3 rigidContactForce;
                PmFloat3 liquidContactForce;
                PmFloat3 softBodyContactForce;
                PmFloat3 ropeContactForce;
            } PmClothVertexState;
        """, functions="""
            PmResult pmCmdWriteCloths(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t firstCloth,
                uint32_t count, const PmClothInfo* pCloths);
            PmResult pmCmdWriteCloths2(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t firstCloth,
                uint32_t count, const PmClothStateInfo* pCloths);
            PmResult pmCmdCopyClothState(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t clothIndex,
                const PmBufferRange* pDestination);
            PmResult pmCmdCopyClothSurface(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t clothIndex,
                const PmBufferRange* pDestination);
        """, text="PmClothInfo is the API 0.2 descriptor. PmClothStateInfo is the ABI-compatible extensible write path and copies the same resource data. Its pNext is NULL unless intrinsic surface self-contact is requested by chaining exactly one PmClothSelfContactState. enabled is PM_FALSE or PM_TRUE; PM_FALSE preserves ordinary cloth behavior and PM_TRUE requests the exact (cloth, cloth, self, contact, bidirectional, endpoint-native) capability. Unknown or nested state is invalid. Enabled self-contact requires finite vertices and nonnegative inverse masses plus distinct, in-range, nondegenerate, nonduplicated triangles. Invalid topology returns INVALID_ARGUMENT before any write is recorded. A valid enabled request returns FEATURE_NOT_PRESENT until its operator and driver qualification tasks advertise the capability, without replacing live state.<br/><br/>A separate-system contact edge uses PM_COUPLING_CLOTH_CLOTH through PmCouplingStateInfo with first &lt; second, CONTACT and ENDPOINT_NATIVE. It consumes simulation coupling capacity and requests the exact separate-system capability. Duplicate or reversed endpoints are invalid; a well-formed edge remains FEATURE_NOT_PRESENT until advertised. Intrinsic self-contact consumes no graph edge, and neither request implies attachment, containment or one-sided collision."),
        c("rope_resources", "Rope resources", types="""
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
        """, functions="""
            PmResult pmCmdWriteRopes(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t firstRope,
                uint32_t count, const PmRopeInfo* pRopes);
            PmResult pmCmdCopyRopeState(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t ropeIndex,
                const PmBufferRange* pDestination);
        """),
        c("soft_body_resources", "Soft-body resources", types="""
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
            typedef struct PmSoftBodyStateInfo {
                PmStructureType sType;
                const void* pNext;
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
            } PmSoftBodyStateInfo;
            typedef struct PmSoftBodySelfContactState {
                PmStructureType sType;
                const void* pNext;
                PmBool32 enabled;
            } PmSoftBodySelfContactState;
            typedef struct PmSoftBodyNodeState {
                PmFloat3 position;
                PmFloat3 velocity;
                PmFloat3 rigidContactForce;
                PmFloat3 clothContactForce;
                PmFloat3 liquidContactForce;
                PmFloat3 ropeContactForce;
            } PmSoftBodyNodeState;
        """, functions="""
            PmResult pmCmdWriteSoftBodies(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t firstSoftBody,
                uint32_t count, const PmSoftBodyInfo* pSoftBodies);
            PmResult pmCmdWriteSoftBodies2(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t firstSoftBody,
                uint32_t count, const PmSoftBodyStateInfo* pSoftBodies);
            PmResult pmCmdCopySoftBodyState(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t softBodyIndex,
                const PmBufferRange* pDestination);
            PmResult pmCmdCopySoftBodySurface(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t softBodyIndex,
                const PmBufferRange* pDestination);
        """, text="PmSoftBodyInfo is the API 0.2 descriptor. PmSoftBodyStateInfo is the ABI-compatible extensible write path and copies the same resource data. Its pNext is NULL unless intrinsic surface self-contact is requested by chaining exactly one PmSoftBodySelfContactState. enabled is PM_FALSE or PM_TRUE; PM_FALSE preserves ordinary soft-body behavior and PM_TRUE requests the exact (soft body, soft body, self, contact, bidirectional, endpoint-native) capability. Unknown or nested state is invalid. The self-contact request requires a finite, consistently wound, closed two-manifold surface with distinct, nondegenerate triangle vertices and valid normalized bindings. Invalid topology returns INVALID_ARGUMENT before any write is recorded. A valid enabled request returns FEATURE_NOT_PRESENT until its operator and driver qualification tasks advertise the capability, without replacing live state.<br/><br/>A separate-system surface-contact edge uses PM_COUPLING_SOFT_BODY_SOFT_BODY through PmCouplingStateInfo with first &lt; second, CONTACT and ENDPOINT_NATIVE. It consumes simulation coupling capacity and requests the exact separate-system capability. Duplicate or reversed endpoints are invalid; a well-formed edge remains FEATURE_NOT_PRESENT until advertised. Intrinsic self-contact consumes no graph edge, and neither request implies attachment, containment or volume inversion prevention."),
        c("smoke_resources", "Smoke resources", types="""
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
        """, functions="""
            PmResult pmCmdWriteSmoke(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t firstSmoke,
                uint32_t count, const PmSmokeInfo* pSmoke);
            PmResult pmCmdCopySmokeState(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t smokeIndex,
                const PmBufferRange* pDestination);
        """),
        c("emitters", "Emitters & destruction", types="""
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
        """, functions="""
            PmResult pmCmdWriteParticleEmitters(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t firstEmitter,
                uint32_t count, const PmParticleEmitterInfo* pEmitters);
            PmResult pmCmdWriteParticleDestroyers(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t firstDestroyer,
                uint32_t count, const PmParticleDestroyerInfo* pDestroyers);
        """, text="Emitter and destroy-plane slots borrow the current generation of fluidIndex. Replacing a fluid retires the endpoints aimed at that fluid slot; record their active definitions again to bind the replacement generation. Replacing or removing one endpoint first retires its old generation. If construction of its replacement fails during execution, that endpoint remains inactive and other fluid populations and endpoints remain live.<br/><br/>Smoke emission is part of PmSmokeInfo rather than a separate endpoint resource. Replacing a smoke slot therefore replaces its inline emitter with the smoke generation."),
        c("paint_resources", "Paint fields & rules", types="""
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
        """, functions="""
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
        """, text="Paint field descriptions and UV values are copied during recording; rigid-target geometry handles remain borrowed. Writing replaces stable field and rule slots, and an inactive entry removes its slot. An active field target must be a live body or cloth at execution. A rigid field additionally requires built geometry whose vertex count matches vertexCount. Width, height and vertexCount must fit the capacities declared when the simulation was created.<br/><br/>A live field pins its target generation, and a live rule pins its source and target-field generations. Disable dependent rules before replacing their target field, and disable fields before replacing their body or cloth. Rejection of an attempted dependency removal preserves that live object. If an old field was removed but later validation or allocation of its replacement fails, that field slot remains inactive; unrelated fields remain valid.<br/><br/>A rigid-target field retains its geometry reference until that field is replaced, removed, or the simulation is reset. Body and other paint references to the same geometry remain valid. Reset retires all paint fields and rules. Keep every borrowed geometry and its dependencies alive through recorded and pending commands and while a live field refers to it.", links=("geometry", "lifecycle", "simulation_init")),
        c("event_outputs", "Constraint, contact & debug output", types="""
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
        """, functions="""
            PmResult pmCmdCopyConstraintStates(PmCommandBuffer commandBuffer,
                PmSimulation simulation, uint32_t firstConstraint,
                uint32_t count, const PmBufferRange* pDestination);
            PmResult pmCmdCopyContactEvents(PmCommandBuffer commandBuffer,
                PmSimulation simulation, const PmBufferRange* pDestination);
            PmResult pmCmdCopyDebugData(PmCommandBuffer commandBuffer,
                PmSimulation simulation, const PmBufferRange* pDestination);
        """),
    ]),
    Section("commands", "Command recording & lifecycle", "Record now; execute when a queue receives the buffer", [
        c("command_pool", "Command pools", types="""
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
        """, functions="""
            PmResult pmCreateCommandPool(PmDevice device,
                const PmCommandPoolCreateInfo* pInfo,
                PmCommandPool* pPool);
            void pmDestroyCommandPool(PmDevice device,
                PmCommandPool pool);
            PmResult pmResetCommandPool(PmDevice device,
                PmCommandPool pool);
        """, text="A pool belongs to one family and is externally synchronized, including all recording operations on its buffers. Use one pool per recording thread. Reset returns every buffer to Initial; no buffer may be Pending. Destroying a pool frees its command buffers, after completion."),
        c("command_buffers", "Allocate, record & reset", types="""
            typedef struct PmCommandBufferBeginInfo {
                PmStructureType sType;
                const void* pNext;
                PmFlags flags;
            } PmCommandBufferBeginInfo;
        """, functions="""
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
        """, text="Primary buffers only. Begin discards prior recording and starts from Initial, Executable or Invalid; Recording and Pending reject Begin. End requires Recording and no latched error or open query. Individual reset requires RESET_BUFFER on the pool. No simultaneous submission of one command buffer. Host input arrays are copied at record time; no device work occurs while recording."),
        c("lifecycle", "Legal states & resource lifetime", rows=[("Initial", "Begin → Recording; may free/reset"), ("Recording", "pmCmd* only; End → Executable"), ("Executable", "QueueSubmit → Pending; may reset/free"), ("Pending", "No edits, reset, free or resubmit"), ("Completion", "Reusable → Executable; one-time → Invalid"), ("Invalid", "Begin → Recording; reset → Initial, or free")], text="Rejected argument/state validation does not enqueue a command and recording may continue. A latched recording allocation failure makes End fail until command-buffer reset. A submit failure before acceptance leaves the command buffer executable and its fence unsignaled, so the submission may be retried. Destruction drains pending work in the reference drivers, but does not rewrite recorded references. After destroying a referenced resource, reset or free every dependent command buffer before reuse. Application code must preserve pipeline, geometry, buffers, query pools and memory until completion. Pipeline bindings do not carry across buffers.", links=("submission", "fences")),
        c("simulation_init", "Reset & particle initialization", types="""
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
        """, functions="""
            PmResult pmCmdResetSimulation(
                PmCommandBuffer commandBuffer,
                PmSimulation simulation);
            PmResult pmCmdSetFluidParticles(
                PmCommandBuffer commandBuffer,
                PmSimulation simulation,
                uint32_t fluidIndex,
                const PmBufferView* pParticles);
        """, text="Reset is required after binding memory. It clears active bodies, joints, particles, loads, sleep history and step status; it preserves capacities. SetFluidParticles replaces one fluid population from PARTICLE_STATE input. count ≤ capacity; count zero clears the population and ignores range/format/stride. Nonzero input requires PARTICLE_INPUT usage. Temperature is carried state; the liquid model does not solve heat transfer."),
        c("step", "Bind & step", types="""
            enum {
                PM_STEP_COLLECT_CONTACT_EVENTS_BIT = 0x1
            };
            typedef struct PmStepInfo {
                float timestep;
                uint32_t substeps;
                float gravity[3];
                PmFlags flags;
            } PmStepInfo;
        """, functions="""
            PmResult pmCmdBindPhysicsPipeline(
                PmCommandBuffer commandBuffer,
                PmPhysicsPipeline pipeline);
            PmResult pmCmdStepSimulation(
                PmCommandBuffer commandBuffer,
                PmSimulation simulation,
                const PmStepInfo* pStep);
        """, text="timestep > 0 seconds; 1 ≤ substeps ≤ maxSubsteps. Each substep advances timestep/substeps. Time advances only by the supplied timestep. PM_STEP_COLLECT_CONTACT_EVENTS_BIT records rigid and liquid contact events for later copy; without it the step avoids event-generation work and the copied event count is zero. A compatible pipeline must be bound. The simulation must be initialized, geometry built, and prior writes visible. Two submissions must not mutate one simulation concurrently.", links=("pipelines", "barriers")),
    ], "lifecycle"),
    Section("transfer", "Data movement & output", "Renderer-visible buffers without exposing private solver layouts", [
        c("buffer_copy", "Buffer transfers", types="""
            typedef struct PmBufferCopy {
                PmDeviceSize srcOffset;
                PmDeviceSize dstOffset;
                PmDeviceSize size;
            } PmBufferCopy;
        """, functions="""
            PmResult pmCmdCopyBuffer(
                PmCommandBuffer commandBuffer,
                PmBuffer src, PmBuffer dst,
                uint32_t count,
                const PmBufferCopy* pRegions);
        """, text="Use TRANSFER_SRC and TRANSFER_DST buffer usage. Regions must be in bounds, nonzero, four-byte aligned, and must not overlap through aliases. Host staging writes require a flush if noncoherent and must complete before submission. Insert a transfer-write dependency before a subsequent build or input read."),
        c("outputs", "Copy simulation output", functions="""
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
        """, text="Destination requires SIMULATION_OUTPUT and TRANSFER_DST usage. Rigid records are tightly packed PmRigidState values in slot order; inactive slots produce zero position/velocities and identity orientation. Fluid output begins with PmDomainOutputHeader followed by active PmFluidParticleOutput records in population order. The destination must cover the addressed rigid count, or the fluid header plus declared particle capacity. Bytes beyond active records are unchanged.<br/><br/>These commands execute in TRANSFER and read simulation state after a PHYSICS_WRITE → TRANSFER_READ dependency. A subsequent renderer needs a TRANSFER_WRITE dependency. Output copies make the portable contract independent of private SoA solver storage.", links=("formats", "barriers", "handoff")),
        c("frame_recipe", "Frame sequence (pseudocode)", code="""
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
        """, text="Updates and state changes are recorded commands. The next frame needs a dependency on this frame's simulation reads/writes, not only on the renderer buffer. Reusing one output buffer requires waiting for its renderer consumer.", links=("example", "submission"), tone="warm"),
        c("reusable_work", "Reusable lists & frames in flight", text="Recorded commands capture scalar settings and host records by value. Editing the original C struct does not change an executable command buffer. Re-record changing inputs or use GPU input buffers with explicit ownership and dependencies.<br/><br/>Reusable buffers can return to Executable after completion. Multiple frames in flight need separate command buffers and output slots, plus timeline dependencies that serialize mutation of one simulation. More command buffers do not make one physics state safely concurrent. Use distinct simulations for independent workloads."),
        c("no_hidden_work", "Allocation and execution boundaries", text="The implementation may allocate CPU command-recording storage in a pool and GPU code/driver bookkeeping at pipeline creation. All physics persistent/scratch storage comes from the declared bindings. Record and submit must not grow physics device allocations. Geometry rebuild, reset, state uploads and result copies are visible commands; the application can put them on its own frame graph.", tone="warm"),
    ]),
    Section("sync", "Submission & synchronization", "Execution order, memory visibility and ownership are separate obligations", [
        c("stage_access", "Stage and access scopes", types="""
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
        """, text="READ/WRITE are generic device accesses narrowed by the stage mask. UPDATE covers reset, body/constraint/state writes and particle initialization. TRANSFER covers buffer and output copies. ALL_COMMANDS covers device stages, not HOST. A host flush is not a GPU completion wait."),
        c("barriers", "Memory & buffer barriers", types="""
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
        """, functions="""
            PmResult pmCmdPipelineBarrier(
                PmCommandBuffer commandBuffer,
                uint32_t memoryBarrierCount,
                const PmMemoryBarrier* pMemoryBarriers,
                uint32_t bufferBarrierCount,
                const PmBufferMemoryBarrier* pBufferBarriers);
        """, text="Memory barriers cover all matching device accesses, including opaque simulation storage. Buffer barriers narrow scope and can transfer ownership. Use IGNORED for both family indices when ownership is unchanged. For a family transfer, record a release on the source and a matching acquire on the destination, with a semaphore between queues. Release uses zero destination stage/access; acquire uses zero source stage/access. Matching ranges and family pairs are required.", links=("handoff",)),
        c("semaphores", "Timeline semaphores", types="""
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
        """, functions="""
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
        """, text="Waits complete at counter ≥ value. Signal values increase strictly and fit within the device's maximum pending difference. Core signals occur through submission, not a host signal call. A timeline is not reset. Use separate timelines when producers cannot order their signal values."),
        c("submission", "Queue submission", types="""
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
        """, functions="""
            PmResult pmQueueSubmit(PmQueue queue,
                const PmSubmitInfo* pInfo, PmFence fence);
            PmResult pmQueueWaitIdle(PmQueue queue);
            PmResult pmDeviceWaitIdle(PmDevice device);
        """, text="Submit accepts Executable buffers from the queue's family. A successful call means accepted, not complete. Execution failures are deferred until fence, timeline, query or simulation completion; referenced objects remain pending until native work drains. A rejected submit leaves its command buffers and fence reusable. Wait scopes name the first consuming device stages; signal scopes must be ALL_COMMANDS. Signals publish prior writes; waits make them visible in the chosen stages. Ownership transfers may still be necessary. fence may be NULL. WaitIdle is useful for shutdown, not required between frames."),
        c("fences", "Host completion fences", types="""
            enum { PM_FENCE_SIGNALED_BIT = 0x1 };
        """, functions="""
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
        """, text="Submit requires an unsignaled fence not associated with pending work. Get returns SUCCESS or NOT_READY. Wait accepts zero for polling and PM_TIMEOUT_INFINITE for an unbounded wait; expiration returns TIMEOUT. Only reset a fence after its associated work completes. A fence synchronizes GPU completion with the CPU, not another GPU queue."),
    ]),
]

SECTIONS += [
    Section("interop", "Native API interoperability", "CUDA, Metal, DirectX 12 and Vulkan", [
        c("native_support", "Backend capabilities", types="""
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
        """, functions="""
            PmResult pmGetPhysicalDeviceInteropProperties(
                PmPhysicalDevice physicalDevice,
                PmNativeBackend backend,
                PmNativeInteropProperties* pProperties);
        """, text="Query each native backend on the selected physical device before importing objects. An unsupported backend returns FEATURE_NOT_PRESENT and zero properties. Each import requires its own capability; support for one backend does not imply support for another. The vulkanInterop feature additionally gates the typed Vulkan extension entry points.<br/><br/>PmNativeHandle is an in-process carrier, not a file descriptor, Win32 shared handle, or serialized object. Encode native pointer handles through uintptr_t, then uint64_t; encode integer handles without changing their bits. Decode using the original native type. Zero means no object, except a CUDA physical-device ordinal may be zero. Imported native objects remain owned by the application. PM wrappers must be destroyed, and their work completed, before native objects are released.", links=("features", "properties")),
        c("native_device", "Device and queue import", types="""
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
        """, functions="""
            PmResult pmImportNativeDevice(
                PmInstance instance,
                const PmNativeDeviceImportInfo* pInfo,
                PmDevice* pDevice);
            PmResult pmGetNativeDevice(PmDevice device,
                PmNativeDeviceInfo* pNativeDevice);
            PmResult pmGetNativeQueue(PmQueue queue,
                PmNativeQueueInfo* pNativeQueue);
        """, text="Import borrows a device/context and at least one explicit queue/stream. enabledFeatures selects PM features; enabledNativeFeatures records native capabilities already enabled by the caller, not requests to enable them. Unknown bits fail. TIMELINE_ENABLED is required for shared counter synchronization. SYNCHRONIZATION2 is required only for Vulkan 1.3 import, and must be zero for other backends. GetNativeDevice and GetNativeQueue return borrowed handles, including for PM-created native devices and queues; they do not transfer ownership.<br/><br/>CUDA uses instance = 0, physicalDevice = CUdevice ordinal, device = CUcontext, queue = a non-default CUstream from that context. Metal uses instance = physicalDevice = 0, device = id&lt;MTLDevice&gt;, queue = id&lt;MTLCommandQueue&gt;. DirectX 12 uses instance = physicalDevice = 0, device = ID3D12Device*, queue = ID3D12CommandQueue* of DIRECT or COMPUTE type. Vulkan uses VkInstance, VkPhysicalDevice, VkDevice and VkQueue. Verify adapter identity before combining native objects.<br/><br/>Vulkan retains native queue family/index values. Other backends use familyIndex = 0 and contiguous queueIndex values starting at zero; PM exposes these as its imported family. Native queues must belong to the supplied device. Serialize host-side access when both APIs submit through the same native queue. Imports do not adopt a caller's partially recorded command list or encoder.", links=("submission", "queues")),
        c("native_buffer", "Buffer import", types="""
            typedef struct PmNativeBufferInfo {
                PmNativeHandle buffer;
                PmDeviceSize offset;
                PmDeviceSize size;
                PmFlags usage;
                uint64_t nativeUsage;
                uint32_t owningQueueFamilyIndex;
            } PmNativeBufferInfo;
        """, functions="""
            PmResult pmImportNativeBuffer(PmDevice device,
                const PmNativeBufferInfo* pInfo,
                PmBuffer* pBuffer);
            PmResult pmGetNativeBuffer(PmDevice device,
                PmBuffer buffer,
                PmNativeBufferInfo* pNativeBuffer);
        """, text="The imported buffer is already allocated and bound. size is nonzero; offset + size must fit the native allocation without overflow and satisfy PM alignment requirements. All PM offsets are relative to this imported range. No pmBindBufferMemory call is permitted on the wrapper. usage declares PM buffer usage and must be compatible with native storage. Get reports the same borrowed native object and range.<br/><br/>CUDA buffer is a CUdeviceptr, not a host pointer; its context must match. Metal buffer is id&lt;MTLBuffer&gt;; device and storage mode must permit GPU access. DirectX 12 buffer is an ID3D12Resource* with BUFFER dimension and appropriate resource flags; it enters and leaves PM ownership in D3D12_RESOURCE_STATE_COMMON. PM performs its internal state transitions. Set nativeUsage to D3D12_RESOURCE_FLAGS for DirectX 12 and to VkBufferUsageFlags for Vulkan; set it to zero for CUDA and Metal. Vulkan buffers use exclusive sharing and require usages matching the typed import contract.<br/><br/>owningQueueFamilyIndex is a Vulkan family for Vulkan; other backends use zero for PM ownership or PM_QUEUE_FAMILY_EXTERNAL for native ownership. On those backends, a PM buffer barrier with EXTERNAL → 0 acquires native storage; 0 → EXTERNAL releases it. The application brackets native access with the matching timeline wait/signal. Queue execution ordering does not remove resource-state or memory-visibility requirements.", links=("barriers", "vulkan_buffer")),
        c("native_timeline", "Timeline import", functions="""
            PmResult pmImportNativeSemaphore(PmDevice device,
                PmNativeHandle nativeSemaphore,
                PmSemaphore* pSemaphore);
            PmResult pmGetNativeSemaphore(PmDevice device,
                PmSemaphore semaphore,
                PmNativeHandle* pNativeSemaphore);
        """, text="Import borrows the native counter object for the device backend. CUDA accepts CUexternalSemaphore imported as a timeline semaphore or D3D12 fence, not a CUevent or binary semaphore. Metal accepts id&lt;MTLSharedEvent&gt;; DirectX 12 accepts ID3D12Fence*; Vulkan accepts a timeline VkSemaphore. The object must be compatible with the native device/context and its counter values must obey PM's increasing-signal rule. Get returns the borrowed object; if no compatible native representation exists it clears the output and returns FEATURE_NOT_PRESENT. CUDA external-handle import and destruction remain the application's responsibility.", links=("semaphores",)),
        c("native_mapping", "Native object mapping", rows=[
            ("CUDA", "CUcontext / CUstream; CUdeviceptr; CUexternalSemaphore"),
            ("Metal", "MTLDevice / MTLCommandQueue; MTLBuffer; MTLSharedEvent"),
            ("DirectX 12", "ID3D12Device / CommandQueue; Resource; Fence"),
            ("Vulkan", "VkDevice / VkQueue; VkBuffer; timeline VkSemaphore")]),
        c("vulkan_device", "Typed Vulkan device adapter", extension=True, types="""
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
        """, functions="""
            PmResult pmImportVulkanDeviceEXT(
                PmInstance instance,
                const PmVulkanDeviceImportInfo* pInfo,
                PmDevice* pDevice);
            PmResult pmGetVulkanDeviceEXT(PmDevice device,
                PmVulkanDeviceInfo* pNativeDevice);
        """, text="Declared in interop_vulkan.h. Requires Vulkan 1.3, timelineSemaphore and synchronization2 actually enabled at native device creation. Caller must report enabled state truthfully. Match an enumerated physical-device UUID with vulkanInterop support. Imports borrow every native object; PM destroys only its wrappers and internal children. Get returns borrowed handles for either an imported or PM-owned native device; never destroy PM-owned handles through Vulkan. Queue family/index values retain their Vulkan meanings."),
        c("vulkan_buffer", "Typed Vulkan buffer adapter", extension=True, types="""
            typedef struct PmVulkanBufferImportInfo {
                VkBuffer buffer;
                VkDeviceSize size;
                VkBufferUsageFlags nativeUsage;
                PmFlags usage;
                uint32_t owningQueueFamilyIndex;
            } PmVulkanBufferImportInfo;
        """, functions="""
            PmResult pmImportVulkanBufferEXT(
                PmDevice device,
                const PmVulkanBufferImportInfo* pInfo,
                PmBuffer* pBuffer);
            PmResult pmGetVulkanBufferEXT(PmDevice device,
                PmBuffer buffer, VkBuffer* pNativeBuffer);
        """, text="The native buffer must be bound, large enough, exclusive-sharing, on the same VkDevice and created with all declared native usages. PM input buffers need STORAGE_BUFFER; transfer source/destination usages must match. SIMULATION_OUTPUT needs TRANSFER_DST. Native renderer usages such as VERTEX_BUFFER remain the renderer's responsibility. Destroying an imported PM wrapper does not destroy its VkBuffer or memory."),
        c("vulkan_semaphore", "Typed Vulkan timeline adapter", extension=True, functions="""
            PmResult pmImportVulkanSemaphoreEXT(
                PmDevice device, VkSemaphore semaphore,
                PmSemaphore* pSemaphore);
            PmResult pmGetVulkanSemaphoreEXT(
                PmDevice device, PmSemaphore semaphore,
                VkSemaphore* pNativeSemaphore);
        """, text="Import borrows an existing timeline semaphore from the same device. Binary semaphores are invalid. Get returns a borrowed native handle; do not destroy it through Vulkan if PM owns it. The same increasing counter values can be used in PmSubmitInfo and VkSemaphoreSubmitInfo. No CPU wait is needed merely to connect the two queues."),
        c("handoff", "Native work → physics → native work", text="Before overwriting a shared output slot, physics waits for the native consumer's done value, acquires the buffer, solves, copies state, and releases the buffer. It signals completion only after the writes and release. The native consumer waits, establishes its required visibility and resource state, reads the data, then releases it and signals a later done value. The same pattern applies to rendering, compute, or another physics producer.<br/><br/>CUDA enqueues external-semaphore waits and signals on streams. Metal encodes shared-event waits and signals on command buffers. DirectX 12 uses command-queue Wait/Signal with a fence and the appropriate resource transitions. Vulkan uses timeline semaphore waits/signals plus matching queue-family release/acquire where necessary. A shared queue does not eliminate memory hazards; host calls sharing that queue must be serialized. No CPU idle wait is required merely to connect supported native queues.", links=("barriers", "submission", "native_timeline"), tone="warm"),
        c("stage_mapping", "Native synchronization", rows=[("CUDA", "Stream order + external timeline waits/signals"), ("Metal", "Command buffers + shared events + resource hazards"), ("DirectX 12", "Queue fences + resource barriers"), ("Vulkan", "Timeline semaphores + memory/ownership barriers")], text="PM stages describe logical dependencies, not a native shader ABI. CUDA and Metal may conservatively synchronize complete earlier operations where a finer PM stage is unavailable. DirectX 12 must also transition shared resources at the ownership boundary. For Vulkan, TRANSFER maps to ALL_TRANSFER, UPDATE to COMPUTE_SHADER | ALL_TRANSFER, GEOMETRY_BUILD and PHYSICS to COMPUTE_SHADER, and ALL_COMMANDS to ALL_COMMANDS; PM read/write accesses map to MEMORY_READ/MEMORY_WRITE. The native consumer selects its actual consuming stage and access."),
        c("interop_boundary", "Sharing boundary", text="Importing an existing in-process native object is distinct from importing operating-system external-memory handles or sharing across processes. Allocate and import external resources through the native APIs first; pass the resulting compatible device-local objects to PM only when the relevant capability is reported. This contract does not promise arbitrary cross-API or cross-device sharing, handle export, or a host-copy fallback. Unsupported imports return FEATURE_NOT_PRESENT. The typed Vulkan adapters have the same ownership and capability requirements as the backend-neutral calls.", tone="warm"),
    ], "handoff"),
    Section("queries", "Timing, work & memory queries", "Measure GPU execution separately from host overhead", [
        c("query_pool", "Query pools", types="""
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
        """, functions="""
            PmResult pmCreateQueryPool(PmDevice device,
                const PmQueryPoolCreateInfo* pInfo,
                PmQueryPool* pPool);
            void pmDestroyQueryPool(PmDevice device,
                PmQueryPool pool);
            PmResult pmCmdResetQueryPool(
                PmCommandBuffer commandBuffer,
                PmQueryPool pool, uint32_t first,
                uint32_t count);
        """, text="Query pools own implementation-managed result storage, accounted as library bytes. Reset slots before first use and after consuming prior results. Do not reset results still in flight. timestampQueries/statisticsQueries must be enabled for the selected type. queryCount is positive and bounded by maxQueriesPerPool."),
        c("timestamps", "GPU timestamps", functions="""
            PmResult pmCmdWriteTimestamp(
                PmCommandBuffer commandBuffer,
                PmStageFlags stage,
                PmQueryPool pool, uint32_t query);
        """, text="Use ALL_COMMANDS for a portable before/after measurement. A timestamp becomes available once earlier commands have completed and the sample has been written. Timestamp values are unsigned 64-bit ticks; only timestampValidBits are meaningful. GPU nanoseconds = modular tick delta × timestampPeriodNanoseconds. Samples must be on the same queue, and duration must be less than one wrap period. They do not measure CPU recording, queue-submit overhead or wall-clock time."),
        c("step_stats", "Step statistics & execution status", types="""
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
        """, functions="""
            PmResult pmCmdBeginStatisticsQuery(
                PmCommandBuffer commandBuffer,
                PmQueryPool pool, uint32_t query);
            PmResult pmCmdEndStatisticsQuery(
                PmCommandBuffer commandBuffer,
                PmQueryPool pool, uint32_t query);
            PmResult pmGetSimulationStatus(
                PmDevice device, PmSimulation simulation,
                PmResult* pExecutionResult);
        """, text="Queries accumulate across enclosed steps: work counters sum, peak fields and high-water bytes take maxima. executionResult is the first failing PmResult, or SUCCESS. Do not nest statistics queries; Begin and End must be in one buffer. After observing completion, GetSimulationStatus exposes the last step status even when statistics are disabled. It returns NOT_READY while simulation work is pending. Overflow invalidates that step's output and requires reinitialization. When an enclosed step completes with CAPACITY_EXCEEDED, its completed statistics result remains readable and returns SUCCESS; executionResult contains CAPACITY_EXCEEDED and overflowCount contains the generated overflow count. Other failed query submissions return their completion error."),
        c("query_results", "Readback & availability", types="""
            enum {
                PM_QUERY_RESULT_WAIT_BIT = 0x1,
                PM_QUERY_RESULT_WITH_AVAILABILITY_BIT = 0x2
            };
        """, functions="""
            PmResult pmGetQueryPoolResults(PmDevice device,
                PmQueryPool pool, uint32_t first,
                uint32_t count, size_t dataSize,
                void* pData, PmDeviceSize stride,
                PmFlags flags);
        """, text="A timestamp result is uint64_t; statistics use PmStepStatistics (88 bytes). WITH_AVAILABILITY appends uint64_t, zero/unavailable or one/available, after each result. stride is an eight-byte multiple large enough for a record; pData is eight-byte aligned; dataSize covers all records. Without WAIT, unavailable result bytes remain untouched and NOT_READY is returned; availability fields are still written. WAIT has no timeout: use a fence or semaphore wait first for bounded waiting."),
        c("memory_budget", "Device budget & library accounting", types="""
            typedef enum PmMemoryBudgetConfidence {
                PM_MEMORY_BUDGET_CONFIDENCE_UNKNOWN = 0,
                PM_MEMORY_BUDGET_CONFIDENCE_IMPLEMENTATION_ESTIMATE = 1,
                PM_MEMORY_BUDGET_CONFIDENCE_DRIVER_REPORTED = 2,
                PM_MEMORY_BUDGET_CONFIDENCE_OS_REPORTED = 3
            } PmMemoryBudgetConfidence;
            typedef struct PmMemoryBudget {
                uint32_t heapCount;
                PmDeviceSize budget[16];
                PmDeviceSize usage[16];
                PmMemoryBudgetConfidence confidence[16];
                PmDeviceSize applicationAllocationBytes;
                PmDeviceSize libraryDeviceBytes;
                PmDeviceSize libraryHostBytes;
                PmDeviceSize borrowedBufferBytes;
            } PmMemoryBudget;
        """, functions="""
            PmResult pmGetDeviceMemoryBudget(
                PmDevice device, PmMemoryBudget* pBudget);
        """, text="Requires enabled memoryBudget. Physical heap size, returned by pmGetPhysicalDeviceMemoryProperties, is capacity and not a usable budget. budget and usage are a live per-heap snapshot; confidence identifies their source, not a guarantee or quality ordering. UNKNOWN entries have zero budget and usage and must never admit work. IMPLEMENTATION_ESTIMATE can omit other processes and driver-private allocations. DRIVER_REPORTED and OS_REPORTED can change immediately after the query and are not reservations.<br/><br/>Application bytes sum live owned pmAllocateMemory blocks. Library device/host bytes count only allocations tracked by the implementation; zero does not prove that the driver has no private allocations. Borrowed bytes sum live imported buffer ranges and can double-count aliases; they do not state physical residency or unique allocation size. Usage may include borrowed and otherwise untracked bytes, so accounting categories are not additive with each other or heap usage. Counter overflow fails the operation rather than wrapping. Resource requirements describe bytes an application must bind, not availability.", links=("memory_types", "requirements")),
    ]),
    Section("capabilities", "Features, limits & formats", "Capability decisions are made before allocation and pipeline creation", [
        c("interaction_capabilities", "Interaction capabilities", text="Enumerate records to discover exact supported family, scope, mode, direction and material-profile combinations. Family order is canonical: equal families for SELF, SAME_DOMAIN and SEPARATE_SYSTEM; increasing family values for DISTINCT_FAMILY. A key that violates this shape or contains an unknown enum returns INVALID_ARGUMENT. A well-formed but unsupported combination returns FEATURE_NOT_PRESENT.<br/><br/>maxGraphEdgesPerSimulation is a driver limit for that exact record. NOT_APPLICABLE means the mode is intrinsic and has no graph edge. SIMULATION_CAPACITY means no narrower device limit is imposed beyond PmSimulationCreateInfo::couplingCapacity. A finite value is an additional per-simulation ceiling. Enumerators return only supported records; broad feature bits never imply a missing record.<br/><br/>Smoke particle and grid profiles are independent records. Thermal and phase modes identify the fixed API 0.2 transfer law; their smoke profile still identifies the representation. EXTERNAL_TO_FIRST/SECOND identifies an external source such as the current fluid-smoke heater. TRANSFER directions consume membership from one family and create it in the other.", links=("features", "coupling", "limits")),
        c("limits", "Physical-device limits", types="""
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
        """, functions="""
            void pmGetPhysicalDeviceLimits(
                PmPhysicalDevice physicalDevice,
                PmPhysicalDeviceLimits* pLimits);
        """, text="Limits are maxima for one resource or field, not performance targets. Memory pressure can still make a legal allocation fail. Every count and numerical budget must be checked against the corresponding limit before creation/recording."),
        c("formats", "Format support query", types="""
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
        """, functions="""
            void pmGetPhysicalDeviceFormatProperties(
                PmPhysicalDevice physicalDevice,
                PmFormat format,
                PmFormatProperties* pProperties);
        """, text="UNDEFINED or an unsupported format returns zero properties. Supported formats have the fixed layouts below; implementation-specific packed formats are not inferred. Query flags express usage support, not a separate simulation model."),
        c("format_layouts", "Portable buffer layouts", rows=[("R32_UINT", "4 bytes / alignment 4 / unsigned index"), ("R32_SFLOAT", "4 bytes / alignment 4 / IEEE binary32"), ("R32G32B32_SFLOAT", "12 bytes / alignment 4 / x,y,z"), ("R32G32B32A32_SFLOAT", "16 bytes / alignment 4 / x,y,z,w"), ("PARTICLE_STATE", "28 bytes / alignment 4 / PmParticleState"), ("FLUID_PARTICLE_OUTPUT", "48 bytes / alignment 4 / PmFluidParticleOutput"), ("RIGID_STATE", "52 bytes / alignment 4 / PmRigidState")], text="Record arrays are little-endian. These buffer layouts have fixed offsets, regardless of a shader language's native vector packing. PmRigidBodyInfo and other handle-containing host descriptors are not GPU binary layouts. Do not memcpy them into storage buffers. Scalar fields are 32-bit unless explicitly declared otherwise."),
        c("feature_matrix", "Dependency checklist", rows=[("Rigid contacts", "rigidBodies"), ("Rigid joints", "rigidBodies + rigidConstraints"), ("One liquid", "particleLiquids"), ("Liquid ↔ bodies", "both domains + liquidRigidCoupling"), ("Liquid ↔ liquid", "particleLiquids + liquidLiquidCoupling"), ("GPU timing", "timestampQueries + queue timestamp bits"), ("Solver counters", "statisticsQueries"), ("Native object sharing", "Per-backend import capabilities"), ("Typed Vulkan adapters", "vulkanInterop + required native features")], text="Report a feature only when its required behavior is supported. Requesting unsupported features or coupling pairs fails with FEATURE_NOT_PRESENT.", links=("features", "native_support")),
        c("threading", "Threading & data races", text="Physical-device queries are safe to call concurrently. Distinct command pools can record independently. Host access to the same queue, pool, fence, mapped range or mutable resource must be externally synchronized. Destroying an object must also be synchronized against queries that use it.<br/><br/>GPU races are a separate concern. Serialize writes to a simulation, coordinate shared input/output buffers, and avoid changing CPU-mapped bytes while a pending command may read them. Callback implementations must support concurrency across independently accessed objects."),
        c("not_capabilities", "Core scope", text="Core 0.2 defines primary command buffers, explicit memory binding, rigid triangle geometry, rigid constraints, particle liquids and the listed couplings. Secondary buffers, simultaneous reuse, sparse resources, shader injection and cross-device migration are outside this specification.<br/><br/>Cloth, ropes, soft bodies, smoke, emitters and paint require domain-specific extensions. Unknown pNext structures are errors.", links=("features",), tone="warm"),
    ]),
]

SECTIONS += [
    Section("example", "A frame, recorded explicitly", "Record, submit and consume a simulation frame", [
        c("example_record", "Record a rigid-body frame", code="@example_record", text="TRY returns the first non-success result. On any recording error, reset the buffer before trying again. This example requires initialized resources, completed prior command/query use, and a renderer output buffer in the same queue family. All referenced objects and memory must remain alive until completion.", links=("lifecycle", "handoff")),
        c("example_submit", "Submit a frame between timeline values", code="@example_submit", text="The renderer signals renderDone after its previous read. Physics signals physicsDone after copying output. The renderer waits for that new physics value at its actual consuming stage. completion is an unsignaled fence, or NULL. Semaphore signal values must be globally monotonic.", links=("submission",)),
        c("example_prepare", "Before the first frame", rows=[("Instance / device", "Enumerate; query support; enable features"), ("Geometry / buffers", "Create, allocate, bind, upload and build"), ("Simulation", "Declare capacities; bind both memory ranges"), ("Pipeline", "Set rigid state and optional liquid materials"), ("Initialization list", "Reset state; write bodies and constraints"), ("Frame objects", "Pool, buffers, timelines, queries and fence")], text="Initialize geometry and simulation state before recording a frame. Make all initialization writes visible to the first physics step.", tone="warm"),
        c("example_after", "After submission", text="Wait or poll completion before reusing a command buffer or query slot. Read simulation status before accepting results after suspected overflow. Read timestamps with availability, using the queue's period and valid-bit count. A CPU readback additionally needs a device-to-host dependency and invalidation for noncoherent memory.<br/><br/>Do not call DeviceWaitIdle every frame to compensate for missing dependencies. Track the resources and completion values that each consumer actually uses.", links=("timestamps", "query_results", "mapping")),
    ]),
]


def structure_name(name):
    return "PM_STRUCTURE_TYPE_" + re.sub(r"(?<!^)(?=[A-Z])", "_", name[2:]).upper()


def apply_registry():
    """Replace declarations with the canonical API registry contents."""
    global PRELUDE, STRUCTURE_TYPE_VALUES
    registry = Path(__file__).resolve().parents[2] / "registry" / "pm.xml"
    root = ET.parse(registry).getroot()
    PRELUDE = root.findtext("prelude") or ""
    indexed = {card.key: card for section in SECTIONS for card in section.cards}
    registry_keys = set()
    for node in root.findall("./cards/card"):
        key = node.attrib["key"]
        registry_keys.add(key)
        if key not in indexed:
            raise RuntimeError(f"registry card {key!r} has no documentation card")
        card = indexed[key]
        card.types = (node.findtext("types") or "").strip()
        card.functions = (node.findtext("commands") or "").strip()
        card.extension = node.attrib.get("extension") == "true"
        card.emit_header = node.attrib.get("emit-header", "true") == "true"
    documented = {key for key, card in indexed.items()
                  if card.types or card.functions}
    missing = documented - registry_keys
    if missing:
        raise RuntimeError(f"declaration cards missing from registry: {sorted(missing)}")

    STRUCTURE_TYPE_VALUES = {}
    for node in root.findall("./structure-types/type"):
        name = node.attrib["name"]
        value = int(node.attrib["value"], 0)
        if name in STRUCTURE_TYPE_VALUES:
            raise RuntimeError(f"duplicate structure type name: {name}")
        if value == 0 or value in STRUCTURE_TYPE_VALUES.values():
            raise RuntimeError(f"invalid or duplicate structure type value: {value}")
        STRUCTURE_TYPE_VALUES[name] = value

    declared_structure_types = {
        name for section in SECTIONS for card in section.cards
        for name, body in re.findall(
            r"typedef struct (Pm\w+) \{(.*?)\}", card.types, re.S)
        if "PmStructureType sType;" in body
    }
    registered_structure_types = set(STRUCTURE_TYPE_VALUES)
    if declared_structure_types != registered_structure_types:
        raise RuntimeError(
            "structure type registry mismatch; missing="
            f"{sorted(declared_structure_types - registered_structure_types)}, "
            "extra="
            f"{sorted(registered_structure_types - declared_structure_types)}")


apply_registry()


STRUCTURE_TYPES = "typedef enum PmStructureType {\n    PM_STRUCTURE_TYPE_UNDEFINED = 0,\n"
STRUCTURE_TYPES += "\n".join(
    f"    {structure_name(name)} = {value},"
    for name, value in sorted(STRUCTURE_TYPE_VALUES.items(), key=lambda item: item[1]))
STRUCTURE_TYPES += "\n} PmStructureType;"

SECTIONS.append(Section("fundamentals", "Fundamental declarations", "Types and identifiers", [
    c("base_types", "Types, handles & constants",
      types=PRELUDE[PRELUDE.index("#define PM_API_VERSION_0_2"):], emit_header=False,
      text="Handles are opaque pointers, not structures applications may dereference. PMAPI_ATTR supplies the platform's loader import/export visibility and PMAPI_CALL supplies its calling convention. All declarations use C linkage in C++. PmBool32 is PM_FALSE or PM_TRUE. Each command has a PFN_pmCommandName function-pointer typedef with exactly the displayed return type, parameters and PMAPI_CALL convention."),
    c("structure_types", "Structure types", types=STRUCTURE_TYPES, emit_header=False,
      text="Every extensible input structure sets sType to the identifier for its exact structure type. pNext is NULL unless a supported extension defines the chained structure. An identifier does not enable a feature or extension."),
]))
