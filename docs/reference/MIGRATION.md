# Command API implementation plan

## Truth at the starting point

As inspected at `2f5e3b7`, the installed C++ APIs expose backend-specific
`World` objects. They own simulation allocation and expose immediate edits,
stepping, asynchronous frame tokens, and GPU views. The CUDA/Metal rigid
solver uses shared Slang AVBD numerical functions, but substantial native
geometry and dispatch code remains. The Slang geometry mirror and optional
adapter are not a complete production replacement. D3D12 has different
implementation coverage. See `docs/SLANG.md` and `docs/CONTACT_SOLVER.md`.

The separate Vulkan branch `codex/vulkan-vertical-slice-20261009`, inspected
at `fc22235`, is an experimental rigid-only vertical slice. Its borrowed
native context and timeline completion are useful
precedents, but it is not evidence that the full AVBD constraints and fluids
already execute on Vulkan.

## Migration gates

1. **Contract and placeholder coverage.** The generated C headers are installed
   separately from `World`. All declared commands have opt-in, fail-closed stubs.
   C/C++ linkage, complete public types, symbol coverage and failure outputs are
   tested. This does not complete any of the runtime gates below; see `API-AUDIT.md`.
2. **Instance/device layer.** Real device enumeration, UUIDs, features,
   limits, queues, and validation. Reject unsupported requests; never invent
   device properties or report planned features as supported.
   Implement the version query and instance/device procedure lookups, with
   exact-name matching, NULL for unavailable commands, and owner-scoped
   function-pointer lifetimes. Generated `PFN_pm*` typedefs are not a loader
   implementation.
3. **Allocator separation.** Extract persistent and scratch requirements
   from `World`. Implement explicit memory type selection, binding,
   non-coherent mapping rules, and accounting. Verify no hidden physics
   device allocation occurs during command recording/submission.
4. **Resource and command execution.** Implement stable slots, geometry
   build, reset/upload/readback commands, lifecycle validation, lifetime
   checks, query availability, and bounded capacity behavior. All referenced
   state stays alive while pending. Build a real end-to-end linked example.
5. **One Slang rigid path.** Execute the same geometry/contact/joint code
   across supported targets. Test stack, point, hinge, slider, spring, motor,
   dump-truck and mixed contact/joint scenes. Do not wrap the older Vulkan
   slice as if it already meets this gate.
6. **Fluid pipelines and coupling.** Port parameter semantics with units,
   material compatibility validation and per-pair capability reporting.
   Require two-way momentum tests for two-way coupling and isolation tests
   when a pair is disabled. Add remaining domains through typed extensions.
7. **External pipeline integration.** Implement native imports for each advertised
   CUDA, Metal, DirectX 12 or Vulkan capability; demonstrate bidirectional native
   work/physics timeline handoffs without a host idle wait. Test the backend's
   resource-state, visibility and ownership requirements, borrowed object ownership,
   round-trip output buffers, cancellation/device-loss and shutdown.
8. **Application adoption.** Move gallery code and then the compatibility
   `World` wrapper onto this API. Preserve scene behavior and calibrated GPU
   workloads. Only then deprecate backend-specific public entry points.

## Required conformance coverage

| Area | Evidence required before marking implemented |
| --- | --- |
| Discovery | Count/fill enumeration, short array returns, UUID match, feature rejection |
| Memory | Every offset/type/alignment error, noncoherent ranges, arena accounting, OOM |
| Resources | Unbound use rejected, uploads ordered, indexed triangle bounds, capacities |
| Commands | All legal/illegal state transitions; failed recording cannot submit |
| Synchronization | Timeline monotonicity, unsignaled fences, visibility, ownership transfer |
| Lifetime | Pending objects cannot be freed/rebound; borrowed native objects survive teardown |
| Physics | Contacts and joints on one solver; fluids and couplings match accepted scenes |
| Statistics | Timestamp units, unavailable query, reset/reuse, overflow reporting, byte totals |
| Interop | Shared device and queue, foreign-queue handoff, supported external handles |
| Performance | Idle/collision frame rates and memory against per-device repository data |

Documentation and unsupported-stub checks are not these runtime conformance
tests. The running solver has not been switched to the command API. Runtime
migration requires all applicable gates above.
