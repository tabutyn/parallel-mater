# Driver migration and qualification

Parallel Mater exposes one public C command API through the loader. The gallery
and repository examples use that API directly. Physics algorithms live in the
shared Slang sources; device adapters provide allocation, dispatch, transfer,
synchronization, queries, and native-handle services without implementing a
second solver or simulation schedule.

The Slang/CUDA driver is the first qualified implementation. Native CUDA calls
inside its device-services and dispatch layers are private driver machinery,
not an alternate public API. Additional drivers must advertise only the
features and interoperability paths they implement and pass the same command,
physics, gallery, and architecture checks before being reported as available.

## Qualification gates

1. **ABI and dispatch.** Export the private driver ABI expected by the loader,
   implement exact-name procedure lookup, preserve owner-scoped lifetimes, and
   pass generated symbol and function-pointer coverage checks.
2. **Discovery.** Report real device identity, features, limits, formats,
   queues, memory types, and extensions. Reject unsupported requests; never
   report planned capabilities as implemented.
3. **Memory and resources.** Implement explicit allocation and binding,
   alignment and range validation, host visibility rules, accounting, imported
   resource ownership, and bounded-capacity failure behavior.
4. **Commands and synchronization.** Preserve command-buffer lifecycle,
   asynchronous queue submission, timeline and fence semantics, resource
   ownership transfers, query availability, and deferred error reporting.
5. **Shared physics.** Dispatch the common Slang geometry, AVBD constraints,
   fluids, cloth, rope, soft bodies, smoke, and coupling kernels. A device
   adapter must not contain independent collision, constraint, or coupling
   algorithms.
6. **Interop.** Validate every advertised native-device, buffer, and timeline
   path with bidirectional native-work/physics handoffs. Cover resource state,
   visibility, ownership, borrowed-object lifetime, device loss, and shutdown.
7. **Application qualification.** Run the complete shared gallery catalog,
   controller/input replays, focused scene regressions, conformance corpus, and
   device-profile performance checks through the public API.

## Required conformance coverage

| Area | Evidence required before qualification |
| --- | --- |
| Discovery | Count/fill enumeration, short-array returns, UUID match, feature rejection |
| Memory | Offset/type/alignment errors, noncoherent ranges, accounting, OOM |
| Resources | Unbound use rejection, ordered uploads, indexed triangle bounds, capacities |
| Commands | Legal and illegal lifecycle transitions; failed recording cannot submit |
| Synchronization | Timeline monotonicity, unsignaled fences, visibility, ownership transfer |
| Lifetime | Pending objects cannot be freed or rebound; borrowed objects survive teardown |
| Physics | All domains and couplings use the shared kernels and accepted scene behavior |
| Statistics | Timestamp units, unavailable results, reset/reuse, overflow, byte totals |
| Interop | Shared device/queue, foreign-queue handoff, supported external handles |
| Gallery | Complete catalog, controls, reset/switching, renderer handoff, headless replay |
| Performance | Idle/collision frame rates and memory against repository device profiles |

Generated documentation and compile-only shader checks do not qualify a
driver. Unsupported targets remain unadvertised and must return the documented
capability errors instead of silently selecting another implementation.
