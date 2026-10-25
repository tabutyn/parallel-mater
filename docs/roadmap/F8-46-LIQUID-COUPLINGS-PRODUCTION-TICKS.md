# F8-46 liquid couplings in production fixed ticks

F8-46 connects the supported liquid interaction graph to the native public
`realtime-single-step` schedule. The installed matrix is:

- equal-material liquid/liquid contact, pressure and viscosity;
- liquid/rigid contact and containment;
- liquid/cloth contact and containment;
- liquid/soft-body and liquid/rope contact; and
- particle-smoke or grid-smoke drag and thermal exchange with liquid.

Miscible and immiscible liquid pairs do not enter the equal-material path.
Phase-transfer storage mutation remains F8-48 work; the thermal operator runs
without consuming or publishing phase transfers in this change.

## Fixed coupling pass

The boundary stages live registrations in canonical family, mode and stable
edge order and pins both endpoint generations plus the public graph revision.
The liquid family then executes four fixed sweeps after every domain solve and
before the one atomic commit. Every operator sees the same staged endpoint
state and F8-10 spatial epoch. No operator predicts, integrates, advances time,
publishes another spatial epoch, waits on the host, or commits independently.

The existing production operators remain the equations used by the public
graph. Equal-material populations use the combined pair pressure/contact and
viscosity entries. Rigid geometry, cloth, rope and soft-body use their existing
two-endpoint contact/containment entries. Particle and grid smoke use their
existing drag entries, while the exchange entry performs thermal response with
boiling disabled until F8-48 owns phase admission. A liquid pair emits and
sorts its fixed combined cell keys once in the first sweep; later sweeps reuse
that frozen order and do not rebuild or republish the shared spatial epoch.

## Bounded work and failure

Liquid registration capacity is fixed at 16 entries. The runtime derives the
exact active mode count before device work, rejects unsupported material
profiles and stale endpoint generations, and accounts every dispatch,
workgroup, clear and numerical sweep in the native telemetry ceiling. The
F8-18 tick contract verifies four complete canonical visits against the same
epoch counters and graph revision.

Abort resets every participating domain and coupling contract. A public mixed
fixture installs nine graph edges across two liquids, rigid, cloth, soft body,
rope and particle smoke. Together with intrinsic fixed-schedule registrations
it reports 20 coupling edges, 12 family sweeps, one time advance, one spatial
build/publication and zero intra-tick readbacks. A one-below workgroup limit
returns `PM_ERROR_CAPACITY_EXCEEDED`; all five copied domain outputs remain
byte-for-byte unchanged and graph removal succeeds afterward.

## Qualification

The complete 102-test host suite passes: 87 tests execute and 15 unadvertised
CPU-domain cases explicitly skip. Both the failure-injection and production
CPU drivers execute the new public mixed graph. Current CUDA and Vulkan drivers
compile, the protected F8 closure remains exact, and an Android arm64 Vulkan
Gallery APK builds successfully.

This host has no NVIDIA device nodes and `adb devices` reports no attached
Pixel 7, so it cannot honestly supply fresh integrated physical execution.
Immutable F8-18 physical CUDA evidence covers the four-sweep liquid coupling
contract and every supported family/mode, while the earlier F5 records cover
the specialized production operators. Those records are provenance, not a
claim that this dirty branch ran on unavailable hardware. Final current-
candidate physical qualification remains owned by the F8-52-routed platform
gates.

Exact commands and hashes are in
[`evidence/F8-46-liquid-couplings-production-ticks.json`](evidence/F8-46-liquid-couplings-production-ticks.json).
