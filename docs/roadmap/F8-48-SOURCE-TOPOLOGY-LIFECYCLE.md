# F8-48 source and topology lifecycle in the common tick

F8-48 completes the dynamic lifecycle owned by the installed
`realtime-single-step` schedule. Particle-liquid sources, destroy planes and
liquid-to-particle-smoke phase changes now cross the same staged tick boundary
as the domain and coupling work delivered by F8-39 through F8-47. No
source-specific scheduler, spatial rebuild or platform-local physics path was
added.

## Source admission and removal

Enabled emitters write fixed-capacity proposals before the shared spatial
index is built. The common admission pass decides those proposals together
with the ordinary domain leaves and records accepted, rejected and
capacity-miss counters. Accepted particles enter the staged liquid arrays;
rejected proposals remain observable as source counters and consume stable ID
ranges without allowing reuse. That deliberate gap policy is the F8-09 stable
identity contract.

Destroy planes mark the staged liquid population after the solve. One gather
compacts all removals; the public active count and particle output change only
at the normal atomic commit. The runtime's GPU-resident live count drives
predict, query, solve, coupling and export dispatches. A source acceptance,
capacity rejection, outflow and later capacity recovery therefore retain one
time advance, one spatial build, one publication and the same fixed dispatch
and workgroup budget.

## Phase transfer

The final liquid/smoke coupling sweep records hot liquid candidates in
persistent fixed-capacity proposal storage. The next tick boundary pins the
source and destination generations, checks aggregate destination capacity,
and either admits every pending proposal or admits none. A successful boundary
removes the source particles and copies their position, velocity, density,
pressure, vorticity and thermal state into particle smoke before that tick's
shared index publication. Failure leaves source mass, stable IDs and the
pending request intact.

The public fixture starts one liquid particle at 150 C against a 100 C boiling
threshold. Its discovery tick retains one liquid and zero smoke particles; the
next tick publishes zero liquid and one smoke particle. A full destination
instead retains the liquid stable ID and both populations. Replacing that
destination invalidates the old generation, rediscovery retains the same
source ID, and the following boundary completes the transfer. Successful and
blocked cases both execute four coupling sweeps with unchanged domain
iteration counts.

## Atomic lifecycle coverage

The focused installed-API fixture covers cold source spawn, capacity
rejection, outflow, recovered capacity, reset, successful and blocked phase
admission, destination replacement and generation-safe rediscovery. The broad
production fixture continues through the existing mixed rigid, liquid, cloth,
soft-body, rope, particle-smoke and grid-smoke graph, including topology/state
replacement and reset paths. Every observed lifecycle tick reports one time
advance, one spatial build, one publication and zero intra-tick host
readbacks. Failure never publishes a partial endpoint update or adds solver
iterations.

## Qualification

The complete host suite passed: 102 tests discovered, 87 executed successfully
and 15 unadvertised CPU-domain tests skipped explicitly. All 418 conformance
tests, all 87 roadmap tests and the protected realtime candidate closure check
passed. CPU execution passed, while current CUDA and desktop Vulkan driver
targets compiled successfully from the shared schedule.

An exact-revision Android arm64 Vulkan build ran on a Google Pixel 7
(`panther`, Mali-G710, Android 17/API 37). Command, async, liquid and broad
mixed production tests passed through the repository capture runner. The
focused lifecycle test also passed and its trace contains source proposal,
shared admission/finalization, destroy/gather and liquid/smoke kernels. There
were no Vulkan validation errors.

Physical Metal and D3D12 devices were unavailable on this Linux host, so this
card claims shared-source compilation for those adapters, not physical-device
execution or performance. Final whole-candidate performance, sustained
thermal and platform qualification remain the F8-52 and platform-leaf gates.
Exact commands, revisions, hashes and assertions are in
[`evidence/F8-48-source-topology-lifecycle.json`](evidence/F8-48-source-topology-lifecycle.json).
