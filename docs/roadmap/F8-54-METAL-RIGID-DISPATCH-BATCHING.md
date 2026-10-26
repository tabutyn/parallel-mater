# F8-54 Metal rigid dispatch batching

F8-54 removes the body-count-sized Metal command graph from the production
fixed-tick rigid solver. Metal now issues one preparation dispatch and one
solve dispatch for the shared Slang AVBD implementation. The solve kernel
retains the same eight algebraic sweeps and advances time once.

The production path does not contain Metal-specific physics equations. Metal
binds the shared `solve_avbd_kernel`, while CUDA, D3D12, Vulkan, Metal and the
CPU reference continue to compile the same Slang source for their targets.

## Execution and accounting

The solve dispatch uses one cooperative workgroup. Its 32 lanes traverse the
prepared active bodies and constraints on the device and synchronize between
dependent color and sweep phases with workgroup barriers. Empty colors no
longer produce host-issued no-op commands. The prepared schedule remains on
the GPU; batching adds no phase-local readback, host wait or global GPU wait.

Metal records the actual command count before and after the launch and rejects
a mismatch with the fixed schedule telemetry. Portable tests independently
freeze a ceiling of one solve dispatch and one solve workgroup for 8, 64, 128
and 386 bodies, including a sparse-capacity case. These ceilings were committed
before the implementation and reject the former
`8 * (body_count + 1) + 6` solve graph.

## Physical Apple GPU result

The public production fixture ran on an Apple M4 with identical inputs before
and after the change:

| Counter | Before | After |
| --- | ---: | ---: |
| Total dispatches | 112 | 35 |
| Domain dispatches | 88 | 11 |
| Total workgroups | 1,513 | 1,436 |
| Domain workgroups | 1,489 | 1,412 |
| Barriers | 117 | 40 |
| Time advances | 1 | 1 |
| Rigid sweeps | 8 | 8 |

The domain dispatch count fell by 87.5 percent. A physical Metal comparison
also ran 8, 64, 128 and 386-body fixtures through both the batched kernel and
the retained staged test oracle. All body states, constraint states and
prepared schedules passed their existing tolerances. The largest fixture
replaced 3,102 solve dispatches with one.

The production test passed twenty separate cold-start processes in succession.
The complete macOS Metal preset and the focused shared-solver, architecture and
physical kernel suites pass. The independent CPU/reference preset builds and
passes all 102 configured tests (15 unavailable-domain cases skip as designed).
The unsigned arm64 iPad Gallery CI artifact also builds against the iPhoneOS
26.5 SDK. Final installed Mac/iPad Gallery comparison is intentionally
assigned to F8-56.

Exact commands, revisions, counters and diagnostic timings are in
[`evidence/F8-54-metal-rigid-dispatch-batching.json`](evidence/F8-54-metal-rigid-dispatch-batching.json).
