# F8-45 mechanical couplings in production fixed ticks

F8-45 connects the existing F8-17 rigid and applicable F8-19 deformable
interaction operators to the public native `realtime-single-step` schedule.
The installed matrix is exact:

- rigid/cloth, rigid/soft-body and rigid/rope contact;
- rigid/rope attachment;
- cloth/soft-body contact;
- cloth/rope attachment; and
- soft-body/rope contact plus attachment.

No additional capability is advertised. Liquid interactions remain F8-46 and
smoke interactions remain F8-47.

## Fixed coupling pass

Live rigid/deformable registrations are staged in canonical family, mode and
identity order at the tick boundary. Rigid contacts are intrinsic edges for
each live cloth, soft body and rope. Rigid rope attachments come from the
installed rope state. The three deformable pairs come from public
`PmCouplingStateInfo` graph creation.

After one prediction per live domain and one common F8-10 spatial
build/query/publication, the schedule runs four rigid coupling sweeps and four
deformable coupling sweeps when their families are present. Both endpoints use
the same staged state and spatial epoch. Coupling performs no integration,
index build, host readback or wait. One commit publishes all participating
domains together.

Rigid/cloth, rigid/soft-body, rigid/rope and cloth/soft-body retain their
existing specialized contact generation and reciprocal application. The
frozen generic F8-17 entries resolve moving rope/rigid attachments. The frozen
generic F8-19 entries resolve cloth/rope and soft-body/rope endpoint
constraints; soft-body/rope scatters the reciprocal impulse through its
precomputed surface binding.

## Storage and failure behavior

Simulation creation reserves generation tables, 64 immutable constraint
records, proposal/impulse/correction scratch and rope/rigid reduction scratch.
Boundary staging clears all reciprocal force and correction streams. No tick
grows storage.

Registrations pin endpoint generations and graph revision. The existing public
graph lifecycle rejects invalid or stale replacement without replacing the
active graph. Coupling edge capacity is checked before device work. Telemetry
limits include boundary staging and every fixed coupling dispatch. A
one-below-limit mixed-family tick returns `PM_ERROR_CAPACITY_EXCEEDED`; cloth,
soft-body and rope public output stays byte-for-byte unchanged and the same
capacity remains reusable.

## Qualification

The public fixture installs cloth, soft-body and rope together, creates all
three public deformable edges in non-canonical input order, retains the three
intrinsic rigid contact edges, and reports seven edges, eight family sweeps,
one time advance, one spatial build/publication and zero intra-tick readbacks.
It separately verifies rigid/rope contact plus moving attachment as two edges
and four sweeps.

The protected F8-17 and F8-19 Slang closures are unchanged. Their recorded
physical CUDA quality gates cover every supported mode, reciprocal residuals,
thin crossing, 100:1 mass ratio, attachment separation, endpoint direction,
stale/rebound generations and atomic failure. This change rebuilt the current
CUDA and Vulkan production drivers, Android arm64 Vulkan artifact and Metal
bindings. The NVIDIA kernel driver and Pixel 7 were unavailable for a fresh
end-to-end physical run; that absence is recorded rather than presented as new
device execution.

Exact commands and hashes are in
[`evidence/F8-45-mechanical-couplings-production-ticks.json`](evidence/F8-45-mechanical-couplings-production-ticks.json).
