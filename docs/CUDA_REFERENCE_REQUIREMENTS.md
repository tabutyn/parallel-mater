# Driver qualification reference

CUDA is the first qualified device adapter, not a separate physics API or
solver. Every adapter runs the same cases through `parallel-mater-conformance`,
which creates gallery scene data and submits it exclusively through the public
`pm*` command API.

For a qualification capture, use a clean source revision and run all 30 cases.
Retain the runner provenance, raw results, comparison report, exact assets,
toolchain and driver identity, and any first-divergence trace. Never refresh
goldens merely to accept a mismatch. Gallery validation separately covers all
canonical scenes, controls, reset and scene switching, debug views, and the
complete Tab catalog.

The retained 2026-10-06 diagnostic package remains under
`conformance/v1/references/cuda-20261006`. Historical CUDA-versus-Metal porting
notes and obsolete native-backend instructions are preserved in
[the legacy requirements](history/CUDA-REFERENCE-REQUIREMENTS-LEGACY.md).
