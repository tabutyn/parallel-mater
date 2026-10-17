# Performance qualification

Parallel Mater qualifies performance per physical device, driver, operating
system, solver identity, scene revision, and output resolution. The gallery
selects only reviewed entries from `config/device-profiles.json`; an unknown
device uses the conservative default and is not treated as qualified.

The brick workload targets approximately 60 FPS outside difficult contact and
30 FPS during difficult contact. Record median and p95 frame time separately
for quiet and collision phases, keep physics quality fixed, and investigate a
repeatable regression greater than 10 percent before accepting it. A profile
must also pass scene stability and control checks.

The current runtime has one command API, one shared schedule, and one set of
Slang physics shaders. Backend names in profiles identify the qualified device
adapter, not an alternate physics implementation. New measurements must name
the source revision, Slang/compiler versions, driver, device, power mode,
resolution, scene settings, and solver identity.

Historical measurements and rejected experiments are preserved in
[the legacy performance log](history/PERFORMANCE-LEGACY.md). They are evidence
for their recorded revisions, not current build instructions or limits.

## Headless qualification gate

`benchmarks/qualification.py` is the independent performance consumer. It uses
only Python's standard library and the installed conformance evaluator; it has
no gallery, renderer, CUDA or private-solver dependency. Run it with a raw
benchmark measurement and the raw physical-quality result produced by the same
driver binary, device, source revision and toolchain:

```text
python3 benchmarks/qualification.py \
  --measurement /path/to/benchmark-measurement.json \
  --quality-result /path/to/core-quality-result.json \
  --report /path/to/qualified-benchmark.json
```

The measurement schema is `parallel-mater-benchmark-measurement/v1`. It records
a clean source revision; driver binary, device and toolchain identity; a hashed
workload revision; the quality profile/result hashes; and at least 20 samples.
For this gate, the `parallel-mater-conformance-quality-result/v1` input must
carry matching `source`, `toolchain`, and enriched `driver` fields. The driver
field includes `name`, `device`, `version`, and `binary_sha256`; this binds the
physical result to the measured binary instead of accepting a pass from another
build or adapter.
Each sample contains a positive integer duration and operation count plus these
public memory observations in bytes:

- application allocation;
- library device and host allocation;
- borrowed buffers;
- per-step scratch high-water usage.

Timing uses a host monotonic clock. Memory comes from
`pmGetDeviceMemoryBudget` and `PmStepStatistics.scratchHighWaterBytes`. The gate
computes aggregate operations/second and nearest-rank median/p95 time per
operation rather than trusting submitted summaries. It retains each memory
field's measured peak.

The output schema is `parallel-mater-benchmark-qualification-report/v1`.
Throughput, timing and memory objects each repeat the immutable quality-profile
SHA-256. A result is `qualified` only when the raw quality matrix passes and its
source, driver binary, device, toolchain, profile hash and result hash match the
benchmark. Failed, missing, unsupported or stale quality evidence emits an
`unqualified` report and a nonzero exit. Its diagnostic metrics are not a
publishable qualified score.
