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
