# F8-77: Windows public completion notifications

Public fence and timeline waits now block on submission completion notifications
instead of 50-microsecond polling sleeps. The existing queue publishes its epoch
after native work drains, captures retire, completion status publishes and queue
bookkeeping completes. Observing the epoch before checking public predicates
prevents a completion between checking and blocking from being lost. A single
steady-clock deadline preserves finite timeout budgets across unrelated wakes.

Exact clean Release source: `701a3f12ef434381a5baa0b62a8f4a1473505a6a`, tree `5f8f8348ce258bffa380b16eb87f2e652255b0db`.
All **146 registered tests pass with zero skips**, retaining every prior 144 test.
Focused queue/public/async regressions cover ownership retirement, notification
races, simultaneous waiters, ready/pending states, zero/finite/large/infinite
timeout, wait-any/all, timelines, validation/foreign handles and failed work.
The relocated SDK passes all 12 checks.

The corrected harness explicitly permits no-device status only for optional
Metal/Vulkan/D3D12 cases. A valid device still executes every assertion; CUDA,
CPU and host cases stay strict. A local zero-device mock verifies optional
unavailability in both modes, while optional-flag executions on the real CUDA
device still run and pass the complete fixture. Windows DLL PATH setup now
appends rather than overwrites per-test mode settings; this fresh registration
records the actual validation set/unset operations alongside DLL lookup.
The original raw evidence remains preserved. This V2 report supersedes its
registered-test mode acceptance; standalone timing/validation captures were
unaffected by the earlier CTest metadata issue.
An independent consumer compiled only against relocated SDK headers/import library
also passes ordinary finite wake and finite-expiry checks in both modes.

Normal installed-public rigid-wall samples use unchanged counts, 600 warmup and
7,200 measured ticks, with no failed ticks and no discarded stalls:

| Mode | Before physics p95 ms | After physics p95 ms | Before GPU p95 ms | After GPU p95 ms |
| --- | ---: | ---: | ---: | ---: |
| plain | 16.0607 | 4.9832 | 6.0088 | 4.8906 |
| validation | 16.0499 | 4.4264 | 7.4650 | 4.2906 |

The baseline is the complete retained F8-76 capture, compiled at
`68e7d8f3af24c47d8d4d1da374d7ef600b9f04e2`. Its compiled-source identity
is distinct from the later claim checkout, which adds only documentation/task
metadata. The shorter consumer status/yield diagnostic is retained as historical
diagnosis and does not replace these normal full samples. Physics critical-path
time and native GPU time remain separate measurements; percentile differences
are not a measured pure-wait interval.

Only the adapter-binding protected closure changes. API/ABI, shared kernels,
runtime, fixed-tick schedule/telemetry, contracts, public oracle, work budgets and
Gallery closures are byte-identical to the accepted ownership repair. There is
no global timer/power change, busy-spin workaround, new GPU drain, physics fork,
workload reduction or raised budget. Existing worker dependency scheduling is
unchanged; this task replaces public completion polling.

Reviewable report: [F8-77-WINDOWS-PUBLIC-COMPLETION-WAITS-V2.json](F8-77-WINDOWS-PUBLIC-COMPLETION-WAITS-V2.json). Raw samples, exact commands,
source diff, test logs/JUnit, SDK report/hashes, device/compiler/memory identities
and reproduction helpers: [F8-77-WINDOWS-PUBLIC-COMPLETION-WAITS-V2.zip](F8-77-WINDOWS-PUBLIC-COMPLETION-WAITS-V2.zip), SHA-256
`a2eb3d1111b2a25279fbb39380f7af07fd96d4f9e8bdc28a21dbaa5f10b6bf43`. Every archive member was hash-verified.

F8-78 still owns complete five-workload/five-negative-control acceptance at the
unchanged 10 ms physics p95, 16.667 ms frame p95 and 0.99 realtime-rate gates.
F8-74 waits for successful follow-up evidence before freezing a replacement.
Candidate v9 is unchanged; this is not final F8-26 qualification.
