# F8-77: Windows public completion notifications

Public fence and timeline waits now block on submission completion notifications
instead of 50-microsecond polling sleeps. The existing queue publishes its epoch
after native work drains, captures retire, completion status publishes and queue
bookkeeping completes. Observing the epoch before checking public predicates
prevents a completion between checking and blocking from being lost. A single
steady-clock deadline preserves finite timeout budgets across unrelated wakes.

Exact clean Release source: `ff4675be97511ab6d89296d78a72a9750369bd1a`, tree `03eb1c710d2499ef647eb3affe005c9f7c0bb785`.
All **146 registered tests pass with zero skips**, retaining every prior 144 test.
Focused queue/public/async regressions cover ownership retirement, notification
races, simultaneous waiters, ready/pending states, zero/finite/large/infinite
timeout, wait-any/all, timelines, validation/foreign handles and failed work.
The relocated SDK passes all 12 checks.
An independent consumer compiled only against relocated SDK headers/import library
also passes ordinary finite wake and finite-expiry checks in both modes.

Normal installed-public rigid-wall samples use unchanged counts, 600 warmup and
7,200 measured ticks, with no failed ticks and no discarded stalls:

| Mode | Before physics p95 ms | After physics p95 ms | Before GPU p95 ms | After GPU p95 ms |
| --- | ---: | ---: | ---: | ---: |
| plain | 16.0607 | 4.0197 | 6.0088 | 3.9357 |
| validation | 16.0499 | 4.0497 | 7.4650 | 3.9444 |

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

Reviewable report: [F8-77-WINDOWS-PUBLIC-COMPLETION-WAITS.json](F8-77-WINDOWS-PUBLIC-COMPLETION-WAITS.json). Raw samples, exact commands,
source diff, test logs/JUnit, SDK report/hashes, device/compiler/memory identities
and reproduction helpers: [F8-77-WINDOWS-PUBLIC-COMPLETION-WAITS.zip](F8-77-WINDOWS-PUBLIC-COMPLETION-WAITS.zip), SHA-256
`46fd0266f95223f515e04babcc85828d74585568be009c4deecad1910d5968b3`. Every archive member was hash-verified.

F8-78 still owns complete five-workload/five-negative-control acceptance at the
unchanged 10 ms physics p95, 16.667 ms frame p95 and 0.99 realtime-rate gates.
F8-74 waits for successful follow-up evidence before freezing a replacement.
Candidate v9 is unchanged; this is not final F8-26 qualification.
