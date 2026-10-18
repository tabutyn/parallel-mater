# F4-07: borrowed device and queue handoff evidence

Date: 2026-10-18. Host: Windows 11 x64, Visual Studio 2022/MSVC
19.44.35228, Windows SDK 10.0.26100.0. Claim token
`4d3de35b-769c-4bdf-9358-099ddd6f2e89`.

## Contract

The common compute boundary now represents native device, queue, buffer and
completion ownership without native API types. Borrowed objects cannot carry a
release callback. Owned objects require one and release exactly once across
move, reset and destruction.

The common native-device and native-queue getters now return borrowed wrappers
instead of raw pointers. Metal uses those wrappers for its actual device and
queue exports, so the no-destroy rule is exercised by a production adapter and
is not only a mock convention.

Queue handoff identifies each native API, device and queue and passes an explicit
list of prior completions into command-list creation. Common validation rejects
null, unidentified, different-API and different-device priors before adapter
wait encoding.
Adapters may accept same-device cross-queue waits only when they can encode the
native synchronization primitive without a CPU wait. The submitted consumer
returns an explicit next completion and retains all prior completions through
its terminal state.

Metal reports its device/queue identity. Same-queue prior work uses native
`MTLCommandQueue` ordering; different-queue work fails explicitly until the
thin adapter supplies `MTLSharedEvent` handoff. No physics equation, Slang
schedule or public ABI changed.

The existing generated specification already defines the public mapping:
device, queue, buffer and timeline imports are borrowed; native getters do not
transfer ownership; interop capabilities advertise import support; timeline
wait/signal values provide prior/next ordering; buffer barriers transfer external
ownership; and D3D12 buffers enter and leave PM in `COMMON`. This implementation
therefore did not duplicate those declarations in a Windows physics path.

## Portable acceptance evidence

Configured and built the standalone solver suite:

```powershell
cmake -S tests/solver -B build-f4-03-portable
cmake --build build-f4-03-portable --config Release
ctest --test-dir build-f4-03-portable -C Release --output-on-failure
```

PASS: 11/11. The new `parallel-mater-compute-handoff-contract-tests` case proves:

- borrowed destruction invokes no release and owned move/destruction releases
  exactly once;
- wait encoding precedes consumer encoding/submission and never calls the prior
  completion's host wait;
- same-device/different-queue prior work reaches the adapter;
- null, unidentified, foreign-API/device and adapter-unsupported priors fail;
- the next completion carries the consumer queue identity; and
- dropping the caller's prior token does not release it until the next
  completion becomes terminal.

The existing argument, dependency, lifetime, memory-arena, owner-slot,
resource-plan, contact, AVBD and architecture tests passed in the same run.

## Windows adapter evidence

The repository D3D12 build compiled the common handoff suite together with the
shared Slang D3D12 driver. The focused Windows run passed 6/6: four common
compute contract/dependency/lifetime/handoff tests, the native D3D12 command API
test on the GTX 1080, and the architecture guard. The D3D12 case completed its
physical command path without adding a Windows-only handoff or physics path.

The four common tests were also rebuilt with MSVC `/W4 /WX /wd4458 /EHsc` and
passed 4/4. The new type-erased ownership and queue-handoff code introduces no
warning under the repository's existing exception and shadow-warning policy.
The handoff contract separately passed 100/100 repeated executions.

## Limits

This gate specifies and tests the common ownership/handoff primitives. It does
not claim native imported-device support in a driver that does not advertise it,
nor qualify full physics/renderer alternation. F4-08 owns the mock adapter with
simultaneous live frames, indirect resources, import failure and exactly-once
teardown/completion injection. Native cross-queue adapters remain responsible
for translating prior/next tokens into their API fence/event/timeline commands.
