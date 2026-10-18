# F4-08: engine handoff mock evidence

Date: 2026-10-18. Host: Windows 11 x64, Visual Studio 2022/MSVC
19.44.35228, Windows SDK 10.0.26100.0. Claim token
`b3e36088-206c-4567-a665-5ef2ac2af7e1`.

## Adapter and scenario

The portable test adapter imports borrowed, backend-scoped mock device, queue
and buffer objects through the F4-07 common ownership boundary. The mock engine
keeps sole ownership of one device, separate physics and renderer queues, and
shared scene/output buffers. The adapter creates PM-side buffer wrappers and
encodes cross-queue dependency jobs; it has no backend API types or physics
implementation.

The qualified scenario records three complete frames without completing GPU
work. Physics writes the scene buffer and renderer copies it to the output. Each
renderer waits for its physics completion and the next physics submission waits
for the prior renderer, producing:

```text
physics 0 -> renderer 0 -> physics 1 -> renderer 1 -> physics 2 -> renderer 2
```

All six submissions remain live simultaneously. The caller drops every physics
and renderer token. Common queue ownership and prior/next retention keep the
chain alive with zero host waits. Adapter teardown drains the native jobs in
dependency order, signals six jobs once, invokes six native completion waits,
destroys six completion tokens, and releases both imported wrappers once.

## Failure and ownership evidence

Before accepting work, the adapter rejects:

- a queue belonging to another device;
- a native object from another API domain;
- a queue object passed to buffer import; and
- a buffer belonging to another device.

These failures create no imported wrapper and accept no submission. Failed
construction, ordinary adapter teardown and completion teardown destroy no
borrowed native object. After the adapter is gone, the engine explicitly
destroys its three buffers, three queues and two devices exactly once.

## Validation

Configured and built the standalone solver suite, then ran all portable tests:

```powershell
cmake -S tests/solver -B build-f4-03-portable
cmake --build build-f4-03-portable --config Release
ctest --test-dir build-f4-03-portable -C Release --output-on-failure
```

PASS: 12/12, including common argument, dependency, lifetime, handoff, mock
engine, resource-plan, AVBD and architecture coverage.

The mock target was also compiled under MSVC `/W4 /WX /wd4458 /EHsc` and passed
100/100 repeated executions.

## Limits

The mock qualifies common ownership, ordering and failure semantics. It does not
claim native imported-device/timeline support, native cross-queue fence/event
translation, physical solver quality or performance. Those remain backend and
physical-evidence gates.
