# F4-99: common-resource foundation exit review

Date: 2026-10-18. Review host: Windows 11 x64, NVIDIA GeForce GTX 1080,
Visual Studio 2022/MSVC 19.44.35228, Windows SDK 10.0.26100.0. Claim token
`6084a4df-5b34-4c87-885f-8ff1891fa1ce`.

## Exit-contract review

The original F4 exit requires a mock adapter to verify resource/dependency
behavior, native details to remain inside thin adapters, and engine handoff to
have a tested common contract. It does not claim universal native imports or an
engine plugin. The merged F4-01 through F4-08 task cards are all `done`, have
nonempty evidence and completed lease history, and cover the exit as follows:

- F4-01 gives every direct, record and indirect resource a device/allocation
  identity, typed access, checked half-open range and retained owner.
- F4-02 retains those owners through terminal completion, deferred failure and
  device teardown independently of caller tokens.
- F4-03 derives exact RAW, WAR and WAW ordering without a global or host wait.
- F4-04 uses common Slang reflection for Vulkan and D3D12 binding layouts and
  rejects unsupported or inconsistent ABI shapes.
- F4-05 distinguishes physical capacity, live budget provenance, tracked
  allocations and borrowed bytes without wrapping failed snapshots.
- F4-06 consumes fresh snapshots in a shared conservative policy, rejects
  unknown/malformed values and reserves borrowed bytes plus 30% headroom.
- F4-07 defines borrowed native objects and prior/next completion ownership
  without exposing native API types in the common scheduler.
- F4-08 holds six alternating physics/render submissions live in a portable
  engine mock, drops caller handles, rejects invalid imports and proves
  dependency-ordered, exactly-once teardown without host waits.

The review found no uncovered F4 leaf. Remaining work is already bounded:
P1-01 inventories every missing adapter/domain slice; P2-01 ports this contract
to one native handoff fixture; P2-02 through P2-04 build the engine plugins.
Physical/capability evidence and optimization remain F5/P1 and P3 work. Metal is
currently the only production adapter on the common command/completion surface,
and no reference driver advertises qualified imported-device/timeline handoff.
Those limits remain explicit in `docs/QUALIFICATION.md` and
`docs/COMMON-RESOURCE-BOUNDARY.md`.

## Current review validation

```text
PASS generated public API/reference check
PASS reflected ABI and DXIL binding-generator tests, 2/2
PASS common compute contract/dependency/lifetime/handoff tests, 4/4
PASS engine-independent handoff mock, 1/1
PASS loader budget mock and Gallery fake-resource admission tests, 2/2
PASS strict MSVC /W4 /WX portable suite, 12/12
PASS native D3D12 command API on NVIDIA GeForce GTX 1080, 61.22 s
PASS roadmap card/dependency/lock/history/index check, 169 cards
```

These results close the common F4 contract. They do not qualify native engine
interop, physical solver behavior, performance, or every adapter's adoption of
the common C++ execution surface.
