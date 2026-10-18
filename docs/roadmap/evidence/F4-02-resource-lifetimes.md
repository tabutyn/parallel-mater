# F4-02: common command/completion lifetime evidence

Date: 2026-10-18. Host: Thomass-MacBook-Air, Apple M4, macOS 26.5.2
(25F84), Apple Clang 21.0.0 (clang-2100.1.1.101), arm64.
Claim: PR #232, token `e74e6bc9-e94d-447c-a85a-0d323438b8e0`.
Base: `236158f` (merged claim after F4-01 completion #231).
Revalidated after integrating main's Vulkan smoke work (`11049bc`) and the
subsequent claim-only update (`c287000`).

## Acceptance evidence

`tests/solver/compute_lifetime_tests.cpp` supplies a non-owning mock encoder and
completion implementation. Weak references prove the shared layer, rather than
the adapter, owns direct arguments, by-value record pointers, indirect bindings,
kernels, copy/fill allocations and upload staging buffers.

Verified cases:

- Client buffers, arguments, command lists and completion tokens may all be
  dropped while work remains pending. Dropping them does not wait or free owners.
- Poll/wait reclaims terminal owners; timestamps and native completion handles
  remain usable. Early timestamp requests do not release pending allocations.
- Two submissions retain the same buffer until both finish.
- Unsubmitted list destruction, native submit allocation failure, missing native
  completion and partial encoding failure cancel without enqueueing or leaking.
- Transfer validation rejects bad/overflowing/overlapping ranges while keeping
  prior valid commands executable. Double submit and post-submit recording fail.
- Deferred execution errors release owners only when terminal. Device teardown
  drains later work even after an earlier completion reports an error.
- A token remains usable after device teardown; an unsubmitted list surviving
  teardown cannot enqueue. Empty submissions have valid completions.
- A condition-variable-gated native wait runs on another thread. Polling,
  collection, new recording and new submission all return before that wait is
  released: no incidental GPU wait enters the queue's collection path.
- The real Metal host scheduler is also exercised with an injected-failure
  mock device. Encoding and submit failures discard canceled lists; the next
  operation succeeds with a fresh batch. Ordinary validation failures preserve
  prior valid commands, and public-service errors do not poison sticky launch
  errors. The scheduler allocates its pending-token slot before acceptance.

The scheduler regression initially failed with `scheduler retained cancelled
command after failed submit`. It passes after the scheduler checks local
recording state and discards canceled lists. No tolerance or expected output was
relaxed to turn this failure into a pass.

## Commands and observed results

```sh
cmake --build build-gallery-portable-verify --target parallel-mater-compute-contract-tests parallel-mater-compute-lifetime-tests parallel-mater-metal-schedule-recovery-tests -j 8
ctest --test-dir build-gallery-portable-verify -R '^parallel-mater-(compute-(contract|lifetime)|metal-schedule-recovery)-tests$' --repeat until-fail:20 --output-on-failure
```

PASS: all three tests passed 20 repetitions (60 executions), including the
deterministic concurrent-wait case and the real scheduler's failure recovery.

```sh
cmake -S tests/solver -B /private/tmp/parallel-mater-f4-02-sanitizers.ZMAjju/build-asan -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -g -Wall -Wextra' -DCMAKE_BUILD_TYPE=Debug
cmake --build /private/tmp/parallel-mater-f4-02-sanitizers.ZMAjju/build-asan --target parallel-mater-compute-contract-tests parallel-mater-compute-lifetime-tests parallel-mater-metal-schedule-recovery-tests -j 8
ctest --test-dir /private/tmp/parallel-mater-f4-02-sanitizers.ZMAjju/build-asan -R '^parallel-mater-(compute-(contract|lifetime)|metal-schedule-recovery)-tests$' --output-on-failure
cmake -S tests/solver -B /private/tmp/parallel-mater-f4-02-sanitizers.ZMAjju/build-tsan -DCMAKE_CXX_FLAGS='-fsanitize=thread -g -Wall -Wextra' -DCMAKE_BUILD_TYPE=Debug
cmake --build /private/tmp/parallel-mater-f4-02-sanitizers.ZMAjju/build-tsan --target parallel-mater-compute-contract-tests parallel-mater-compute-lifetime-tests parallel-mater-metal-schedule-recovery-tests -j 8
ctest --test-dir /private/tmp/parallel-mater-f4-02-sanitizers.ZMAjju/build-tsan -R '^parallel-mater-(compute-(contract|lifetime)|metal-schedule-recovery)-tests$' --output-on-failure
```

PASS: warning-enabled sanitizer builds and 3/3 tests in each configuration; no
AddressSanitizer, UndefinedBehaviorSanitizer or ThreadSanitizer diagnostics. The temporary output
directory was created with `mktemp -d`; use a fresh temporary directory to repeat.

```sh
cmake --build build-macos-metal-preview --target parallel-mater-driver-slang-metal parallel-mater-slang-metal-kernel-tests -j 8
cmake --build build-macos-metal-preview --target parallel-mater-slang-metal-command-tests parallel-mater-slang-metal-async-tests -j 8
ctest --test-dir build-macos-metal-preview -R '^parallel-mater-slang-metal-(kernel|command|async)-tests$' --output-on-failure
python3 -B tests/solver/check_slang_architecture.py
```

PASS: native driver builds; 3/3 Apple-GPU tests pass, no skips. Kernel tests
include dropped-token copy/dispatch resources and a new device-teardown test:
enqueue private-buffer fill/copy, drop the token, destroy the device, then verify
all 64 host-visible output bytes. A surviving command list rejects submission
after device destruction. Public command and asynchronous submission tests also
pass. The Slang architecture check reports no native physics kernels or direct
physics launches in the runtime.

## Limits

This qualifies shared ownership behavior and its Metal integration, not physical
solver quality, performance, external engine integration, or borrowed device/
queue imports. Other adapter execution interfaces still need migration. Idle
queues may retain dropped-token resources until the next collection or teardown.
The portable test is in the standard shader-portability CI matrix; local results
above are macOS/Apple-GPU evidence, not a substitute for Windows/Linux CI or
physical-device qualification on other GPUs/iPad.
