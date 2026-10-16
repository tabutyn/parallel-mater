# Contributing to Parallel Mater

The code and specification use the repository's MIT license. Contributions must
retain dependency licenses and establish rights to any imported source/assets.
The current project owns its specification and extension namespace. Future
standards-group adoption is a goal, not current endorsement or governance.

## Claiming roadmap work

Roadmap work uses one leaf task per reservation and completion PR. Read the
[lease workflow](docs/roadmap/WORKFLOW.md), push a claim-only PR, and wait for its
merge before implementation. The task's lease token on current main establishes
ownership. Renew, release, reclaim and complete through the documented PR flow;
do not rely on an open PR or an expired reservation. The generated index and
machine-readable cards let other machines see the same work queue.

## Change ownership

`registry/pm.xml` defines the public command contract. Change it before generated
headers/dispatch, document normative behavior and add public-API positive and
negative evidence. Shared physical equations belong in Slang. Native adapters
own device services and execution details; engines and Gallery remain clients.
Do not introduce private solver types into the C SDK or portable conformance.

For a new command, structure, enum or extension, reserve the name/value in a
reviewed registry change before relying on it. Use the project namespace for
project features and an explicitly allocated vendor namespace for vendor
extensions. Document requirements, feature discovery, unsupported behavior,
ownership, threading, pending-operation lifetime, errors, version negotiation
and tests. Do not borrow another standards organization's namespace. Experimental
features stay unadvertised until those contracts and their evidence exist.

## Compatibility and review

Treat public layout or behavior changes as compatibility changes, including
parameter units/defaults and deferred failure behavior. State the affected API,
driver and layer versions separately. Never reinterpret an existing field to
mean a different quantity. Use explicit negotiation and test old/new and invalid
requests against an independently built implementation. Follow the pre-1.0
[version policy](docs/SDK-RELEASES.md).

Each change should state its observable trigger/result and the evidence that
tests it. Keep contract tests independent of the implementation being tested;
negative controls must establish which layer rejects an invalid operation.
Run the generated-contract and installed-SDK gates for API/package changes and
the affected lifecycle/physics cases for runtime changes. Preserve original
failure data. A tolerance or golden change requires separate physical reasoning
and review; performance gains never justify reduced physical quality.

A release reviewer checks the exact source/toolchain/device tuple, complete
case scope, unsupported outcomes and unresolved failures. Authors cannot turn
execution into qualification by renaming a report. Record limitations with the
capability claim; mixed-family and distinct fluid–fluid systems each need their
own lifecycle and physical evidence. Publish benchmark timings only alongside
the fixed quality gate and its result.

## Worktrees and concurrent work

Use HTTPS remotes on workstations. Inspect local/submodule state before merging
or pushing; never stage, reset or delete another worktree's unrelated changes.
Preserve dirty legacy work with its source revision, binary patch, complete
working files and verified hashes before reconciliation. Port a behavior with
its regression into the current architecture, rather than reviving retired
runtime/compiler shims. Retire a checkout only when its unique work has a recorded
disposition and no active toolchain depends on it.
