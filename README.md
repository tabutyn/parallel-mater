# ParallelMater

ParallelMater is a planned MIT-licensed CUDA C++ physics library. The first
implementation milestone couples particle fluid with analytic rigid bodies;
later solvers will be added only after the small public API is proven by the
gallery application.

This branch is an **API review RFC**. It contains exact C++ declarations,
contracts, and implementation stages, but intentionally contains no physics
implementation yet.

## Design goals

- One owning `World` coordinates simulation and cross-system coupling.
- Fluid and rigid-body resources use stable, generation-checked handles.
- One `step` call advances a complete fixed frame; applications do not invoke
  solver-internal phases.
- CUDA allocations remain owned by the library while renderers borrow explicit
  device views.
- Synchronous convenience calls and stream-ordered asynchronous calls share the
  same semantics.
- The gallery is simultaneously the example suite, visual regression surface,
  and basis of a small progression game.
- Rendering, OptiX, scene objectives, authored controls, and campaign state are
  application code—not physics API concepts.

Start with [the proposed API](docs/API.md), then review
[the gallery boundary](docs/GALLERY.md) and [the staged implementation
plan](docs/ROADMAP.md).

## Configure the proposal checks

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The current test validates header shape only. Passing it does not imply that a
physics implementation exists.
