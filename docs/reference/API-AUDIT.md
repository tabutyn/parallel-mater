# Command API implementation audit

`api-coverage.json` is generated from `registry/pm.xml` and the loader entry
points. Generation fails when a declared command has no loader implementation
or forwarding route, or has more than one route. A routed command is not a
claim that every installed driver supports it; applications query each
physical device's features, limits, formats, and extensions.

The Slang/CUDA driver currently implements discovery, explicit memory and
buffer binding, command recording and transfer, queue submission, timeline
synchronization, fences, queries, memory budgets, limits, formats, native
buffer access, and the gallery's physics domains and couplings.

The Slang/D3D12 driver implements discovery, explicit memory and buffer
binding, transfer command recording, barriers, queue submission, fences,
memory budgets, limits, formats, and native handles. It also qualifies the
embedded shared AVBD DXIL schedule during device creation, but returns no
physics feature bits until the full command schedule is available.

Typed Vulkan functions have concrete validation entry points and explicitly
report unavailable capability when the selected device is not a Vulkan-imported
device. They are compiled when Vulkan headers are available.

The private CUDA reference runtime lives under `src/simulation/`. It is driver
implementation detail and is not installed as an SDK interface.
