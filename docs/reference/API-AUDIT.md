# Command API implementation audit

`api-coverage.json` is generated from the same declaration metadata as the
headers and from definitions found in `src/api/`. Generation fails when a
declared command has no definition or has more than one definition.

The CUDA command-runtime foundation currently implements instance and device
discovery, explicit memory and buffer binding, command recording and transfer,
queue submission, timeline synchronization, fences, queries, memory budgets,
limits, formats, and CUDA native-buffer access. Physics commands are present but
remain capability-gated until their shared Slang schedules are connected. The
device therefore does not advertise physics features prematurely.

Typed Vulkan functions have concrete validation entry points and explicitly
report unavailable capability when the selected device is not a Vulkan-imported
device. They are compiled when Vulkan headers are available.

The legacy inventory is retained only as a deletion checklist. It is not part
of the public API and will disappear with the backend-specific `World` APIs
after the CUDA conformance and gallery gates pass.
