# Runtime evidence

- API contract build: passed.
- Real Blender re-export and loader contract: passed for all authored scenes.
- GPU rigid constraint solver suite: passed all eight types, breaking, stale
  handles, body retention, and live enable/disable.
- GPU scene load/instantiate/30-frame step: passed all eight scenes.
- OptiX headless 120-frame render: passed all eight scenes.
- Complete CUDA/OptiX CTest suite: 81/81 passed in 417.25 seconds.

The adjacent render files are captured from the actual gallery executable.
Formal Form and Runtime approval remain pending reviewer inspection; this
package records evidence without self-approving those gates.
