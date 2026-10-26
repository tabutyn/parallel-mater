# F8-55 shared BVH coupling and batching

F8-55 removes the remaining production coupling broad phases that ignored the
single fixed-tick spatial epoch. Geometry contact consumers now read the
generation-tagged candidate lists published once before the four coupling
sweeps. Explicit attachments remain identity-addressed, and regular-grid
stencils remain grids.

## Production interaction map

| Pair / mode | Search and candidate path | Dispatch path |
|---|---|---|
| Rigid/rigid contact | Existing rigid proxies and common BVH candidates | Existing batched rigid manifold path |
| Rigid/cloth contact | Cloth vertex queries against rigid body proxies; mesh BVH is narrow phase only | Candidate filter, cloth correction, batched body correction |
| Rigid/soft contact | Soft node queries against rigid body proxies; generation and topology checked | Candidate filter, node correction, batched body correction |
| Rigid/rope contact | Rope node queries against rigid body proxies | Candidate filter and batched node/body work |
| Liquid/rigid contact and paint | Liquid point queries against rigid body proxies; mesh BVH is narrow phase only | One candidate list reused by every rigid sweep; private body hash removed |
| Liquid/liquid pressure, viscosity and phase exchange | Existing bounded particle cell-neighbor stencil | Existing liquid pair batches; no unnecessary BVH query |
| Liquid/cloth contact and containment | Liquid point queries against cloth triangle leaves | Candidate contact/containment passes and cloth force apply |
| Liquid/rope contact and drag | Liquid point queries against rope segment leaves | Candidate contact pass and rope apply |
| Liquid/soft contact and recovery | Liquid point queries against soft surface triangle leaves | Candidate detect/recover/solve/apply; private per-edge surface refit removed |
| Liquid/smoke drag, thermal and phase | Particle cell neighbors or smoke-grid sampling | Existing regular-grid/particle stencil |
| Soft/cloth mechanical contact | Soft node queries against cloth triangle leaves | Candidate detect and reciprocal soft/cloth apply |
| Rope/soft attachment | Declared rope endpoint plus declared soft triangle/barycentric identity | Compatible endpoint records staged together; plural sample/scatter/apply calls |
| Rope/cloth attachment | Declared rope endpoint plus declared cloth vertex identity | Both compatible endpoints share one sample/project/apply sequence |
| Smoke/rigid contact | Smoke point queries against rigid proxies; body mesh BVH remains narrow phase | Candidate-gated contact plus batched impulse apply |
| Smoke/cloth contact or wind | Smoke point queries against cloth triangle leaves; grid smoke samples its grid | Candidate contact/wind passes or documented grid sampling |
| Smoke/rope contact or wind | Smoke point queries against rope segment leaves; grid smoke samples its grid | Candidate contact/wind passes or documented grid sampling |
| Smoke/soft contact or wind | Smoke point queries against soft surface triangle leaves; grid smoke samples its grid | Candidate contact/wind passes or documented grid sampling |

`PM_COUPLING_SMOKE_SMOKE`, `PM_COUPLING_SOFT_BODY_SOFT_BODY`,
`PM_COUPLING_CLOTH_CLOTH`, and `PM_COUPLING_ROPE_ROPE` remain reserved and
return `PM_ERROR_FEATURE_NOT_PRESENT`; they are not production consumers.

## Candidate contract

Rigid bodies, cloth triangles, soft-body surface triangles, rope segments and
point leaves are published with family, owner slot/generation, topology
revision, primitive kind/slot/generation, query mask and admission state.
Cloth vertices, soft nodes, rope nodes, liquids and smoke particles emit
separate swept query envelopes. The envelope covers previous/current position,
the interaction support radius and the fixed-tick correction allowance.

Consumers reject query overflow, invalid admission and stale identity rather
than truncating or accepting a replaced endpoint. A randomized host oracle
compares swept point/triangle candidates with brute force, including thin
crossings, sparse resources and mixed families. CUDA tests cover mixed leaves,
query-only points, a soft/cloth hit and stale-generation rejection.

Exactly one `build_spatial_index_device` call remains in the production tick.
Its publication is reused by all four coupling sweeps. The removed paths are
the fluid/rigid body grid, fluid/soft private refit and unconditional
soft-node/all-cloth-triangle scan. Mesh-local BVHs remain valid narrow phases
after the common body candidate gate.

## Batching and independent gates

Rigid/deformable attachment records are constraint-aligned arrays. Rigid/rope,
rope/cloth and rope/soft endpoints are grouped by compatible owner/mode before
dispatch. Adding another compatible endpoint increases record/device work; it
does not reproduce the complete host launch sequence.

The public mixed fixture freezes limits outside `native_realtime_limits`:

- total: at most 512 dispatches, 640 workgroups, 65,536 moved bytes and 768 barriers;
- candidate phase: at most 16 dispatches and 96 workgroups;
- coupling phase: at most 288 dispatches and 288 workgroups;
- exactly one spatial build, one publication and zero intra-tick host readbacks.

The pre-change scaling rejected by source tests is:

- soft/cloth detection performs `soft_nodes * cloth_triangles * 4` inner
  triangle tests per tick;
- fluid/soft performs five launches per sweep and rebuilds/refits its private
  surface tree; the shared path performs four and consumes the common epoch;
- liquid/rigid performs two private body-grid setup launches per sweep;
- rigid/rope and rope/cloth attachments perform three launches per endpoint
  per sweep; compatible endpoints now share three launches;
- rope/soft performs seven launches for every record per sweep; the staged
  path has a fixed compatible batch plus record-scaled device work.

Architecture negative controls fail if the retired production calls, exhaustive
loops or singular per-record helpers return. The limits above are intentionally
independent of the runtime admission formula. F8-56 retains the matched
installed Mac/iPad before/after capture obligation.

## Verification

On Windows 11 build 26200, NVIDIA GeForce GTX 1080, driver 582.66, CUDA 12.9,
Slang 2026.18, the installed public CUDA fixture reported:

```text
dispatches=467 workgroups=530 bytes=37843 barriers=628
candidate-dispatches=11 candidate-workgroups=59
coupling-dispatches=259 coupling-workgroups=259
builds=1 publications=1
```

The focused host/CUDA suite passed 8/8. The same public coupling fixture passed
through D3D12 on the GTX 1080. DXIL compilation, Metal generation, CUDA mirror
validation and Vulkan SPIR-V mirror validation all passed. Full installed
Gallery and Apple-device performance comparison remains F8-56.
