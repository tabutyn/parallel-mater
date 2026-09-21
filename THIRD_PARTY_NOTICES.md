# Third-party notices

ParallelMater's installed physics library has no bundled third-party source.
The optional gallery fetches the following pinned development dependencies at
configure time:

- **cgltf** (`85cd62382dfea638278962690cf515023f33ed00`) is copyright Johannes
  Kuhlmann and contributors and is distributed under the MIT License.
- **NVIDIA OptiX development headers** (`v9.1.0`, commit
  `f1f6dd803f3159992d248178f6e09421c6eb8b6d`) are copyright NVIDIA Corporation
  and are distributed under the 3-Clause BSD License included in that source
  package. Use of the NVIDIA OptiX runtime remains subject to NVIDIA's own
  license.

The repository does not copy either dependency into its source tree. Their
notices and license texts are available in the exact revisions fetched by
CMake.
