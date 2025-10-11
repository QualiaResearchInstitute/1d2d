# Hyperbolic Atlas Data Model

The CPU compositor now exposes a reusable atlas describing the hyperbolic
coordinate system applied during curvature warps. The atlas is generated once
per `(width, height, mode, curvatureStrength)` tuple and reused by both the CPU
pipeline and developer tooling.

## CPU Structure

`createHyperbolicAtlas({ width, height, curvatureStrength, mode })` returns:

| Field         | Shape                              | Description                                                                                              |
| ------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------- | ------ | ------------------ |
| `coords`      | `Float32Array[width * height * 2]` | Euclidean sample positions `(x, y)` for the warped pixel.                                                |
| `polar`       | `Float32Array[width * height * 2]` | Hyperbolic polar coordinates `(r, θ)` where `r = 2·scale·artanh(ρ)` and `θ = atan2`.                     |
| `jacobians`   | `Float32Array[width * height * 4]` | Jacobian of the disk projection w.r.t. image-space `(x, y)`. Layout: `[∂x'/∂x, ∂x'/∂y, ∂y'/∂x, ∂y'/∂y]`. |
| `areaWeights` | `Float32Array[width * height]`     | Hyperbolic area measure per pixel (`                                                                     | det(J) | · 4 / (1 - ρ²)²`). |
| `metadata`    | `HyperbolicAtlasMetadata`          | Scalar parameters (`curvatureScale`, `diskLimit`, centre, etc.).                                         |

All arrays are dense `Float32Array`s using row-major ordering (`y * width + x`).

## GPU Packaging

`packageHyperbolicAtlasForGpu(atlas)` emits a single `Float32Array` (`stride = 9`)
with per-pixel records:

```
[ sampleX, sampleY, radius, theta,
  jacobianXX, jacobianXY, jacobianYX, jacobianYY,
  hyperbolicAreaWeight ]
```

Recommended WebGL2 upload strategy (minimises format conversion while staying
within `RGBA32F`/`R32F`):

1. Texture A (`RGBA32F`) — pack `sampleX`, `sampleY`, `radius`, `theta`.
2. Texture B (`RGBA32F`) — pack `jacobianXX`, `jacobianXY`, `jacobianYX`, `jacobianYY`.
3. Texture C (`R32F`) — pack `hyperbolicAreaWeight`.

WebGPU/compute can alternatively consume the raw buffer via SSBO / storage
texture using the provided stride metadata.

The renderer stores the most recent GPU package (`GpuRenderer#setHyperbolicAtlas`)
so downstream passes or developer probes can serialise or upload the data
without re-running the CPU generator.

## Coordinate Guarantees

- Radial lines remain monotonic over the interior of the disk.
- Hyperbolic area of a disk of radius `r` grows ~`exp(r)`; the precomputed
  `areaWeights` encode this scaling per texel.
- The atlas respects both Poincaré and Klein projections; the metadata specifies
  `mode` so clients can reconstruct the mapping analytically if required.

## Educative Ruler Overlay

Phase 5 introduces an optional “hyperbolic ruler” overlay in the UI. When enabled,
the app reuses the atlas to render a geodesic axis and labelled tick marks:

- The guide walks a 45° geodesic (`θ = -π/4`) so the annotations stay clear of the
  most common crop aspect ratios.
- Tick radii are sampled in hyperbolic units (`σ = 2·scale·artanh(ρ)`); each tick
  reports both the hyperbolic arc length and its Euclidean pixel distance from the
  atlas centre. This keeps the overlay educational for users who are new to
  curvature-driven warps.
- A runtime slider (`hyperbolicGuideSpacing`, persisted in presets/exports) chooses
  the spacing between tick marks. The value is clamped to `[0.25, 2.5]` hyper units
  so the guide remains legible across 720p–4K canvases.
- If curvature strength drops to zero the atlas is discarded and the overlay
  automatically disables itself, matching the runtime toggle and exported preset
  payload.

Because the overlay reuses the atlas metadata rather than duplicating curvature
maths, it adds no extra CPU cost and stays perfectly in sync with whichever
projection (Poincaré or Klein) the renderer is using.
