# Rainbow Perimeter Lab – GPU Phase

This repo now ships a dual-path renderer that keeps the original CPU reference implementation as a debugging baseline while elevating the default path to a WebGL2 pipeline. The notes below capture the architecture, shader math, worker handshakes, and the manual regression steps that gate release sign‑off.

## Architecture Overview

| Layer                      | CPU reference                                                                       | GPU pipeline                                                                                                                                            |
| -------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Kuramoto evolution         | `stepKuramotoState` + `deriveKuramotoFieldsCore` in `App.tsx` / `kuramotoCore.ts`   | Worker-driven OA step writing float buffers (`kuramotoWorker.ts`) that are uploaded to textures each frame                                              |
| Rim + wallpaper compositor | CPU loop in `renderFrameCore` (per-pixel LMS offsets, wallpaper sampling, coupling) | Fragment shader in `gpuRenderer.ts` mirrors the same math: normal-aligned offsets, comb filters, chirality mixing, wallpaper warp, and surface coupling |
| Frame transport            | Canvas 2D `putImageData`                                                            | Full-screen quad, RGBA + RGBA32F textures, `texelFetch` in shader                                                                                       |
| Diagnostics                | `drawFrameCpu`                                                                      | `drawFrameGpu` (default)                                                                                                                                |

The GPU path consumes three textures:

1. Base photo (RGBA8)
2. Precomputed edge field (`gx`, `gy`, `|∇I|`, padding) in `RGBA32F`
3. Kuramoto-derived field (`gradX`, `gradY`, vorticity, coherence) in `RGBA32F`

All textures and shader math keep a **top-left origin** to match the CPU path; the shader samples with integer texcoords (via `texelFetch`) so parity checks stay exact.

## Worker Protocol (OA simulation)

`src/kuramotoWorker.ts` listens for:

| Message        | Payload                                                  | Effect                                                                        |
| -------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `init`         | `{ width, height, params, qInit, buffers, seed }`        | Seeds state slices & derived buffer views                                     |
| `tick`         | `{ dt, timestamp, frameId }`                             | Evolves state, computes derived field, posts `frame` with transferable buffer |
| `updateParams` | `{ params }`                                             | Hot-swaps OA coefficients                                                     |
| `simulate`     | `{ frameCount, dt, params, width, height, qInit, seed }` | Batch-simulates frames for regression harness                                 |
| `reset`        | `{ qInit }`                                              | Reinitialises the field with the requested twist                              |

Frames are recycled with `postMessage({ kind: "frame", buffer, ... }, [buffer])` to keep GC pressure off.

## Shader Notes (`src/gpuRenderer.ts`)

- Vertex shader maps clip-space quad → `[0, width) × [0, height)` coordinates; fragment shader uses `texelFetch` to preserve parity with the CPU bilinear sampling.
- LMS rim pipeline is ported straight into GLSL: Gaussian offsets, crystalline comb (`uKernelK0`, `uKernelQ`), chirality windup, and surface coupling gates.
- Wallpaper fallback is CPU-generated vectors piped as uniforms; when the Kuramoto field is active the shader swaps to reading gradients from the float texture.
- Rim/warp masks reuse the CPU thresholds so any mismatch shows up immediately in parity tests.
- Orientation: texture rows are consumed without any `UNPACK_FLIP_Y_WEBGL`; sampling is done with the same top-left origin as the CPU path.

## Diagnostics & Toggles

UI controls (left panel):

- **Use GPU renderer** – live switch between CPU canvas and WebGL path. CPU mode keeps the worker hot but bypasses the shader, which is useful when QA spots visual drift.
- **Telemetry logging** – enables per-phase timers; anomalies (e.g. `renderGpu` > 10 ms) emit `console.warn` once per second until they drop back under threshold.
- **Run GPU parity check** – executes the 3-scene regression harness and reports worst mismatch vs the 0.5 % tolerance.
- **Measure render throughput (120 frames)** – gathers average CPU/GPU frame cost and prints the throughput gain. (Set the canvas to 1000×1000 with OA + Surface enabled before running to match the acceptance target.)

Developer hooks (`window` globals):

| Function                                        | Purpose                                                              |
| ----------------------------------------------- | -------------------------------------------------------------------- |
| `__setRenderBackend("gpu" \| "cpu")`            | Programmatic toggle (uses the same cleanup/reupload path as the UI)  |
| `__setTelemetryEnabled(true/false)`             | Master switch for anomaly logging                                    |
| `__getTelemetryHistory()`                       | Returns the rolling 6‑second telemetry buffer (phase, ms, timestamp) |
| `__runFrameRegression(frameCount?)`             | CPU vs worker sanity sweep                                           |
| `__runGpuParityCheck()`                         | GPU vs CPU image comparison (same as UI button)                      |
| `__measureRenderPerformance(frameCount?)`       | Returns `cpuMs`, `gpuMs`, fps, and throughput gain                   |
| `__setFrameProfiler(enabled, samples?, label?)` | Collects raw canvas frame timings                                    |

## Behind the Scenes – QRI Motivation

Phase 5 leans into the Qualia Research Institute goal of making psychedelic-style
visualisations _legible_. The new hyperbolic ruler overlay is more than eye
candy: it exposes the atlas’ hyperbolic arc length right next to the Euclidean
pixel distance so researchers can document how curvature manipulations reshape
perceptual space. The toggle/slider pair is fully preset-aware, meaning QRI
playbooks can now ship “explainer” presets that carry both the curvature warp
and the teaching overlay into exported captures. This keeps the Tactile
Visualizer narrative—clarity, measurability, and reproducibility—front and
centre while still letting artists disable the guide when they just want the
look.

## Manual Regression Checklist

1. `npm run build` – ensures TypeScript + shader strings compile.
2. Launch dev server, upload a test photo, toggle **Use GPU renderer** on.
3. Run **Run GPU parity check** – expect each scene `percent` ≤ 0.5 % and `maxDelta` ≤ 1.
4. Set canvas to 1000×1000, enable OA + Surface, run **Measure render throughput** – expect ≥ 5× `throughputGain`.
5. Optionally toggle to CPU, confirm visuals match, then back to GPU.
6. Review console for `[telemetry]` warnings; none should persist under normal presets.

Document decisions & deltas in QA sign-off:

- Worst-case parity % / delta
- GPU vs CPU frame cost
- Any telemetry warnings observed

## File Guide

- `src/App.tsx` – runtime orchestration, UI, telemetry, and diagnostic toggles.
- `src/gpuRenderer.ts` – shaders + texture management (WebGL2).
- `src/kuramotoWorker.ts` – OA evolution + derived fields in a dedicated thread.
- `overview.md` – in-depth description of the rim/wallpaper pipeline and controls.

Happy rendering! For further tweaks, the shader uniforms are intentionally kept 1:1 with the CPU code so parity stays tractable. Pull the diagnostic toggles when experimenting—they exist so regression data is always one click away.
