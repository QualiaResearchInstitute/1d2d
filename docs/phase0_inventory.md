# Phase 0 Inventory

## Snapshot

- Canonical preset: `Rainbow Rims + DMT Kernel Effects` (`App.tsx:34-72`) drives both CPU and GPU renderers.
- CPU compositor: `renderFrameCore` (`App.tsx:1005-1356`) renders into a `Uint8ClampedArray` via sampled edge data and optional Kuramoto fields.
- GPU path: `drawFrameGpu` (`App.tsx:1420-1633`) streams the same buffers into `createGpuRenderer` (`gpuRenderer.ts:471-632`) and draws a full-screen quad in WebGL2.
- Kuramoto support exists in two flavors: synchronous CPU refs (`App.tsx:629-798`) and a WebWorker mirror (`kuramotoWorker.ts:1-200`) that feeds derived fields back to the UI.

## 1.5D Edge Rim Path

### CPU data products

| Data product                    | Producer (ref)                                                | Format / dims                                                          | Consumers                                                                          | Notes                                                                 |
| ------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Base pixels (`basePixelsRef`)   | Upload handler (`App.tsx:2408-2439`)                          | `ImageData` (`Uint8ClampedArray` RGBA) sized to current canvas         | CPU compositor (`renderFrameCore`); GPU static upload (`refreshGpuStaticTextures`) | Holds the canonical source frame used for rim compositing.            |
| Edge field (`edgeDataRef`)      | Sobel pass in upload handler (`App.tsx:2441-2474`)            | `EdgeField` with `Float32Array` buffers (`gx`,`gy`,`mag`)              | CPU compositor, GPU edge texture upload                                            | `mag` normalized to [0,1] for thresholds.                             |
| Frame buffer (`frameBufferRef`) | Allocated in `getFrameBuffer` (`App.tsx:632-672`)             | Recycled `ImageData` + `Uint8ClampedArray` backing store               | CPU renderer output + CPU preview                                                  | Ensures CPU path writes into a stable buffer for diagnostics/capture. |
| Rim metrics cache               | `normTargetRef` / `lastObsRef` (`App.tsx:538-540`)            | scalar refs                                                            | CPU + GPU frame gain calculation                                                   | Tracks average rim energy for normalization.                          |
| Kuramoto overlays (optional)    | Derived arrays set in `ensureKurCpuState` (`App.tsx:629-652`) | `Float32Array` views over `ArrayBuffer` (`gradX`,`gradY`,`vort`,`coh`) | CPU compositor rim/warp coupling                                                   | Populated either locally or via worker.                               |

### GPU staging

| Texture / buffer | Loader (ref)                                                             | Layout                                                    | Downstream                                       | Notes                                                        |
| ---------------- | ------------------------------------------------------------------------ | --------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| `uBaseTex`       | `uploadBase` (`gpuRenderer.ts:523-537`)                                  | RGBA8, matches `basePixelsRef`                            | Sampled in fragment shader for base color        | Bound to texture unit 0.                                     |
| `uEdgeTex`       | `uploadEdge` (`gpuRenderer.ts:539-560`)                                  | RGBA32F: `(gx, gy, mag, 0)`                               | Supplies edge vectors/weights to shader          | Texture unit 1, kept in sync with CPU Sobel buffers.         |
| Rim uniforms     | `RenderUniforms` payload (`App.tsx:1586-1623`, `gpuRenderer.ts:561-720`) | Scalars/vectors: thresholds, offsets, jitter, gain, blend | Fragment shader blocks controlling rim synthesis | Includes DMT-inflated `kEff`, `muJ`, phase pin toggles, etc. |

## 2D Surface Wallpaper / Warp

### CPU data products

| Data product                              | Producer (ref)                                      | Format / dims                                         | Consumers                                      | Notes                                           |
| ----------------------------------------- | --------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------- |
| Orientation cache (`orientationCacheRef`) | `getOrientationCache` (`App.tsx:676-685`)           | `Float32Array` pairs (`cos`,`sin`) sized to `nOrient` | CPU compositor, GPU uniforms                   | Recomputed only when orientation count changes. |
| Wallpaper gradient sampler                | `wallpaperAt` (`App.tsx:976-1001`)                  | Pure function returning `{gx, gy}`                    | CPU compositor fallback when Kuramoto disabled | Aggregates symmetry ops + alive “breath”.       |
| Surface warp parameters                   | CPU loop in `renderFrameCore` (`App.tsx:1288-1347`) | Scalars: mask, warp vectors, rim energy               | CPU compositor                                 | Applies warp sample + blends with base pixels.  |

### GPU staging

| Uniform set           | Loader                                                                       | Fields                                                                                                                   | Notes                                               |
| --------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| Wallpaper ops         | `toGpuOps` + `gl.uniform4fv` (`App.tsx:1462-1472`, `gpuRenderer.ts:603-626`) | Up to 8 ops encoded as `[kind, angle, 0, 0]`                                                                             | Implements wallpaper symmetry in shader.            |
| Orientation harmonics | `gl.uniform1fv` set (`App.tsx:1466-1472`, `gpuRenderer.ts:627-645`)          | `uOrientCos`, `uOrientSin`, `uOrientCount`                                                                               | Matches CPU orientation cache for per-frame warps.  |
| Surface controls      | `render` call (`App.tsx:1586-1623`)                                          | `uWarpAmp`, `uSurfaceBlend`, `uSurfaceRegion`, `uSurfEnabled`, `uCoupleE2S`, `uEtaAmp`, `uUseWallpaper`, `uCanvasCenter` | Drive surface warp amplitude, gating, and coupling. |

## Kuramoto Field

### Core state & derived buffers

| Data product                          | Producer (ref)                                                                                 | Format / dims                                                 | Consumers                    | Notes                                              |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------- | -------------------------------------------------- |
| Kuramoto state (`KuramotoState`)      | `createKuramotoState` (`kuramotoCore.ts:41-48`)                                                | Width × height complex field (`Zr`,`Zi`)                      | CPU integrator and worker    | Allocated per resolution.                          |
| Derived buffer (`createDerivedViews`) | `ensureKurCpuState` (`App.tsx:629-652`); worker pool (`kuramotoWorker.ts:118-140`)             | Shared `ArrayBuffer` sliced into `gradX`,`gradY`,`vort`,`coh` | CPU compositor, GPU upload   | `derivedBufferSize` controls allocation footprint. |
| CPU stepping                          | `stepKuramotoState` & `deriveKuramotoFieldsCore` (`App.tsx:788-798`, `kuramotoCore.ts:92-171`) | Euler step + OA derivatives                                   | Local deterministic baseline | Uses `createNormalGenerator` for noise (seedable). |

### Worker pipeline

| Stage             | Location                                       | Behavior                                                       | Notes                                               |
| ----------------- | ---------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------- |
| Init / pool setup | `handleInit` (`kuramotoWorker.ts:143-153`)     | Seeds state, takes shared buffers from main thread             | `buffers` array handed in via transferable objects. |
| Frame ticks       | `handleTick` (`kuramotoWorker.ts:118-140`)     | Steps state, fills derived buffer, posts back (`KurFrameView`) | `queueDepth` reported for diagnostics.              |
| Batch simulate    | `handleSimulate` (`kuramotoWorker.ts:161-184`) | Offline capture path; returns array of derived buffers         | Useful for regression capture later.                |
| Buffer return     | `releaseFrameToWorker` (`App.tsx:815-823`)     | Recycles buffers back to worker pool                           | Prevents allocations mid-run.                       |

### GPU staging

| Texture / uniform | Loader                                                                    | Layout                                                                          | Notes                                               |
| ----------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------- |
| `uKurTex`         | `uploadKur` (`gpuRenderer.ts:561-575`)                                    | RGBA32F storing `(gradX, gradY, vort, coh)`                                     | Bound to unit 2; toggled by `uKurEnabled`.          |
| Kuramoto toggles  | `render` call (`App.tsx:1586-1623`)                                       | `uKurEnabled`, `uCoupleS2E`, `uAlphaPol`, `uGammaOff`, `uKSigma`, `uGammaSigma` | Gate coupling into rims/surfaces; re-used on CPU.   |
| Worker sync flags | `kurSyncRef` / `workerInflightRef` (`App.tsx:541-554`, `App.tsx:833-907`) | Scalars tracking worker state                                                   | Control whether CPU or worker field feeds renderer. |

## Compositor & Instrumentation Hooks

- Telemetry: `telemetryRef` (`App.tsx:580-614`) records `frame`, `renderGpu`, `renderCpu`, `kuramoto` timings; output includes thresholds for logging.
- Rim energy normalization: `normTargetRef` / `lastObsRef` used on both CPU and GPU paths to stabilize brightness (`App.tsx:538-540`, `App.tsx:1494-1633`).
- Frame gain + capture: shared between CPU (`renderFrameCore`) and GPU (`drawFrameGpu`), enabling consistent metrics for subsequent baseline comparisons.
- Frame metrics stream: `metricsRef` pushes CPU/GPU `RainbowFrameMetrics` summaries and exposes `window.__getFrameMetrics()` (`App.tsx:451-456`, `App.tsx:873-1065`).
