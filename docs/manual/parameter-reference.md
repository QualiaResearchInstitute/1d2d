# Parameter Reference

The tables below summarise the key controls exposed through manifests and the CLI/SDK surfaces. Values map directly to the physics pipeline in `src/pipeline/rainbowFrame.ts` unless otherwise stated.

## Tracer controls

| Parameter       | Description                                                          | Default    |
| --------------- | -------------------------------------------------------------------- | ---------- |
| `edgeThreshold` | Sobel magnitude threshold that gates rim synthesis.                  | 0.22       |
| `beta2`         | Dispersion term controlling chromatic offsets along the edge normal. | 1.1        |
| `sigma`         | Rim Gaussian width; smaller values yield sharper rims.               | 1.4        |
| `jitter`        | Temporal jitter amplitude used to add micro-motion without drift.    | 0.5        |
| `microsaccade`  | Enables stochastic orientation shaking to mimic saccades.            | true       |
| `phasePin`      | Locks the global phase reference to avoid drift.                     | true       |
| `thetaMode`     | `gradient` uses ∇θ from Kuramoto; `global` applies a fixed angle.    | `gradient` |
| `thetaGlobal`   | Global fallback orientation in radians when `thetaMode = global`.    | 0          |

## Composer controls

| Parameter      | Description                                               |
| -------------- | --------------------------------------------------------- |
| `blend`        | Base blend factor between source image and Indra overlay. |
| `surfaceBlend` | Weight applied to the surface wallpaper layer.            |
| `rimAlpha`     | Scalar applied to rim opacity before compositing.         |
| `rimEnabled`   | Toggles rim synthesis entirely.                           |

## Kuramoto parameters

Mapped to `KuramotoParams` in `src/kuramotoCore.ts`.

| Parameter                            | Description                                     |
| ------------------------------------ | ----------------------------------------------- |
| `K0`                                 | Coupling strength between oscillators.          |
| `alphaKur`                           | Phase lag controlling synchronisation speed.    |
| `gammaKur`                           | Line width term anchoring oscillator stability. |
| `epsKur`                             | Noise magnitude injected per step.              |
| `omega0`                             | Mean natural frequency.                         |
| `fluxX`, `fluxY`                     | Imposed phase flux along X/Y.                   |
| `smallWorldWeight`, `p_sw`           | Small-world rewiring weight and probability.    |
| `smallWorldDegree`, `smallWorldSeed` | Graph degree and RNG seed for rewiring.         |

## Hyperbolic and wallpaper controls

| Parameter           | Description                                      |
| ------------------- | ------------------------------------------------ |
| `warpAmp`           | Scalar applied to the hyperbolic warp field.     |
| `curvatureStrength` | Amount of hyperbolic curvature applied.          |
| `curvatureMode`     | `poincare` or `klein` projection.                |
| `wallpaperGroup`    | Wallpaper symmetry (p4, p6m, etc.).              |
| `surfaceRegion`     | Region influenced (`surfaces`, `edges`, `both`). |

## SU(7) projector

| Parameter        | Description                                                           |
| ---------------- | --------------------------------------------------------------------- |
| `gain`           | Projector gain applied to SU(7) gate outputs.                         |
| `projectorMode`  | Rendering mode (`identity`, `directRgb`, `overlaySplit`, `hopfLens`). |
| `decimationMode` | Sampling scheme (`hybrid`, `stride`, `edges`).                        |

## Metrics

The CLI, REST API, and SDKs return a consistent subset of the `RainbowFrameMetrics` structure:

- `metrics.rimMean` – average rim energy
- `metrics.warpMean` – average surface warp magnitude
- `metrics.coherenceMean` – mean Kuramoto coherence |Z|
- `metrics.indraIndex` – composite perceptual score (0–1)

See `src/pipeline/rainbowFrame.ts` for the complete metric payload, including parallax, motion energy, and SU(7) guardrails.
