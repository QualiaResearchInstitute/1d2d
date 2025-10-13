# Phase 10 Developer Onboarding Guide

This guide streamlines setup for the Rainbow Perimeter Lab renderer. Following it end to end keeps onboarding under 30 minutes while covering builds, tests, baselines, serialization, performance knobs, and GPU/Python tooling that the phase now depends on.

## 0–5 min — Environment & Install

1. **Prerequisites**
   - Node.js ≥ 20, npm ≥ 10
   - Python 3 (only if you post-process diagnostics)
   - `ffmpeg` + `ffprobe` on `PATH` (required for the offline renderer examples)
   - WebGL2-capable GPU (Chrome/Edge/Firefox latest)
2. **Clone & install**
   ```bash
   git clone <repo>
   cd <repo>
   npm install
   ```
3. **Sanity check** – run `npm run dev` and hit the Vite URL. Upload a sample image; the GPU/CPU toggle should work out of the box.

## 5–15 min — Build & Test Matrix

| Command                   | Purpose                                          | Notes                                                                                  |
| ------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `npm run build`           | TS project references + Vite production bundle   | Runs the shader string bundling paths; fail here before shipping.                      |
| `npm test`                | Node test runner over all suites                 | Includes property-based suites powered by `fast-check` (dev dep) and SU7 parity tests. |
| `npm run coverage`        | Compiles test build then runs coverage with `c8` | Outputs HTML in `coverage/`.                                                           |
| `npm run test -- --watch` | Focus on a spec during development               | Combine with `--test-name-pattern=<regex>`.                                            |

All of the above wire through `tsconfig.tests.json` / `tsconfig.baseline.json`, so failures usually mean the build graph needs attention rather than missing local tsconfigs.

## 15–22 min — Baselines & Cross-Tier Checks

The baseline harness exercises CPU ↔ GPU equivalence, SU7 presets, and telemetry budgets. Every command writes status to `baseline-dist/` and `baseline/metrics`.

| Command                    | What it does                                                                                                                                               |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `npm run baseline`         | Rebuilds harness, renders canonical scenes, saves PNG/PPM frames plus canonical JSON + BLAKE3 hashes for the SU7 preset (`baseline/metrics/su7-preset.*`). |
| `npm run baseline:check`   | Compares current outputs against the committed baseline. Fails on mismatched hashes, image drift, or preset alterations.                                   |
| `npm run cross-tier:check` | Runs the cross-tier validation matrix (Node timeline + worker evolution) and asserts telemetry stays within guard rails.                                   |

Typical workflow: `baseline` only when visuals intentionally change, `baseline:check` in CI/pre-push, `cross-tier:check` before cutting release notes.

## 22–26 min — Serialization & Preset Format

Canonical serialization lives in `src/serialization/canonicalJson.ts` and powers the preset snapshots stored under `baseline/metrics/`:

- **Deterministic normalisation** (`writeCanonicalJson`):
  - Drops `undefined`, functions, and symbols unless inside arrays (where they demote to `null`).
  - Sorts object keys & `Map` entries, coerces `Set`→array, and canonicalises binary data by expanding typed arrays to numeric lists.
  - Normalises numbers (rejects non-finite, converts `-0` to `0`).
- **Blake3 hashing** (`hashCanonicalJsonString`): generates the 256-bit digest saved alongside presets for regression.
- **Round-trip helpers** (`readCanonicalJson`) keep `-0` squashing symmetric during checks.

_Workflow snippet_:

```ts
import { hashCanonicalJson } from '../src/serialization/canonicalJson.js';

const payload = {
  preset: 'Rainbow Rims + DMT Kernel Effects',
  gates: currentSu7GateList,
};

const { json, hash } = hashCanonicalJson(payload);
// Save json to metrics/su7-preset.json and hash to metrics/su7-preset.hash
```

## 26–30 min — Performance Knobs & Offline Renderer

Performance budgets now ship with the offline renderer (`scripts/videoPipeline.ts`):

- **Budgets** (`OFFLINE_RENDER_BUDGETS`): frame ≤ 75 ms, RSS ≤ 4 GB, heap ≤ 2 GB, CPU load ≤ 1200 % (across cores).
- **Watchdog tolerance**: 20 % slack (`OFFLINE_WATCHDOG_TOLERANCE`) before logging violations that end up in the metadata JSON.
- **Tile height**: 128 rows per chunk to keep memory reuse predictable on 4K/10-bit exports.

The renderer converts RGBA8 CPU output to 10-bit buffers, encodes PNG intermediates as `rgba64le`, and emits summary telemetry (performance history + SU7 drift stats) in `<output>.meta.json`.

_Example run_:

```bash
npm run video:process -- \
  --input demos/rainbow-sequence.mov \
  --output out/rainbow-offline.mov \
  --timeline demos/rainbow.timeline.json \
  --keep-temp
```

Inspect the console:

- `[video]` summary with frame count & avg frame time,
- `[watchdog]` either “budgets respected” or violation count,
- `[su7]` determinant drift + fallback stats.
  Check `out/rainbow-offline.mov.meta.json` for full performance history.

## Preset Canon & Hashing

Baseline presets live under `baseline/metrics/su7-preset.json`. They use the canonical JSON above plus the SU7 runtime schema from `src/pipeline/su7/types.ts`. Any change to seeds, gate schedules, or projector settings must flow through `npm run baseline` so the BLAKE3 hash in `su7-preset.hash` updates in lockstep.

## GPU Kernel Notes

The SU7 GPU path runs through `src/pipeline/su7/gpuKernel.wgsl`:

- Workgroup size `64`, one invocation per vector.
- Storage buffers: flattened 7×7 complex unitary, input vectors, output vectors, plus uniform block for stride/count.
- Inner loop performs complex multiply-add in WGSL to maintain parity with the CPU math.

Fallback handling (branch cuts / determinant issues) is covered in `src/pipeline/su7/geodesic.ts`, which:

- Re-unitarises start/end matrices (`polar_reunitarize`).
- Attempts a logarithmic geodesic; if reconstruction error, branch guard (`π − 1e-4`), or determinant proximity to `-I` trips, it logs a fallback and blends via polar interpolation.
- Records fallback counts into `telemetry.geodesicFallbacks`, surfaced in offline metadata.

## Runnable Examples

1. **Dev server parity check**

   ```bash
   npm run dev
   # In UI: upload sample → toggle GPU renderer → Diagnostics → Run GPU parity check
   ```

   Watch the console for `[telemetry] renderGpu` anomalies; none should persist.

2. **Baseline smoke**

   ```bash
   npm run baseline:check
   ```

   Expect “SU7 preset hash match” and zero frame discrepancies.

3. **Cross-tier validation**

   ```bash
   npm run cross-tier:check
   ```

   Ensures timeline serialization, worker evolution, and tracer half-life remain inside tolerances.

4. **Offline exporter** – see example earlier; confirm metadata budgets stay under limits.

## Troubleshooting

| Symptom                                         | Fix                                                                                                                                        |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `ffmpeg exited with code 1` during video export | Verify `ffmpeg`/`ffprobe` are installed and accessible. Use `--keep-temp` to inspect staged PNGs.                                          |
| `baseline:check` reports preset hash mismatch   | Regenerate via `npm run baseline`, review `baseline/metrics/` diffs, and commit hash/json together.                                        |
| WebGL parity drifts > 0.5 %                     | Run `npm run cross-tier:check`; if clean, inspect recent shader changes vs `renderRainbowFrame` CPU reference.                             |
| Node heap breach when processing large footage  | Lower `OFFLINE_TILE_HEIGHT` in `scripts/videoPipeline.ts`, or split input (watchdog logs will point to frames exceeding RSS/heap budgets). |
| Tests fail with missing typed array exports     | Run `npm run build` once—tsc emits type stubs needed by some suites.                                                                       |

## Further Reading

- `README.md` – architecture overview and regression checklist.
- `overview.md` – visual pipeline deep dive.
- `docs/phase6_validation_rollout.md` – prior ops phase context (baseline expectations).

Welcome aboard! With these steps you can install, run tests, regenerate baselines, and ship offline renders without spelunking through the entire codebase. For open questions, check telemetry logs or the SU7 metadata in your offline exports—they now serve as first-line diagnostics.
