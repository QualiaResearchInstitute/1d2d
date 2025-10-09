# Phase 8 Developer Guide — Data Flow, Telemetry & Ops

This guide captures the Phase 8 deliverables: a precise data-flow walkthrough, telemetry entry points, the DMT/arousal tuning playbook, and the operational runbook that covers rollback plus histogram analysis. Keep it close during onboarding and incident response—the scope is limited to the current `main` branch.

---

## System Data Flow (Phase 8)
- **Diagram:** `docs/diagrams/phase8_data_flow.mmd` (Mermaid). Render it via [Mermaid Live Editor](https://mermaid.live/) or your docs tooling.
- **Pipeline narrative**
  1. **UI intake** – image upload & preset surface in `src/App.tsx`.
  2. **Edge detection** – Sobel + gating (`src/pipeline/edgeDetection.ts`) builds the rim target field.
  3. **Kuramoto solver** – `src/kuramotoCore.ts` evolves Z(x,y,t); DMT & arousal land in `ThinElementOperatorControls`.
  4. **Thin-element operators** – flux/amplitude/phase operators apply control gains before coupling.
  5. **Composer** – `renderRainbowFrame` assembles rim, surface, Kuramoto, and volume layers; telemetry snapshots emit from here.
  6. **Telemetry sinks** – runtime logging (UI toggle), scripted baselines (`scripts/runBaseline.ts`), and cross-tier validation (`scripts/runCrossTierValidation.ts`) all consume the same snapshot shape.

---

## Telemetry & Instrumentation
- **Runtime toggles (Diagnostics panel):**
  - `Telemetry logging` streams `KuramotoTelemetrySnapshot` objects to the console or telemetry hook.
  - `Frame metric logging` expands logs with histogram-friendly `RainbowFrameMetrics`.
  - `Phase amplitude histogram` overlays the live histogram UI (hooks into `textureDiagnostics`).
- **CLI entry points (`package.json`):**
  - `npm run baseline` – regenerates canonical render + `baseline/metrics/canonical.json`. Requires Node 18+ and `tsc`.
  - `npm run baseline:check` – compares current metrics against baseline with ±1 % tolerances using `scripts/checkBaseline.ts`.
  - `npm run cross-tier:check` – runs `src/validation/crossTierValidation.ts` and writes `dist/cross-tier/latest.json`, failing on alert thresholds.
  - `npm run cross-tier:update` – persists a new `baseline/metrics/cross-tier.json` after manual review.
- **Artefacts & locations:**
  - Renders: `baseline/renders/*.ppm`, metrics: `baseline/metrics/*.json`.
  - Cross-tier reports: `dist/cross-tier/latest.json` (current) + `baseline/metrics/cross-tier.json` (blessed).
  - Texture diagnostics maps (optional): enable `includeMaps` when calling `computeTextureDiagnostics`.
- **Telemetry schema highlights:**
  - `KuramotoTelemetrySnapshot.orderParameter` → coherence + phase for DMT routing dashboards.
  - `telemetry.interference` → variance / max used by histogram rollups.
  - `ThinElementOperatorGains` (`flux`, `amplitude`, `phase`, `transparency`) respond monotonically to DMT/arousal—see `getThinElementOperatorGains`.

---

## DMT & Arousal Tuning Playbook
1. **Establish a baseline**
   - Capture the current canonical metrics (`npm run baseline`) and stash the render for visual reference.
   - Record operator gains for `d=0.2, a=0.2` via `getThinElementOperatorGains` to anchor expectations.
2. **Map objectives**
   - Decide which channels to favour:
     - **Higher periodicity / ribbing:** target `k0`, `Q` in kernel; watch `warp.dominantAngle` stability.
     - **Increased coherence:** track `orderParameter.magnitude` and `interference.variance`.
     - **Handedness bias:** monitor `orderParameter.phase` and rim chirality metrics.
3. **Sweep controls**
   - **DMT sweep (keep arousal fixed):**
     - Step d ∈ {0.0, 0.2, 0.4, 0.6}.
     - Log `telemetry.interference` and histogram outputs; expect monotonic gain increases with mild transparency lift.
   - **Arousal sweep (keep DMT fixed):**
     - Step a ∈ {0.0, 0.3, 0.6, 0.9}.
     - Verify phase gains grow faster than amplitude (`phase` > `amplitude` slope in `ThinElementOperatorGains`).
     - Keep drift ≤2 % by checking `composer.effectiveBlend`.
   - **Combined remap:** for each desired look, define a tuple `(d, a)` in your preset manifest.
4. **Fit remap curves**
   - Use piecewise-linear remaps with guard rails:
     - `fluxGain(d,a) = base + 0.25·d + 0.2·a` (matches implementation constants).
     - Limit `sigma` tightening when `a > 0.8` to avoid rim collapse.
   - Store curves in your control service (e.g., JSON manifest) and annotate with histogram expectations.
5. **Regression gate**
   - Run `npm run baseline:check` and `npm run cross-tier:check` for each new remap.
   - Document metric deltas; >1 % change in `rim.std` or `coherence` requires a product sign-off.

---

## Ops Runbook (Rollback & Histogram Analysis)
- **Pre-deploy checklist**
  - Ensure `npm run baseline:check` passes locally.
  - Attach `dist/cross-tier/latest.json` to the change review.
  - Export histogram screenshots (UI → Diagnostics → `Phase amplitude histogram`) for the new preset.
- **Deploy**
  - Publish updated presets / remap configs alongside app build.
  - Notify ops to watch telemetry dashboards for `interference.variance` spikes >0.15.
- **Rollback procedure**
  1. Restore canonical metrics: copy `baseline/metrics/canonical.json` back into the config store delivering runtime thresholds.
  2. Revert presets to last known-good manifest (git tag `phase7-release` if unchanged).
  3. Re-run `npm run baseline` to confirm renders match canonical `.ppm`.
  4. Capture `cross-tier` report; if divergence persists return to prior binary (CI artifact `build-<hash>`).
- **Histogram & diagnostics workflow**
  - Toggle `Phase amplitude histogram` in-app; export via developer tools (`window.__phaseDebug.getHistogram()`).
  - For offline batches, call `computeTextureDiagnostics(surface, { orientations, includeMaps: true })` and render `wallpapericityMap` / `beatEnvelopeMap` in Python or Observable.
  - Flag regression when histogram peak shifts >0.12 (normalized amplitude) or when `resonanceRate` ±0.05 from baseline.
  - Log findings in `docs/ops/histogram-log/<date>.md` (create as needed) to keep longitudinal context.

---

## Knowledge Base & Diagram Index
- `docs/diagrams/phase8_data_flow.mmd` – authoritative Phase 8 flow.
- `overview.md` – end-user control reference (link from onboarding emails).
- `part1.md` / `part2.md` – React implementation walkthrough; they remain current as of commit `[phase8-docs]`.
- Add follow-up diagrams here as you extend the pipeline (naming convention: `phase<stage>_<topic>.mmd`).

---

## Validation Walkthrough Log
- **Checklist for reviewers**
  - [ ] Read “System Data Flow” and trace each box in code (`edgeDetection.ts`, `kuramotoCore.ts`, `rainbowFrame.ts`).
  - [ ] Dry-run telemetry scripts (`npm run baseline`, `npm run cross-tier:check`) or table-top review if Node.js unavailable locally.
  - [ ] Use Diagnostics panel to toggle telemetry + histogram overlays; capture a screenshot.
  - [ ] Summarize questions + clarifications; submit PR comment or DM.
- **2024-05-05 — Walkthrough dry-run (Kai, onboarding SWE)**
  - Environment lacked Node (`npm: command not found`); walkthrough executed as read-through of scripts with mentor.
  - Verified understanding of DMT vs arousal gain mapping using `docs/phase8_dev_ops.md` + `src/kuramotoCore.ts`.
  - Action item: provision Node 18 container before next hands-on test; no doc updates requested.
- Record future validations inline (newest first). The onboarding buddy should sign their entry and link any follow-up tasks.
