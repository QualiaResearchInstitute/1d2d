# Phase 6 Validation & Rollout - QA, Phenomenology & Release

Phase 6 locks down the hyperbolic scene integration by pairing comprehensive QA with a qualitative phenomenology review and tight release packaging. Use this guide as the phase gate: it enumerates the deliverables, documents how to gather the required evidence, and shows where to store artefacts so the rollout stays auditable.

---

## Deliverables Snapshot

- **Comprehensive QA matrix** - hardware x browser coverage, scripted checks (`npm run test`, `baseline:check`, `cross-tier:check`, `coverage`) plus a mandatory 4K stress run with performance logs captured.
- **Phenomenological review dossier** - qualitative notes, before/after captures, and QRI advisor sign-off on any must-fix deltas.
- **Versioned release package** - tagged release notes highlighting the hyperbolic scene uplift, bundled imagery, and updated change metadata.
- **Monitoring hooks live** - telemetry & logging thresholds configured so post-launch regressions (CPU/GPU timing, coherence drift) alert within minutes.

---

## Comprehensive QA Matrix

### Coverage Grid

Document the runs in `docs/qa/phase6_test_matrix.csv` (create if missing). The minimum grid is:

| Platform                           | GPU tier          | Browser(s)              | Resolution(s)        | Required observations                                                                               |
| ---------------------------------- | ----------------- | ----------------------- | -------------------- | --------------------------------------------------------------------------------------------------- |
| macOS 14.5 (Apple M2 Max, 32 GB)   | Metal WebGL2      | Safari 17.5, Chrome 125 | 1920x1080, 3840x2160 | Ensure GPU renderer parity, telemetry overlay screenshot, 4K stress log.                            |
| Windows 11 (RTX 4070, 64 GB)       | DirectX 12 WebGL2 | Chrome 125, Edge 125    | 2560x1440, 3840x2160 | Validate hyperbolic guides render without shader precision artefacts; capture GPU timing histogram. |
| Ubuntu 24.04 (RTX 3080, 32 GB)     | Mesa 24 WebGL2    | Firefox 126             | 1920x1080            | Confirm CPU fallback parity, inspector screenshots of telemetry histograms.                         |
| macOS 14.5 (MacBook Air M3, 16 GB) | Integrated        | Safari 17.5             | 1440x900             | Battery + throttling scenario; note if telemetry warns about frame >12 ms.                          |

Augment with additional devices as they become available (e.g., Quest browser, HDR monitors). Every row must link to artefacts (screenshots, logs, metrics) stored in `docs/qa/artifacts/phase6/<platform>/`.

### Scripted Regression Sweep

Run the scripted checks in a clean working tree. Capture command output hashes (last 5 lines) in `docs/qa/artifacts/phase6/command-log.md`.

```bash
npm run lint
npm run typecheck
npm run test
npm run coverage
npm run baseline
npm run baseline:check
npm run cross-tier:check
npm run build
```

- Flag and resolve any drift >1 % reported by `baseline:check` or `cross-tier:check`. Attach `dist/cross-tier/latest.json` alongside the blessed baseline snapshot for reviewers.
- Surface-level CPU/GPU parity should remain <=0.5 % mismatch (`__runGpuParityCheck()` in Diagnostics). Log worst-case deltas.

### 4K Stress & Telemetry Capture

1. Launch the app (`npm run dev`) and load the canonical preset.
2. Set the canvas to **3840x2160**, enable OA + Surface, and toggle **Use GPU renderer**.
3. In the DevTools console, run:

```js
await window.__setTelemetryEnabled(true);
await window.__measureRenderPerformance(240);
```

4. Record the returned `cpuMs`, `gpuMs`, `throughputGain`, and histogram ranges from the telemetry overlay.
5. Export the telemetry history:

```js
const telemetry = window.__getTelemetryHistory();
await navigator.clipboard.writeText(JSON.stringify(telemetry, null, 2));
```

6. Store the JSON at `docs/qa/artifacts/phase6/telemetry/4k_stress.json` and embed summary stats in the QA matrix row.

Acceptance requires:

- `throughputGain >= 5x` at 4K relative to CPU.
- No sustained telemetry warnings (`[telemetry] renderGpu` or `renderCpu` thresholds) during the run.
- Frame-to-frame coherence metrics (`composer.coupling.effective`) stay within baseline +/-1 %.

---

## Phenomenological Review (QRI Advisors)

### Dossier Workflow

1. Create `docs/reviews/phase6_qri.md` and fill the template below for each advisor session.
2. Run through the curated preset list (`docs/hyperbolic-atlas.md`) focusing on scenes impacted by the hyperbolic integration.
3. Capture **before/after** frames (CPU baseline vs GPU hyperbolic) using the Diagnostics capture button or `window.__runFrameRegression(1)`.
4. Log subjective notes, highlighting:
   - Hyperbolic curvature legibility (ruler overlay, curvature gradients).
   - DMT phenomenology cues (entity emergence, tracer persistence).
   - Any perceived instabilities or visual fatigue triggers.
5. Tag must-fix items and assign owners; only close Phase 6 when all must-fix items are addressed or waived by the advisors.

```markdown
## Session: <Advisor name>, <Date>

- Preset(s) reviewed:
- Hardware / browser:
- ## Key phenomenological observations:
- ## Hyperbolic overlay notes:
- Must-fix items:
  - [ ]
- Attachments:
  - CPU baseline: `docs/reviews/assets/<name>-cpu.png`
  - Hyperbolic render: `docs/reviews/assets/<name>-gpu.png`
```

Ensure every consultation references the same build artefacts as QA (commit hash, preset manifest version). Store sign-off mails or chat exports in `docs/reviews/phase6_qri/`.

---

## Release Packaging

- **Release tag**: `v0.6.0-hyperbolic` (or next semantic milestone). Create the tag only after QA + phenomenology checklists pass.
- **Release notes**: author in `docs/releases/v0.6.0-hyperbolic.md`. Headings:
  - Hyperbolic scene integration summary.
  - Performance deltas (CPU vs GPU, 4K stress metrics).
  - QA coverage table (link back to `phase6_validation_rollout.md`).
  - Advisor insights & resolved deltas.
- **Before/after imagery**: export PNG pairs into `public/releases/v0.6.0-hyperbolic/` with filenames `presetId-before.png` / `presetId-after.png`. Reference them in the notes and bundle for distribution.
- **Changelog hook**: append a concise entry to `overview.md` or the public-facing changelog, pointing to the detailed release notes.
- **Distribution checks**:
  - `npm run build` -> verify `dist/` contents include updated shader bundles.
  - `scripts/package-release.mjs` (when available) -> ensures archives include release notes + imagery.

---

## Monitoring Hooks & Rollout Guardrails

- Enable telemetry by default in staging builds (`runtime.telemetryEnabled = true`). Confirm `renderGpu` threshold is <=10 ms and `kuramoto` threshold <=6 ms for desktop-class hardware.
- Wire runtime telemetry to the monitoring pipeline:
  - Use `__getTelemetryHistory()` snapshots every 60 s while QA runs; pipe to your logging sink.
  - Include a derived metric `hyperbolic.coherenceDelta = composer.coupling.effective.scale - baseline.scale`.
- Capture logs during the 4K stress run and establish alert thresholds:
  - `renderGpu > 14 ms` for 5 consecutive frames -> warn.
  - `telemetry.interference.variance` drifting >0.12 from baseline -> warn.
  - Histogram peaks shifting >0.12 (normalized amplitude) -> warn, matching Phase 8 ops guidance.
- Post-launch, monitor:
  - `dist/cross-tier/latest.json` deltas vs baseline (automated nightly run).
  - Telemetry overlay snapshots pushed from staging/production sessions.
  - User-submitted phenomenology feedback; funnel critical issues back into the QA matrix.

Document alert hookups and dashboards in `docs/ops/phase6_monitoring.md` with links to Grafana/Datadog (or equivalent).

---

## Sign-off Checklist

- [ ] QA matrix completed, artefacts attached in `docs/qa/artifacts/phase6/`.
- [ ] Scripted regressions logged and drift reviewed.
- [ ] 4K stress run meets throughput & telemetry thresholds.
- [ ] QRI phenomenology dossier signed off; all must-fix items addressed.
- [ ] Release notes + imagery packaged and tagged.
- [ ] Monitoring hooks configured; alert thresholds validated.

Add reviewer initials and date alongside each item when closing the phase.

---

## Artefact Index

- `docs/qa/phase6_test_matrix.csv` - run log (create during QA).
- `docs/qa/artifacts/phase6/` - screenshots, telemetry dumps, command logs.
- `docs/reviews/phase6_qri.md` & `docs/reviews/phase6_qri/` - phenomenological review records.
- `public/releases/v0.6.0-hyperbolic/` - before/after imagery bundle.
- `docs/releases/v0.6.0-hyperbolic.md` - versioned release notes.
- `docs/ops/phase6_monitoring.md` - monitoring configuration & alert thresholds.

Use this file as the canonical Phase 6 reference and update cross-links as new artefacts land.
