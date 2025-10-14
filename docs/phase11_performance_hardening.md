# Phase 11 Performance Hardening & QA

Phase 11 locks the product for release by pushing rendering throughput, stamping out performance regressions, and validating long-haul stability. Treat this file as the source of truth for the hardening campaign: it defines the artefacts to capture, the profiling loops to run, and the exit guardrails that gate the public build.

---

## Deliverables Snapshot

- **Performance report** – Before/after metrics for each canonical scene, stored at `docs/perf/phase11/performance-report.md` with linked raw logs in `docs/perf/phase11/artifacts/<scenario>/`.
- **Optimised code paths** – GPU/CPU changes annotated inline plus a companion changelog in `docs/perf/phase11/code-notes.md` explaining rationale, shader tuning, and trade-offs.
- **Stability verification** – Soak-test evidence (memory, temp, FPS traces) and leak triage notes captured in `docs/perf/phase11/stability/`.
- **Resolved bug log** – Triage table in `docs/perf/phase11/bug-scrub.csv` tracking every Phase 11 fix, owner, repro link, and test coverage.
- **Expanded test suite & CI proof** – New/updated tests committed plus CI screenshots or run IDs archived at `docs/perf/phase11/ci/`.

Keep the directory tree under version control; when artefacts are too large, save summary stats alongside a pointer to the storage location.

---

## Performance Profiling Playbook

### Scenario Matrix

Document each profiling run in `docs/perf/phase11/scenario-matrix.csv` using the grid below as a baseline. Add columns for commit hash, FPS, average GPU ms, CPU ms, peak memory, and optimisation notes.

| Scene preset                                | Hardware target            | Renderer path | Resolution | Measurement tools                                                   |
| ------------------------------------------- | -------------------------- | ------------- | ---------- | ------------------------------------------------------------------- |
| 4-branch polarization (canonical baseline)  | MacBook Air M1 16 GB       | Metal GPU     | 1920×1080  | Xcode Instruments (Time Profiler + Metal), GPU Frame Capture        |
| Hyperbolic atlas w/ diagnostics overlays    | MacBook Pro M2 Max 32 GB   | Metal GPU     | 3840×2160  | Xcode Instruments (Metal System Trace), Chrome DevTools Performance |
| High-density oscillator manifest (Kuramoto) | Mac Studio Ultra           | Metal GPU     | 2560×1440  | Xcode Instruments + telemetry overlay                               |
| Headless CLI render (baseline pipeline)     | GitHub-hosted macOS runner | CPU fallback  | 1920×1080  | `time`, Node profiling hooks, `baseline:check`                      |

Add Windows/Linux scenarios if the fallback paths are expected to function; mark non-Metal toolchains explicitly.

### GPU Optimisation Workflow

1. **Capture a baseline trace** – Use Xcode > Product > Profile > Metal System Trace to record 5 s of the scene. Export the trace (`.trace` bundle) to `docs/perf/phase11/artifacts/<scene>/gpu/baseline.trace`.
2. **Identify hotspots** – Look for long-running kernels, high threadgroup occupancy waste, or pipeline state compilation on the hot path. Summarise findings in `code-notes.md` with timestamps.
3. **Tune shader & pipeline** – Consider:
   - Right-sizing threadgroup/threadgroup memory for Apple GPU SM architecture.
   - Switching to half-float (`half`) buffers when visual delta ≤ 0.5 % (record the comparison images).
   - Pre-warming Metal pipeline states at app start (`MTLCompileOptions` with fast Math) to avoid runtime stutters.
   - Reducing GPU↔CPU synchronisation by batching buffer updates and ensuring `MTLCommandBuffer.commit()` happens before waiting on completion.
4. **Validate improvement** – Re-run the trace and log delta FPS + GPU ms in the scenario matrix. Retain both traces for audit.
5. **Regression guard** – If a shader trade-off is acceptable, add a comment in the shader and an entry in `code-notes.md` describing when to revisit (e.g., “Half precision OK up to 4K on M2; watch for banding on HDR panels”).

### CPU Simulation & Scheduling

- Profile the Kuramoto scheduler with Instruments’ Time Profiler. Target scenarios where oscillator count >8 k. Note idle time per worker thread.
- If CPU load dominates, experiment with:
  - Work stealing via the Kuramoto worker queue (align with the message protocol in `src/kuramotoWorker.ts`).
  - Decimating oscillator updates to every Nth frame; maintain deterministic ordering by seeding RNG per frame and storing the last update tick.
  - Memoising frame-invariant matrices (phase coupling, adjacency) and invalidating only when inputs change.
- Benchmark each change with Chrome DevTools’ Performance tab (renderer process) and Node’s `--cpu-prof` for CLI flows. Store JSON/CPuprofile exports beside the GPU traces.

---

## Memory & Resource Leak Campaign

1. **Soak setups** – Run the heaviest preset for ≥2 h on M1 and M2 hardware. Automate with a script that toggles camera paths and parameter presets every 10 min so resources churn.
2. **Monitoring** – Record:
   - `Activity Monitor` samples (export as `.spindump`) every 30 min.
   - `memory_pressure` CLI output snapshots (`/usr/bin/memory_pressure -l critical`) piped to log.
   - GPU `IORegistry` telemetry via `powermetrics --samplers smc,gpu_power -i 120`.
3. **Leak triage** – Track suspicious growth in `docs/perf/phase11/stability/leak-log.md` with suspected subsystem, repro steps, fix status, and whether it reproduces after 30 min restart.
4. **Cleanup audit** – Review:
   - Texture lifetime (ensure `MTLTexture` references release when scenes swap).
   - Event listeners in React components (verify `useEffect` cleanup).
   - Timeline or history buffers (cap length / reuse arrays).
5. **Mitigation hooks** – If thermal throttling occurs, add a “Performance cap” toggle (FPS limit or quality reduction). Document fallback in the performance report.

---

## Determinism & Precision Validation

- Re-run the deterministic render harness (`npm run baseline` + `npm run baseline:check`) after each optimisation branch lands. Record result hashes in `docs/perf/phase11/determinism-log.md` with commit IDs.
- Add GPU-vs-GPU comparisons (M1 vs M2) by exporting rendered frame PNGs (via existing capture pipeline) and hashing with `blake3`. Store the hash table at `docs/perf/phase11/artifacts/determinism/frame_hashes.json`.
- For multi-threaded CPU updates, enforce ordering with explicit barriers or deterministic scheduling seeds. Document the approach in `code-notes.md`.
- When switching precision (e.g., half floats), run image diff with tolerance <= 0.5 % using the baseline tooling; attach diff heatmaps.

---

## Cross-Platform Sanity Sweep

Even if release targets Apple Silicon, verify fallback paths and document gaps.

| Platform               | Renderer               | Must-run checks                                    | Artefact location                              |
| ---------------------- | ---------------------- | -------------------------------------------------- | ---------------------------------------------- |
| macOS 14.5 (M1 Air)    | Metal GPU              | Full profiling suite, soak, determinism            | `docs/perf/phase11/artifacts/macos-m1/`        |
| macOS 14.5 (M2 Pro)    | Metal GPU              | 4K trace, telemetry overlay, thermal log           | `docs/perf/phase11/artifacts/macos-m2pro/`     |
| Windows 11 (RTX 4070)  | WebGPU/WebGL2 fallback | Sanity render, FPS log, shader compatibility notes | `docs/perf/phase11/artifacts/windows-rtx4070/` |
| Ubuntu 24.04 (GPU TBD) | WebGL2                 | CLI baseline render, CPU fallback perf             | `docs/perf/phase11/artifacts/linux/`           |

If a platform fails, document repro steps, stack traces, and workaround proposals in `bug-scrub.csv`.

---

## Bug Scrub & Regression Closure

- Consolidate the backlog from previous phases into `bug-scrub.csv`. Mark severity (`blocker`, `major`, `minor`), owner, and status (`open`, `in-progress`, `fixed`, `waived`).
- For each fix, add:
  - Repro link (issue tracker URL or commit hash).
  - Validation evidence (test case, video, screenshot).
  - Regression test reference (`tests/...` path).
- Adopt a daily triage ritual: review new findings from profiling and soak runs, assign owners, and refuse to close the phase while any blocker remains open.
- Keep a lightweight “Known Issues” appendix in the performance report for deferred low-priority items.

---

## Test Suite Expansion & Automation

- Extend deterministic rendering tests to cover the high-density oscillator manifest and at least one shader tuned during Phase 11. Commit fixtures under `tests/baseline/manifests/phase11/`.
- Add unit tests around:
  - Worker scheduling utilities (simulate thread churn).
  - Resource cleanup (ensure texture pools release handles).
  - Any newly introduced throttling or cap logic.
- Update CI to run `npm run lint`, `npm run typecheck`, `npm run test`, `npm run baseline:check`, and `npm run cross-tier:check`. Publish artefact URLs or run IDs into `docs/perf/phase11/ci/run-log.md`.
- Consider adding a nightly GitHub Actions workflow for soak-smoke: run `npm run baseline` on a macOS runner with a 30 min timeout and capture metrics.

---

## Reporting & Evidence Packaging

1. **Performance report** – Summarise each optimisation: scene, change, raw numbers, relative gain, regression risk, follow-up recommendations.
2. **Change log alignment** – Cross-link PRs/commits that landed optimisations; highlight any config toggles users may need.
3. **Stability digest** – Graph memory/GPU temperature trends (attach CSV + quick chart) and state the outcome (“No leak detected after 2 h”).
4. **Determinism confirmation** – Include hash tables and note any tolerance adjustments.
5. **Release note prep** – Provide user-facing highlights: expected FPS by hardware tier, new settings (caps), and known issues.

For quick reviews, keep a `README.md` in `docs/perf/phase11/` pointing to top-level artefacts.

---

## Exit Criteria Checklist

- [ ] All scenarios in the profiling matrix meet or exceed target FPS (e.g., ≥60 FPS on M1 at 1080p, ≥120 FPS on M2 Pro for canonical scenes).
- [ ] No unchecked memory growth over 2 h soak tests (≤5 % variance) and no sustained thermal throttling without mitigation.
- [ ] Determinism harness passes across hardware variants; hash diffs within tolerance.
- [ ] Bug scrub shows zero open blockers and all majors with mitigation or documented waiver.
- [ ] CI runs clean with the expanded test suite; soak automation (if implemented) reports green.
- [ ] Performance report, stability logs, and change notes committed and reviewed.

When every box is checked and reviewers sign off, Phase 11 is complete and the build is ready for release packaging.
