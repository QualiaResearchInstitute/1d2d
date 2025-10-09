# Phase 0 Acceptance Checklist

- [x] Inventory doc maps edge/surface/Kuramoto buffers and uniforms (`docs/phase0_inventory.md`).
- [x] Baseline metrics + render stored under `baseline/` (`canonical.ppm`, `canonical.json`).
- [x] Instrumentation exposes per-frame metrics via `window.__getFrameMetrics()`.
- [x] `npm run baseline` regenerates canonical artifacts.
- [x] `npm run baseline:check` verifies current metrics within ≤1% of stored baseline.
- [ ] Extend coverage: enable surface warp preset + capture follow-up baseline (Phase 1).
