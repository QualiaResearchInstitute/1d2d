# Baseline Artifacts

- `renders/canonical.ppm` — CPU-generated canonical frame for "Rainbow Rims + DMT Kernel Effects" at 256×256.
- `renders/canonical.hash` — BLAKE3-256 of the canonical render bytes.
- `renders/canonical-64.ppm` — 64×64 downsampled golden preview derived from the canonical frame.
- `renders/canonical-64.hash` — BLAKE3-256 of the thumbnail render bytes.
- `metrics/canonical.json` — Aggregated rim/warp/∇θ/compositor metrics plus golden hashes.
- `metrics/su7-preset.json` — Canonical SU7 preset snapshot captured as canonical JSON.
- `metrics/su7-preset.hash` — BLAKE3-256 of the preset snapshot JSON.

Rebuild via:

```bash
npm run baseline
```

This recompiles the Node harness and overwrites the artifacts.
