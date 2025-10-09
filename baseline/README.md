# Baseline Artifacts

- `renders/canonical.ppm` — CPU-generated canonical frame for "Rainbow Rims + DMT Kernel Effects" at 256×256.
- `metrics/canonical.json` — Aggregated rim/warp/∇θ/compositor metrics captured from the same frame.

Rebuild via:

```bash
npm run baseline
```

This recompiles the Node harness and overwrites the artifacts.
