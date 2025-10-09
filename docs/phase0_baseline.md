# Phase 0 Baseline Metrics

- **Preset**: Rainbow Rims + DMT Kernel Effects
- **Frame**: 256×256 CPU render (Kuramoto enabled, surfaces disabled)
- **Artifacts**: `baseline/renders/canonical.ppm`, `baseline/metrics/canonical.json`

## Key Metrics

| Metric | Value | Notes |
| --- | ---: | --- |
| Rim energy mean | 0.2713 | Stable chroma envelope across active edges |
| Rim energy max | 3.4424 | Peak localized at high-contrast rim seed |
| Warp magnitude mean | 0.0572 | Kuramoto-driven flow; surfaces currently disabled => coupling reserve |
| Warp dominant angle | 0.0037 rad | Near-horizontal alignment, confirming q=1 twist |
| ∇θ magnitude mean | 0.0572 | Mirrors warp mean (same field) |
| Coherence mean | 0.7943 | Healthy Kuramoto order; std = 0.0426 |
| Effective blend | 0.58 | DMT transparency uplift captured |
| Surface blend mean | 0 | Surface warp off → weak coupling zone flagged |
| Observed rim average (`obsAverage`) | 0.1012 | Feeds normalization loop |

**Weak coupling zone:** surface compositor metrics remain zero because `surfEnabled=false`; this is intentional for the canonical preset but should be revisited when surface tests are added.

## Regeneration

```bash
npm run baseline        # rebuild harness + refresh artifacts
npm run baseline:check  # compare live metrics against stored baseline (≤1% drift)
```
