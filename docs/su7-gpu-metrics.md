# SU(7) GPU Transform Metrics

## Per-Pixel Floating-Point Budget

- 49 complex multiply-accumulate operations (7 output components × 7 input components).
- Each complex MAC expands to 4 real multiplies and 4 real additions.
- Total core cost per pixel: **196 multiplies + 196 additions ≈ 392 FLOPs**.
- Post-transform magnitude sampling (7 complex magnitudes) adds ~21 multiplies, 14 additions, and 7 square roots.
- Optional projector weighting reuses the transformed magnitudes without extra multiplies.

## Memory Traffic

- C7 vectors stored in four RGBA32F textures ⇒ 16 floats ⇒ **64 bytes per pixel**.
- Norm channel piggybacks on the last texture (no additional allocation).
- Unitary tiles reside in an 8×(4·tiles) RGBA32F atlas ⇒ 16 floats per tile row; accessed sparsely via integer fetch.
- GPU kernel uploads reuse persistent buffers; only the 64 B/pixel vector payload is refreshed each frame.

## Validation

- `tests/su7GpuParity.test.ts` runs a synthetic parity harness that compares the GPU kernel output against the CPU packed multiply, enforcing an RMS error ≤ 1e-6 (spec limit 1e-5).
- `Su7GpuKernel` retains a 240-sample ring buffer; the telemetry overlay surfaces median/mean dispatch times, drift, and warning state (triggered when the median exceeds the baseline by >10%).
- The overlay shows the active backend (`gpu` when WebGPU is available, otherwise the CPU fallback) and the most recent dispatch duration in milliseconds and vector count, providing per-frame profiling context for 1080p preview targets.
