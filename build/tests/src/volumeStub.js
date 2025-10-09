import { makeResolution } from "./fields/contracts.js";
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const wrapPi = (theta) => {
    const twoPi = Math.PI * 2;
    return theta - Math.floor((theta + Math.PI) / twoPi) * twoPi - Math.PI;
};
const mulberry32 = (seed) => {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let r = Math.imul(t ^ (t >>> 15), 1 | t);
        r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
};
export const createVolumeStubState = (width, height, seed = 2024) => {
    const total = width * height;
    const basePhase = new Float32Array(total);
    const baseDepth = new Float32Array(total);
    const baseIntensity = new Float32Array(total);
    const rng = mulberry32(seed);
    for (let i = 0; i < total; i++) {
        basePhase[i] = (rng() - 0.5) * Math.PI;
        baseDepth[i] = rng();
        baseIntensity[i] = 0.4 + 0.6 * rng();
    }
    const state = {
        width,
        height,
        time: 0,
        phase: new Float32Array(total),
        depth: new Float32Array(total),
        intensity: new Float32Array(total),
        basePhase,
        baseDepth,
        baseIntensity
    };
    stepVolumeStub(state, 0);
    return state;
};
export const stepVolumeStub = (state, dt) => {
    const { width, height, basePhase, baseDepth, baseIntensity, phase, depth, intensity } = state;
    const total = width * height;
    state.time += dt;
    const t = state.time;
    for (let y = 0; y < height; y++) {
        const ny = height > 1 ? (y / (height - 1)) * 2 - 1 : 0;
        for (let x = 0; x < width; x++) {
            const nx = width > 1 ? (x / (width - 1)) * 2 - 1 : 0;
            const idx = y * width + x;
            const radial = Math.hypot(nx, ny);
            const swirl = Math.atan2(ny, nx);
            const carrier = radial * 1.75 + swirl * 0.5;
            phase[idx] = wrapPi(basePhase[idx] + 0.75 * Math.sin(carrier + t * 0.9));
            const envelope = clamp(0.35 + 0.65 * Math.sin(radial * 3.2 - t * 0.6), 0, 1);
            depth[idx] = clamp(0.2 + 0.6 * baseDepth[idx] + 0.4 * envelope * (0.5 + 0.5 * Math.cos(swirl * 2.2 - t * 0.4)), 0, 1);
            intensity[idx] = clamp(baseIntensity[idx] *
                (0.55 + 0.45 * (0.5 + 0.5 * Math.sin(radial * 2.4 + swirl * 1.1 + t * 0.7))), 0, 1.5);
        }
    }
    if (total <= 0)
        return;
};
export const snapshotVolumeStub = (state) => ({
    kind: "volume",
    resolution: makeResolution(state.width, state.height),
    phase: state.phase,
    depth: state.depth,
    intensity: state.intensity
});
export const ingestVolumeRecording = (recording) => {
    const { width, height, phase, depth, intensity } = recording;
    const total = width * height;
    if (phase.length !== total || depth.length !== total || intensity.length !== total) {
        throw new Error(`[volumeStub] recording length mismatch (phase=${phase.length}, depth=${depth.length}, intensity=${intensity.length}, expected=${total})`);
    }
    return {
        kind: "volume",
        resolution: makeResolution(width, height),
        phase: Float32Array.from(phase),
        depth: Float32Array.from(depth),
        intensity: Float32Array.from(intensity)
    };
};
