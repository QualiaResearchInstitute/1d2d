import test from 'node:test';
import assert from 'node:assert/strict';
import { makeResolution } from '../src/fields/contracts.js';
import { computeTextureDiagnostics } from '../src/pipeline/textureDiagnostics.js';
import { renderRainbowFrame } from '../src/pipeline/rainbowFrame.js';
import { createDefaultSu7RuntimeParams } from '../src/pipeline/su7/types.js';
import { createKernelSpec } from '../src/kernel/kernelSpec.js';
const clamp01 = (value) => Math.max(0, Math.min(1, value));
const createSurfaceFromFn = (width, height, fn) => {
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const value = clamp01(fn(x, y));
            const v = Math.round(value * 255);
            rgba[idx + 0] = v;
            rgba[idx + 1] = v;
            rgba[idx + 2] = v;
            rgba[idx + 3] = 255;
        }
    }
    return {
        kind: 'surface',
        resolution: makeResolution(width, height),
        rgba,
    };
};
const createUniformSurface = (width, height, value) => createSurfaceFromFn(width, height, () => value);
const createStripeSurface = (width, height, angle, frequency) => {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return createSurfaceFromFn(width, height, (x, y) => {
        const nx = (x + 0.5) / width - 0.5;
        const ny = (y + 0.5) / height - 0.5;
        const coord = nx * cos + ny * sin;
        const phase = 2 * Math.PI * frequency * coord;
        return 0.5 + 0.5 * Math.sin(phase);
    });
};
const createBeatSurface = (width, height, baseFreq, deltaAngle) => {
    const cosB = Math.cos(deltaAngle);
    const sinB = Math.sin(deltaAngle);
    return createSurfaceFromFn(width, height, (x, y) => {
        const nx = (x + 0.5) / width - 0.5;
        const ny = (y + 0.5) / height - 0.5;
        const coordA = nx;
        const coordB = nx * cosB + ny * sinB;
        const s1 = Math.sin(2 * Math.PI * baseFreq * coordA);
        const s2 = Math.sin(2 * Math.PI * (baseFreq * 1.1) * coordB + 0.35);
        return 0.5 + 0.3 * s1 + 0.3 * s2;
    });
};
const createFlatRimField = (width, height) => {
    const total = width * height;
    return {
        kind: 'rim',
        resolution: makeResolution(width, height),
        gx: new Float32Array(total),
        gy: new Float32Array(total),
        mag: new Float32Array(total),
    };
};
test('texture diagnostics near zero for uniform surface', () => {
    const surface = createUniformSurface(32, 32, 0.5);
    const result = computeTextureDiagnostics(surface, { orientations: [0, Math.PI / 2] });
    assert.equal(result.sampleCount, 32 * 32);
    assert.ok(result.wallpapericityMean < 1e-3);
    assert.ok(result.beatEnergyMean < 1e-3);
});
test('texture diagnostics wallpapericity rises for striped surface', () => {
    const surface = createStripeSurface(64, 64, 0, 6);
    const result = computeTextureDiagnostics(surface, { orientations: [0, Math.PI / 2] });
    assert.equal(result.sampleCount, 64 * 64);
    assert.ok(result.wallpapericityMean > 0.08);
    assert.ok(result.beatEnergyMean >= 0);
});
test('texture diagnostics beat energy spikes for near-resonant stripes', () => {
    const baseSurface = createStripeSurface(64, 64, 0, 6);
    const beatSurface = createBeatSurface(64, 64, 6, Math.PI / 7);
    const baseMetrics = computeTextureDiagnostics(baseSurface, { orientations: [0, Math.PI / 7] });
    const beatMetrics = computeTextureDiagnostics(beatSurface, { orientations: [0, Math.PI / 7] });
    assert.ok(beatMetrics.beatEnergyMean > baseMetrics.beatEnergyMean + 0.005);
    assert.ok(beatMetrics.resonanceRate > baseMetrics.resonanceRate);
});
const buildRenderInput = (surface, rim) => {
    const { width, height, texels } = surface.resolution;
    return {
        width,
        height,
        timeSeconds: 0,
        out: new Uint8ClampedArray(texels * 4),
        surface,
        rim,
        phase: null,
        volume: null,
        kernel: createKernelSpec({
            gain: 2.4,
            k0: 0.22,
            Q: 3.6,
            anisotropy: 0.8,
            chirality: 1.15,
            transparency: 0.28,
        }),
        dmt: 0.25,
        arousal: 0.2,
        blend: 0.4,
        normPin: true,
        normTarget: 0.6,
        lastObs: 0.6,
        lambdaRef: 520,
        lambdas: { L: 560, M: 530, S: 420 },
        beta2: 1.2,
        microsaccade: false,
        alive: false,
        phasePin: true,
        edgeThreshold: 0.18,
        wallpaperGroup: 'off',
        surfEnabled: true,
        orientationAngles: [0, Math.PI / 6],
        thetaMode: 'gradient',
        thetaGlobal: 0,
        polBins: 16,
        jitter: 0.4,
        coupling: {
            rimToSurfaceBlend: 0.2,
            rimToSurfaceAlign: 0.2,
            surfaceToRimOffset: 0.2,
            surfaceToRimSigma: 0.15,
            surfaceToRimHue: 0.2,
            kurToTransparency: 0,
            kurToOrientation: 0,
            kurToChirality: 0,
            volumePhaseToHue: 0,
            volumeDepthToWarp: 0,
        },
        sigma: 3.2,
        contrast: 1.1,
        rimAlpha: 1,
        rimEnabled: true,
        displayMode: 'color',
        surfaceBlend: 0.3,
        surfaceRegion: 'both',
        warpAmp: 1,
        curvatureStrength: 0,
        curvatureMode: 'poincare',
        kurEnabled: false,
        debug: undefined,
        su7: createDefaultSu7RuntimeParams(),
        composer: undefined,
    };
};
test('renderRainbowFrame exposes texture metrics with beat spikes', () => {
    const width = 64;
    const height = 64;
    const rim = createFlatRimField(width, height);
    const singleSurface = createStripeSurface(width, height, 0, 5);
    const beatSurface = createBeatSurface(width, height, 5, Math.PI / 6);
    const baseInput = buildRenderInput(singleSurface, rim);
    const beatInput = buildRenderInput(beatSurface, rim);
    const baseResult = renderRainbowFrame({ ...baseInput, out: baseInput.out.slice() });
    const beatResult = renderRainbowFrame({ ...beatInput, out: beatInput.out.slice() });
    assert.ok(baseResult.metrics.texture.earlyVision.wallpapericity > 0.05);
    assert.ok(baseResult.metrics.texture.crystallizer.sampleCount > 0);
    assert.ok(beatResult.metrics.texture.beatEnergy > baseResult.metrics.texture.beatEnergy + 0.005);
    assert.ok(beatResult.metrics.texture.earlyVision.beatEnergy >
        baseResult.metrics.texture.earlyVision.beatEnergy);
});
