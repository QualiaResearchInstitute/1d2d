import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ingestVolumeRecording, createVolumeStubState, stepVolumeStub, snapshotVolumeStub, } from '../src/volumeStub.js';
import { renderRainbowFrame } from '../src/pipeline/rainbowFrame.js';
import { createDefaultSu7RuntimeParams } from '../src/pipeline/su7/types.js';
import { createKernelSpec } from '../src/kernel/kernelSpec.js';
import { makeResolution, } from '../src/fields/contracts.js';
const readFixture = async () => {
    const fixturePath = join(process.cwd(), 'tests', 'fixtures', 'volume-preroll.json');
    const raw = await readFile(fixturePath, 'utf8');
    const parsed = JSON.parse(raw);
    const recording = {
        width: parsed.meta.width,
        height: parsed.meta.height,
        phase: parsed.phase,
        depth: parsed.depth,
        intensity: parsed.intensity,
    };
    return { recording, meta: parsed.meta };
};
const makeSurfaceField = (width, height) => {
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const xf = width > 1 ? x / (width - 1) : 0;
            const yf = height > 1 ? y / (height - 1) : 0;
            rgba[idx + 0] = Math.round(90 + 120 * xf);
            rgba[idx + 1] = Math.round(70 + 140 * yf);
            rgba[idx + 2] = Math.round(110 + 60 * (1 - xf));
            rgba[idx + 3] = 255;
        }
    }
    return {
        kind: 'surface',
        resolution: makeResolution(width, height),
        rgba,
    };
};
const makeRimField = (width, height) => {
    const total = width * height;
    const gx = new Float32Array(total);
    const gy = new Float32Array(total);
    const mag = new Float32Array(total);
    const cx = (width - 1) * 0.5;
    const cy = (height - 1) * 0.5;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            const dx = x - cx;
            const dy = y - cy;
            gx[idx] = dx * 0.01;
            gy[idx] = dy * 0.01;
            mag[idx] = Math.hypot(dx, dy) * 0.02 + (x % 2 === 0 ? 0.05 : 0.0);
        }
    }
    return {
        kind: 'rim',
        resolution: makeResolution(width, height),
        gx,
        gy,
        mag,
    };
};
test('volume stub matches prerecorded fixture', async () => {
    const { recording, meta } = await readFixture();
    const volume = ingestVolumeRecording(recording);
    const stub = createVolumeStubState(meta.width, meta.height, meta.seed);
    for (let i = 0; i < meta.steps; i++) {
        stepVolumeStub(stub, meta.dt);
    }
    const snapshot = snapshotVolumeStub(stub);
    assert.equal(snapshot.phase.length, volume.phase.length);
    const tolerance = 1e-5;
    for (let i = 0; i < snapshot.phase.length; i++) {
        assert.ok(Math.abs(snapshot.phase[i] - volume.phase[i]) <= tolerance, `phase[${i}]`);
        assert.ok(Math.abs(snapshot.depth[i] - volume.depth[i]) <= tolerance, `depth[${i}]`);
        assert.ok(Math.abs(snapshot.intensity[i] - volume.intensity[i]) <= tolerance, `intensity[${i}]`);
    }
});
const CANONICAL_KERNEL = createKernelSpec({
    gain: 2.2,
    k0: 0.18,
    Q: 3.8,
    anisotropy: 0.7,
    chirality: 1.1,
    transparency: 0.32,
});
const makeInput = (volume, surface, rim) => {
    const total = surface.resolution.texels;
    return {
        width: surface.resolution.width,
        height: surface.resolution.height,
        timeSeconds: 1.2,
        out: new Uint8ClampedArray(total * 4),
        surface,
        rim,
        phase: null,
        volume,
        kernel: CANONICAL_KERNEL,
        dmt: 0.35,
        arousal: 0.25,
        blend: 0.42,
        normPin: true,
        normTarget: 0.6,
        lastObs: 0.6,
        lambdaRef: 520,
        lambdas: { L: 560, M: 530, S: 420 },
        beta2: 1.4,
        microsaccade: false,
        alive: false,
        phasePin: true,
        edgeThreshold: 0.18,
        wallpaperGroup: 'off',
        surfEnabled: true,
        orientationAngles: [0, Math.PI / 2],
        thetaMode: 'gradient',
        thetaGlobal: 0,
        polBins: 16,
        jitter: 0.4,
        coupling: {
            rimToSurfaceBlend: 0.25,
            rimToSurfaceAlign: 0.35,
            surfaceToRimOffset: 0.3,
            surfaceToRimSigma: 0.2,
            surfaceToRimHue: 0.3,
            kurToTransparency: 0,
            kurToOrientation: 0,
            kurToChirality: 0,
            volumePhaseToHue: 1.4,
            volumeDepthToWarp: 1.2,
        },
        sigma: 3.2,
        contrast: 1.15,
        rimAlpha: 1,
        rimEnabled: true,
        displayMode: 'color',
        surfaceBlend: 0.38,
        surfaceRegion: 'both',
        warpAmp: 1.4,
        curvatureStrength: 0,
        curvatureMode: 'poincare',
        kurEnabled: false,
        debug: undefined,
        su7: createDefaultSu7RuntimeParams(),
    };
};
test('volume feed modulates pipeline outputs', async () => {
    const { recording, meta } = await readFixture();
    const volume = ingestVolumeRecording(recording);
    const surface = makeSurfaceField(meta.width, meta.height);
    const rim = makeRimField(meta.width, meta.height);
    const inputWithVolume = makeInput(volume, surface, rim);
    const inputWithoutVolume = makeInput(null, surface, rim);
    const resultVolume = renderRainbowFrame(inputWithVolume);
    const resultNoVolume = renderRainbowFrame(inputWithoutVolume);
    assert.equal(resultVolume.metrics.volume.sampleCount, meta.width * meta.height);
    assert.ok(resultVolume.metrics.volume.phaseStd > 0.05);
    assert.ok(resultVolume.metrics.volume.depthGradMean > 0);
    assert.equal(resultNoVolume.metrics.volume.sampleCount, 0);
    const volumeEnergy = resultVolume.metrics.composer.fields.volume.energy;
    const baselineEnergy = resultNoVolume.metrics.composer.fields.volume.energy;
    assert.ok(volumeEnergy > baselineEnergy, `expected volume composer energy to rise (${volumeEnergy} vs ${baselineEnergy})`);
});
