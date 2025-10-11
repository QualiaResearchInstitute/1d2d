import test from 'node:test';
import assert from 'node:assert/strict';

import {
  renderRainbowFrame,
  type RainbowFrameInput,
  createDefaultComposerConfig,
  type CouplingConfig,
} from '../src/pipeline/rainbowFrame.js';
import { createDefaultSu7RuntimeParams, type Su7RuntimeParams } from '../src/pipeline/su7/types.js';
import { embedToC7 } from '../src/pipeline/su7/embed.js';
import {
  computeProjectorEnergy,
  buildScheduledUnitary,
  computeUnitaryError,
  type Su7ScheduleContext,
} from '../src/pipeline/su7/math.js';
import { applySU7 } from '../src/pipeline/su7/unitary.js';
import { createKernelSpec, type KernelSpec } from '../src/kernel/kernelSpec.js';
import { makeResolution, type SurfaceField, type RimField } from '../src/fields/contracts.js';

const FRAME_WIDTH = 12;
const FRAME_HEIGHT = 10;
const TOTAL_TEXELS = FRAME_WIDTH * FRAME_HEIGHT;

const TEST_KERNEL: KernelSpec = createKernelSpec({
  gain: 1.6,
  k0: 0.18,
  Q: 3.0,
  anisotropy: 0.52,
  chirality: 0.58,
  transparency: 0.24,
});

const TEST_COUPLING: CouplingConfig = {
  rimToSurfaceBlend: 0.3,
  rimToSurfaceAlign: 0.35,
  surfaceToRimOffset: 0.22,
  surfaceToRimSigma: 0.3,
  surfaceToRimHue: 0.42,
  kurToTransparency: 0.18,
  kurToOrientation: 0.28,
  kurToChirality: 0.25,
  volumePhaseToHue: 0.5,
  volumeDepthToWarp: 0.45,
};

const approx = (actual: number, expected: number, eps = 1e-6) => {
  assert.ok(Math.abs(actual - expected) <= eps, `expected ${expected}, got ${actual}`);
};

const buildSurfaceField = (width: number, height: number): SurfaceField => {
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

const buildRimField = (
  width: number,
  height: number,
  { zeroMag = false }: { zeroMag?: boolean } = {},
): RimField => {
  const gx = new Float32Array(width * height);
  const gy = new Float32Array(width * height);
  const mag = new Float32Array(width * height);
  const cx = (width - 1) * 0.5;
  const cy = (height - 1) * 0.5;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const dx = x - cx;
      const dy = y - cy;
      gx[idx] = dx * 0.015;
      gy[idx] = dy * 0.015;
      mag[idx] = zeroMag ? 0 : Math.hypot(dx, dy) * 0.02 + (x % 3 === 0 ? 0.05 : 0.02);
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

const createFrameInput = (
  su7: Su7RuntimeParams,
  { zeroRim = false }: { zeroRim?: boolean } = {},
): RainbowFrameInput => {
  const surface = buildSurfaceField(FRAME_WIDTH, FRAME_HEIGHT);
  const rim = buildRimField(FRAME_WIDTH, FRAME_HEIGHT, { zeroMag: zeroRim });
  return {
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    timeSeconds: 0.5,
    out: new Uint8ClampedArray(TOTAL_TEXELS * 4),
    surface,
    rim,
    phase: null,
    volume: null,
    kernel: TEST_KERNEL,
    dmt: 0.32,
    arousal: 0.3,
    blend: 0.41,
    normPin: true,
    normTarget: 0.6,
    lastObs: 0.6,
    lambdaRef: 520,
    lambdas: { L: 560, M: 530, S: 420 },
    beta2: 1.25,
    microsaccade: false,
    alive: false,
    phasePin: true,
    edgeThreshold: 0.2,
    wallpaperGroup: 'off',
    surfEnabled: true,
    orientationAngles: [0, Math.PI / 2],
    thetaMode: 'gradient',
    thetaGlobal: 0,
    polBins: 8,
    jitter: 0.36,
    coupling: TEST_COUPLING,
    sigma: 2.8,
    contrast: 1.12,
    rimAlpha: 1,
    rimEnabled: true,
    displayMode: 'color',
    surfaceBlend: 0.33,
    surfaceRegion: 'both',
    warpAmp: 0.22,
    curvatureStrength: 0,
    curvatureMode: 'poincare',
    kurEnabled: false,
    su7,
    composer: createDefaultComposerConfig(),
    attentionHooks: undefined,
  };
};

const initSu7 = <T extends Record<string, unknown>>(overrides: T): Su7RuntimeParams & T =>
  ({
    ...createDefaultSu7RuntimeParams(),
    ...overrides,
  }) as Su7RuntimeParams & T;

const cloneRuntime = <T extends Record<string, unknown>>(
  runtime: Su7RuntimeParams & T,
): Su7RuntimeParams & T => {
  const scheduleClone = runtime.schedule.map((stage) => ({ ...stage })) as typeof runtime.schedule;
  const projectorClone = { ...runtime.projector };
  return {
    ...runtime,
    schedule: scheduleClone,
    projector: projectorClone,
  };
};

const renderFrame = <T extends Record<string, unknown>>(
  runtime: Su7RuntimeParams & T,
  opts?: { zeroRim?: boolean },
) => {
  const su7 = cloneRuntime(runtime);
  const input = createFrameInput(su7 as Su7RuntimeParams, opts);
  const result = renderRainbowFrame(input);
  return {
    pixels: Uint8ClampedArray.from(input.out),
    metrics: result.metrics,
  };
};

const maxChannelDelta = (a: Uint8ClampedArray, b: Uint8ClampedArray): number => {
  let max = 0;
  for (let i = 0; i < a.length; i += 4) {
    max = Math.max(
      max,
      Math.abs(a[i + 0] - b[i + 0]),
      Math.abs(a[i + 1] - b[i + 1]),
      Math.abs(a[i + 2] - b[i + 2]),
    );
  }
  return max;
};

const countChangedPixels = (a: Uint8ClampedArray, b: Uint8ClampedArray): number => {
  let count = 0;
  for (let i = 0; i < a.length; i += 4) {
    if (a[i + 0] !== b[i + 0] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2]) {
      count++;
    }
  }
  return count;
};

const countDecimatedPixels = (width: number, height: number, stride: number): number => {
  let total = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (x % stride === 0 && y % stride === 0) {
        total++;
      }
    }
  }
  return total;
};

test('computeProjectorEnergy scales with identity weight', () => {
  const energyHalf = computeProjectorEnergy({ id: 'identity', weight: 0.5 });
  approx(energyHalf, 7 * 0.25, 1e-9);
  const energyUnit = computeProjectorEnergy({ id: 'identity', weight: 1 });
  approx(energyUnit, 7, 1e-9);
});

test('per-pixel SU7 norms remain invariant after transformation', () => {
  const su7 = initSu7({
    enabled: true,
    gain: 0.75,
    preset: 'random',
    seed: 1337,
    projector: { id: 'identity', weight: 0 },
    decimationStride: 1,
    decimationMode: 'hybrid',
  });
  const surface = buildSurfaceField(FRAME_WIDTH, FRAME_HEIGHT);
  const rim = buildRimField(FRAME_WIDTH, FRAME_HEIGHT);
  const { vectors } = embedToC7({
    surface,
    rim,
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    gauge: 'rim',
  });
  const unitary = buildScheduledUnitary(su7);
  let maxDelta = 0;
  let sumDelta = 0;
  let samples = 0;
  for (const vector of vectors) {
    if (!vector) continue;
    const beforeNorm = Math.sqrt(
      vector.reduce((acc, cell) => acc + cell.re * cell.re + cell.im * cell.im, 0),
    );
    const transformed = applySU7(unitary, vector);
    const afterNorm = Math.sqrt(
      transformed.reduce((acc, cell) => acc + cell.re * cell.re + cell.im * cell.im, 0),
    );
    const delta = Math.abs(afterNorm - beforeNorm);
    if (delta > maxDelta) {
      maxDelta = delta;
    }
    sumDelta += delta;
    samples += 1;
  }
  assert.ok(maxDelta <= 1e-6, `max delta ${maxDelta}`);
  assert.ok(samples > 0);
  assert.ok(sumDelta / samples <= 1e-7, `mean delta ${sumDelta / samples}`);
});

test('projector energy stays within unit luminance budget', () => {
  const su7 = initSu7({
    enabled: true,
    gain: 0.85,
    preset: 'random',
    seed: 2025,
    projector: { id: 'identity', weight: 1 },
    decimationStride: 1,
    decimationMode: 'hybrid',
  });
  const { metrics } = renderFrame(su7);
  assert.ok(
    metrics.su7.projectorEnergy <= 1 + 1e-6,
    `projectorEnergy=${metrics.su7.projectorEnergy}`,
  );
  assert.ok(metrics.su7.projectorEnergy >= 0);
});

test('identity unitary and projector leave baseline output unchanged', () => {
  const baseline = renderFrame(createDefaultSu7RuntimeParams());
  const identity = initSu7({
    enabled: true,
    gain: 0,
    preset: 'identity',
    seed: 0,
    projector: { id: 'identity', weight: 1 },
    decimationStride: 1,
    decimationMode: 'hybrid',
  });
  const identityResult = renderFrame(identity);
  assert.equal(countChangedPixels(baseline.pixels, identityResult.pixels), 0);
  assert.equal(maxChannelDelta(baseline.pixels, identityResult.pixels), 0);
  assert.equal(identityResult.metrics.su7.normDeltaMax, 0);
  assert.equal(identityResult.metrics.su7.projectorEnergy, 0);
});

test('seeded SU7 runtime produces stable outputs across runs', () => {
  const seeded = initSu7({
    enabled: true,
    gain: 0.68,
    preset: 'random',
    seed: 424242,
    projector: { id: 'identity', weight: 0.65 },
    decimationStride: 1,
    decimationMode: 'hybrid',
  });
  const first = renderFrame(seeded);
  const second = renderFrame(seeded);
  assert.equal(maxChannelDelta(first.pixels, second.pixels), 0);
  assert.deepEqual(first.metrics.su7, second.metrics.su7);
});

test('contextual SU7 scheduling stays unitary and responds to flow cues', () => {
  const contextualParams = initSu7({
    enabled: true,
    gain: 1.05,
    preset: 'random',
    seed: 2468,
    schedule: [
      { gain: 0.42, index: 2, spread: 0.7 },
      { gain: 0.35, spread: 1.1, label: 'context' },
    ],
    projector: { id: 'identity', weight: 0 },
  });
  const context: Su7ScheduleContext = {
    dmt: 0.45,
    arousal: 0.55,
    flow: {
      angle: Math.PI / 4,
      magnitude: 0.9,
      coherence: 0.7,
      axisBias: new Float32Array([0.9, 1.2, 1.0, 0.7, 0.6, 0.8, 1.1]),
      gridSize: 2,
      gridVectors: new Float32Array([0.4, 0.15, 0.05, -0.2, 0.18, 0.22, -0.12, 0.08]),
    },
    curvatureStrength: 0.3,
    parallaxRadial: 0.25,
    volumeCoverage: 0.4,
  };
  const legacy = buildScheduledUnitary(contextualParams);
  const contextual = buildScheduledUnitary(contextualParams, context);
  const unitaryError = computeUnitaryError(contextual);
  assert.ok(unitaryError < 1e-6, `unitary error ${unitaryError}`);
  let diffMagnitude = 0;
  let maxOffDiag = 0;
  for (let row = 0; row < 7; row++) {
    for (let col = 0; col < 7; col++) {
      const deltaRe = contextual[row][col].re - legacy[row][col].re;
      const deltaIm = contextual[row][col].im - legacy[row][col].im;
      diffMagnitude += Math.abs(deltaRe) + Math.abs(deltaIm);
      if (row !== col) {
        maxOffDiag = Math.max(
          maxOffDiag,
          Math.hypot(contextual[row][col].re, contextual[row][col].im),
        );
      }
    }
  }
  assert.ok(
    diffMagnitude > 0.05,
    `expected contextual schedule to differ from legacy (diff=${diffMagnitude})`,
  );
  assert.ok(maxOffDiag > 1e-3, `expected contextual mixing pulses, maxOffDiag=${maxOffDiag}`);
});

test('decimation stride limits SU7 mixing footprint', () => {
  const baseline = renderFrame(createDefaultSu7RuntimeParams(), { zeroRim: true });
  const stride = 3;
  const decimated = initSu7({
    enabled: true,
    gain: 1,
    preset: 'random',
    seed: 777,
    projector: { id: 'identity', weight: 1 },
    decimationStride: stride,
    decimationMode: 'stride',
  });
  const result = renderFrame(decimated, { zeroRim: true });
  const changed = countChangedPixels(baseline.pixels, result.pixels);
  const expectedMax = countDecimatedPixels(FRAME_WIDTH, FRAME_HEIGHT, stride);
  assert.ok(changed > 0, 'expected SU7 mix to alter at least one pixel');
  assert.ok(changed <= expectedMax, `changed ${changed} exceeds decimated budget ${expectedMax}`);
  assert.ok(maxChannelDelta(baseline.pixels, result.pixels) > 0);
});
