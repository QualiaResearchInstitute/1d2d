import test from 'node:test';
import assert from 'node:assert/strict';

import {
  renderRainbowFrame,
  computeHyperbolicFlowScale,
  type CouplingConfig,
  type CurvatureMode,
} from '../src/pipeline/rainbowFrame.js';
import { createDefaultSu7RuntimeParams } from '../src/pipeline/su7/types.js';
import { createHyperbolicAtlas, type HyperbolicAtlas } from '../src/hyperbolic/atlas.js';
import { createKernelSpec } from '../src/kernel/kernelSpec.js';
import { makeResolution, type SurfaceField, type RimField } from '../src/fields/contracts.js';

const SURFACE_HIST_BINS = 32;

const createSurface = (width: number, height: number, value: number): SurfaceField => {
  const rgba = new Uint8ClampedArray(width * height * 4);
  const v = Math.max(0, Math.min(255, Math.round(value * 255)));
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    rgba[idx + 0] = v;
    rgba[idx + 1] = v;
    rgba[idx + 2] = v;
    rgba[idx + 3] = 255;
  }
  return {
    kind: 'surface',
    resolution: makeResolution(width, height),
    rgba,
  };
};

const createRimField = (width: number, height: number): RimField => {
  const total = width * height;
  return {
    kind: 'rim',
    resolution: makeResolution(width, height),
    gx: new Float32Array(total),
    gy: new Float32Array(total),
    mag: new Float32Array(total),
  };
};

const zeroCoupling: CouplingConfig = {
  rimToSurfaceBlend: 0,
  rimToSurfaceAlign: 0,
  surfaceToRimOffset: 0,
  surfaceToRimSigma: 0,
  surfaceToRimHue: 0,
  kurToTransparency: 0,
  kurToOrientation: 0,
  kurToChirality: 0,
  volumePhaseToHue: 0,
  volumeDepthToWarp: 0,
};

const createSurfaceDebugRequest = (orientationCount: number, texels: number) => {
  const magnitudeHist = new Float32Array(orientationCount * SURFACE_HIST_BINS);
  const phases = Array.from({ length: orientationCount }, () => new Float32Array(texels));
  const magnitudes = Array.from({ length: orientationCount }, () => new Float32Array(texels));
  return {
    phases,
    magnitudes,
    magnitudeHist,
    orientationCount,
    flowVectors: {
      x: new Float32Array(texels),
      y: new Float32Array(texels),
      hyperbolicScale: new Float32Array(texels),
    },
  };
};

const renderFrame = (
  atlas: HyperbolicAtlas | null,
  curvatureStrength: number,
  surfaceDebug: ReturnType<typeof createSurfaceDebugRequest>,
  curvatureMode: CurvatureMode = 'poincare',
) => {
  const width = surfaceDebug.phases[0].length
    ? Math.round(Math.sqrt(surfaceDebug.phases[0].length))
    : 0;
  const height = width;
  const out = new Uint8ClampedArray(width * height * 4);
  return renderRainbowFrame({
    width,
    height,
    timeSeconds: 0,
    out,
    surface: createSurface(width, height, 0.5),
    rim: createRimField(width, height),
    phase: null,
    volume: null,
    kernel: createKernelSpec({
      gain: 1.1,
      k0: 0.2,
      Q: 2.1,
      anisotropy: 0.35,
      chirality: 0.25,
      transparency: 0.4,
    }),
    dmt: 0,
    arousal: 0,
    blend: 0.4,
    normPin: false,
    normTarget: 0.6,
    lastObs: 0.6,
    lambdaRef: 520,
    lambdas: { L: 560, M: 530, S: 420 },
    beta2: 0.15,
    microsaccade: false,
    alive: false,
    phasePin: false,
    edgeThreshold: 0,
    wallpaperGroup: 'off',
    surfEnabled: true,
    orientationAngles: [0],
    thetaMode: 'gradient',
    thetaGlobal: 0,
    polBins: 0,
    jitter: 0,
    coupling: zeroCoupling,
    sigma: 1,
    contrast: 1,
    rimAlpha: 1,
    rimEnabled: false,
    displayMode: 'color',
    surfaceBlend: 0.35,
    surfaceRegion: 'surfaces',
    warpAmp: 0.2,
    curvatureStrength,
    curvatureMode,
    hyperbolicAtlas: atlas,
    kurEnabled: false,
    debug: {
      surface: surfaceDebug,
    },
    su7: createDefaultSu7RuntimeParams(),
    composer: undefined,
  });
};

test('hyperbolic flow vectors apply sinh-based scaling', () => {
  const width = 24;
  const height = 24;
  const texels = width * height;
  const orientationCount = 1;
  const flatDebug = createSurfaceDebugRequest(orientationCount, texels);
  renderFrame(null, 0, flatDebug);

  const atlas = createHyperbolicAtlas({
    width,
    height,
    curvatureStrength: 0.6,
    mode: 'poincare',
  });
  const curvedDebug = createSurfaceDebugRequest(orientationCount, texels);
  renderFrame(atlas, 0.6, curvedDebug);

  const targetY = Math.floor(height * 0.75);
  const targetX = Math.floor(width * 0.8);
  const idx = targetY * width + targetX;
  const flatFlowX = flatDebug.flowVectors!.x[idx];
  const flatFlowY = flatDebug.flowVectors!.y[idx];
  const flatMag = Math.hypot(flatFlowX, flatFlowY);

  const curvedFlowX = curvedDebug.flowVectors!.x[idx];
  const curvedFlowY = curvedDebug.flowVectors!.y[idx];
  const curvedMag = Math.hypot(curvedFlowX, curvedFlowY);
  const recordedScale = curvedDebug.flowVectors!.hyperbolicScale![idx];

  const hyperRadius = atlas.polar[idx * 2];
  const expectedScale = computeHyperbolicFlowScale(hyperRadius, 0.6);

  assert.ok(
    Math.abs(recordedScale - expectedScale) < 1e-3,
    `recorded scale ${recordedScale} differs from expected ${expectedScale}`,
  );

  assert.ok(flatMag > 1e-6, 'expected non-zero baseline flow magnitude');

  const jacOffset = idx * 4;
  const jacobians = atlas.jacobians;
  const atlasSampleScale = atlas.metadata.maxRadius;
  const j11 = jacobians[jacOffset] * atlasSampleScale;
  const j12 = jacobians[jacOffset + 1] * atlasSampleScale;
  const j21 = jacobians[jacOffset + 2] * atlasSampleScale;
  const j22 = jacobians[jacOffset + 3] * atlasSampleScale;
  const det = j11 * j22 - j12 * j21;
  assert.ok(Math.abs(det) > 1e-6, 'expected invertible jacobian at sample pixel');
  const invDet = 1 / det;
  const flatHyperX = (j22 * flatFlowX - j12 * flatFlowY) * invDet;
  const flatHyperY = (-j21 * flatFlowX + j11 * flatFlowY) * invDet;
  const transformedMag = Math.hypot(flatHyperX, flatHyperY);
  const expectedRatio = expectedScale * (transformedMag / flatMag);
  const ratio = curvedMag / flatMag;
  const relError = Math.abs(ratio - expectedRatio) / Math.max(expectedRatio, 1e-6);
  assert.ok(
    relError < 0.12,
    `scaled magnitude ratio ${ratio} deviates from expected ${expectedRatio} (rel err ${relError})`,
  );
});

test('rainbow frame curvature/projection combinations stay stable', () => {
  const width = 32;
  const height = 32;
  const texels = width * height;
  const orientationCount = 1;
  const combos: Array<{ strength: number; mode: CurvatureMode }> = [
    { strength: 0, mode: 'poincare' },
    { strength: 0.6, mode: 'poincare' },
    { strength: 0.6, mode: 'klein' },
  ];

  const hyperAverages: number[] = [];

  for (const { strength, mode } of combos) {
    const atlas =
      strength > 1e-4
        ? createHyperbolicAtlas({
            width,
            height,
            curvatureStrength: strength,
            mode,
          })
        : null;
    const debug = createSurfaceDebugRequest(orientationCount, texels);
    const result = renderFrame(atlas, strength, debug, mode);

    const { metrics } = result;
    assert.ok(
      Number.isFinite(metrics.compositor.effectiveBlend),
      'effective blend should be finite',
    );
    assert.ok(
      metrics.compositor.surfaceCount > 0,
      'surface compositor metrics should accumulate samples',
    );
    assert.ok(Number.isFinite(metrics.parallax.radialSlope), 'parallax slope should be finite');
    assert.ok(
      Number.isFinite(metrics.motionEnergy.parallaxMean),
      'motion energy stats should be finite',
    );

    const hyperScales = debug.flowVectors.hyperbolicScale;
    let hyperSum = 0;
    let positiveCount = 0;
    for (let i = 0; i < hyperScales.length; i++) {
      const value = hyperScales[i];
      assert.ok(Number.isFinite(value), 'hyperbolic scale contained NaN/Inf');
      hyperSum += value;
      if (value > 1e-4) {
        positiveCount++;
      }
    }
    const hyperAvg = hyperSum / hyperScales.length;
    hyperAverages.push(hyperAvg);

    if (strength <= 1e-4) {
      assert.ok(
        positiveCount === 0 && hyperAvg < 1e-5,
        'flat curvature should not accumulate hyperbolic scale',
      );
    } else {
      assert.ok(
        positiveCount > 0 && hyperAvg > 1e-4,
        `expected hyperbolic scale to register for ${mode} projection`,
      );
    }
  }

  assert.ok(
    hyperAverages[1] > 0 && hyperAverages[2] > 0,
    'curved cases should produce positive hyperbolic scale averages',
  );
  const projectionGap = Math.abs(hyperAverages[1] - hyperAverages[2]);
  assert.ok(projectionGap > 5e-4, 'poincaré vs klein projections should diverge measurably');
});

test('branch attention hook captures orientation summaries with hyperbolic gain', () => {
  const width = 16;
  const height = 16;
  const texels = width * height;
  const orientationAngles = [0, Math.PI / 2];
  const surfaceDebug = createSurfaceDebugRequest(orientationAngles.length, texels);
  const atlas = createHyperbolicAtlas({
    width,
    height,
    curvatureStrength: 0.55,
    mode: 'poincare',
  });

  let orientationSummaryCount = 0;
  let lastOrientationSummaryAverageGain = 0;

  renderRainbowFrame({
    width,
    height,
    timeSeconds: 0,
    out: new Uint8ClampedArray(texels * 4),
    surface: createSurface(width, height, 0.5),
    rim: createRimField(width, height),
    phase: null,
    volume: null,
    kernel: createKernelSpec({
      gain: 1.15,
      k0: 0.18,
      Q: 2.2,
      anisotropy: 0.4,
      chirality: 0.3,
      transparency: 0.45,
    }),
    dmt: 0,
    arousal: 0,
    blend: 0.4,
    normPin: false,
    normTarget: 0.6,
    lastObs: 0.6,
    lambdaRef: 520,
    lambdas: { L: 560, M: 530, S: 420 },
    beta2: 0.2,
    microsaccade: false,
    alive: false,
    phasePin: false,
    edgeThreshold: 0,
    wallpaperGroup: 'off',
    surfEnabled: true,
    orientationAngles,
    thetaMode: 'gradient',
    thetaGlobal: 0,
    polBins: 0,
    jitter: 0,
    coupling: zeroCoupling,
    sigma: 1,
    contrast: 1,
    rimAlpha: 1,
    rimEnabled: false,
    displayMode: 'color',
    surfaceBlend: 0,
    surfaceRegion: 'surfaces',
    warpAmp: 0,
    curvatureStrength: 0.55,
    curvatureMode: 'poincare',
    hyperbolicAtlas: atlas,
    kurEnabled: false,
    debug: {
      surface: surfaceDebug,
    },
    su7: createDefaultSu7RuntimeParams(),
    composer: undefined,
    attentionHooks: {
      onOrientation(summary) {
        orientationSummaryCount += 1;
        const gainMean =
          summary.entries.reduce((acc, entry) => acc + entry.hyperbolicGain, 0) /
          Math.max(summary.entries.length, 1);
        lastOrientationSummaryAverageGain = gainMean;
      },
    },
  });

  assert.equal(orientationSummaryCount, 1, 'expected one orientation summary');
  assert.ok(
    lastOrientationSummaryAverageGain > 1.01,
    `expected hyperbolic gain > 1, got ${lastOrientationSummaryAverageGain}`,
  );
});
