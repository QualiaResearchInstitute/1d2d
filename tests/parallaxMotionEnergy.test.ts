import test from "node:test";
import assert from "node:assert/strict";

import {
  makeResolution,
  type SurfaceField,
  type RimField,
  type PhaseField
} from "../src/fields/contracts.js";
import {
  renderRainbowFrame,
  type CouplingConfig
} from "../src/pipeline/rainbowFrame.js";
import { createKernelSpec } from "../src/kernel/kernelSpec.js";

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
    kind: "surface",
    resolution: makeResolution(width, height),
    rgba
  };
};

const createRimField = (width: number, height: number): RimField => {
  const total = width * height;
  return {
    kind: "rim",
    resolution: makeResolution(width, height),
    gx: new Float32Array(total),
    gy: new Float32Array(total),
    mag: new Float32Array(total)
  };
};

const createRadialPhaseField = (width: number, height: number): PhaseField => {
  const total = width * height;
  const gradX = new Float32Array(total);
  const gradY = new Float32Array(total);
  const vort = new Float32Array(total);
  const coh = new Float32Array(total);
  const amp = new Float32Array(total);
  const cx = (width - 1) * 0.5;
  const cy = (height - 1) * 0.5;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const dx = x - cx;
      const dy = y - cy;
      gradX[idx] = dx / Math.max(width, 1);
      gradY[idx] = dy / Math.max(height, 1);
      vort[idx] = 0;
      coh[idx] = 0.6;
      amp[idx] = 0.6;
    }
  }
  return {
    kind: "phase",
    resolution: makeResolution(width, height),
    gradX,
    gradY,
    vort,
    coh,
    amp
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
  volumeDepthToWarp: 0
};

test("parallax metrics report positive radial slope for radial gradients", () => {
  const width = 32;
  const height = 32;
  const surface = createSurface(width, height, 0.5);
  const rim = createRimField(width, height);
  const phase = createRadialPhaseField(width, height);
  const out = new Uint8ClampedArray(width * height * 4);

  const result = renderRainbowFrame({
    width,
    height,
    timeSeconds: 0,
    out,
    surface,
    rim,
    phase,
    volume: null,
    kernel: createKernelSpec({
      gain: 1.4,
      k0: 0.18,
      Q: 2.2,
      anisotropy: 0.5,
      chirality: 0.3,
      transparency: 0.45
    }),
    dmt: 0,
    blend: 0.4,
    normPin: false,
    normTarget: 0.6,
    lastObs: 0.6,
    lambdaRef: 520,
    lambdas: { L: 560, M: 530, S: 420 },
    beta2: 0.1,
    microsaccade: false,
    alive: false,
    phasePin: true,
    edgeThreshold: 1,
    wallpaperGroup: "off",
    surfEnabled: false,
    orientationAngles: [],
    thetaMode: "gradient",
    thetaGlobal: 0,
    polBins: 0,
    jitter: 0,
    coupling: zeroCoupling,
    sigma: 1,
    contrast: 1,
    rimAlpha: 1,
    rimEnabled: false,
    displayMode: "color",
    surfaceBlend: 0,
    surfaceRegion: "surfaces",
    warpAmp: 0,
    kurEnabled: true,
    debug: undefined,
    composer: undefined
  });

  const parallax = result.metrics.parallax;
  assert.ok(parallax.sampleCount > 0, "expected parallax samples");
  assert.ok(parallax.radialSlope > 0.02, `radial slope too small (${parallax.radialSlope})`);
  assert.ok(parallax.tagConsistency > 0.7, `tag consistency low (${parallax.tagConsistency})`);
});

test("motion energy captures cross-branch phase shift for multi-orientation overlays", () => {
  const width = 32;
  const height = 32;
  const surface = createSurface(width, height, 0.5);
  const rim = createRimField(width, height);
  const out = new Uint8ClampedArray(width * height * 4);

  const result = renderRainbowFrame({
    width,
    height,
    timeSeconds: 0.5,
    out,
    surface,
    rim,
    phase: null,
    volume: null,
    kernel: createKernelSpec({
      gain: 1.2,
      k0: 0.16,
      Q: 2.4,
      anisotropy: 0.4,
      chirality: 0.2,
      transparency: 0.3
    }),
    dmt: 0.1,
    blend: 0.35,
    normPin: false,
    normTarget: 0.6,
    lastObs: 0.6,
    lambdaRef: 520,
    lambdas: { L: 560, M: 530, S: 420 },
    beta2: 0.6,
    microsaccade: false,
    alive: false,
    phasePin: false,
    edgeThreshold: 1,
    wallpaperGroup: "off",
    surfEnabled: true,
    orientationAngles: [0, Math.PI / 2],
    thetaMode: "gradient",
    thetaGlobal: 0,
    polBins: 0,
    jitter: 0,
    coupling: zeroCoupling,
    sigma: 1.5,
    contrast: 1,
    rimAlpha: 1,
    rimEnabled: false,
    displayMode: "color",
    surfaceBlend: 0.2,
    surfaceRegion: "surfaces",
    warpAmp: 0.2,
    kurEnabled: false,
    debug: undefined,
    composer: undefined
  });

  const motionEnergy = result.metrics.motionEnergy;
  assert.ok(motionEnergy.branchCount >= 2, "expected multiple motion branches");
  assert.equal(motionEnergy.source, "orientation");
  assert.ok(
    motionEnergy.phaseShiftMean > 0.05,
    `phase shift too small (${motionEnergy.phaseShiftMean})`
  );
});

