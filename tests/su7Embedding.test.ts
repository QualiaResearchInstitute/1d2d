import test from 'node:test';
import assert from 'node:assert/strict';

import {
  makeResolution,
  type PhaseField,
  type RimField,
  type SurfaceField,
  type VolumeField,
} from '../src/fields/contracts.js';
import { embedToC7, type GaugeMode } from '../src/pipeline/su7/embed.js';
import type { C7Vector } from '../src/pipeline/su7/types.js';

const createSurface = (width: number, height: number, rgba: Uint8ClampedArray): SurfaceField => ({
  kind: 'surface',
  resolution: makeResolution(width, height),
  rgba,
});

const createRim = (
  width: number,
  height: number,
  gx: Float32Array,
  gy: Float32Array,
  mag: Float32Array,
): RimField => ({
  kind: 'rim',
  resolution: makeResolution(width, height),
  gx,
  gy,
  mag,
});

const createPhase = (
  width: number,
  height: number,
  data: {
    gradX: Float32Array;
    gradY: Float32Array;
    vort: Float32Array;
    coh: Float32Array;
    amp: Float32Array;
  },
): PhaseField => ({
  kind: 'phase',
  resolution: makeResolution(width, height),
  gradX: data.gradX,
  gradY: data.gradY,
  vort: data.vort,
  coh: data.coh,
  amp: data.amp,
});

const createVolume = (width: number, height: number, depth: Float32Array): VolumeField => ({
  kind: 'volume',
  resolution: makeResolution(width, height),
  phase: new Float32Array(width * height),
  depth,
  intensity: new Float32Array(width * height),
});

const normOf = (vector: C7Vector): number => {
  let sum = 0;
  for (const cell of vector) {
    sum += cell.re * cell.re + cell.im * cell.im;
  }
  return Math.sqrt(sum);
};

const approx = (actual: number, expected: number, eps = 1e-6) => {
  assert.ok(Math.abs(actual - expected) <= eps, `expected ${expected}, got ${actual}`);
};

const runEmbed = (params: {
  surface?: SurfaceField | null;
  rim?: RimField | null;
  phase?: PhaseField | null;
  volume?: VolumeField | null;
  width?: number;
  height?: number;
  gauge?: GaugeMode;
}) => embedToC7(params).vectors;

test('vorticity channel dominates when other signals absent', () => {
  const width = 1;
  const height = 1;
  const phase = createPhase(width, height, {
    gradX: new Float32Array([0]),
    gradY: new Float32Array([0]),
    vort: new Float32Array([0.5]),
    coh: new Float32Array([0]),
    amp: new Float32Array([0]),
  });

  const vectors = runEmbed({ phase, gauge: 'none' });
  assert.equal(vectors.length, 1);
  const v = vectors[0];
  approx(normOf(v), 1, 1e-6);
  v.forEach((cell, idx) => {
    if (idx === 4) {
      approx(cell.re, 0, 1e-6);
      approx(cell.im, 1, 1e-6);
    } else {
      approx(cell.re, 0, 1e-6);
      approx(cell.im, 0, 1e-6);
    }
  });
});

test('rim LMS embedding follows configured phase offsets', () => {
  const width = 1;
  const height = 1;
  const rgba = new Uint8ClampedArray([255, 0, 0, 255]);
  const surface = createSurface(width, height, rgba);
  const rim = createRim(
    width,
    height,
    new Float32Array([0]),
    new Float32Array([0]),
    new Float32Array([1]),
  );

  const vectors = runEmbed({ surface, rim, gauge: 'none' });
  const v = vectors[0];
  approx(normOf(v), 1, 1e-6);
  approx(v[0].re, 0.8951237744555596, 1e-6);
  approx(v[0].im, 0, 1e-6);
  approx(v[1].re, -0.22146794585744847, 1e-6);
  approx(v[1].im, 0.38359373447301415, 1e-6);
  approx(v[2].re, -0.025304269576305827, 1e-6);
  approx(v[2].im, -0.04382828055458102, 1e-6);
  for (let i = 3; i < 7; i++) {
    approx(v[i].re, 0, 1e-6);
    approx(v[i].im, 0, 1e-6);
  }
});

test('gauge rotation aligns phase gradient to rim orientation', () => {
  const width = 1;
  const height = 1;
  const rim = createRim(
    width,
    height,
    new Float32Array([0]),
    new Float32Array([1]),
    new Float32Array([0]),
  );
  const phase = createPhase(width, height, {
    gradX: new Float32Array([1]),
    gradY: new Float32Array([0]),
    vort: new Float32Array([0]),
    coh: new Float32Array([0]),
    amp: new Float32Array([0]),
  });

  const vectors = runEmbed({ rim, phase, gauge: 'rim' });
  const v = vectors[0];
  approx(normOf(v), 1, 1e-6);
  v.forEach((cell, idx) => {
    if (idx === 3) {
      approx(cell.re, 0, 1e-6);
      approx(cell.im, -1, 1e-6);
    } else {
      approx(cell.re, 0, 1e-6);
      approx(cell.im, 0, 1e-6);
    }
  });
});

test('volume depth gradient maps to radial phase when available', () => {
  const width = 2;
  const height = 2;
  const depth = new Float32Array([0, 2, 1, 3]);
  const volume = createVolume(width, height, depth);

  const vectors = runEmbed({ volume, gauge: 'none' });
  assert.equal(vectors.length, width * height);
  const v = vectors[3];
  approx(normOf(v), 1, 1e-6);
  for (let i = 0; i < 6; i++) {
    approx(v[i].re, 0, 1e-6);
    approx(v[i].im, 0, 1e-6);
  }
  approx(v[6].re, Math.SQRT1_2, 1e-6);
  approx(v[6].im, Math.SQRT1_2, 1e-6);
});

test('falls back to default vector when no signals provided', () => {
  const vectors = runEmbed({ width: 1, height: 1, gauge: 'none' });
  assert.equal(vectors.length, 1);
  const v = vectors[0];
  approx(normOf(v), 1, 1e-6);
  approx(v[0].re, 1, 1e-6);
  approx(v[0].im, 0, 1e-6);
  for (let i = 1; i < 7; i++) {
    approx(v[i].re, 0, 1e-6);
    approx(v[i].im, 0, 1e-6);
  }
});
