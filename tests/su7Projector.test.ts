import test from 'node:test';
import assert from 'node:assert/strict';

import { makeResolution, type RimField, type SurfaceField } from '../src/fields/contracts.js';
import { embedToC7 } from '../src/pipeline/su7/embed.js';
import { projectSu7Vector } from '../src/pipeline/su7/projector.js';
import { applySU7, buildUnitary } from '../src/pipeline/su7/unitary.js';
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

const approx = (actual: number, expected: number, tol = 1e-2) => {
  assert.ok(Math.abs(actual - expected) <= tol, `expected ${expected}, got ${actual}`);
};

test('identity projector reconstructs base color after SU7 embedding', () => {
  const width = 1;
  const height = 1;
  const rgba = new Uint8ClampedArray([192, 96, 48, 255]);
  const surface = createSurface(width, height, rgba);
  const rim = createRim(
    width,
    height,
    new Float32Array([0]),
    new Float32Array([0]),
    new Float32Array([1]),
  );

  const { vectors, norms } = embedToC7({ surface, rim, gauge: 'none' });
  assert.equal(vectors.length, 1);
  const vector = vectors[0];
  const norm = norms[0];
  const unitary = buildUnitary({ preset: 'identity' });
  const transformed = applySU7(unitary, vector);
  const baseColor: [number, number, number] = [rgba[0] / 255, rgba[1] / 255, rgba[2] / 255];

  const projection = projectSu7Vector({
    vector: transformed,
    norm,
    projector: { id: 'identity', weight: 1 },
    gain: 1,
    frameGain: 1,
    baseColor,
  });

  assert.ok(projection);
  const diffR = Math.abs(projection.rgb[0] - baseColor[0]);
  const diffG = Math.abs(projection.rgb[1] - baseColor[1]);
  const diffB = Math.abs(projection.rgb[2] - baseColor[2]);
  const maxDelta = Math.max(diffR, diffG, diffB);
  assert.ok(maxDelta < 0.2, `expected color delta < 0.2, saw ${maxDelta}`);
  approx(projection.mix, 1, 1e-6);
});

const buildSyntheticVector = (entries: number[]): { vector: C7Vector; norm: number } => {
  const squares = entries.reduce((sum, value) => sum + value * value, 0);
  const norm = Math.sqrt(squares);
  const inv = norm > 0 ? 1 / norm : 1;
  const vector = entries.map((value) => ({ re: value * inv, im: 0 })) as C7Vector;
  return { vector, norm };
};

test('composer-weight projector emits per-field weight multipliers', () => {
  const { vector, norm } = buildSyntheticVector([0.5, 0.35, 0.25, 0.8, 0.15, 0.1, 0.4]);
  const baseColor: [number, number, number] = [0.35, 0.42, 0.3];
  const projection = projectSu7Vector({
    vector,
    norm,
    projector: { id: 'composerWeights', weight: 1 },
    gain: 1,
    frameGain: 1,
    baseColor,
  });
  assert.ok(projection);
  assert.ok(projection.composerWeights);
  const weights = projection.composerWeights!;
  assert.ok(weights.surface && weights.rim);
  assert.ok(weights.surface! > weights.rim!);
  assert.ok(weights.volume && weights.volume! > 0.2);
  assert.ok(projection.mix > 0);
});

test('overlay split projector provides overlay mixes bounded by main mix', () => {
  const { vector, norm } = buildSyntheticVector([0.45, 0.35, 0.25, 0.65, 0.3, 0.2, 0.5]);
  const baseColor: [number, number, number] = [0.28, 0.33, 0.4];
  const projection = projectSu7Vector({
    vector,
    norm,
    projector: { id: 'overlaySplit', weight: 0.8 },
    gain: 1,
    frameGain: 1,
    baseColor,
  });
  assert.ok(projection);
  assert.ok(projection.overlays && projection.overlays.length > 0);
  const mixSum = projection.overlays!.reduce((sum, overlay) => sum + overlay.mix, 0);
  assert.ok(mixSum <= projection.mix + 1e-6);
  projection.overlays!.forEach((overlay) => {
    overlay.rgb.forEach((channel) => {
      assert.ok(channel >= 0 && channel <= 1);
    });
  });
});

test('flux overlay sample adds overlay entry and respects mix bounds', () => {
  const { vector, norm } = buildSyntheticVector([0.4, 0.3, 0.2, 0.5, 0.25, 0.15, 0.35]);
  const baseColor: [number, number, number] = [0.32, 0.41, 0.38];
  const projection = projectSu7Vector({
    vector,
    norm,
    projector: { id: 'identity', weight: 1 },
    gain: 1,
    frameGain: 1,
    baseColor,
    fluxOverlay: {
      energy: 0.85,
      energyScale: 1,
      dirX: 1,
      dirY: 0.1,
    },
  });
  assert.ok(projection);
  assert.ok(projection.overlays && projection.overlays.length > 0, 'flux overlay missing');
  const fluxOverlay = projection.overlays!.at(-1)!;
  assert.ok(fluxOverlay.mix > 0, 'flux overlay mix should be positive');
  assert.ok(fluxOverlay.mix <= projection.mix + 1e-6, 'flux overlay exceeds primary mix');
  fluxOverlay.rgb.forEach((channel) => {
    assert.ok(channel >= 0 && channel <= 1, 'flux overlay color out of range');
  });
});
