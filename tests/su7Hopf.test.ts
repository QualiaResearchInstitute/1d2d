import test from 'node:test';
import assert from 'node:assert/strict';

import {
  computeHopfCoordinates,
  hopfBaseToRgb,
  hopfFiberToRgb,
} from '../src/pipeline/su7/geodesic.js';
import {
  projectSu7Vector,
  resolveHopfLenses,
  type HopfLensProjection,
} from '../src/pipeline/su7/projector.js';
import {
  createDefaultSu7RuntimeParams,
  sanitizeSu7RuntimeParams,
  type C7Vector,
  type Complex,
  type Su7ProjectorDescriptor,
} from '../src/pipeline/su7/types.js';

const wrapAngle = (theta: number): number => {
  const tau = Math.PI * 2;
  let t = theta;
  t = t - Math.floor((t + Math.PI) / tau) * tau;
  return t - Math.PI;
};

const multiplyComplex = (a: Complex, b: Complex): Complex => ({
  re: a.re * b.re - a.im * b.im,
  im: a.re * b.im + a.im * b.re,
});

const createRealVector = (entries: number[]): { vector: C7Vector; norm: number } => {
  const squares = entries.reduce((sum, value) => sum + value * value, 0);
  const norm = Math.sqrt(squares);
  const inv = norm > 0 ? 1 / norm : 1;
  const vector = entries.map((value) => ({ re: value * inv, im: 0 })) as C7Vector;
  return { vector, norm };
};

test('computeHopfCoordinates is invariant to global phase', () => {
  const sqrt2 = Math.sqrt(2);
  const a: Complex = { re: 1 / sqrt2, im: 0 };
  const b: Complex = { re: 0, im: 1 / sqrt2 };
  const base = computeHopfCoordinates(a, b);
  const phase = { re: Math.cos(Math.PI / 3), im: Math.sin(Math.PI / 3) };
  const rotated = computeHopfCoordinates(multiplyComplex(a, phase), multiplyComplex(b, phase));

  base.base.forEach((component, idx) => {
    assert.ok(
      Math.abs(component - rotated.base[idx]) < 1e-6,
      'base component changed under global phase',
    );
  });
  const expectedFiber = base.fiber + Math.PI / 3;
  const expectedVec = {
    x: Math.cos(expectedFiber),
    y: Math.sin(expectedFiber),
  };
  const actualVec = {
    x: Math.cos(rotated.fiber),
    y: Math.sin(rotated.fiber),
  };
  const fiberError = Math.hypot(expectedVec.x - actualVec.x, expectedVec.y - actualVec.y);
  assert.ok(fiberError < 1e-6, 'fiber did not shift by the global phase');
  assert.ok(
    Math.abs(base.magnitude - rotated.magnitude) < 1e-6,
    'magnitude changed under phase rotation',
  );
});

test('hopf projector emits overlays and metrics per lens', () => {
  const { vector, norm } = createRealVector([0.42, 0.33, 0.27, 0.51, 0.18, 0.46, 0.39]);
  const projector: Su7ProjectorDescriptor = {
    id: 'hopflens',
    weight: 0.9,
    hopf: {
      lenses: [
        { axes: [0, 1], baseMix: 0.8, fiberMix: 0.6 },
        { axes: [2, 3], baseMix: 0.7, fiberMix: 0.5 },
        { axes: [4, 5], baseMix: 1.0, fiberMix: 0.4 },
        { axes: [5, 6], baseMix: 0.5, fiberMix: 0.5 },
      ],
    },
  };
  const result = projectSu7Vector({
    vector,
    norm,
    projector,
    gain: 1,
    frameGain: 1,
    baseColor: [0.35, 0.4, 0.45],
  });
  assert.ok(result, 'hopf projection should produce a result');
  const hopfEntries = result!.hopf ?? [];
  assert.ok(hopfEntries.length > 0, 'expected hopf lens projections');
  assert.ok(hopfEntries.length <= 3, 'lens projections should be limited to three lenses');
  assert.ok((result!.overlays ?? []).length <= 6, 'overlay limit exceeded');
  hopfEntries.forEach((entry: HopfLensProjection) => {
    assert.ok(entry.baseMix <= result!.mix + 1e-6, 'base mix exceeds projector mix');
    assert.ok(entry.fiberMix <= result!.mix + 1e-6, 'fiber mix exceeds projector mix');
    const rgbBase = hopfBaseToRgb(entry.base);
    const rgbFiber = hopfFiberToRgb(entry.fiber);
    rgbBase.forEach((channel) => assert.ok(channel >= 0 && channel <= 1, 'base color invalid'));
    rgbFiber.forEach((channel) => assert.ok(channel >= 0 && channel <= 1, 'fiber color invalid'));
  });
});

test('resolveHopfLenses sanitizes axes, mixes, and limits count', () => {
  const descriptor: Su7ProjectorDescriptor = {
    id: 'hopflens',
    hopf: {
      lenses: [
        { axes: [0, 1], baseMix: 1.2, fiberMix: -0.1, controlTarget: 'base' },
        { axes: [7, 9], baseMix: 0.5, fiberMix: 0.9, controlTarget: 'fiber' },
        { axes: [2, 2], baseMix: 0.6, fiberMix: 0.4 },
        { axes: [3, 4], baseMix: 0.5, fiberMix: 0.5 },
      ],
    },
  };
  const lenses = resolveHopfLenses(descriptor);
  assert.equal(lenses.length, 3);
  lenses.forEach((lens) => {
    assert.ok(lens.axes[0] >= 0 && lens.axes[0] <= 6);
    assert.ok(lens.axes[1] >= 0 && lens.axes[1] <= 6);
    assert.ok(lens.baseMix! >= 0 && lens.baseMix! <= 1);
    assert.ok(lens.fiberMix! >= 0 && lens.fiberMix! <= 1);
  });
  assert.equal(lenses[0].controlTarget, 'base');
  assert.equal(lenses[1].controlTarget, 'fiber');
});

test('sanitizeSu7RuntimeParams preserves hopf lens control targets', () => {
  const runtime = createDefaultSu7RuntimeParams();
  runtime.projector = {
    id: 'hopflens',
    hopf: {
      lenses: [
        { axes: [0, 1], baseMix: 0.8, fiberMix: 0.6, controlTarget: 'base' },
        { axes: [2, 3], baseMix: 0.7, fiberMix: 0.5, controlTarget: 'none' },
      ],
    },
  };
  const sanitized = sanitizeSu7RuntimeParams(runtime, createDefaultSu7RuntimeParams());
  const lenses = resolveHopfLenses(sanitized.projector);
  assert.equal(lenses[0].controlTarget, 'base');
  assert.equal(lenses[1].controlTarget, 'none');
});
