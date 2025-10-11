import test from 'node:test';
import assert from 'node:assert/strict';

import { computeCouplingWeight, computeCouplingWeights } from '../src/kuramotoCore.js';
import { getCouplingKernelParams } from '../src/kernel/kernelSpec.js';

test('DMT coupling kernel yields Mexican-hat profile', () => {
  const params = getCouplingKernelParams('dmt');
  const near = computeCouplingWeight(0, params);
  const mid = computeCouplingWeight(2.5, params);
  const edge = computeCouplingWeight(params.radius - 0.01, params);
  const outside = computeCouplingWeight(params.radius + 0.1, params);
  assert.ok(near < 0, `near value expected negative, received ${near}`);
  assert.ok(mid > 0, `mid value expected positive, received ${mid}`);
  assert.ok(edge >= 0, `edge value expected non-negative, received ${edge}`);
  assert.equal(outside, 0);
});

test('5-MeO coupling kernel stays uniformly positive', () => {
  const params = getCouplingKernelParams('5meo');
  const near = computeCouplingWeight(0, params);
  const mid = computeCouplingWeight(params.radius * 0.5, params);
  const edge = computeCouplingWeight(params.radius - 1e-3, params);
  const outside = computeCouplingWeight(params.radius + 1e-3, params);
  assert.ok(near > 0);
  assert.ok(mid > 0);
  assert.ok(edge > 0);
  assert.equal(outside, 0);
});

test('vectorized coupling path reuses provided buffer', () => {
  const params = getCouplingKernelParams('dmt');
  const distances = new Float32Array([0, 1, params.radius + 1]);
  const out = new Float32Array(distances.length);
  const result = computeCouplingWeights(distances, params, out);
  assert.strictEqual(result, out);
  assert.ok(out[0] < 0);
  assert.ok(out[1] !== 0);
  assert.equal(out[2], 0);
});
