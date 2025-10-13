import test from 'node:test';
import assert from 'node:assert/strict';

import { createGeodesicMorph } from '../src/pipeline/su7/geodesic.js';
import {
  compose_dense,
  computeUnitaryError,
  phase_gate,
  polar_reunitarize,
  su2_embed,
} from '../src/pipeline/su7/math.js';
import { DEFAULT_SU7_TELEMETRY, type Complex, type Complex7x7 } from '../src/pipeline/su7/types.js';

const frobeniusDiff = (a: Complex7x7, b: Complex7x7): number => {
  let sum = 0;
  for (let row = 0; row < 7; row++) {
    for (let col = 0; col < 7; col++) {
      const lhs = a[row][col];
      const rhs = b[row][col];
      const dr = lhs.re - rhs.re;
      const di = lhs.im - rhs.im;
      sum += dr * dr + di * di;
    }
  }
  return Math.sqrt(sum);
};

const makeDiagonal = (values: readonly number[]): Complex7x7 => {
  const rows: Complex[][] = [];
  for (let row = 0; row < 7; row++) {
    const cells: Complex[] = [];
    for (let col = 0; col < 7; col++) {
      if (row === col) {
        cells.push({ re: values[row] ?? 1, im: 0 });
      } else {
        cells.push({ re: 0, im: 0 });
      }
    }
    rows.push(cells);
  }
  return rows as Complex7x7;
};

test('geodesic morph stays unitary along the path', () => {
  const start = phase_gate([0, 0, 0, 0, 0, 0, 0]);
  const rotation = su2_embed(0, 3, Math.PI / 5, Math.PI / 3);
  const diagonal = phase_gate([0.2, -0.15, 0.35, -0.25, 0.18, -0.12, -0.21]);
  const end = compose_dense(rotation, diagonal);
  const morph = createGeodesicMorph(start, end);
  assert.equal(morph.method, 'geodesic');

  const samples = [0, 0.25, 0.5, 0.75, 1];
  for (const t of samples) {
    const unitary = morph.evaluate(t);
    const err = computeUnitaryError(unitary);
    assert.ok(err <= 1e-6, `unitary error ${err} at t=${t}`);
  }

  const startCheck = morph.evaluate(0);
  const endCheck = morph.evaluate(1);
  assert.ok(
    frobeniusDiff(startCheck, polar_reunitarize(start)) <= 1e-6,
    'geodesic morph mismatches start unitary',
  );
  assert.ok(
    frobeniusDiff(endCheck, polar_reunitarize(end)) <= 1e-6,
    'geodesic morph mismatches end unitary',
  );
});

test('branch-cut fallback uses polar interpolation and logs telemetry', () => {
  const start = phase_gate([0, 0, 0, 0, 0, 0, 0]);
  const end = makeDiagonal([-1, -1, -1, -1, -1, -1, 1]);
  const telemetry = { ...DEFAULT_SU7_TELEMETRY };
  const morph = createGeodesicMorph(start, end, { telemetry });
  assert.equal(morph.method, 'polar');
  assert.equal(telemetry.geodesicFallbacks, DEFAULT_SU7_TELEMETRY.geodesicFallbacks + 1);

  const mid = morph.evaluate(0.5);
  const err = computeUnitaryError(mid);
  assert.ok(err <= 1e-6, `fallback morph unitary error ${err}`);

  const final = morph.evaluate(1);
  assert.ok(
    frobeniusDiff(final, polar_reunitarize(end)) <= 1e-6,
    'fallback morph did not converge to target unitary',
  );
});
