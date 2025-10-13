import test from 'node:test';
import assert from 'node:assert/strict';

import {
  su2_embed,
  phase_gate,
  compose_dense,
  polar_reunitarize,
  logm_su,
  expm_su,
  computeUnitaryError,
  computeDeterminantDrift,
} from '../src/pipeline/su7/math.js';
import { cloneComplex7x7, type Complex, type Complex7x7 } from '../src/pipeline/su7/types.js';

const createZeroMatrix = (): Complex7x7 =>
  new Array(7)
    .fill(null)
    .map(() => new Array(7).fill(null).map(() => ({ re: 0, im: 0 })) as Complex[]) as Complex7x7;

const mulberry32 = (seed: number): (() => number) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let c = Math.imul(t ^ (t >>> 15), 1 | t);
    c ^= c + Math.imul(c ^ (c >>> 7), 61 | c);
    return ((c ^ (c >>> 14)) >>> 0) / 4294967296;
  };
};

const naiveMultiply = (a: Complex7x7, b: Complex7x7): Complex7x7 => {
  const result = createZeroMatrix();
  for (let row = 0; row < 7; row++) {
    for (let col = 0; col < 7; col++) {
      let re = 0;
      let im = 0;
      for (let k = 0; k < 7; k++) {
        const lhs = a[row][k];
        const rhs = b[k][col];
        re += lhs.re * rhs.re - lhs.im * rhs.im;
        im += lhs.re * rhs.im + lhs.im * rhs.re;
      }
      result[row][col] = { re, im };
    }
  }
  return result;
};

const addMatrices = (a: Complex7x7, b: Complex7x7): Complex7x7 => {
  const result = createZeroMatrix();
  for (let row = 0; row < 7; row++) {
    for (let col = 0; col < 7; col++) {
      result[row][col] = {
        re: a[row][col].re + b[row][col].re,
        im: a[row][col].im + b[row][col].im,
      };
    }
  }
  return result;
};

const conjugateTranspose = (matrix: Complex7x7): Complex7x7 => {
  const result = createZeroMatrix();
  for (let row = 0; row < 7; row++) {
    for (let col = 0; col < 7; col++) {
      const entry = matrix[col][row];
      result[row][col] = { re: entry.re, im: -entry.im };
    }
  }
  return result;
};

const frobeniusDiff = (a: Complex7x7, b: Complex7x7): number => {
  let sum = 0;
  for (let row = 0; row < 7; row++) {
    for (let col = 0; col < 7; col++) {
      const diffRe = a[row][col].re - b[row][col].re;
      const diffIm = a[row][col].im - b[row][col].im;
      sum += diffRe * diffRe + diffIm * diffIm;
    }
  }
  return Math.sqrt(sum);
};

const buildSkewHermitian = (seed: number): Complex7x7 => {
  const rng = mulberry32(seed);
  const result = createZeroMatrix();
  const diagImag = new Float64Array(7);
  for (let i = 0; i < 7; i++) {
    diagImag[i] = (rng() - 0.5) * 0.2;
  }
  const meanDiag = diagImag.reduce((acc, value) => acc + value, 0) / 7;
  for (let row = 0; row < 7; row++) {
    result[row][row] = { re: 0, im: diagImag[row] - meanDiag };
    for (let col = row + 1; col < 7; col++) {
      const re = (rng() - 0.5) * 0.3;
      const im = (rng() - 0.5) * 0.3;
      result[row][col] = { re, im };
      result[col][row] = { re: -re, im: im };
    }
  }
  return result;
};

test('su2_embed constructs an SU(2) pulse embedded in SU(7)', () => {
  const matrix = su2_embed(1, 4, Math.PI / 4, Math.PI / 3);
  const expected = {
    diag: Math.cos(Math.PI / 4),
    upperRe: -Math.sin(Math.PI / 4) * Math.cos(Math.PI / 3),
    upperIm: -Math.sin(Math.PI / 4) * Math.sin(Math.PI / 3),
  };
  assert.ok(Math.abs(matrix[1][1].re - expected.diag) < 1e-12);
  assert.strictEqual(matrix[1][1].im, 0);
  assert.ok(Math.abs(matrix[4][4].re - expected.diag) < 1e-12);
  assert.strictEqual(matrix[4][4].im, 0);
  assert.ok(Math.abs(matrix[1][4].re - expected.upperRe) < 1e-12);
  assert.ok(Math.abs(matrix[1][4].im - expected.upperIm) < 1e-12);
  assert.ok(Math.abs(matrix[4][1].re + expected.upperRe) < 1e-12);
  assert.ok(Math.abs(matrix[4][1].im - expected.upperIm) < 1e-12);
});

test('phase_gate enforces zero-mean locking', () => {
  const gate = phase_gate([0, 0.5, 1, 1.5, 2, 2.5, 3]);
  let sum = 0;
  const phases: number[] = [];
  for (let i = 0; i < 7; i++) {
    const entry = gate[i][i];
    const phase = Math.atan2(entry.im, entry.re);
    phases.push(phase);
    sum += phase;
  }
  const mean = sum / 7;
  assert.ok(Math.abs(mean) <= 1e-12);
  const detPhase = phases.reduce((acc, value) => acc + value, 0);
  assert.ok(Math.abs(detPhase) <= 1e-12);
});

test('compose_dense matches naive multiply to machine precision', () => {
  const rng = mulberry32(2024);
  const a = createZeroMatrix();
  const b = createZeroMatrix();
  for (let row = 0; row < 7; row++) {
    for (let col = 0; col < 7; col++) {
      a[row][col] = { re: rng() - 0.5, im: rng() - 0.5 };
      b[row][col] = { re: rng() - 0.5, im: rng() - 0.5 };
    }
  }
  const optimized = compose_dense(a, b);
  const baseline = naiveMultiply(a, b);
  const diff = frobeniusDiff(optimized, baseline);
  assert.ok(diff <= 1e-11, `compose_dense deviation ${diff}`);
});

test('polar_reunitarize produces special unitary matrices within tolerance', () => {
  const base = compose_dense(
    su2_embed(0, 3, Math.PI / 5, Math.PI / 7),
    phase_gate([0.1, 0.2, 0.3, -0.4, 0.5, -0.2, -0.5]),
  );
  const perturbed = cloneComplex7x7(base);
  perturbed[0][1].re += 2e-3;
  perturbed[2][4].im -= 1.5e-3;
  perturbed[5][6].re += 8e-4;
  const repaired = polar_reunitarize(perturbed);
  const err = computeUnitaryError(repaired);
  const drift = computeDeterminantDrift(repaired);
  assert.ok(err <= 1e-7, `unitary error ${err}`);
  assert.ok(drift <= 1e-7, `determinant drift ${drift}`);
});

test('expm_su(logm_su(U)) recovers U within tolerance', () => {
  const skew = buildSkewHermitian(77);
  const unitary = polar_reunitarize(expm_su(skew));
  const log = logm_su(unitary);
  const reconstructed = polar_reunitarize(expm_su(log));
  const diff = frobeniusDiff(reconstructed, unitary);
  assert.ok(diff <= 3e-6, `exp(log(U)) mismatch ${diff}`);

  const adjoint = conjugateTranspose(log);
  const skewCheck = frobeniusDiff(addMatrices(log, adjoint), createZeroMatrix());
  assert.ok(skewCheck <= 1e-10, `log not skew-Hermitian ${skewCheck}`);

  const trace = log.reduce(
    (acc, row, idx) => ({ re: acc.re + row[idx].re, im: acc.im + row[idx].im }),
    { re: 0, im: 0 },
  );
  assert.ok(Math.abs(trace.re) <= 1e-10);
  assert.ok(Math.abs(trace.im) <= 1e-10);

  const err = computeUnitaryError(unitary);
  const drift = computeDeterminantDrift(unitary);
  assert.ok(err <= 1e-7, `exp(X) not unitary: ${err}`);
  assert.ok(drift <= 1e-7, `exp(X) det drift ${drift}`);
});
