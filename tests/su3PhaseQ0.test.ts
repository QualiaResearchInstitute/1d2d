import test from 'node:test';
import assert from 'node:assert/strict';

import {
  su3_project,
  su3_exp,
  su3_log,
  su3_mul,
  su3_frobNorm,
  su3_haar,
  type Complex3x3,
} from '../src/qcd/su3.js';
import {
  su3_embed,
  computeUnitaryError,
  computeDeterminantDrift,
} from '../src/pipeline/su7/math.js';

type Complex = {
  re: number;
  im: number;
};

const complexAdd = (a: Complex, b: Complex): Complex => ({ re: a.re + b.re, im: a.im + b.im });

const complexSub = (a: Complex, b: Complex): Complex => ({ re: a.re - b.re, im: a.im - b.im });

const complexMul = (a: Complex, b: Complex): Complex => ({
  re: a.re * b.re - a.im * b.im,
  im: a.re * b.im + a.im * b.re,
});

const createZeroMatrix = (): Complex3x3 =>
  new Array(3)
    .fill(null)
    .map(() => new Array(3).fill(null).map(() => ({ re: 0, im: 0 })) as Complex[]) as Complex3x3;

const conjugateTranspose = (matrix: Complex3x3): Complex3x3 => {
  const result = createZeroMatrix();
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const entry = matrix[col][row];
      result[row][col] = { re: entry.re, im: -entry.im };
    }
  }
  return result;
};

const matrixMultiply = (a: Complex3x3, b: Complex3x3): Complex3x3 => {
  const result = createZeroMatrix();
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      let re = 0;
      let im = 0;
      for (let k = 0; k < 3; k++) {
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

const frobeniusDiff = (a: Complex3x3, b: Complex3x3): number => {
  let sum = 0;
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const diffRe = a[row][col].re - b[row][col].re;
      const diffIm = a[row][col].im - b[row][col].im;
      sum += diffRe * diffRe + diffIm * diffIm;
    }
  }
  return Math.sqrt(sum);
};

const unitaryError3 = (matrix: Complex3x3): number => {
  const gram = matrixMultiply(conjugateTranspose(matrix), matrix);
  const identity = createZeroMatrix();
  for (let i = 0; i < 3; i++) {
    identity[i][i] = { re: 1, im: 0 };
  }
  return frobeniusDiff(gram, identity);
};

const determinant3 = (matrix: Complex3x3): Complex => {
  const a = matrix[0][0];
  const b = matrix[0][1];
  const c = matrix[0][2];
  const d = matrix[1][0];
  const e = matrix[1][1];
  const f = matrix[1][2];
  const g = matrix[2][0];
  const h = matrix[2][1];
  const i = matrix[2][2];

  const term1 = complexMul(a, complexSub(complexMul(e, i), complexMul(f, h)));
  const term2 = complexMul(b, complexSub(complexMul(d, i), complexMul(f, g)));
  const term3 = complexMul(c, complexSub(complexMul(d, h), complexMul(e, g)));
  return complexAdd(complexSub(term1, term2), term3);
};

const mulberry32 = (seed: number): (() => number) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let c = Math.imul(t ^ (t >>> 15), 1 | t);
    c ^= c + Math.imul(c ^ (c >>> 7), 61 | c);
    return ((c ^ (c >>> 14)) >>> 0) / 4294967296;
  };
};

const buildRandomMatrix = (seed: number): Complex3x3 => {
  const rng = mulberry32(seed);
  const result = createZeroMatrix();
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      result[row][col] = { re: rng() - 0.5, im: rng() - 0.5 };
    }
  }
  return result;
};

test('su3_project produces a special unitary within tight tolerance', () => {
  const raw = buildRandomMatrix(2024);
  const projected = su3_project(raw);
  const unitaryError = unitaryError3(projected);
  assert.ok(unitaryError <= 1e-7, `unitary error ${unitaryError}`);
  const det = determinant3(projected);
  const detMagnitude = Math.hypot(det.re, det.im);
  assert.ok(Math.abs(detMagnitude - 1) <= 1e-7, `determinant magnitude ${detMagnitude}`);
});

test('su3_exp and su3_log round-trip SU(3) matrices', () => {
  const base = su3_haar(0.75, mulberry32(11));
  const coeffs = su3_log(base);
  const rebuilt = su3_exp(coeffs);
  const diff = frobeniusDiff(rebuilt, base);
  assert.ok(diff <= 1e-7, `round-trip deviation ${diff}`);
});

test('su3_embed injects SU(3) blocks into SU(7) cleanly', () => {
  const block = su3_haar(1, mulberry32(7));
  const embedded = su3_embed(block);
  const unitaryError = computeUnitaryError(embedded);
  assert.ok(unitaryError <= 1e-7, `embedded unitary error ${unitaryError}`);
  const detDrift = computeDeterminantDrift(embedded);
  assert.ok(detDrift <= 1e-7, `determinant drift ${detDrift}`);
});

test('su3_mul matches naive multiply and su3_frobNorm matches manual norm', () => {
  const lhs = su3_haar(0.6, mulberry32(101));
  const rhs = su3_haar(0.6, mulberry32(202));
  const optimized = su3_mul(lhs, rhs);
  const baseline = matrixMultiply(lhs, rhs);
  const diff = frobeniusDiff(optimized, baseline);
  assert.ok(diff <= 1e-12, `su3_mul deviation ${diff}`);

  const normRef = Math.sqrt(
    baseline.reduce(
      (acc, row) =>
        acc + row.reduce((inner, cell) => inner + cell.re * cell.re + cell.im * cell.im, 0),
      0,
    ),
  );
  const normActual = su3_frobNorm(baseline);
  assert.ok(
    Math.abs(normActual - normRef) <= 1e-12,
    `frobenius norm mismatch ${normActual} vs ${normRef}`,
  );
});

test('su3_haar sampler generates unitary with determinant near one', () => {
  const sample = su3_haar(0.35, mulberry32(909));
  const unitaryError = unitaryError3(sample);
  assert.ok(unitaryError <= 1e-7, `haar unitary error ${unitaryError}`);
  const det = determinant3(sample);
  const detMagnitude = Math.hypot(det.re, det.im);
  assert.ok(Math.abs(detMagnitude - 1) <= 1e-7, `haar determinant magnitude ${detMagnitude}`);
});
