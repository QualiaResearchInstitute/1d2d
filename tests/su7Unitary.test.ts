import test from 'node:test';
import assert from 'node:assert/strict';

import { computeUnitaryError } from '../src/pipeline/su7/math.js';
import { applySU7, buildUnitary, type BuildUnitaryOptions } from '../src/pipeline/su7/unitary.js';
import type { C7Vector, Complex, Complex7x7 } from '../src/pipeline/su7/types.js';

const EPSILON = 1e-6;

const complexAdd = (a: Complex, b: Complex): Complex => ({
  re: a.re + b.re,
  im: a.im + b.im,
});

const complexSub = (a: Complex, b: Complex): Complex => ({
  re: a.re - b.re,
  im: a.im - b.im,
});

const complexMul = (a: Complex, b: Complex): Complex => ({
  re: a.re * b.re - a.im * b.im,
  im: a.re * b.im + a.im * b.re,
});

const complexConj = (a: Complex): Complex => ({ re: a.re, im: -a.im });

const complexAbs2 = (a: Complex): number => a.re * a.re + a.im * a.im;

const complexDiv = (a: Complex, b: Complex): Complex => {
  const denom = complexAbs2(b);
  if (denom <= Number.EPSILON) {
    return { re: 0, im: 0 };
  }
  const conj = complexConj(b);
  const num = complexMul(a, conj);
  return { re: num.re / denom, im: num.im / denom };
};

const cloneMatrix = (matrix: Complex7x7): Complex7x7 =>
  matrix.map((row) => row.map((entry) => ({ re: entry.re, im: entry.im }))) as Complex7x7;

const computeDeterminant = (matrix: Complex7x7): Complex => {
  const working = cloneMatrix(matrix);
  let det: Complex = { re: 1, im: 0 };
  let sign = 1;

  for (let i = 0; i < 7; i++) {
    let pivotRow = i;
    let pivotMag = Math.sqrt(complexAbs2(working[i][i]));
    for (let r = i + 1; r < 7; r++) {
      const mag = Math.sqrt(complexAbs2(working[r][i]));
      if (mag > pivotMag) {
        pivotMag = mag;
        pivotRow = r;
      }
    }

    if (pivotMag <= Number.EPSILON) {
      return { re: 0, im: 0 };
    }

    if (pivotRow !== i) {
      const tmp = working[i];
      working[i] = working[pivotRow];
      working[pivotRow] = tmp;
      sign *= -1;
    }

    const pivot = working[i][i];
    det = complexMul(det, pivot);

    for (let r = i + 1; r < 7; r++) {
      const factor = complexDiv(working[r][i], pivot);
      for (let c = i; c < 7; c++) {
        working[r][c] = complexSub(working[r][c], complexMul(factor, working[i][c]));
      }
    }
  }

  if (sign === -1) {
    det = { re: -det.re, im: -det.im };
  }
  return det;
};

const normOf = (vector: C7Vector): number => {
  let sum = 0;
  for (const cell of vector) {
    sum += cell.re * cell.re + cell.im * cell.im;
  }
  return Math.sqrt(sum);
};

const buildVector = (seed: number): C7Vector => {
  const rng = (() => {
    let t = seed >>> 0;
    return () => {
      t += 0x6d2b79f5;
      let c = Math.imul(t ^ (t >>> 15), 1 | t);
      c ^= c + Math.imul(c ^ (c >>> 7), 61 | c);
      return ((c ^ (c >>> 14)) >>> 0) / 4294967296;
    };
  })();

  const entries: Complex[] = [];
  for (let i = 0; i < 7; i++) {
    const angle = 2 * Math.PI * rng();
    const radius = Math.sqrt(-Math.log(Math.max(rng(), 1e-12)));
    entries.push({
      re: radius * Math.cos(angle),
      im: radius * Math.sin(angle),
    });
  }
  const vector = entries as C7Vector;
  const norm = normOf(vector);
  return vector.map((entry) => ({
    re: entry.re / norm,
    im: entry.im / norm,
  })) as C7Vector;
};

const approx = (actual: number, expected: number, tol = EPSILON) => {
  assert.ok(Math.abs(actual - expected) <= tol, `expected ${expected}, got ${actual}`);
};

const checkSU7 = (matrix: Complex7x7) => {
  const err = computeUnitaryError(matrix);
  assert.ok(err <= EPSILON, `matrix not unitary: error ${err}`);
  const det = computeDeterminant(matrix);
  const phase = Math.atan2(det.im, det.re);
  approx(phase, 0, 1e-6);
  const magnitude = Math.sqrt(det.re * det.re + det.im * det.im);
  approx(magnitude, 1, 1e-6);
};

const createUnitary = (options?: BuildUnitaryOptions) => buildUnitary(options);

test('buildUnitary returns deterministic SU(7) for given seed', () => {
  const first = createUnitary({ seed: 42 });
  const second = createUnitary({ seed: 42 });
  checkSU7(first);
  checkSU7(second);
  for (let row = 0; row < 7; row++) {
    for (let col = 0; col < 7; col++) {
      approx(first[row][col].re, second[row][col].re, 1e-9);
      approx(first[row][col].im, second[row][col].im, 1e-9);
    }
  }
});

test('applySU7 preserves complex vector norm', () => {
  const unitary = createUnitary({ seed: 1337 });
  const vector = buildVector(99);
  const normBefore = normOf(vector);
  const result = applySU7(unitary, vector);
  const normAfter = normOf(result);
  approx(normBefore, 1, 1e-9);
  approx(normAfter, 1, 1e-6);
});

test('raising preset rotates expected subspace', () => {
  const theta = Math.PI / 5;
  const unitary = createUnitary({ preset: 'raise:01', theta });
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  approx(unitary[0][0].re, cos, 1e-6);
  approx(unitary[1][1].re, cos, 1e-6);
  approx(unitary[0][1].re, -sin, 1e-6);
  approx(unitary[1][0].re, sin, 1e-6);
  for (let idx = 2; idx < 7; idx++) {
    approx(unitary[idx][idx].re, 1, 1e-6);
    approx(unitary[idx][idx].im, 0, 1e-6);
  }
  checkSU7(unitary);
});

test('schedule composes pulses before projection', () => {
  const thetaA = Math.PI / 7;
  const thetaB = Math.PI / 9;
  const scheduled = createUnitary({
    preset: 'identity',
    theta: thetaA,
    schedule: [
      { preset: 'raise:01', theta: thetaA },
      { preset: 'raise:23', theta: thetaB },
    ],
  });
  checkSU7(scheduled);
  const basis: C7Vector = [
    { re: 1, im: 0 },
    { re: 0, im: 0 },
    { re: 0, im: 0 },
    { re: 0, im: 0 },
    { re: 0, im: 0 },
    { re: 0, im: 0 },
    { re: 0, im: 0 },
  ];
  const result = applySU7(scheduled, basis);
  approx(normOf(result), 1, 1e-6);
});

test('random seeds yield well-conditioned special unitaries', () => {
  const seeds = Array.from({ length: 24 }, (_, idx) => (idx + 1) * 97);
  for (const seed of seeds) {
    const randomUnitary = createUnitary({ seed, preset: 'random' });
    checkSU7(randomUnitary);

    const scheduled = createUnitary({
      seed,
      preset: 'random',
      schedule: [
        { preset: 'raise:01', theta: 0.05 * (seed % 11) },
        { preset: 'raise:34', theta: 0.035 * (seed % 7) },
      ],
    });
    checkSU7(scheduled);
  }
});
