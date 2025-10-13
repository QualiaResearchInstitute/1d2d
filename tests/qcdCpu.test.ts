import test from 'node:test';
import assert from 'node:assert/strict';

import { GaugeLattice } from '../src/qcd/lattice.js';
import {
  runWilsonCpuUpdate,
  computeAveragePlaquette,
  initializeGaugeField,
  applyApeSmear,
} from '../src/qcd/updateCpu.js';

type Complex = { re: number; im: number };
type Complex3Vector = [Complex, Complex, Complex];
type Complex3x3 = [Complex3Vector, Complex3Vector, Complex3Vector];

const mulberry32 = (seed: number): (() => number) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let c = Math.imul(t ^ (t >>> 15), 1 | t);
    c ^= c + Math.imul(c ^ (c >>> 7), 61 | c);
    return ((c ^ (c >>> 14)) >>> 0) / 4294967296;
  };
};

const createZeroMatrix = (): Complex3x3 =>
  new Array(3).fill(null).map(
    () =>
      [
        { re: 0, im: 0 },
        { re: 0, im: 0 },
        { re: 0, im: 0 },
      ] as Complex3Vector,
  ) as Complex3x3;

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

const matrixMultiply = (lhs: Complex3x3, rhs: Complex3x3): Complex3x3 => {
  const result = createZeroMatrix();
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      let re = 0;
      let im = 0;
      for (let k = 0; k < 3; k++) {
        const a = lhs[row][k];
        const b = rhs[k][col];
        re += a.re * b.re - a.im * b.im;
        im += a.re * b.im + a.im * b.re;
      }
      result[row][col] = { re, im };
    }
  }
  return result;
};

const unitaryError = (matrix: Complex3x3): number => {
  const gram = matrixMultiply(conjugateTranspose(matrix), matrix);
  let sum = 0;
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const expected = row === col ? 1 : 0;
      const diffRe = gram[row][col].re - expected;
      const diffIm = gram[row][col].im;
      sum += diffRe * diffRe + diffIm * diffIm;
    }
  }
  return Math.sqrt(sum);
};

const determinant = (matrix: Complex3x3): Complex => {
  const a = matrix[0][0];
  const b = matrix[0][1];
  const c = matrix[0][2];
  const d = matrix[1][0];
  const e = matrix[1][1];
  const f = matrix[1][2];
  const g = matrix[2][0];
  const h = matrix[2][1];
  const i = matrix[2][2];

  const mul = (x: Complex, y: Complex): Complex => ({
    re: x.re * y.re - x.im * y.im,
    im: x.re * y.im + x.im * y.re,
  });
  const sub = (x: Complex, y: Complex): Complex => ({ re: x.re - y.re, im: x.im - y.im });
  const add = (x: Complex, y: Complex): Complex => ({ re: x.re + y.re, im: x.im + y.im });

  const term1 = mul(a, sub(mul(e, i), mul(f, h)));
  const term2 = mul(b, sub(mul(d, i), mul(f, g)));
  const term3 = mul(c, sub(mul(d, h), mul(e, g)));
  return add(sub(term1, term2), term3);
};

const computePlaquetteReference = (lattice: GaugeLattice): number => {
  const width = lattice.width;
  const height = lattice.height;
  let sum = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const xp1 = (x + 1) % width;
      const yp1 = (y + 1) % height;

      const Ux = lattice.getLinkMatrix(x, y, 'x');
      const Uy = lattice.getLinkMatrix(x, y, 'y');
      const UxForward = lattice.getLinkMatrix(x, yp1, 'x');
      const UyForward = lattice.getLinkMatrix(xp1, y, 'y');

      const plaquette = matrixMultiply(
        matrixMultiply(matrixMultiply(Ux, UyForward), conjugateTranspose(UxForward)),
        conjugateTranspose(Uy),
      );

      sum += (plaquette[0][0].re + plaquette[1][1].re + plaquette[2][2].re) / 3;
    }
  }

  return sum / (width * height);
};

test('Wilson CPU update yields monotone plaquette history and reference agreement', () => {
  const lattice = new GaugeLattice({ width: 6, height: 6 });
  const betaSchedule = [2.25, 3.0, 4.0, 5.25];
  const result = runWilsonCpuUpdate(lattice, {
    betaSchedule,
    sweepsPerBeta: 3,
    thermalizationSweeps: 5,
    overRelaxationSteps: 2,
    startMode: 'cold',
    seed: 1234,
  });

  assert.equal(
    result.plaquetteHistory.length,
    betaSchedule.length,
    'plaquette history length mismatch',
  );

  for (let i = 0; i < result.plaquetteHistory.length - 1; i++) {
    const current = result.plaquetteHistory[i]!;
    const next = result.plaquetteHistory[i + 1]!;
    assert.ok(
      current <= next + 1e-6,
      `plaquette sequence is not monotone at index ${i}: ${current} -> ${next}`,
    );
  }

  const measured = result.plaquetteHistory.at(-1)!;
  const reference = computePlaquetteReference(lattice);
  assert.ok(
    Math.abs(measured - reference) <= 1e-3,
    `plaquette mismatch ${measured} vs ${reference}`,
  );
});

test('Link updates preserve unitarity and determinant within tolerance', () => {
  const lattice = new GaugeLattice({ width: 5, height: 5 });
  runWilsonCpuUpdate(lattice, {
    betaSchedule: [2.5, 3.25, 4.5],
    sweepsPerBeta: 4,
    thermalizationSweeps: 6,
    overRelaxationSteps: 2,
    startMode: 'hot',
    seed: 8181,
  });

  let maxUnitaryError = 0;
  let maxDeterminantDrift = 0;

  for (let y = 0; y < lattice.height; y++) {
    for (let x = 0; x < lattice.width; x++) {
      for (const axis of lattice.axes) {
        const link = lattice.getLinkMatrix(x, y, axis);
        const uError = unitaryError(link);
        if (uError > maxUnitaryError) {
          maxUnitaryError = uError;
        }
        const det = determinant(link);
        const detMagnitude = Math.hypot(det.re, det.im);
        const drift = Math.abs(detMagnitude - 1);
        if (drift > maxDeterminantDrift) {
          maxDeterminantDrift = drift;
        }
      }
    }
  }

  assert.ok(maxUnitaryError <= 5e-7, `unitary drift ${maxUnitaryError}`);
  assert.ok(maxDeterminantDrift <= 5e-7, `determinant drift ${maxDeterminantDrift}`);
});

test('APE smearing toggles cleanly and modifies hot configurations', () => {
  const lattice = new GaugeLattice({ width: 4, height: 4 });
  initializeGaugeField(lattice, 'hot', mulberry32(2024));

  const baseline = new Float32Array(lattice.data);

  applyApeSmear(lattice, 0, 3);
  assert.deepEqual(
    Array.from(lattice.data),
    Array.from(baseline),
    'zero-alpha smear altered lattice',
  );

  const smeared = new GaugeLattice({ width: 4, height: 4 }, new Float32Array(baseline));
  applyApeSmear(smeared, 0.55, 2);

  let maxDelta = 0;
  for (let i = 0; i < baseline.length; i++) {
    const delta = Math.abs(smeared.data[i] - baseline[i]);
    if (delta > maxDelta) {
      maxDelta = delta;
    }
  }
  assert.ok(maxDelta > 1e-5, `smearing failed to adjust links (max delta ${maxDelta})`);

  const plaquetteBefore = computePlaquetteReference(
    new GaugeLattice({ width: 4, height: 4 }, new Float32Array(baseline)),
  );
  const plaquetteAfter = computeAveragePlaquette(smeared);
  assert.ok(
    plaquetteAfter >= plaquetteBefore - 1e-3,
    `smearing degraded plaquette ${plaquetteAfter} vs ${plaquetteBefore}`,
  );
});
