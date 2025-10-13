import test from 'node:test';
import assert from 'node:assert/strict';
import fc from 'fast-check';

import {
  expm_su,
  logm_su,
  polar_reunitarize,
  computeUnitaryError,
  computeDeterminantDrift,
} from '../src/pipeline/su7/math.js';
import { cloneComplex7x7, type Complex, type Complex7x7 } from '../src/pipeline/su7/types.js';

const createZeroMatrix = (): Complex7x7 =>
  new Array(7)
    .fill(null)
    .map(() => new Array(7).fill(null).map(() => ({ re: 0, im: 0 })) as Complex[]) as Complex7x7;

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

const isFiniteMatrix = (matrix: Complex7x7): boolean => {
  for (let row = 0; row < 7; row++) {
    for (let col = 0; col < 7; col++) {
      const entry = matrix[row][col];
      if (!Number.isFinite(entry.re) || !Number.isFinite(entry.im)) {
        return false;
      }
    }
  }
  return true;
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

const skewHermitianArb = (() => {
  const entry = fc.tuple(
    fc.double({ min: -0.05, max: 0.05 }),
    fc.double({ min: -0.05, max: 0.05 }),
  );
  const diag = fc.array(fc.double({ min: -0.05, max: 0.05 }), { minLength: 7, maxLength: 7 });
  const upper = fc.array(entry, { minLength: 21, maxLength: 21 });
  return fc.tuple(upper, diag).map(([upperEntries, diagImag]) => {
    const matrix = createZeroMatrix();
    let idx = 0;
    for (let row = 0; row < 7; row++) {
      for (let col = row + 1; col < 7; col++) {
        const [re, im] = upperEntries[idx++];
        matrix[row][col] = { re, im };
        matrix[col][row] = { re: -re, im };
      }
    }
    const mean = diagImag.reduce((acc, value) => acc + value, 0) / 7;
    for (let i = 0; i < 7; i++) {
      matrix[i][i] = { re: 0, im: diagImag[i] - mean };
    }
    return matrix;
  });
})();

const nearUnitaryArb = skewHermitianArb.chain((skew) => {
  const unitary = expm_su(skew);
  const noise = fc.array(fc.double({ min: -1, max: 1 }), { minLength: 98, maxLength: 98 });
  return noise.map((values) => {
    const perturbed = cloneComplex7x7(unitary);
    let idx = 0;
    for (let row = 0; row < 7; row++) {
      for (let col = 0; col < 7; col++) {
        const deltaRe = values[idx++] * 1e-3;
        const deltaIm = values[idx++] * 1e-3;
        perturbed[row][col] = {
          re: perturbed[row][col].re + deltaRe,
          im: perturbed[row][col].im + deltaIm,
        };
      }
    }
    return perturbed;
  });
});

test('expm/log round-trip maintains SU(7) properties', async () => {
  await fc.assert(
    fc.property(skewHermitianArb, (skew) => {
      const unitary = polar_reunitarize(expm_su(skew));
      fc.pre(isFiniteMatrix(unitary));
      fc.pre(computeUnitaryError(unitary) < 0.5);
      const log = logm_su(unitary);
      const roundTrip = polar_reunitarize(expm_su(log));
      fc.pre(isFiniteMatrix(roundTrip));

      const unitaryError = computeUnitaryError(unitary);
      const detDrift = computeDeterminantDrift(unitary);
      assert.ok(unitaryError <= 1e-7, `exp(X) unitary error ${unitaryError}`);
      assert.ok(detDrift <= 1e-7, `exp(X) determinant drift ${detDrift}`);

      const diff = frobeniusDiff(roundTrip, unitary);
      assert.ok(diff <= 3e-6, `round-trip mismatch ${diff}`);

      const skewCheck = frobeniusDiff(
        addMatrices(log, conjugateTranspose(log)),
        createZeroMatrix(),
      );
      assert.ok(skewCheck <= 1e-9, `log result not skew-Hermitian ${skewCheck}`);

      const trace = log.reduce(
        (acc, row, idx) => ({
          re: acc.re + row[idx].re,
          im: acc.im + row[idx].im,
        }),
        { re: 0, im: 0 },
      );
      assert.ok(Math.abs(trace.re) <= 1e-9);
      assert.ok(Math.abs(trace.im) <= 1e-9);
    }),
    { numRuns: 25 },
  );
});

test('polar_reunitarize restores special unitarity for near-unitary matrices', async () => {
  await fc.assert(
    fc.property(nearUnitaryArb, (matrix) => {
      const baselineErr = computeUnitaryError(matrix);
      fc.pre(baselineErr < 0.5);
      const repaired = polar_reunitarize(matrix);
      const err = computeUnitaryError(repaired);
      const drift = computeDeterminantDrift(repaired);
      assert.ok(err <= 1e-7, `polar error ${err}`);
      assert.ok(drift <= 1e-7, `polar drift ${drift}`);
    }),
    { numRuns: 25 },
  );
});
