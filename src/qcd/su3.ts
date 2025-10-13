import { type Complex, type Complex7x7 } from '../pipeline/su7/types.js';

export type Complex3Vector = [Complex, Complex, Complex];
export type Complex3x3 = [Complex3Vector, Complex3Vector, Complex3Vector];

const DIM3 = 3;
const DIM7 = 7;
const EPSILON = 1e-12;

const complex = (re: number, im: number): Complex => ({ re, im });

const createZero3 = (): Complex3x3 =>
  new Array(DIM3)
    .fill(null)
    .map(() => new Array(DIM3).fill(null).map(() => complex(0, 0)) as Complex3Vector) as Complex3x3;

const createIdentity3 = (): Complex3x3 => {
  const result = createZero3();
  for (let i = 0; i < DIM3; i++) {
    result[i][i] = complex(1, 0);
  }
  return result;
};

const createIdentity7 = (): Complex7x7 => {
  const rows: Complex[][] = [];
  for (let i = 0; i < DIM7; i++) {
    const row: Complex[] = [];
    for (let j = 0; j < DIM7; j++) {
      row.push(complex(i === j ? 1 : 0, 0));
    }
    rows.push(row);
  }
  return rows as Complex7x7;
};

const cloneComplex3x3 = (matrix: Complex3x3): Complex3x3 =>
  matrix.map(
    (row) => row.map((entry) => complex(entry.re, entry.im)) as Complex3Vector,
  ) as Complex3x3;

const complexAdd = (a: Complex, b: Complex): Complex => complex(a.re + b.re, a.im + b.im);

const complexSub = (a: Complex, b: Complex): Complex => complex(a.re - b.re, a.im - b.im);

const complexMul = (a: Complex, b: Complex): Complex =>
  complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);

const complexConj = (value: Complex): Complex => complex(value.re, -value.im);

const complexAbs2 = (value: Complex): number => value.re * value.re + value.im * value.im;

const complexDiv = (a: Complex, b: Complex): Complex => {
  const denom = complexAbs2(b);
  if (denom <= EPSILON) {
    return complex(0, 0);
  }
  const conj = complexConj(b);
  const num = complexMul(a, conj);
  return complex(num.re / denom, num.im / denom);
};

const complexScale = (value: Complex, scalar: number): Complex =>
  complex(value.re * scalar, value.im * scalar);

const complexMulImag = (value: Complex): Complex => complex(-value.im, value.re);

const complexMulNegImag = (value: Complex): Complex => complex(value.im, -value.re);

const matrixAdd = (a: Complex3x3, b: Complex3x3): Complex3x3 => {
  const result = createZero3();
  for (let row = 0; row < DIM3; row++) {
    for (let col = 0; col < DIM3; col++) {
      result[row][col] = complexAdd(a[row][col], b[row][col]);
    }
  }
  return result;
};

const matrixSub = (a: Complex3x3, b: Complex3x3): Complex3x3 => {
  const result = createZero3();
  for (let row = 0; row < DIM3; row++) {
    for (let col = 0; col < DIM3; col++) {
      result[row][col] = complexSub(a[row][col], b[row][col]);
    }
  }
  return result;
};

const matrixScale = (matrix: Complex3x3, scalar: number): Complex3x3 => {
  const result = createZero3();
  for (let row = 0; row < DIM3; row++) {
    for (let col = 0; col < DIM3; col++) {
      result[row][col] = complexScale(matrix[row][col], scalar);
    }
  }
  return result;
};

const matrixLinearCombination = (
  terms: readonly { matrix: Complex3x3; weight: number }[],
): Complex3x3 => {
  const result = createZero3();
  for (const { matrix, weight } of terms) {
    if (!Number.isFinite(weight) || weight === 0) continue;
    for (let row = 0; row < DIM3; row++) {
      for (let col = 0; col < DIM3; col++) {
        const entry = matrix[row][col];
        result[row][col].re += entry.re * weight;
        result[row][col].im += entry.im * weight;
      }
    }
  }
  return result;
};

const conjugateTranspose3 = (matrix: Complex3x3): Complex3x3 => {
  const result = createZero3();
  for (let row = 0; row < DIM3; row++) {
    for (let col = 0; col < DIM3; col++) {
      result[row][col] = complexConj(matrix[col][row]);
    }
  }
  return result;
};

const matrixMultiply3 = (a: Complex3x3, b: Complex3x3): Complex3x3 => {
  const result = createZero3();
  for (let row = 0; row < DIM3; row++) {
    for (let col = 0; col < DIM3; col++) {
      let sum = complex(0, 0);
      for (let k = 0; k < DIM3; k++) {
        sum = complexAdd(sum, complexMul(a[row][k], b[k][col]));
      }
      result[row][col] = sum;
    }
  }
  return result;
};

const matrixMinusIdentity3 = (matrix: Complex3x3): Complex3x3 => {
  const result = createZero3();
  for (let row = 0; row < DIM3; row++) {
    for (let col = 0; col < DIM3; col++) {
      const identity = row === col ? complex(1, 0) : complex(0, 0);
      result[row][col] = complexSub(matrix[row][col], identity);
    }
  }
  return result;
};

const addScaledIdentity3 = (matrix: Complex3x3, scalar: number): Complex3x3 => {
  const result = cloneComplex3x3(matrix);
  for (let i = 0; i < DIM3; i++) {
    result[i][i] = complexAdd(result[i][i], complex(scalar, 0));
  }
  return result;
};

const frobeniusNorm3 = (matrix: Complex3x3): number => {
  let sum = 0;
  for (let row = 0; row < DIM3; row++) {
    for (let col = 0; col < DIM3; col++) {
      const entry = matrix[row][col];
      sum += entry.re * entry.re + entry.im * entry.im;
    }
  }
  return Math.sqrt(sum);
};

const matrixOneNorm3 = (matrix: Complex3x3): number => {
  let max = 0;
  for (let col = 0; col < DIM3; col++) {
    let sum = 0;
    for (let row = 0; row < DIM3; row++) {
      const entry = matrix[row][col];
      sum += Math.hypot(entry.re, entry.im);
    }
    if (sum > max) {
      max = sum;
    }
  }
  return max;
};

const solveLinear3 = (lhs: Complex3x3, rhs: Complex3x3): Complex3x3 => {
  const a = cloneComplex3x3(lhs);
  const result = cloneComplex3x3(rhs);

  for (let pivotIndex = 0; pivotIndex < DIM3; pivotIndex++) {
    let pivotRow = pivotIndex;
    let pivotMag = complexAbs2(a[pivotIndex][pivotIndex]);
    for (let candidate = pivotIndex + 1; candidate < DIM3; candidate++) {
      const magnitude = complexAbs2(a[candidate][pivotIndex]);
      if (magnitude > pivotMag) {
        pivotMag = magnitude;
        pivotRow = candidate;
      }
    }

    if (pivotMag <= EPSILON) {
      a[pivotIndex][pivotIndex] = complex(1, 0);
      pivotMag = 1;
    }

    if (pivotRow !== pivotIndex) {
      const tempRow = a[pivotIndex];
      a[pivotIndex] = a[pivotRow];
      a[pivotRow] = tempRow;
      const tempResult = result[pivotIndex];
      result[pivotIndex] = result[pivotRow];
      result[pivotRow] = tempResult;
    }

    const pivot = a[pivotIndex][pivotIndex];

    for (let row = pivotIndex + 1; row < DIM3; row++) {
      const entry = a[row][pivotIndex];
      if (complexAbs2(entry) <= EPSILON) {
        a[row][pivotIndex] = complex(0, 0);
        continue;
      }
      const factor = complexDiv(entry, pivot);
      for (let col = pivotIndex; col < DIM3; col++) {
        a[row][col] = complexSub(a[row][col], complexMul(factor, a[pivotIndex][col]));
      }
      for (let col = 0; col < DIM3; col++) {
        result[row][col] = complexSub(
          result[row][col],
          complexMul(factor, result[pivotIndex][col]),
        );
      }
      a[row][pivotIndex] = complex(0, 0);
    }
  }

  for (let row = DIM3 - 1; row >= 0; row--) {
    const pivot = a[row][row];
    for (let col = 0; col < DIM3; col++) {
      let value = result[row][col];
      for (let k = row + 1; k < DIM3; k++) {
        value = complexSub(value, complexMul(a[row][k], result[k][col]));
      }
      result[row][col] = complexDiv(value, pivot);
    }
  }

  return result;
};

const determinant3 = (matrix: Complex3x3): Complex => {
  const a = matrix[0][0];
  const b = matrix[0][1];
  const cEntry = matrix[0][2];
  const d = matrix[1][0];
  const e = matrix[1][1];
  const f = matrix[1][2];
  const g = matrix[2][0];
  const h = matrix[2][1];
  const i = matrix[2][2];

  const term1 = complexMul(a, complexSub(complexMul(e, i), complexMul(f, h)));
  const term2 = complexMul(b, complexSub(complexMul(d, i), complexMul(f, g)));
  const term3 = complexMul(cEntry, complexSub(complexMul(d, h), complexMul(e, g)));
  return complexAdd(complexSub(term1, term2), term3);
};

const polarInverseSqrt3 = (matrix: Complex3x3): Complex3x3 => {
  let trace = 0;
  for (let i = 0; i < DIM3; i++) {
    trace += matrix[i][i].re;
  }
  let scale = trace / DIM3;
  if (!Number.isFinite(scale) || scale <= 0) {
    scale = 1;
  }
  let Y = matrixScale(matrix, 1 / scale);
  let Z = createIdentity3();
  let residual = Number.POSITIVE_INFINITY;

  for (let iter = 0; iter < 12; iter++) {
    const YZ = matrixMultiply3(Z, Y);
    const correction = createZero3();
    for (let row = 0; row < DIM3; row++) {
      for (let col = 0; col < DIM3; col++) {
        const entry = YZ[row][col];
        correction[row][col] = complex(-entry.re, -entry.im);
      }
      correction[row][row].re += DIM3;
    }
    const halfCorrection = matrixScale(correction, 0.5);
    const nextY = matrixMultiply3(Y, halfCorrection);
    const nextZ = matrixMultiply3(halfCorrection, Z);
    const gram = matrixMultiply3(conjugateTranspose3(nextY), nextY);
    const error = frobeniusNorm3(matrixMinusIdentity3(gram));
    Y = nextY;
    Z = nextZ;
    if (error < 1e-12 || Math.abs(error - residual) < 1e-14) {
      break;
    }
    residual = error;
  }

  return matrixScale(Z, 1 / Math.sqrt(scale));
};

const normalizeDeterminant = (matrix: Complex3x3): Complex3x3 => {
  const det = determinant3(matrix);
  const angle = -Math.atan2(det.im, det.re) / DIM3;
  const correction = complex(Math.cos(angle), Math.sin(angle));
  const result = createZero3();
  for (let row = 0; row < DIM3; row++) {
    for (let col = 0; col < DIM3; col++) {
      result[row][col] = complexMul(matrix[row][col], correction);
    }
  }
  return result;
};

const polarReunitarize3 = (matrix: Complex3x3): Complex3x3 => {
  const base = cloneComplex3x3(matrix);
  const gram = matrixMultiply3(conjugateTranspose3(base), base);
  const invSqrt = polarInverseSqrt3(gram);
  const unitary = matrixMultiply3(base, invSqrt);
  return normalizeDeterminant(unitary);
};

const makeSkewHermitian3 = (matrix: Complex3x3): Complex3x3 => {
  const conjugate = conjugateTranspose3(matrix);
  return matrixScale(matrixSub(matrix, conjugate), 0.5);
};

const computeTrace3 = (matrix: Complex3x3): Complex => {
  let trace = complex(0, 0);
  for (let i = 0; i < DIM3; i++) {
    trace = complexAdd(trace, matrix[i][i]);
  }
  return trace;
};

const matrixSquareRootUnitary3 = (unitary: Complex3x3): Complex3x3 => {
  let Y = cloneComplex3x3(unitary);
  let Z = createIdentity3();
  const identity = createIdentity3();

  for (let iter = 0; iter < 12; iter++) {
    const Yinv = solveLinear3(Y, identity);
    const Zinv = solveLinear3(Z, identity);
    const nextY = matrixScale(matrixAdd(Y, Zinv), 0.5);
    const nextZ = matrixScale(matrixAdd(Z, Yinv), 0.5);
    const diffY = frobeniusNorm3(matrixSub(nextY, Y));
    const diffZ = frobeniusNorm3(matrixSub(nextZ, Z));
    Y = nextY;
    Z = nextZ;
    if (diffY < 1e-12 && diffZ < 1e-12) {
      break;
    }
  }

  return polarReunitarize3(Y);
};

const sqrtOneThird = 1 / Math.sqrt(3);

const gellMannMatrices: Complex3x3[] = [
  [
    [complex(0, 0), complex(1, 0), complex(0, 0)] as Complex3Vector,
    [complex(1, 0), complex(0, 0), complex(0, 0)] as Complex3Vector,
    [complex(0, 0), complex(0, 0), complex(0, 0)] as Complex3Vector,
  ] as Complex3x3,
  [
    [complex(0, 0), complex(0, -1), complex(0, 0)] as Complex3Vector,
    [complex(0, 1), complex(0, 0), complex(0, 0)] as Complex3Vector,
    [complex(0, 0), complex(0, 0), complex(0, 0)] as Complex3Vector,
  ] as Complex3x3,
  [
    [complex(1, 0), complex(0, 0), complex(0, 0)] as Complex3Vector,
    [complex(0, 0), complex(-1, 0), complex(0, 0)] as Complex3Vector,
    [complex(0, 0), complex(0, 0), complex(0, 0)] as Complex3Vector,
  ] as Complex3x3,
  [
    [complex(0, 0), complex(0, 0), complex(1, 0)] as Complex3Vector,
    [complex(0, 0), complex(0, 0), complex(0, 0)] as Complex3Vector,
    [complex(1, 0), complex(0, 0), complex(0, 0)] as Complex3Vector,
  ] as Complex3x3,
  [
    [complex(0, 0), complex(0, 0), complex(0, -1)] as Complex3Vector,
    [complex(0, 0), complex(0, 0), complex(0, 0)] as Complex3Vector,
    [complex(0, 1), complex(0, 0), complex(0, 0)] as Complex3Vector,
  ] as Complex3x3,
  [
    [complex(0, 0), complex(0, 0), complex(0, 0)] as Complex3Vector,
    [complex(0, 0), complex(0, 0), complex(1, 0)] as Complex3Vector,
    [complex(0, 0), complex(1, 0), complex(0, 0)] as Complex3Vector,
  ] as Complex3x3,
  [
    [complex(0, 0), complex(0, 0), complex(0, 0)] as Complex3Vector,
    [complex(0, 0), complex(0, 0), complex(0, -1)] as Complex3Vector,
    [complex(0, 0), complex(0, 1), complex(0, 0)] as Complex3Vector,
  ] as Complex3x3,
  [
    [complex(sqrtOneThird, 0), complex(0, 0), complex(0, 0)] as Complex3Vector,
    [complex(0, 0), complex(sqrtOneThird, 0), complex(0, 0)] as Complex3Vector,
    [complex(0, 0), complex(0, 0), complex(-2 * sqrtOneThird, 0)] as Complex3Vector,
  ] as Complex3x3,
];

const buildHermitianFromCoefficients = (coeffs: ArrayLike<number>): Complex3x3 => {
  const result = createZero3();
  for (let idx = 0; idx < 8; idx++) {
    const weight =
      idx < coeffs.length && Number.isFinite(coeffs[idx]) ? (coeffs[idx] as number) : 0;
    if (weight === 0) continue;
    const basis = gellMannMatrices[idx];
    for (let row = 0; row < DIM3; row++) {
      for (let col = 0; col < DIM3; col++) {
        const entry = basis[row][col];
        result[row][col].re += entry.re * weight;
        result[row][col].im += entry.im * weight;
      }
    }
  }
  return result;
};

const traceProduct = (a: Complex3x3, b: Complex3x3): Complex => {
  let sum = complex(0, 0);
  for (let row = 0; row < DIM3; row++) {
    for (let k = 0; k < DIM3; k++) {
      sum = complexAdd(sum, complexMul(a[row][k], b[k][row]));
    }
  }
  return sum;
};

const makeHermitianFromSkew = (skew: Complex3x3): Complex3x3 => {
  const result = createZero3();
  for (let row = 0; row < DIM3; row++) {
    for (let col = 0; col < DIM3; col++) {
      result[row][col] = complexMulNegImag(skew[row][col]);
    }
  }
  return result;
};

const makeSkewFromHermitian = (hermitian: Complex3x3): Complex3x3 => {
  const result = createZero3();
  for (let row = 0; row < DIM3; row++) {
    for (let col = 0; col < DIM3; col++) {
      result[row][col] = complexMulImag(hermitian[row][col]);
    }
  }
  return result;
};

const expmSkew3 = (skew: Complex3x3): Complex3x3 => {
  const input = makeSkewHermitian3(skew);
  const trace = computeTrace3(input);
  const mean = complex(trace.re / DIM3, trace.im / DIM3);
  for (let i = 0; i < DIM3; i++) {
    input[i][i] = complexSub(input[i][i], mean);
  }

  const theta13 = 2.29;
  const norm = matrixOneNorm3(input);
  let s = 0;
  if (norm > theta13) {
    s = Math.max(0, Math.ceil(Math.log2(norm / theta13)));
  }
  const scale = 1 / (1 << s);
  const scaled = matrixScale(input, scale);

  const X2 = matrixMultiply3(scaled, scaled);
  const X4 = matrixMultiply3(X2, X2);
  const X6 = matrixMultiply3(X4, X2);

  const b = [
    64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000,
    10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1,
  ];

  const identity = createIdentity3();
  const innerU = matrixLinearCombination([
    { matrix: X6, weight: b[13] },
    { matrix: X4, weight: b[11] },
    { matrix: X2, weight: b[9] },
  ]);
  const X6InnerU = matrixMultiply3(X6, innerU);
  const polyU = matrixLinearCombination([
    { matrix: X6InnerU, weight: 1 },
    { matrix: X6, weight: b[7] },
    { matrix: X4, weight: b[5] },
    { matrix: X2, weight: b[3] },
    { matrix: identity, weight: b[1] },
  ]);
  const U = matrixMultiply3(scaled, polyU);

  const innerV = matrixLinearCombination([
    { matrix: X6, weight: b[12] },
    { matrix: X4, weight: b[10] },
    { matrix: X2, weight: b[8] },
  ]);
  const X6InnerV = matrixMultiply3(X6, innerV);
  const V = matrixLinearCombination([
    { matrix: X6InnerV, weight: 1 },
    { matrix: X6, weight: b[6] },
    { matrix: X4, weight: b[4] },
    { matrix: X2, weight: b[2] },
    { matrix: identity, weight: b[0] },
  ]);

  const VminusU = matrixSub(V, U);
  const VplusU = matrixAdd(V, U);
  let result = solveLinear3(VminusU, VplusU);

  for (let i = 0; i < s; i++) {
    result = matrixMultiply3(result, result);
  }

  return polarReunitarize3(result);
};

const logmSkew3 = (matrix: Complex3x3): Complex3x3 => {
  let unitary = polarReunitarize3(matrix);
  const identity = createIdentity3();
  let scaling = 0;

  while (scaling < 10) {
    const delta = matrixSub(unitary, identity);
    const norm = frobeniusNorm3(delta);
    if (norm < 0.55) {
      break;
    }
    unitary = matrixSquareRootUnitary3(unitary);
    scaling += 1;
  }

  const UplusI = addScaledIdentity3(unitary, 1);
  const UminusI = matrixSub(unitary, identity);
  const inverse = solveLinear3(UplusI, identity);
  let K = matrixMultiply3(UminusI, inverse);
  K = makeSkewHermitian3(K);

  const K2 = matrixMultiply3(K, K);
  let term = cloneComplex3x3(K);
  let result = matrixScale(term, 2);
  const maxTerms = 18;
  for (let n = 1; n < maxTerms; n++) {
    term = matrixMultiply3(term, K2);
    const coeff = 2 / (2 * n + 1);
    const contribution = matrixScale(term, coeff);
    result = matrixAdd(result, contribution);
    const magnitude = frobeniusNorm3(contribution);
    if (magnitude < 1e-12) {
      break;
    }
  }

  if (scaling > 0) {
    result = matrixScale(result, 1 << scaling);
  }

  result = makeSkewHermitian3(result);
  const trace = computeTrace3(result);
  const mean = complex(trace.re / DIM3, trace.im / DIM3);
  for (let i = 0; i < DIM3; i++) {
    result[i][i] = complexSub(result[i][i], mean);
  }
  return result;
};

const createGaussianGenerator = (rng: () => number): (() => number) => {
  let spare: number | null = null;
  return () => {
    if (spare != null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u = 0;
    while (u <= EPSILON) {
      u = rng();
    }
    const v = rng();
    const radius = Math.sqrt(-2 * Math.log(u));
    const theta = 2 * Math.PI * v;
    spare = radius * Math.sin(theta);
    return radius * Math.cos(theta);
  };
};

export const su3_mul = (lhs: Complex3x3, rhs: Complex3x3): Complex3x3 => matrixMultiply3(lhs, rhs);

export const su3_conjugateTranspose = (matrix: Complex3x3): Complex3x3 =>
  conjugateTranspose3(matrix);

export const su3_applyToVector = (matrix: Complex3x3, vector: Complex3Vector): Complex3Vector => {
  const result: Complex3Vector = [
    { re: 0, im: 0 },
    { re: 0, im: 0 },
    { re: 0, im: 0 },
  ];
  for (let row = 0; row < DIM3; row++) {
    let sumRe = 0;
    let sumIm = 0;
    for (let col = 0; col < DIM3; col++) {
      const coeff = matrix[row][col];
      const entry = vector[col];
      sumRe += coeff.re * entry.re - coeff.im * entry.im;
      sumIm += coeff.re * entry.im + coeff.im * entry.re;
    }
    result[row] = complex(sumRe, sumIm);
  }
  return result;
};

export const su3_frobNorm = (matrix: Complex3x3): number => frobeniusNorm3(matrix);

export const su3_project = (matrix: Complex3x3): Complex3x3 => polarReunitarize3(matrix);

export const su3_exp = (coeffs: ArrayLike<number>): Complex3x3 => {
  const hermitian = buildHermitianFromCoefficients(coeffs);
  const skew = makeSkewFromHermitian(hermitian);
  return expmSkew3(skew);
};

export const su3_log = (matrix: Complex3x3): Float64Array => {
  const skew = logmSkew3(matrix);
  const hermitian = makeHermitianFromSkew(skew);
  const coefficients = new Float64Array(8);
  for (let idx = 0; idx < 8; idx++) {
    const lambda = gellMannMatrices[idx];
    const trace = traceProduct(hermitian, lambda);
    coefficients[idx] = trace.re / 2;
  }
  return coefficients;
};

export const su3_embed = (block: Complex3x3): Complex7x7 => {
  const su3 = su3_project(block);
  const result = createIdentity7();
  for (let row = 0; row < DIM3; row++) {
    for (let col = 0; col < DIM3; col++) {
      result[row][col] = complex(su3[row][col].re, su3[row][col].im);
    }
  }
  return result;
};

export const su3_haar = (beta: number, rng: () => number = Math.random): Complex3x3 => {
  const variance = Number.isFinite(beta) && beta > 0 ? beta : 1;
  const gaussian = createGaussianGenerator(rng);
  const coeffs = new Float64Array(8);
  const scale = Math.sqrt(variance);
  for (let idx = 0; idx < 8; idx++) {
    coeffs[idx] = gaussian() * scale;
  }
  return su3_exp(coeffs);
};
