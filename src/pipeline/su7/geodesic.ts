import { cloneComplex7x7, type Complex, type Complex7x7, type Su7Telemetry } from './types.js';
import {
  compose_dense,
  computeDeterminant,
  computeUnitaryError,
  expm_su,
  logm_su,
  projectToSpecialUnitary,
  polar_reunitarize,
} from './math.js';

const VECTOR_DIM = 7;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_BRANCH_GUARD = 1e-4;
const NEGATIVE_IDENTITY_DET_THRESHOLD = 1e-6;
const POLAR_REGULARIZATION_EPSILON = 1e-3;

type FallbackLogTarget = Pick<Su7Telemetry, 'geodesicFallbacks'>;

export type GeodesicMorphMethod = 'geodesic' | 'polar';

export type GeodesicMorph = {
  readonly start: Complex7x7;
  readonly end: Complex7x7;
  readonly generator: Complex7x7 | null;
  readonly method: GeodesicMorphMethod;
  readonly fallbackReason: string | null;
  readonly fallbackCount: number;
  evaluate: (t: number) => Complex7x7;
};

export type GeodesicMorphOptions = {
  tolerance?: number;
  branchGuard?: number;
  telemetry?: FallbackLogTarget | null;
};

const createZeroMatrix = (): Complex7x7 => {
  const rows: Complex[][] = [];
  for (let row = 0; row < VECTOR_DIM; row++) {
    const cells: Complex[] = [];
    for (let col = 0; col < VECTOR_DIM; col++) {
      cells.push({ re: 0, im: 0 });
    }
    rows.push(cells);
  }
  return rows as Complex7x7;
};

const conjugateTranspose = (matrix: Complex7x7): Complex7x7 => {
  const result = createZeroMatrix();
  for (let row = 0; row < VECTOR_DIM; row++) {
    for (let col = 0; col < VECTOR_DIM; col++) {
      const entry = matrix[col][row];
      result[row][col] = { re: entry.re, im: -entry.im };
    }
  }
  return result;
};

const scaleMatrix = (matrix: Complex7x7, scalar: number): Complex7x7 => {
  const result = createZeroMatrix();
  for (let row = 0; row < VECTOR_DIM; row++) {
    for (let col = 0; col < VECTOR_DIM; col++) {
      const entry = matrix[row][col];
      result[row][col] = {
        re: entry.re * scalar,
        im: entry.im * scalar,
      };
    }
  }
  return result;
};

const blendMatrices = (start: Complex7x7, end: Complex7x7, t: number): Complex7x7 => {
  const oneMinusT = 1 - t;
  const result = createZeroMatrix();
  for (let row = 0; row < VECTOR_DIM; row++) {
    for (let col = 0; col < VECTOR_DIM; col++) {
      const a = start[row][col];
      const b = end[row][col];
      result[row][col] = {
        re: a.re * oneMinusT + b.re * t,
        im: a.im * oneMinusT + b.im * t,
      };
    }
  }
  return result;
};

const computeReconstructionError = (delta: Complex7x7, generator: Complex7x7): number => {
  const reconstructed = expm_su(generator);
  const compare = compose_dense(reconstructed, conjugateTranspose(delta));
  return computeUnitaryError(compare);
};

const complexMagnitude = (value: Complex): number => Math.hypot(value.re, value.im);

const addScaledIdentity = (matrix: Complex7x7, scale: number): Complex7x7 => {
  const result = cloneMatrix(matrix);
  for (let i = 0; i < VECTOR_DIM; i++) {
    const entry = result[i][i];
    result[i][i] = { re: entry.re + scale, im: entry.im };
  }
  return result;
};

const maxImagDiag = (matrix: Complex7x7): number => {
  let max = 0;
  for (let i = 0; i < VECTOR_DIM; i++) {
    const entry = matrix[i][i];
    const imag = Math.abs(entry.im);
    if (imag > max) {
      max = imag;
    }
  }
  return max;
};

const cloneMatrix = (matrix: Complex7x7): Complex7x7 => cloneComplex7x7(matrix);

const recordFallback = (target: FallbackLogTarget | null | undefined, count: number) => {
  if (!target || count <= 0) {
    return;
  }
  target.geodesicFallbacks = (target.geodesicFallbacks ?? 0) + count;
};

export const createGeodesicMorph = (
  start: Complex7x7,
  end: Complex7x7,
  options: GeodesicMorphOptions = {},
): GeodesicMorph => {
  const tolerance = Number.isFinite(options.tolerance)
    ? (options.tolerance as number)
    : DEFAULT_TOLERANCE;
  const branchGuard = Number.isFinite(options.branchGuard)
    ? (options.branchGuard as number)
    : DEFAULT_BRANCH_GUARD;

  const startUnitary = polar_reunitarize(start);
  const endUnitary = polar_reunitarize(end);
  const startClone = cloneMatrix(startUnitary);
  const endClone = cloneMatrix(endUnitary);

  const delta = compose_dense(endClone, conjugateTranspose(startClone));
  const detDeltaPlusIdentity = computeDeterminant(addScaledIdentity(delta, 1));
  const nearNegativeIdentity =
    complexMagnitude(detDeltaPlusIdentity) <= NEGATIVE_IDENTITY_DET_THRESHOLD;

  let generator: Complex7x7 | null = null;
  let fallbackReason: string | null = null;
  let fallbackCount = 0;
  let method: GeodesicMorphMethod = 'geodesic';

  try {
    const candidate = logm_su(delta);
    const reconstructionError = computeReconstructionError(delta, candidate);
    const diagImag = maxImagDiag(candidate);
    const nonFinite = !Number.isFinite(reconstructionError) || !Number.isFinite(diagImag);
    const reconstructionIssue = reconstructionError > tolerance;
    const branchIssue = diagImag >= Math.PI - branchGuard || nearNegativeIdentity;

    if (nonFinite || reconstructionIssue || branchIssue) {
      fallbackReason = nonFinite
        ? 'non-finite-generator'
        : reconstructionIssue
          ? 'reconstruction-error'
          : 'branch-cut';
      fallbackCount = 1;
      method = 'polar';
    } else {
      generator = candidate;
    }
  } catch (error) {
    fallbackReason = (error as Error).message ?? 'logm-error';
    fallbackCount = 1;
    method = 'polar';
  }

  recordFallback(options.telemetry, fallbackCount);

  const evaluate = (t: number): Complex7x7 => {
    if (method === 'geodesic' && generator) {
      const scaled = scaleMatrix(generator, Number.isFinite(t) ? (t as number) : 0);
      const step = expm_su(scaled);
      return compose_dense(step, startClone);
    }
    const mix = blendMatrices(startClone, endClone, Number.isFinite(t) ? (t as number) : 0);
    const stabilized = addScaledIdentity(mix, POLAR_REGULARIZATION_EPSILON);
    return projectToSpecialUnitary(stabilized);
  };

  return {
    start: startClone,
    end: endClone,
    generator,
    method,
    fallbackReason,
    fallbackCount,
    evaluate,
  };
};

const clampUnit = (value: number): number => {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value > 1) return 1;
  if (value < -1) return -1;
  return value;
};

const clamp01 = (value: number): number => (value <= 0 ? 0 : value >= 1 ? 1 : value);

const wrapAngle = (theta: number): number => {
  let t = theta;
  const tau = Math.PI * 2;
  t = t - Math.floor((t + Math.PI) / tau) * tau;
  return t - Math.PI;
};

const complexArgument = (value: Complex): number => Math.atan2(value.im, value.re);

const complexMultiply = (a: Complex, b: Complex): Complex => ({
  re: a.re * b.re - a.im * b.im,
  im: a.re * b.im + a.im * b.re,
});

const complexConjugate = (value: Complex): Complex => ({ re: value.re, im: -value.im });

const scaleComplex = (value: Complex, scalar: number): Complex => ({
  re: value.re * scalar,
  im: value.im * scalar,
});

export type HopfCoordinates = {
  base: [number, number, number];
  fiber: number;
  magnitude: number;
};

export const computeHopfCoordinates = (a: Complex, b: Complex): HopfCoordinates => {
  const magA = complexMagnitude(a);
  const magB = complexMagnitude(b);
  const magnitude = Math.hypot(magA, magB);
  if (magnitude <= 1e-12) {
    return {
      base: [0, 0, 0],
      fiber: 0,
      magnitude: 0,
    };
  }
  const inv = 1 / magnitude;
  const na = scaleComplex(a, inv);
  const nb = scaleComplex(b, inv);
  const prod = complexMultiply(na, complexConjugate(nb));
  const x = clampUnit(2 * prod.re);
  const y = clampUnit(2 * prod.im);
  const nzA = clampUnit(magA * inv);
  const nzB = clampUnit(magB * inv);
  const z = clampUnit(nzA * nzA - nzB * nzB);
  const phaseA = complexArgument(na);
  const phaseB = complexArgument(nb);
  const fiber = wrapAngle(0.5 * (phaseA + phaseB));
  return {
    base: [x, y, z],
    fiber,
    magnitude,
  };
};

export const hopfBaseToRgb = (
  base: readonly [number, number, number],
): [number, number, number] => {
  const r = clamp01(0.5 + 0.5 * clampUnit(base[0]));
  const g = clamp01(0.5 + 0.5 * clampUnit(base[1]));
  const b = clamp01(0.5 + 0.5 * clampUnit(base[2]));
  return [r, g, b];
};

export const hopfFiberToRgb = (fiber: number): [number, number, number] => {
  const angle = wrapAngle(fiber);
  const r = clamp01(0.5 + 0.5 * Math.cos(angle));
  const g = clamp01(0.5 + 0.5 * Math.cos(angle - (2 * Math.PI) / 3));
  const b = clamp01(0.5 + 0.5 * Math.cos(angle + (2 * Math.PI) / 3));
  return [r, g, b];
};
