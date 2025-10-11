import { type C7Vector, type Complex, type Complex7x7 } from './types.js';

export type UnitaryScheduleStage = {
  preset: string;
  theta?: number;
};

export type BuildUnitaryOptions = {
  seed?: number;
  preset?: string;
  theta?: number;
  schedule?: readonly UnitaryScheduleStage[];
};

const DIM = 7;
const EPSILON = 1e-9;
const DEFAULT_SEED = 0;
const DEFAULT_THETA = Math.PI / 12;
const DEFAULT_PRESET = 'random';

const RAISING_PRESETS: Record<string, readonly [number, number]> = {
  'raise:01': [0, 1],
  'raise:12': [1, 2],
  'raise:23': [2, 3],
  'raise:34': [3, 4],
  'raise:45': [4, 5],
  'raise:56': [5, 6],
  'raise:06': [0, 6],
};

const createIdentity = (): Complex7x7 => {
  const rows: Complex[][] = [];
  for (let i = 0; i < DIM; i++) {
    const row: Complex[] = [];
    for (let j = 0; j < DIM; j++) {
      row.push({ re: i === j ? 1 : 0, im: 0 });
    }
    rows.push(row);
  }
  return rows as Complex7x7;
};

const cloneMatrix = (matrix: Complex7x7): Complex7x7 =>
  matrix.map((row) => row.map((entry) => ({ re: entry.re, im: entry.im }))) as Complex7x7;

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
  if (denom <= EPSILON) {
    return { re: 0, im: 0 };
  }
  const conj = complexConj(b);
  const num = complexMul(a, conj);
  return { re: num.re / denom, im: num.im / denom };
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

const buildRandomComplexMatrix = (seed: number): Complex7x7 => {
  const rng = mulberry32(seed);
  const gaussian = createGaussianGenerator(rng);
  const rows: Complex[][] = [];
  for (let i = 0; i < DIM; i++) {
    const row: Complex[] = [];
    for (let j = 0; j < DIM; j++) {
      row.push({ re: gaussian(), im: gaussian() });
    }
    rows.push(row);
  }
  return rows as Complex7x7;
};

const columnVector = (matrix: Complex7x7, column: number): Complex[] =>
  matrix.map((row) => ({ ...row[column] }));

const dotProduct = (a: Complex[], b: Complex[]): Complex => {
  let sum: Complex = { re: 0, im: 0 };
  for (let i = 0; i < DIM; i++) {
    const conj = complexConj(a[i]);
    sum = complexAdd(sum, complexMul(conj, b[i]));
  }
  return sum;
};

const normalizeVector = (vector: Complex[]): Complex[] => {
  let normSq = 0;
  for (let i = 0; i < DIM; i++) {
    normSq += complexAbs2(vector[i]);
  }
  const norm = Math.sqrt(normSq);
  if (norm <= EPSILON) {
    const fallback: Complex[] = new Array(DIM)
      .fill(null)
      .map((_, idx) => ({ re: idx === 0 ? 1 : 0, im: 0 }));
    return fallback;
  }
  return vector.map((entry) => ({
    re: entry.re / norm,
    im: entry.im / norm,
  }));
};

const orthonormalize = (matrix: Complex7x7): Complex7x7 => {
  const qColumns: Complex[][] = [];
  for (let col = 0; col < DIM; col++) {
    let v = columnVector(matrix, col);
    for (let prev = 0; prev < col; prev++) {
      const coef = dotProduct(qColumns[prev], v);
      const scaled = qColumns[prev].map((entry) => complexMul(entry, coef));
      v = v.map((entry, idx) => complexSub(entry, scaled[idx]));
    }
    v = normalizeVector(v);
    qColumns.push(v);
  }

  const orthonormal = createIdentity();
  for (let row = 0; row < DIM; row++) {
    for (let col = 0; col < DIM; col++) {
      orthonormal[row][col] = { ...qColumns[col][row] };
    }
  }
  return orthonormal;
};

const computeDeterminant = (matrix: Complex7x7): Complex => {
  const working = cloneMatrix(matrix);
  let sign = 1;
  let det: Complex = { re: 1, im: 0 };

  for (let i = 0; i < DIM; i++) {
    let pivotRow = i;
    let pivotMag = Math.sqrt(complexAbs2(working[i][i]));
    for (let r = i + 1; r < DIM; r++) {
      const mag = Math.sqrt(complexAbs2(working[r][i]));
      if (mag > pivotMag) {
        pivotMag = mag;
        pivotRow = r;
      }
    }

    if (pivotMag <= EPSILON) {
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

    for (let r = i + 1; r < DIM; r++) {
      const factor = complexDiv(working[r][i], pivot);
      if (Math.sqrt(complexAbs2(factor)) <= EPSILON) continue;
      for (let c = i; c < DIM; c++) {
        working[r][c] = complexSub(working[r][c], complexMul(factor, working[i][c]));
      }
    }
  }

  if (sign === -1) {
    det = { re: -det.re, im: -det.im };
  }

  return det;
};

const enforceSpecialUnitary = (matrix: Complex7x7): Complex7x7 => {
  const orthonormal = orthonormalize(matrix);
  const det = computeDeterminant(orthonormal);
  const magnitude = Math.sqrt(det.re * det.re + det.im * det.im);
  if (magnitude <= EPSILON) {
    return orthonormal;
  }
  const scale = { re: det.re / magnitude, im: -det.im / magnitude };

  for (let row = 0; row < DIM; row++) {
    const entry = orthonormal[row][DIM - 1];
    orthonormal[row][DIM - 1] = complexMul(entry, scale);
  }
  return orthonormal;
};

const multiplyMatrices = (a: Complex7x7, b: Complex7x7): Complex7x7 => {
  const result = createIdentity();
  for (let row = 0; row < DIM; row++) {
    for (let col = 0; col < DIM; col++) {
      let sum: Complex = { re: 0, im: 0 };
      for (let k = 0; k < DIM; k++) {
        sum = complexAdd(sum, complexMul(a[row][k], b[k][col]));
      }
      result[row][col] = sum;
    }
  }
  return result;
};

const buildRaisingPulse = (pair: readonly [number, number], theta: number): Complex7x7 => {
  const [i, j] = pair;
  const result = createIdentity();
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);

  result[i][i] = { re: cos, im: 0 };
  result[j][j] = { re: cos, im: 0 };
  result[i][j] = { re: -sin, im: 0 };
  result[j][i] = { re: sin, im: 0 };
  return result;
};

const hashPresetLabel = (label: string): number => {
  let h = 2166136261;
  for (let i = 0; i < label.length; i++) {
    h ^= label.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
};

const resolvePresetMatrix = (preset: string, seed: number, theta: number): Complex7x7 => {
  if (preset === 'identity') {
    return createIdentity();
  }
  const raising = RAISING_PRESETS[preset];
  if (raising) {
    return buildRaisingPulse(raising, theta);
  }
  const resolvedSeed = preset === 'random' ? seed : seed ^ hashPresetLabel(preset);
  const random = buildRandomComplexMatrix(resolvedSeed);
  return enforceSpecialUnitary(random);
};

const composeSchedule = (
  base: Complex7x7,
  schedule: readonly UnitaryScheduleStage[],
  seed: number,
  fallbackTheta: number,
): Complex7x7 => {
  if (!schedule.length) {
    return base;
  }
  let current = base;
  schedule.forEach((stage, index) => {
    const theta = Number.isFinite(stage.theta) ? (stage.theta as number) : fallbackTheta;
    const stageMatrix = resolvePresetMatrix(stage.preset, seed ^ ((index + 1) * 0x9e3779b9), theta);
    current = multiplyMatrices(stageMatrix, current);
  });
  return enforceSpecialUnitary(current);
};

export const applySU7 = (unitary: Complex7x7, vector: C7Vector): C7Vector => {
  const result: Complex[] = new Array(DIM);
  for (let row = 0; row < DIM; row++) {
    let sum: Complex = { re: 0, im: 0 };
    for (let col = 0; col < DIM; col++) {
      sum = complexAdd(sum, complexMul(unitary[row][col], vector[col]));
    }
    result[row] = sum;
  }
  return result as C7Vector;
};

export const buildUnitary = (options: BuildUnitaryOptions = {}): Complex7x7 => {
  const seed = Number.isFinite(options.seed) ? (options.seed as number) : DEFAULT_SEED;
  const preset = options.preset ?? DEFAULT_PRESET;
  const fallbackTheta =
    Number.isFinite(options.theta) && options.theta != null
      ? (options.theta as number)
      : DEFAULT_THETA;
  const base = resolvePresetMatrix(preset, seed, fallbackTheta);
  const schedule = options.schedule ?? [];
  if (!schedule.length) {
    return enforceSpecialUnitary(base);
  }
  return composeSchedule(base, schedule, seed, fallbackTheta);
};

export const projectToSU7 = (matrix: Complex7x7): Complex7x7 => enforceSpecialUnitary(matrix);
