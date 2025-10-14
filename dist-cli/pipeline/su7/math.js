import {
  DEFAULT_SU7_TELEMETRY,
  cloneComplex7x7,
  cloneSu7ProjectorDescriptor,
  cloneSu7Schedule,
  createDefaultSu7RuntimeParams,
} from './types.js';
import { su3_embed as embedSu3Block } from '../../qcd/su3.js';
const TAU = Math.PI * 2;
const EPSILON = 1e-9;
const createIdentity7 = () => {
  const rows = [];
  for (let i = 0; i < 7; i++) {
    const row = [];
    for (let j = 0; j < 7; j++) {
      row.push({ re: i === j ? 1 : 0, im: 0 });
    }
    rows.push(row);
  }
  return rows;
};
const createZero7 = () => {
  const rows = [];
  for (let i = 0; i < 7; i++) {
    const row = [];
    for (let j = 0; j < 7; j++) {
      row.push({ re: 0, im: 0 });
    }
    rows.push(row);
  }
  return rows;
};
const complexAdd = (a, b) => ({
  re: a.re + b.re,
  im: a.im + b.im,
});
const complexSub = (a, b) => ({
  re: a.re - b.re,
  im: a.im - b.im,
});
const complexMul = (a, b) => ({
  re: a.re * b.re - a.im * b.im,
  im: a.re * b.im + a.im * b.re,
});
const complexConj = (a) => ({ re: a.re, im: -a.im });
const complexAbs2 = (a) => a.re * a.re + a.im * a.im;
const complexAbs = (a) => Math.sqrt(complexAbs2(a));
const complexDiv = (a, b) => {
  const denom = complexAbs2(b);
  if (denom <= EPSILON) {
    return { re: 0, im: 0 };
  }
  const conj = complexConj(b);
  const num = complexMul(a, conj);
  return { re: num.re / denom, im: num.im / denom };
};
const complexScale = (value, scalar) => ({
  re: value.re * scalar,
  im: value.im * scalar,
});
const complexFromPolar = (magnitude, angle) => ({
  re: magnitude * Math.cos(angle),
  im: magnitude * Math.sin(angle),
});
const conjugateTranspose = (matrix) => {
  const result = createIdentity7();
  for (let i = 0; i < 7; i++) {
    for (let j = 0; j < 7; j++) {
      result[i][j] = complexConj(matrix[j][i]);
    }
  }
  return result;
};
const matrixMultiply = (a, b) => {
  const result = createIdentity7();
  for (let i = 0; i < 7; i++) {
    for (let j = 0; j < 7; j++) {
      let sum = { re: 0, im: 0 };
      for (let k = 0; k < 7; k++) {
        sum = complexAdd(sum, complexMul(a[i][k], b[k][j]));
      }
      result[i][j] = sum;
    }
  }
  return result;
};
const addMatrices = (a, b) => {
  const result = createZero7();
  for (let row = 0; row < 7; row++) {
    for (let col = 0; col < 7; col++) {
      const lhs = a[row][col];
      const rhs = b[row][col];
      result[row][col] = { re: lhs.re + rhs.re, im: lhs.im + rhs.im };
    }
  }
  return result;
};
const subtractMatrices = (a, b) => {
  const result = createZero7();
  for (let row = 0; row < 7; row++) {
    for (let col = 0; col < 7; col++) {
      const lhs = a[row][col];
      const rhs = b[row][col];
      result[row][col] = { re: lhs.re - rhs.re, im: lhs.im - rhs.im };
    }
  }
  return result;
};
const scaleMatrix = (matrix, scalar) => {
  const result = createZero7();
  for (let row = 0; row < 7; row++) {
    for (let col = 0; col < 7; col++) {
      const entry = matrix[row][col];
      result[row][col] = { re: entry.re * scalar, im: entry.im * scalar };
    }
  }
  return result;
};
const addScaledIdentity = (matrix, scalar) => {
  const result = cloneComplex7x7(matrix);
  for (let i = 0; i < 7; i++) {
    result[i][i] = complexAdd(result[i][i], { re: scalar, im: 0 });
  }
  return result;
};
const linearCombination = (terms) => {
  const result = createZero7();
  for (const { matrix, weight } of terms) {
    if (!Number.isFinite(weight) || weight === 0) continue;
    for (let row = 0; row < 7; row++) {
      for (let col = 0; col < 7; col++) {
        const entry = matrix[row][col];
        result[row][col].re += entry.re * weight;
        result[row][col].im += entry.im * weight;
      }
    }
  }
  return result;
};
const matrixOneNorm = (matrix) => {
  let max = 0;
  for (let col = 0; col < 7; col++) {
    let sum = 0;
    for (let row = 0; row < 7; row++) {
      const entry = matrix[row][col];
      sum += Math.hypot(entry.re, entry.im);
    }
    if (sum > max) {
      max = sum;
    }
  }
  return max;
};
const makeSkewHermitian = (matrix) => {
  const adjoint = conjugateTranspose(matrix);
  const result = createZero7();
  for (let row = 0; row < 7; row++) {
    for (let col = 0; col < 7; col++) {
      const diff = complexSub(matrix[row][col], adjoint[row][col]);
      result[row][col] = complexScale(diff, 0.5);
    }
  }
  return result;
};
const computeTrace = (matrix) => {
  let trace = { re: 0, im: 0 };
  for (let i = 0; i < 7; i++) {
    trace = complexAdd(trace, matrix[i][i]);
  }
  return trace;
};
const clamp01 = (value) => (Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0);
const wrapAngle = (theta) => {
  if (!Number.isFinite(theta)) return 0;
  let t = theta;
  while (t <= -Math.PI) t += TAU;
  while (t > Math.PI) t -= TAU;
  return t;
};
const VECTOR_DIM = 7;
const safeNumber = (value) => (Number.isFinite(value) ? value : 0);
const createPhaseAccumulator = () => new Array(VECTOR_DIM).fill(0);
const toGatePhaseVector = (values) => {
  const result = [];
  for (let i = 0; i < VECTOR_DIM; i++) {
    const raw = values[i];
    result.push(safeNumber(raw ?? 0));
  }
  return result;
};
const finalizePhaseAngles = (phases) => {
  const total = phases.reduce((acc, value) => acc + value, 0);
  const mean = phases.length > 0 ? total / phases.length : 0;
  const wrapped = phases.map((value) => wrapAngle(value - mean));
  return toGatePhaseVector(wrapped);
};
const finalizePulseAngles = (pulses) => toGatePhaseVector(pulses.map((value) => safeNumber(value)));
const createPhaseGate = (phases, label) => ({
  kind: 'phase',
  label,
  phases: toGatePhaseVector(phases),
});
const createPulseGate = (axis, theta, phase, label) => ({
  kind: 'pulse',
  axis,
  theta: safeNumber(theta),
  phase: safeNumber(phase),
  label,
});
const orthonormalize = (matrix) => {
  const result = createIdentity7();
  const columns = [];
  for (let col = 0; col < 7; col++) {
    const v = new Array(7).fill(null).map((_, idx) => ({ ...matrix[idx][col] }));
    for (let prev = 0; prev < columns.length; prev++) {
      const basis = columns[prev];
      let proj = { re: 0, im: 0 };
      for (let row = 0; row < 7; row++) {
        const conj = complexConj(basis[row]);
        proj = complexAdd(proj, complexMul(conj, v[row]));
      }
      for (let row = 0; row < 7; row++) {
        const scaled = complexMul(basis[row], proj);
        v[row] = complexSub(v[row], scaled);
      }
    }
    let normSq = 0;
    for (let row = 0; row < 7; row++) {
      normSq += complexAbs2(v[row]);
    }
    const norm = normSq > EPSILON ? Math.sqrt(normSq) : 1;
    for (let row = 0; row < 7; row++) {
      result[row][col] = { re: v[row].re / norm, im: v[row].im / norm };
    }
    columns.push(result.map((row) => ({ ...row[col] })));
  }
  return result;
};
export const projectToSpecialUnitary = (matrix) => {
  const orthonormal = orthonormalize(matrix);
  const det = computeDeterminant(orthonormal);
  const magnitude = complexAbs(det);
  if (magnitude <= EPSILON) {
    return orthonormal;
  }
  const correction = {
    re: det.re / magnitude,
    im: -det.im / magnitude,
  };
  for (let row = 0; row < 7; row++) {
    orthonormal[row][6] = complexMul(orthonormal[row][6], correction);
  }
  return orthonormal;
};
const createTwoPlanePulse = (axisA, axisB, theta, phase) => {
  const result = createIdentity7();
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const phaseCos = Math.cos(phase);
  const phaseSin = Math.sin(phase);
  result[axisA][axisA] = { re: cos, im: 0 };
  result[axisB][axisB] = { re: cos, im: 0 };
  result[axisA][axisB] = {
    re: -sin * phaseCos,
    im: -sin * phaseSin,
  };
  result[axisB][axisA] = {
    re: sin * phaseCos,
    im: -sin * phaseSin,
  };
  return result;
};
const matrixMinusIdentity = (matrix) => {
  const result = createIdentity7();
  for (let i = 0; i < 7; i++) {
    for (let j = 0; j < 7; j++) {
      const identity = i === j ? { re: 1, im: 0 } : { re: 0, im: 0 };
      result[i][j] = complexSub(matrix[i][j], identity);
    }
  }
  return result;
};
const frobeniusNorm = (matrix) => {
  let sum = 0;
  for (let i = 0; i < 7; i++) {
    for (let j = 0; j < 7; j++) {
      sum += complexAbs2(matrix[i][j]);
    }
  }
  return Math.sqrt(sum);
};
const mulberry32 = (seed) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let c = Math.imul(t ^ (t >>> 15), 1 | t);
    c ^= c + Math.imul(c ^ (c >>> 7), 61 | c);
    return ((c ^ (c >>> 14)) >>> 0) / 4294967296;
  };
};
const hashLabel = (label) => {
  let h = 2166136261;
  for (let i = 0; i < label.length; i++) {
    h ^= label.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
};
export const computeDeterminant = (matrix) => {
  const working = cloneComplex7x7(matrix);
  let sign = 1;
  let det = { re: 1, im: 0 };
  for (let i = 0; i < 7; i++) {
    let pivotRow = i;
    let pivotMag = complexAbs(working[i][i]);
    for (let r = i + 1; r < 7; r++) {
      const mag = complexAbs(working[r][i]);
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
    for (let r = i + 1; r < 7; r++) {
      const factor = complexDiv(working[r][i], pivot);
      if (complexAbs(factor) <= EPSILON) continue;
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
const solveLinear = (lhs, rhs) => {
  const a = cloneComplex7x7(lhs);
  const result = cloneComplex7x7(rhs);
  for (let pivotIndex = 0; pivotIndex < 7; pivotIndex++) {
    let pivotRow = pivotIndex;
    let pivotMag = complexAbs2(a[pivotIndex][pivotIndex]);
    for (let candidate = pivotIndex + 1; candidate < 7; candidate++) {
      const magnitude = complexAbs2(a[candidate][pivotIndex]);
      if (magnitude > pivotMag) {
        pivotMag = magnitude;
        pivotRow = candidate;
      }
    }
    if (pivotMag <= EPSILON) {
      a[pivotIndex][pivotIndex] = { re: 1, im: 0 };
      pivotMag = 1;
    }
    if (pivotRow !== pivotIndex) {
      const tmpRow = a[pivotIndex];
      a[pivotIndex] = a[pivotRow];
      a[pivotRow] = tmpRow;
      const tmpRes = result[pivotIndex];
      result[pivotIndex] = result[pivotRow];
      result[pivotRow] = tmpRes;
    }
    const pivot = a[pivotIndex][pivotIndex];
    for (let row = pivotIndex + 1; row < 7; row++) {
      const entry = a[row][pivotIndex];
      if (complexAbs2(entry) <= EPSILON) {
        a[row][pivotIndex] = { re: 0, im: 0 };
        continue;
      }
      const factor = complexDiv(entry, pivot);
      for (let col = pivotIndex; col < 7; col++) {
        a[row][col] = complexSub(a[row][col], complexMul(factor, a[pivotIndex][col]));
      }
      for (let col = 0; col < 7; col++) {
        result[row][col] = complexSub(
          result[row][col],
          complexMul(factor, result[pivotIndex][col]),
        );
      }
      a[row][pivotIndex] = { re: 0, im: 0 };
    }
  }
  for (let row = 6; row >= 0; row--) {
    const pivot = a[row][row];
    for (let col = 0; col < 7; col++) {
      let value = result[row][col];
      for (let k = row + 1; k < 7; k++) {
        value = complexSub(value, complexMul(a[row][k], result[k][col]));
      }
      result[row][col] = complexDiv(value, pivot);
    }
  }
  return result;
};
const columnNormDeltas = (matrix) => {
  const deltas = [];
  for (let col = 0; col < 7; col++) {
    let sum = 0;
    for (let row = 0; row < 7; row++) {
      sum += complexAbs2(matrix[row][col]);
    }
    const norm = Math.sqrt(sum);
    deltas.push(Math.abs(norm - 1));
  }
  const max = deltas.reduce((acc, value) => (value > acc ? value : acc), 0);
  const mean = deltas.reduce((acc, value) => acc + value, 0) / deltas.length;
  return { max, mean };
};
const buildPhaseProjector = (seed) => {
  const rng = mulberry32(seed);
  const projector = createIdentity7();
  for (let i = 0; i < 7; i++) {
    const theta = (rng() - 0.5) * Math.PI;
    projector[i][i] = { re: Math.cos(theta), im: Math.sin(theta) };
  }
  return projector;
};
const resolveProjectorMatrix = (descriptor) => {
  if (descriptor.matrix) {
    return cloneComplex7x7(descriptor.matrix);
  }
  if (descriptor.id === 'identity') {
    return createIdentity7();
  }
  const seed = hashLabel(descriptor.id);
  return buildPhaseProjector(seed);
};
const computeProjectorEnergyInternal = (projector, weight) => {
  let energy = 0;
  for (let i = 0; i < 7; i++) {
    for (let j = 0; j < 7; j++) {
      energy += complexAbs2(projector[i][j]);
    }
  }
  return energy * weight * weight;
};
export const computeUnitaryError = (matrix) => {
  const lhs = conjugateTranspose(matrix);
  const gram = matrixMultiply(lhs, matrix);
  const delta = matrixMinusIdentity(gram);
  return frobeniusNorm(delta);
};
export const computeDeterminantDrift = (matrix) => {
  const det = computeDeterminant(matrix);
  const magnitude = complexAbs(det);
  return Math.abs(1 - magnitude);
};
export const computeNormDeltas = (matrix) => columnNormDeltas(matrix);
export const computeProjectorEnergy = (descriptor) => {
  const projector = resolveProjectorMatrix(descriptor);
  const weight =
    typeof descriptor.weight === 'number' && Number.isFinite(descriptor.weight)
      ? descriptor.weight
      : 1;
  return computeProjectorEnergyInternal(projector, weight);
};
export const su2_embed = (axisA, axisB, theta, phi) => {
  const i = Math.trunc(Number.isFinite(axisA) ? axisA : 0);
  const j = Math.trunc(Number.isFinite(axisB) ? axisB : 0);
  if (i === j || i < 0 || j < 0 || i >= 7 || j >= 7) {
    return createIdentity7();
  }
  const angle = Number.isFinite(theta) ? theta : 0;
  const phaseAngle = Number.isFinite(phi) ? phi : 0;
  const result = createIdentity7();
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const phase = complexFromPolar(1, phaseAngle);
  result[i][i] = { re: cos, im: 0 };
  result[j][j] = { re: cos, im: 0 };
  result[i][j] = { re: -sin * phase.re, im: -sin * phase.im };
  result[j][i] = { re: sin * phase.re, im: -sin * phase.im };
  return result;
};
export const phase_gate = (phases) => {
  const diagonal = createIdentity7();
  const buffer = new Float64Array(7);
  let sum = 0;
  for (let i = 0; i < 7; i++) {
    const value = i < phases.length ? phases[i] : 0;
    const finite = Number.isFinite(value) ? value : 0;
    buffer[i] = finite;
    sum += finite;
  }
  const mean = sum / 7;
  for (let i = 0; i < 7; i++) {
    const theta = wrapAngle(buffer[i] - mean);
    diagonal[i][i] = { re: Math.cos(theta), im: Math.sin(theta) };
  }
  return diagonal;
};
export const compose_dense = (lhs, rhs) => {
  const result = createZero7();
  const rhsRe = new Float64Array(49);
  const rhsIm = new Float64Array(49);
  for (let col = 0; col < 7; col++) {
    for (let row = 0; row < 7; row++) {
      const index = col * 7 + row;
      const entry = rhs[row][col];
      rhsRe[index] = entry.re;
      rhsIm[index] = entry.im;
    }
  }
  for (let row = 0; row < 7; row++) {
    const lhsRow = lhs[row];
    for (let col = 0; col < 7; col++) {
      let sumRe = 0;
      let sumIm = 0;
      for (let k = 0; k < 7; k++) {
        const a = lhsRow[k];
        const index = col * 7 + k;
        const br = rhsRe[index];
        const bi = rhsIm[index];
        sumRe += a.re * br - a.im * bi;
        sumIm += a.re * bi + a.im * br;
      }
      result[row][col] = { re: sumRe, im: sumIm };
    }
  }
  return result;
};
export const su3_embed = (block) => embedSu3Block(block);
const polarInverseSqrt = (matrix) => {
  const hermitian = matrix;
  let trace = 0;
  for (let i = 0; i < 7; i++) {
    trace += hermitian[i][i].re;
  }
  let scale = trace / 7;
  if (!Number.isFinite(scale) || scale <= 0) {
    scale = 1;
  }
  let Y = scaleMatrix(hermitian, 1 / scale);
  let Z = createIdentity7();
  let residual = Number.POSITIVE_INFINITY;
  for (let iter = 0; iter < 12; iter++) {
    const YZ = compose_dense(Z, Y);
    const correction = createZero7();
    for (let row = 0; row < 7; row++) {
      for (let col = 0; col < 7; col++) {
        const entry = YZ[row][col];
        correction[row][col] = { re: -entry.re, im: -entry.im };
      }
      correction[row][row].re += 3;
    }
    const halfCorrection = scaleMatrix(correction, 0.5);
    const nextY = compose_dense(Y, halfCorrection);
    const nextZ = compose_dense(halfCorrection, Z);
    const gram = compose_dense(conjugateTranspose(nextY), nextY);
    const error = frobeniusNorm(matrixMinusIdentity(gram));
    Y = nextY;
    Z = nextZ;
    if (error < 1e-10 || Math.abs(error - residual) < 1e-12) {
      break;
    }
    residual = error;
  }
  return scaleMatrix(Z, 1 / Math.sqrt(scale));
};
export const polar_reunitarize = (matrix) => {
  const base = cloneComplex7x7(matrix);
  const gram = compose_dense(conjugateTranspose(base), base);
  const invSqrt = polarInverseSqrt(gram);
  let unitary = compose_dense(base, invSqrt);
  const gramCheck = compose_dense(conjugateTranspose(unitary), unitary);
  const deviation = frobeniusNorm(matrixMinusIdentity(gramCheck));
  if (deviation > 1e-10) {
    unitary = projectToSpecialUnitary(unitary);
  }
  const det = computeDeterminant(unitary);
  const phase = -Math.atan2(det.im, det.re) / 7;
  const correction = complexFromPolar(1, phase);
  for (let row = 0; row < 7; row++) {
    unitary[row][6] = complexMul(unitary[row][6], correction);
  }
  return unitary;
};
const matrixSquareRootUnitary = (unitary) => {
  let Y = cloneComplex7x7(unitary);
  let Z = createIdentity7();
  const identity = createIdentity7();
  for (let iter = 0; iter < 12; iter++) {
    const Yinv = solveLinear(Y, identity);
    const Zinv = solveLinear(Z, identity);
    const nextY = scaleMatrix(addMatrices(Y, Zinv), 0.5);
    const nextZ = scaleMatrix(addMatrices(Z, Yinv), 0.5);
    const diffY = frobeniusNorm(subtractMatrices(nextY, Y));
    const diffZ = frobeniusNorm(subtractMatrices(nextZ, Z));
    Y = nextY;
    Z = nextZ;
    if (diffY < 1e-10 && diffZ < 1e-10) {
      break;
    }
  }
  return projectToSpecialUnitary(Y);
};
export const logm_su = (matrix) => {
  let unitary = polar_reunitarize(matrix);
  const identity = createIdentity7();
  let scaling = 0;
  while (scaling < 10) {
    const delta = subtractMatrices(unitary, identity);
    const norm = frobeniusNorm(delta);
    if (norm < 0.55) {
      break;
    }
    unitary = matrixSquareRootUnitary(unitary);
    scaling += 1;
  }
  const UplusI = addScaledIdentity(unitary, 1);
  const UminusI = subtractMatrices(unitary, identity);
  const inverse = solveLinear(UplusI, identity);
  let K = compose_dense(UminusI, inverse);
  K = makeSkewHermitian(K);
  const K2 = compose_dense(K, K);
  let term = cloneComplex7x7(K);
  let result = scaleMatrix(term, 2);
  const maxTerms = 18;
  for (let n = 1; n < maxTerms; n++) {
    term = compose_dense(term, K2);
    const coeff = 2 / (2 * n + 1);
    const contribution = scaleMatrix(term, coeff);
    result = addMatrices(result, contribution);
    const magnitude = frobeniusNorm(contribution);
    if (magnitude < 1e-10) {
      break;
    }
  }
  const scaleFactor = 1 << scaling;
  if (scaling > 0) {
    result = scaleMatrix(result, scaleFactor);
  }
  result = makeSkewHermitian(result);
  const trace = computeTrace(result);
  const mean = { re: trace.re / 7, im: trace.im / 7 };
  for (let i = 0; i < 7; i++) {
    result[i][i] = complexSub(result[i][i], mean);
  }
  return result;
};
export const expm_su = (skew) => {
  const input = makeSkewHermitian(skew);
  const trace = computeTrace(input);
  const mean = { re: trace.re / 7, im: trace.im / 7 };
  for (let i = 0; i < 7; i++) {
    input[i][i] = complexSub(input[i][i], mean);
  }
  const theta13 = 2.29;
  const norm = matrixOneNorm(input);
  let s = 0;
  if (norm > theta13) {
    s = Math.max(0, Math.ceil(Math.log2(norm / theta13)));
  }
  const scale = 1 / (1 << s);
  const scaled = scaleMatrix(input, scale);
  const X2 = compose_dense(scaled, scaled);
  const X4 = compose_dense(X2, X2);
  const X6 = compose_dense(X4, X2);
  const b = [
    64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000,
    10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1,
  ];
  const identity = createIdentity7();
  const innerU = linearCombination([
    { matrix: X6, weight: b[13] },
    { matrix: X4, weight: b[11] },
    { matrix: X2, weight: b[9] },
  ]);
  const X6InnerU = compose_dense(X6, innerU);
  const polyU = linearCombination([
    { matrix: X6InnerU, weight: 1 },
    { matrix: X6, weight: b[7] },
    { matrix: X4, weight: b[5] },
    { matrix: X2, weight: b[3] },
    { matrix: identity, weight: b[1] },
  ]);
  const U = compose_dense(scaled, polyU);
  const innerV = linearCombination([
    { matrix: X6, weight: b[12] },
    { matrix: X4, weight: b[10] },
    { matrix: X2, weight: b[8] },
  ]);
  const X6InnerV = compose_dense(X6, innerV);
  const V = linearCombination([
    { matrix: X6InnerV, weight: 1 },
    { matrix: X6, weight: b[6] },
    { matrix: X4, weight: b[4] },
    { matrix: X2, weight: b[2] },
    { matrix: identity, weight: b[0] },
  ]);
  const VminusU = subtractMatrices(V, U);
  const VplusU = addMatrices(V, U);
  let result = solveLinear(VminusU, VplusU);
  for (let i = 0; i < s; i++) {
    result = compose_dense(result, result);
  }
  return polar_reunitarize(result);
};
const normalizeLaneId = (value) => {
  if (!value) {
    return 'main';
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed.toLowerCase() : 'main';
};
const resolveScheduleStages = (schedule) => {
  const resolved = [];
  for (let i = 0; i < schedule.length; i++) {
    const stage = schedule[i];
    const gain = Number.isFinite(stage.gain) ? stage.gain : 0;
    const lane = normalizeLaneId(stage.lane);
    const macro = stage.macro === true || lane === 'macro';
    if (!macro && Math.abs(gain) <= 1e-12) {
      continue;
    }
    const spread = Number.isFinite(stage.spread) && stage.spread != null ? stage.spread : 1;
    const rawIndex =
      typeof stage.index === 'number' && Number.isFinite(stage.index) ? stage.index : null;
    const index =
      rawIndex != null ? ((Math.trunc(rawIndex) % VECTOR_DIM) + VECTOR_DIM) % VECTOR_DIM : null;
    const time = typeof stage.time === 'number' && Number.isFinite(stage.time) ? stage.time : i;
    const length =
      typeof stage.length === 'number' && Number.isFinite(stage.length) && stage.length > 0
        ? stage.length
        : 0;
    const phase =
      typeof stage.phase === 'number' && Number.isFinite(stage.phase) ? stage.phase : null;
    resolved.push({
      gain,
      spread,
      index,
      label: stage.label,
      lane,
      macro,
      time,
      length,
      phase,
      order: i,
    });
  }
  resolved.sort((a, b) => {
    if (a.time !== b.time) {
      return a.time < b.time ? -1 : 1;
    }
    if (a.lane !== b.lane) {
      return a.lane < b.lane ? -1 : 1;
    }
    return a.order - b.order;
  });
  return resolved;
};
const computeStageSeed = (baseSeed, stage) => {
  const laneHash = hashLabel(stage.lane);
  const labelHash = stage.label ? hashLabel(stage.label) : 0;
  const indexHash = stage.index != null ? (stage.index & 0xff) * 1315423911 : 0;
  const timeHash = Math.trunc(stage.time * 4096) | 0;
  const lengthHash = Math.trunc(stage.length * 4096) | 0;
  const phaseHash = stage.phase != null ? Math.trunc(stage.phase * 4096) | 0 : 0;
  const seed = baseSeed ^ laneHash ^ labelHash ^ indexHash ^ timeHash ^ lengthHash ^ phaseHash;
  return seed >>> 0;
};
const createStageRng = (baseSeed, stage) => mulberry32(computeStageSeed(baseSeed, stage));
const computeLegacyGateComputation = (params, schedule) => {
  const phases = createPhaseAccumulator();
  const pulses = createPhaseAccumulator();
  const gates = [];
  const baseSeed = (params.seed ?? 0) >>> 0;
  const resolvedStages = resolveScheduleStages(schedule);
  for (const stage of resolvedStages) {
    if (stage.macro) {
      const axis = stage.index ?? 0;
      const theta = safeNumber(stage.gain * stage.spread);
      const phase = safeNumber(stage.phase ?? 0);
      pulses[axis] += theta;
      gates.push(createPulseGate(axis, theta, phase, stage.label));
      continue;
    }
    const lengthFactor = stage.length > 0 ? 1 / (1 + stage.length) : 1;
    const gain = safeNumber(stage.gain * lengthFactor);
    if (!Number.isFinite(gain) || Math.abs(gain) <= 1e-12) {
      continue;
    }
    const contribution = createPhaseAccumulator();
    if (stage.index != null) {
      contribution[stage.index] += gain * stage.spread;
    } else {
      const rng = createStageRng(baseSeed, stage);
      for (let axis = 0; axis < VECTOR_DIM; axis++) {
        const jitter = (rng() - 0.5) * stage.spread * 0.05;
        contribution[axis] += gain * (1 + (axis - 3) * 0.02) + jitter;
      }
    }
    for (let axis = 0; axis < VECTOR_DIM; axis++) {
      phases[axis] += contribution[axis];
    }
    gates.push(createPhaseGate(contribution, stage.label));
  }
  const baseGain = Number.isFinite(params.gain) ? params.gain : 1;
  const baseContribution = new Array(VECTOR_DIM).fill(baseGain);
  for (let axis = 0; axis < VECTOR_DIM; axis++) {
    phases[axis] += baseContribution[axis];
  }
  gates.push(createPhaseGate(baseContribution, 'baseGain'));
  return {
    gates,
    phaseAngles: finalizePhaseAngles(phases),
    pulseAngles: finalizePulseAngles(pulses),
    baseGain,
    chiralityPhase: 0,
  };
};
const computeAdvancedGateComputation = (params, schedule, context) => {
  const axisAngles = new Float64Array(VECTOR_DIM);
  for (let i = 0; i < VECTOR_DIM; i++) {
    axisAngles[i] = (i / VECTOR_DIM) * TAU;
  }
  const phases = createPhaseAccumulator();
  const pulses = createPhaseAccumulator();
  const gates = [];
  const macroEntries = Array.from({ length: VECTOR_DIM }, () => []);
  const macroTotals = new Array(VECTOR_DIM).fill(0);
  const baseGain = Number.isFinite(params.gain) ? params.gain : 1;
  const dmt = clamp01(context.dmt ?? 0);
  const arousal = clamp01(context.arousal ?? 0);
  const modulationBase = 1 + 0.6 * dmt + 0.45 * arousal;
  const flow = context.flow ?? null;
  const axisBias = new Float64Array(VECTOR_DIM);
  for (let i = 0; i < VECTOR_DIM; i++) {
    axisBias[i] = 1;
  }
  let flowCoherence = 0;
  let flowMagnitude = 0;
  let flowAngle = 0;
  let gridEnergy = 0;
  if (flow) {
    flowCoherence = clamp01(flow.coherence);
    flowMagnitude = Math.tanh(flow.magnitude);
    flowAngle = flow.angle;
    if (flow.axisBias.length === VECTOR_DIM) {
      let maxBias = 0;
      for (let i = 0; i < VECTOR_DIM; i++) {
        const value = Math.max(flow.axisBias[i], 0);
        axisBias[i] = value;
        if (value > maxBias) {
          maxBias = value;
        }
      }
      const scale = maxBias > EPSILON ? maxBias : 1;
      for (let i = 0; i < VECTOR_DIM; i++) {
        axisBias[i] = 0.2 + 0.8 * clamp01(axisBias[i] / scale);
      }
    }
    if (flow.gridSize > 0 && flow.gridVectors.length >= flow.gridSize * flow.gridSize * 2) {
      const cells = flow.gridSize * flow.gridSize;
      for (let i = 0; i < cells; i++) {
        const vx = flow.gridVectors[i * 2];
        const vy = flow.gridVectors[i * 2 + 1];
        gridEnergy += Math.hypot(vx, vy);
      }
      gridEnergy = cells > 0 ? Math.tanh(gridEnergy / cells) : 0;
    }
  }
  const curvatureStrength = clamp01(Math.abs(context.curvatureStrength ?? 0));
  const parallaxMag = clamp01(Math.abs(context.parallaxRadial ?? 0));
  const volumeCoverage = clamp01(context.volumeCoverage ?? 0);
  const curvatureBoost = 1 + curvatureStrength * 0.5;
  const parallaxBoost = 1 + parallaxMag * 0.35;
  const volumeBoost = 1 + volumeCoverage * 0.4;
  const baseSeed = (params.seed ?? 0) >>> 0;
  const flowIndex =
    ((Math.round(((((flowAngle % TAU) + TAU) % TAU) / TAU) * VECTOR_DIM) % VECTOR_DIM) +
      VECTOR_DIM) %
    VECTOR_DIM;
  const resolvedStages = resolveScheduleStages(schedule);
  for (const stage of resolvedStages) {
    if (stage.macro) {
      const axis = stage.index ?? 0;
      const amount = safeNumber(stage.gain * stage.spread);
      const phase = safeNumber(stage.phase ?? 0);
      const gate = createPulseGate(axis, amount, phase, stage.label);
      if (gate.kind === 'pulse') {
        macroEntries[axis].push({ amount, gate });
      }
      macroTotals[axis] += amount;
      pulses[axis] += amount;
      gates.push(gate);
      continue;
    }
    const lengthFactor = stage.length > 0 ? 1 / (1 + stage.length) : 1;
    const stageGain = safeNumber(stage.gain * lengthFactor);
    if (!Number.isFinite(stageGain) || Math.abs(stageGain) <= 1e-12) {
      continue;
    }
    const spreadRaw = Math.max(0.05, Math.abs(stage.spread));
    const centerIndex = stage.index != null ? stage.index : flowIndex;
    const centerAngle = axisAngles[centerIndex];
    const spreadAngle = spreadRaw * (TAU / VECTOR_DIM);
    const stageMod = stageGain * modulationBase * curvatureBoost * (0.7 + 0.3 * flowCoherence);
    const contribution = createPhaseAccumulator();
    const rng = createStageRng(baseSeed, stage);
    for (let axis = 0; axis < VECTOR_DIM; axis++) {
      const delta = wrapAngle(axisAngles[axis] - centerAngle);
      const gaussian = Math.exp(-0.5 * Math.pow(delta / (spreadAngle + 1e-6), 2));
      const weight = gaussian * (0.6 + 0.4 * axisBias[axis]);
      const phaseDelta = stageMod * weight;
      contribution[axis] += phaseDelta;
      phases[axis] += phaseDelta;
      const neighbor = (axis + 1) % VECTOR_DIM;
      const neighborDelta = wrapAngle(axisAngles[neighbor] - centerAngle);
      const neighborWeight =
        Math.exp(-0.5 * Math.pow(neighborDelta / (spreadAngle + 1e-6), 2)) *
        (0.6 + 0.4 * axisBias[neighbor]);
      const pairWeight = 0.5 * (weight + neighborWeight);
      const chiralityBias = stage.label && stage.label.toLowerCase().includes('vortex') ? 1.2 : 1;
      pulses[axis] += pairWeight * stageGain * chiralityBias * (0.35 + 0.45 * flowCoherence);
    }
    if (stage.index == null) {
      for (let axis = 0; axis < VECTOR_DIM; axis++) {
        const jitter = (rng() - 0.5) * spreadRaw * 0.03;
        const addition = stageGain * 0.08 * modulationBase + jitter;
        contribution[axis] += addition;
        phases[axis] += addition;
      }
    }
    gates.push(createPhaseGate(contribution, stage.label));
  }
  const baseContribution = new Array(VECTOR_DIM).fill(baseGain);
  for (let axis = 0; axis < VECTOR_DIM; axis++) {
    phases[axis] += baseContribution[axis];
  }
  gates.push(createPhaseGate(baseContribution, 'baseGain'));
  const pulseScale =
    (0.35 + 0.4 * flowCoherence + 0.25 * gridEnergy) * modulationBase * parallaxBoost * volumeBoost;
  const chiralityPhase = (dmt - 0.5) * 0.9 + (arousal - 0.5) * 0.5 + flowMagnitude * 0.25;
  const rawPulseTotals = Array.from(pulses);
  for (let axis = 0; axis < VECTOR_DIM; axis++) {
    const rawTotal = rawPulseTotals[axis];
    let theta = safeNumber(rawTotal * pulseScale);
    theta = Math.tanh(theta);
    pulses[axis] = theta;
    const entries = macroEntries[axis];
    let remaining = theta;
    if (entries.length > 0) {
      const macroTotal = macroTotals[axis];
      for (const entry of entries) {
        const weight =
          Math.abs(rawTotal) > 1e-9
            ? entry.amount / rawTotal
            : Math.abs(macroTotal) > 1e-9
              ? entry.amount / macroTotal
              : 0;
        const macroTheta = safeNumber(theta * weight);
        entry.gate.theta = macroTheta;
        remaining -= macroTheta;
      }
    }
    if (Math.abs(remaining) > 1e-6) {
      gates.push(createPulseGate(axis, remaining, chiralityPhase));
    }
  }
  return {
    gates,
    phaseAngles: finalizePhaseAngles(phases),
    pulseAngles: finalizePulseAngles(pulses),
    baseGain,
    chiralityPhase,
  };
};
export const createSu7GateList = (params, context) => {
  const seed = Math.trunc(Number.isFinite(params.seed) ? params.seed : 0);
  const preset = typeof params.preset === 'string' ? params.preset : 'identity';
  const projector = cloneSu7ProjectorDescriptor(params.projector);
  const fallbackSchedule = createDefaultSu7RuntimeParams().schedule;
  const effectiveSchedule = params.schedule.length > 0 ? params.schedule : fallbackSchedule;
  const scheduleClone = cloneSu7Schedule(effectiveSchedule);
  if (!params.enabled) {
    const zero = toGatePhaseVector(new Array(VECTOR_DIM).fill(0));
    return {
      seed,
      preset,
      schedule: scheduleClone,
      projector,
      gains: {
        baseGain: Number.isFinite(params.gain) ? params.gain : 0,
        phaseAngles: zero,
        pulseAngles: zero,
        chiralityPhase: 0,
      },
      gates: [],
      squashedAppends: 0,
    };
  }
  const computation = context
    ? computeAdvancedGateComputation(params, scheduleClone, context)
    : computeLegacyGateComputation(params, scheduleClone);
  return {
    seed,
    preset,
    schedule: scheduleClone,
    projector,
    gains: {
      baseGain: computation.baseGain,
      phaseAngles: computation.phaseAngles,
      pulseAngles: computation.pulseAngles,
      chiralityPhase: computation.chiralityPhase,
    },
    gates: computation.gates,
    squashedAppends: 0,
  };
};
export const createSu7GateListSnapshot = (params, context) => {
  const list = createSu7GateList(params, context);
  const { gates: _gates, squashedAppends: _squashed, ...snapshot } = list;
  return snapshot;
};
export const buildScheduledUnitary = (params, context) => {
  if (!params.enabled) {
    return createIdentity7();
  }
  const gateList = createSu7GateList(params, context);
  const { gains } = gateList;
  const diag = createIdentity7();
  for (let i = 0; i < VECTOR_DIM; i++) {
    const theta = gains.phaseAngles[i];
    diag[i][i] = { re: Math.cos(theta), im: Math.sin(theta) };
  }
  let current = diag;
  for (let axis = 0; axis < VECTOR_DIM; axis++) {
    const theta = gains.pulseAngles[axis];
    if (Math.abs(theta) <= 1e-6) continue;
    const neighbor = (axis + 1) % VECTOR_DIM;
    const rotation = createTwoPlanePulse(axis, neighbor, theta, gains.chiralityPhase);
    current = matrixMultiply(rotation, current);
  }
  return projectToSpecialUnitary(current);
};
export const enforceUnitaryGuardrail = (matrix, options) => {
  const threshold = options?.threshold ?? 1e-6;
  const force = options?.force === true;
  const baseline = cloneComplex7x7(matrix);
  const rawError = computeUnitaryError(baseline);
  if (!force && rawError <= threshold) {
    return {
      unitary: baseline,
      unitaryError: rawError,
      determinantDrift: computeDeterminantDrift(baseline),
      event: null,
    };
  }
  const corrected = polar_reunitarize(baseline);
  const correctedError = computeUnitaryError(corrected);
  const drift = computeDeterminantDrift(corrected);
  const event = {
    kind: 'autoReorthon',
    before: rawError,
    after: correctedError,
    threshold,
    forced: force,
  };
  return {
    unitary: corrected,
    unitaryError: correctedError,
    determinantDrift: drift,
    event,
  };
};
export const detectFlickerGuardrail = (previousEnergy, currentEnergy, frameTimeMs, options) => {
  const curr = Number.isFinite(currentEnergy) ? Math.max(currentEnergy, 0) : 0;
  const prev =
    previousEnergy != null && Number.isFinite(previousEnergy) ? Math.max(previousEnergy, 0) : null;
  const minEnergy = options?.minEnergy ?? 0.05;
  if (curr < minEnergy && (prev == null || prev < minEnergy)) {
    return null;
  }
  if (prev == null) {
    return null;
  }
  const ms = Number.isFinite(frameTimeMs) ? Math.max(frameTimeMs, 0) : 0;
  if (ms <= 1e-6) {
    return null;
  }
  const ratioThreshold = options?.ratioThreshold ?? 0.12;
  const freqThreshold = options?.frequencyThreshold ?? 30;
  const denom = Math.max(prev, 1e-6);
  const delta = curr - prev;
  const deltaRatio = Math.abs(delta) / denom;
  if (deltaRatio <= ratioThreshold) {
    return null;
  }
  const frequencyHz = 1000 / ms;
  if (!Number.isFinite(frequencyHz) || frequencyHz < freqThreshold) {
    return null;
  }
  return {
    kind: 'flicker',
    frequencyHz,
    deltaRatio,
    energy: curr,
  };
};
export const computeSu7Telemetry = (params, context) => {
  if (!params.enabled) {
    return { ...DEFAULT_SU7_TELEMETRY };
  }
  const unitary = buildScheduledUnitary(params, context);
  const unitaryError = computeUnitaryError(unitary);
  const determinantDrift = computeDeterminantDrift(unitary);
  const { max, mean } = computeNormDeltas(unitary);
  const projectorEnergy = computeProjectorEnergy(params.projector);
  return {
    unitaryError,
    determinantDrift,
    normDeltaMax: max,
    normDeltaMean: mean,
    projectorEnergy,
    geodesicFallbacks: 0,
  };
};
