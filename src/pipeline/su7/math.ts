import {
  DEFAULT_SU7_TELEMETRY,
  cloneComplex7x7,
  createDefaultSu7RuntimeParams,
  type Complex,
  type Complex7x7,
  type Su7ProjectorDescriptor,
  type Su7RuntimeParams,
  type Su7Telemetry,
} from './types.js';

export type Su7ScheduleFlowContext = {
  angle: number;
  magnitude: number;
  coherence: number;
  axisBias: Float32Array;
  gridSize: number;
  gridVectors: Float32Array;
};

export type Su7ScheduleContext = {
  dmt: number;
  arousal: number;
  flow?: Su7ScheduleFlowContext | null;
  curvatureStrength?: number;
  parallaxRadial?: number;
  volumeCoverage?: number;
};

const TAU = Math.PI * 2;
const EPSILON = 1e-9;

const createIdentity7 = (): Complex7x7 => {
  const rows: Complex[][] = [];
  for (let i = 0; i < 7; i++) {
    const row: Complex[] = [];
    for (let j = 0; j < 7; j++) {
      row.push({ re: i === j ? 1 : 0, im: 0 });
    }
    rows.push(row);
  }
  return rows as Complex7x7;
};

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

const complexAbs = (a: Complex): number => Math.sqrt(complexAbs2(a));

const complexDiv = (a: Complex, b: Complex): Complex => {
  const denom = complexAbs2(b);
  if (denom <= EPSILON) {
    return { re: 0, im: 0 };
  }
  const conj = complexConj(b);
  const num = complexMul(a, conj);
  return { re: num.re / denom, im: num.im / denom };
};

const conjugateTranspose = (matrix: Complex7x7): Complex7x7 => {
  const result = createIdentity7();
  for (let i = 0; i < 7; i++) {
    for (let j = 0; j < 7; j++) {
      result[i][j] = complexConj(matrix[j][i]);
    }
  }
  return result;
};

const matrixMultiply = (a: Complex7x7, b: Complex7x7): Complex7x7 => {
  const result = createIdentity7();
  for (let i = 0; i < 7; i++) {
    for (let j = 0; j < 7; j++) {
      let sum: Complex = { re: 0, im: 0 };
      for (let k = 0; k < 7; k++) {
        sum = complexAdd(sum, complexMul(a[i][k], b[k][j]));
      }
      result[i][j] = sum;
    }
  }
  return result;
};

const clamp01 = (value: number): number =>
  Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;

const wrapAngle = (theta: number): number => {
  if (!Number.isFinite(theta)) return 0;
  let t = theta;
  while (t <= -Math.PI) t += TAU;
  while (t > Math.PI) t -= TAU;
  return t;
};

const orthonormalize = (matrix: Complex7x7): Complex7x7 => {
  const result = createIdentity7();
  const columns: Complex[][] = [];
  for (let col = 0; col < 7; col++) {
    const v: Complex[] = new Array(7).fill(null).map((_, idx) => ({ ...matrix[idx][col] }));
    for (let prev = 0; prev < columns.length; prev++) {
      const basis = columns[prev];
      let proj: Complex = { re: 0, im: 0 };
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

const projectToSpecialUnitary = (matrix: Complex7x7): Complex7x7 => {
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

const createTwoPlanePulse = (
  axisA: number,
  axisB: number,
  theta: number,
  phase: number,
): Complex7x7 => {
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

const matrixMinusIdentity = (matrix: Complex7x7): Complex7x7 => {
  const result = createIdentity7();
  for (let i = 0; i < 7; i++) {
    for (let j = 0; j < 7; j++) {
      const identity = i === j ? { re: 1, im: 0 } : { re: 0, im: 0 };
      result[i][j] = complexSub(matrix[i][j], identity);
    }
  }
  return result;
};

const frobeniusNorm = (matrix: Complex7x7): number => {
  let sum = 0;
  for (let i = 0; i < 7; i++) {
    for (let j = 0; j < 7; j++) {
      sum += complexAbs2(matrix[i][j]);
    }
  }
  return Math.sqrt(sum);
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

const hashLabel = (label: string): number => {
  let h = 2166136261;
  for (let i = 0; i < label.length; i++) {
    h ^= label.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
};

const computeDeterminant = (matrix: Complex7x7): Complex => {
  const working = cloneComplex7x7(matrix);
  let sign = 1;
  let det: Complex = { re: 1, im: 0 };

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

const columnNormDeltas = (matrix: Complex7x7): { max: number; mean: number } => {
  const deltas: number[] = [];
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

const buildPhaseProjector = (seed: number): Complex7x7 => {
  const rng = mulberry32(seed);
  const projector = createIdentity7();
  for (let i = 0; i < 7; i++) {
    const theta = (rng() - 0.5) * Math.PI;
    projector[i][i] = { re: Math.cos(theta), im: Math.sin(theta) };
  }
  return projector;
};

const resolveProjectorMatrix = (descriptor: Su7ProjectorDescriptor): Complex7x7 => {
  if (descriptor.matrix) {
    return cloneComplex7x7(descriptor.matrix);
  }
  if (descriptor.id === 'identity') {
    return createIdentity7();
  }
  const seed = hashLabel(descriptor.id);
  return buildPhaseProjector(seed);
};

const computeProjectorEnergyInternal = (projector: Complex7x7, weight: number): number => {
  let energy = 0;
  for (let i = 0; i < 7; i++) {
    for (let j = 0; j < 7; j++) {
      energy += complexAbs2(projector[i][j]);
    }
  }
  return energy * weight * weight;
};

export const computeUnitaryError = (matrix: Complex7x7): number => {
  const lhs = conjugateTranspose(matrix);
  const gram = matrixMultiply(lhs, matrix);
  const delta = matrixMinusIdentity(gram);
  return frobeniusNorm(delta);
};

export const computeDeterminantDrift = (matrix: Complex7x7): number => {
  const det = computeDeterminant(matrix);
  const magnitude = complexAbs(det);
  return Math.abs(1 - magnitude);
};

export const computeNormDeltas = (matrix: Complex7x7): { max: number; mean: number } =>
  columnNormDeltas(matrix);

export const computeProjectorEnergy = (descriptor: Su7ProjectorDescriptor): number => {
  const projector = resolveProjectorMatrix(descriptor);
  const weight =
    typeof descriptor.weight === 'number' && Number.isFinite(descriptor.weight)
      ? descriptor.weight
      : 1;
  return computeProjectorEnergyInternal(projector, weight);
};

const buildScheduledUnitaryLegacy = (params: Su7RuntimeParams): Complex7x7 => {
  const base = createIdentity7();
  const phases = new Float64Array(7);
  const rng = mulberry32((params.seed ?? 0) >>> 0);
  const fallback = createDefaultSu7RuntimeParams().schedule;
  const schedule = params.schedule.length ? params.schedule : fallback;

  schedule.forEach((stage) => {
    const spread = Number.isFinite(stage.spread ?? NaN) ? (stage.spread as number) : 1;
    const gain = Number.isFinite(stage.gain) ? stage.gain : 0;
    if (!Number.isFinite(gain) || gain === 0) return;
    if (typeof stage.index === 'number' && Number.isFinite(stage.index)) {
      const target = ((Math.trunc(stage.index) % 7) + 7) % 7;
      phases[target] += gain * spread;
    } else {
      for (let column = 0; column < 7; column++) {
        const jitter = (rng() - 0.5) * spread * 0.05;
        phases[column] += gain * (1 + (column - 3) * 0.02) + jitter;
      }
    }
  });

  const baseGain = Number.isFinite(params.gain) ? params.gain : 1;
  for (let i = 0; i < 7; i++) {
    phases[i] += baseGain;
  }

  const meanPhase = phases.reduce((acc, value) => acc + value, 0) / phases.length;
  for (let i = 0; i < 7; i++) {
    const theta = wrapAngle(phases[i] - meanPhase);
    base[i][i] = { re: Math.cos(theta), im: Math.sin(theta) };
  }

  return base;
};

const buildScheduledUnitaryAdvanced = (
  params: Su7RuntimeParams,
  context: Su7ScheduleContext,
): Complex7x7 => {
  const axisAngles = new Float64Array(7);
  for (let i = 0; i < 7; i++) {
    axisAngles[i] = (i / 7) * TAU;
  }

  const fallback = createDefaultSu7RuntimeParams().schedule;
  const schedule = params.schedule.length ? params.schedule : fallback;

  const baseGain = Number.isFinite(params.gain) ? params.gain : 1;
  const dmt = clamp01(context.dmt ?? 0);
  const arousal = clamp01(context.arousal ?? 0);
  const modulationBase = 1 + 0.6 * dmt + 0.45 * arousal;

  const flow = context.flow ?? null;
  const axisBias = new Float64Array(7);
  for (let i = 0; i < 7; i++) {
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
    if (flow.axisBias.length === 7) {
      let maxBias = 0;
      for (let i = 0; i < 7; i++) {
        const value = Math.max(flow.axisBias[i], 0);
        axisBias[i] = value;
        if (value > maxBias) {
          maxBias = value;
        }
      }
      const scale = maxBias > EPSILON ? maxBias : 1;
      for (let i = 0; i < 7; i++) {
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

  const phases = new Float64Array(7);
  const pulses = new Float64Array(7);

  const rng = mulberry32((params.seed ?? 0) >>> 0);
  const flowIndex = ((Math.round(((((flowAngle % TAU) + TAU) % TAU) / TAU) * 7) % 7) + 7) % 7;

  schedule.forEach((stage) => {
    const gain = Number.isFinite(stage.gain) ? stage.gain : 0;
    if (!Number.isFinite(gain) || gain === 0) return;
    const spreadRaw =
      Number.isFinite(stage.spread ?? NaN) && stage.spread != null
        ? Math.max(0.05, stage.spread as number)
        : 1;
    const centerIndex =
      typeof stage.index === 'number' && Number.isFinite(stage.index)
        ? ((Math.trunc(stage.index) % 7) + 7) % 7
        : flowIndex;
    const centerAngle = axisAngles[centerIndex];
    const spreadAngle = spreadRaw * (TAU / 7);
    const stageMod = gain * modulationBase * curvatureBoost * (0.7 + 0.3 * flowCoherence);

    for (let axis = 0; axis < 7; axis++) {
      const delta = wrapAngle(axisAngles[axis] - centerAngle);
      const gaussian = Math.exp(-0.5 * Math.pow(delta / (spreadAngle + 1e-6), 2));
      const weight = gaussian * (0.6 + 0.4 * axisBias[axis]);
      phases[axis] += stageMod * weight;
      const neighbor = (axis + 1) % 7;
      const neighborDelta = wrapAngle(axisAngles[neighbor] - centerAngle);
      const neighborWeight =
        Math.exp(-0.5 * Math.pow(neighborDelta / (spreadAngle + 1e-6), 2)) *
        (0.6 + 0.4 * axisBias[neighbor]);
      const pairWeight = 0.5 * (weight + neighborWeight);
      const chiralityBias = stage.label && stage.label.toLowerCase().includes('vortex') ? 1.2 : 1;
      pulses[axis] += pairWeight * gain * chiralityBias * (0.35 + 0.45 * flowCoherence);
    }

    if (!Number.isFinite(stage.index)) {
      for (let axis = 0; axis < 7; axis++) {
        const jitter = (rng() - 0.5) * spreadRaw * 0.03;
        phases[axis] += gain * 0.08 * modulationBase + jitter;
      }
    }
  });

  for (let i = 0; i < 7; i++) {
    phases[i] += baseGain;
  }

  const meanPhase = phases.reduce((acc, value) => acc + value, 0) / phases.length;
  const diag = createIdentity7();
  for (let i = 0; i < 7; i++) {
    const theta = wrapAngle(phases[i] - meanPhase);
    diag[i][i] = { re: Math.cos(theta), im: Math.sin(theta) };
  }

  let current = diag;
  const pulseScale =
    (0.35 + 0.4 * flowCoherence + 0.25 * gridEnergy) * modulationBase * parallaxBoost * volumeBoost;
  const chiralityPhase = (dmt - 0.5) * 0.9 + (arousal - 0.5) * 0.5 + flowMagnitude * 0.25;

  for (let axis = 0; axis < 7; axis++) {
    const neighbor = (axis + 1) % 7;
    let theta = pulses[axis] * pulseScale;
    if (!Number.isFinite(theta)) {
      theta = 0;
    }
    theta = Math.tanh(theta);
    if (Math.abs(theta) <= 1e-6) continue;
    const rotation = createTwoPlanePulse(axis, neighbor, theta, chiralityPhase);
    current = matrixMultiply(rotation, current);
  }

  return projectToSpecialUnitary(current);
};

export const buildScheduledUnitary = (
  params: Su7RuntimeParams,
  context?: Su7ScheduleContext,
): Complex7x7 => {
  if (!params.enabled) {
    return createIdentity7();
  }
  if (!context) {
    return buildScheduledUnitaryLegacy(params);
  }
  return buildScheduledUnitaryAdvanced(params, context);
};

export const computeSu7Telemetry = (
  params: Su7RuntimeParams,
  context?: Su7ScheduleContext,
): Su7Telemetry => {
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
  };
};
