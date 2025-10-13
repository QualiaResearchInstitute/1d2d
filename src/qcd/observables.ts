import { GaugeLattice, type GaugeLinkAxis } from './lattice.js';
import { su3_mul, type Complex3x3 } from './su3.js';
import type { Complex } from '../pipeline/su7/types.js';

export type WilsonRectangle = {
  extentX: number;
  extentY: number;
  axes?: [GaugeLinkAxis, GaugeLinkAxis];
};

export type WilsonLoopMeasurement = {
  extentX: number;
  extentY: number;
  axes: [GaugeLinkAxis, GaugeLinkAxis];
  normalized: Complex;
  magnitude: number;
  value: number;
  sampleCount: number;
};

export type WilsonLoopTable = ReadonlyMap<string, WilsonLoopMeasurement>;

export type PolyakovLoopMeasurement = {
  axis: GaugeLinkAxis;
  extent: number;
  average: Complex;
  magnitude: number;
  sampleCount: number;
};

export type RunningEstimateSnapshot = {
  count: number;
  mean: number;
  variance: number;
  standardDeviation: number;
  standardError: number;
};

export type JackknifeResult = {
  estimate: number;
  jackknifeMean: number;
  bias: number;
  error: number;
  pseudoValues: number[];
};

type LatticeCoord = { x: number; y: number; z: number; t: number };

const MATRIX_DIM = 3;
const NORMALIZATION = 1 / MATRIX_DIM;
const LOOP_KEY_SEPARATOR = 'x';
const MIN_POSITIVE = 1e-12;

const createZeroMatrix = (): Complex3x3 =>
  new Array(MATRIX_DIM)
    .fill(null)
    .map(
      () =>
        new Array(MATRIX_DIM).fill(null).map(() => ({ re: 0, im: 0 }) as Complex) as [
          Complex,
          Complex,
          Complex,
        ],
    ) as Complex3x3;

const createIdentityMatrix = (): Complex3x3 => {
  const matrix = createZeroMatrix();
  for (let i = 0; i < MATRIX_DIM; i++) {
    matrix[i][i] = { re: 1, im: 0 };
  }
  return matrix;
};

const conjugateTranspose = (matrix: Complex3x3): Complex3x3 => {
  const result = createZeroMatrix();
  for (let row = 0; row < MATRIX_DIM; row++) {
    for (let col = 0; col < MATRIX_DIM; col++) {
      const entry = matrix[col][row];
      result[row][col] = { re: entry.re, im: -entry.im };
    }
  }
  return result;
};

const traceComplex = (matrix: Complex3x3): Complex => {
  let re = 0;
  let im = 0;
  for (let i = 0; i < MATRIX_DIM; i++) {
    const entry = matrix[i][i];
    re += entry.re;
    im += entry.im;
  }
  return { re, im };
};

const coerceExtent = (label: string, value: number): number => {
  if (!Number.isFinite(value)) {
    throw new TypeError(`Wilson loop ${label} must be finite (received ${value})`);
  }
  const coerced = Math.floor(value);
  if (coerced <= 0) {
    throw new RangeError(`Wilson loop ${label} must be positive (received ${value})`);
  }
  return coerced;
};

const axisExtent = (lattice: GaugeLattice, axis: GaugeLinkAxis): number => {
  switch (axis) {
    case 'x':
      return lattice.width;
    case 'y':
      return lattice.height;
    case 'z':
      return lattice.depth;
    case 't':
    default:
      return lattice.temporalExtent;
  }
};

const assertAxisActive = (lattice: GaugeLattice, axis: GaugeLinkAxis): void => {
  if (!lattice.axes.includes(axis)) {
    throw new RangeError(`Gauge lattice axis ${axis} is inactive`);
  }
};

const normalizeAxes = (
  lattice: GaugeLattice,
  axes?: [GaugeLinkAxis, GaugeLinkAxis],
): [GaugeLinkAxis, GaugeLinkAxis] => {
  if (!axes) {
    return ['x', lattice.axes.includes('y') ? 'y' : (lattice.axes[1] ?? 'x')];
  }
  const [a, b] = axes;
  if (a === b) {
    throw new RangeError('Wilson loop axes must be distinct');
  }
  assertAxisActive(lattice, a);
  assertAxisActive(lattice, b);
  return [a, b];
};

const forEachSite = (lattice: GaugeLattice, visit: (coord: LatticeCoord) => void) => {
  for (let t = 0; t < lattice.temporalExtent; t++) {
    for (let z = 0; z < lattice.depth; z++) {
      for (let y = 0; y < lattice.height; y++) {
        for (let x = 0; x < lattice.width; x++) {
          visit({ x, y, z, t });
        }
      }
    }
  }
};

const getLink = (lattice: GaugeLattice, coord: LatticeCoord, axis: GaugeLinkAxis) =>
  lattice.getLinkMatrix(coord.x, coord.y, axis, coord.z, coord.t);

const iterateRectangularLoop = (
  lattice: GaugeLattice,
  origin: LatticeCoord,
  extentX: number,
  extentY: number,
  axes: [GaugeLinkAxis, GaugeLinkAxis],
): Complex3x3 => {
  let matrix = createIdentityMatrix();
  let cursor: LatticeCoord = { ...origin };

  const advance = (axis: GaugeLinkAxis, direction: 1 | -1) => {
    if (direction === 1) {
      matrix = su3_mul(matrix, getLink(lattice, cursor, axis));
      cursor = lattice.shiftCoordinate(cursor, axis, 1);
    } else {
      cursor = lattice.shiftCoordinate(cursor, axis, -1);
      matrix = su3_mul(matrix, conjugateTranspose(getLink(lattice, cursor, axis)));
    }
  };

  for (let step = 0; step < extentX; step++) {
    advance(axes[0], 1);
  }
  for (let step = 0; step < extentY; step++) {
    advance(axes[1], 1);
  }
  for (let step = 0; step < extentX; step++) {
    advance(axes[0], -1);
  }
  for (let step = 0; step < extentY; step++) {
    advance(axes[1], -1);
  }
  return matrix;
};

const makeLookupKey = (extentX: number, extentY: number, axes: [GaugeLinkAxis, GaugeLinkAxis]) =>
  `${extentX}${LOOP_KEY_SEPARATOR}${extentY}:${axes[0]}${axes[1]}`;

export const measureWilsonRectangle = (
  lattice: GaugeLattice,
  extentX: number,
  extentY: number,
  axes?: [GaugeLinkAxis, GaugeLinkAxis],
): WilsonLoopMeasurement => {
  const ax = coerceExtent('extentX', extentX);
  const ay = coerceExtent('extentY', extentY);
  const pair = normalizeAxes(lattice, axes);

  let sumRe = 0;
  let sumIm = 0;
  let samples = 0;

  forEachSite(lattice, (coord) => {
    const loop = iterateRectangularLoop(lattice, coord, ax, ay, pair);
    const trace = traceComplex(loop);
    sumRe += trace.re * NORMALIZATION;
    sumIm += trace.im * NORMALIZATION;
    samples += 1;
  });

  const normalized: Complex =
    samples > 0 ? { re: sumRe / samples, im: sumIm / samples } : { re: 0, im: 0 };
  return {
    extentX: ax,
    extentY: ay,
    axes: pair,
    normalized,
    magnitude: Math.hypot(normalized.re, normalized.im),
    value: normalized.re,
    sampleCount: samples,
  };
};

export const measureWilsonLoops = (
  lattice: GaugeLattice,
  rectangles: readonly WilsonRectangle[],
): WilsonLoopMeasurement[] => {
  const unique = new Map<string, WilsonRectangle>();
  rectangles.forEach((rect) => {
    const extentX = coerceExtent('extentX', rect.extentX);
    const extentY = coerceExtent('extentY', rect.extentY);
    const pair = normalizeAxes(lattice, rect.axes);
    unique.set(makeLookupKey(extentX, extentY, pair), {
      extentX,
      extentY,
      axes: pair,
    });
  });

  const results: WilsonLoopMeasurement[] = [];
  for (const rect of unique.values()) {
    results.push(measureWilsonRectangle(lattice, rect.extentX, rect.extentY, rect.axes));
  }
  return results;
};

export const measureWilsonLoopGrid = (
  lattice: GaugeLattice,
  maxExtentX: number,
  maxExtentY: number,
  axes?: [GaugeLinkAxis, GaugeLinkAxis],
): WilsonLoopTable => {
  const extentX = coerceExtent('maxExtentX', maxExtentX);
  const extentY = coerceExtent('maxExtentY', maxExtentY);
  const pair = normalizeAxes(lattice, axes);
  const entries = new Map<string, WilsonLoopMeasurement>();
  for (let dx = 1; dx <= extentX; dx++) {
    for (let dy = 1; dy <= extentY; dy++) {
      const measurement = measureWilsonRectangle(lattice, dx, dy, pair);
      entries.set(makeLookupKey(dx, dy, pair), measurement);
    }
  }
  return entries;
};

export const getWilsonLoopValue = (
  table: WilsonLoopTable,
  extentX: number,
  extentY: number,
  axes: [GaugeLinkAxis, GaugeLinkAxis],
): WilsonLoopMeasurement | undefined => table.get(makeLookupKey(extentX, extentY, axes));

export const computeCreutzRatio = (
  table: WilsonLoopTable,
  extentX: number,
  extentY: number,
  axes: [GaugeLinkAxis, GaugeLinkAxis],
): number => {
  const x = coerceExtent('extentX', extentX);
  const y = coerceExtent('extentY', extentY);
  const r1 = getWilsonLoopValue(table, x + 1, y + 1, axes);
  const r2 = getWilsonLoopValue(table, x, y, axes);
  const r3 = getWilsonLoopValue(table, x + 1, y, axes);
  const r4 = getWilsonLoopValue(table, x, y + 1, axes);

  if (!r1 || !r2 || !r3 || !r4) {
    throw new RangeError(
      `Creutz ratio requires Wilson loops up to (${x + 1}, ${y + 1}) along axes ${axes.join('')}`,
    );
  }

  const numerator = Math.max(Math.abs(r1.value * r2.value), MIN_POSITIVE);
  const denominator = Math.max(Math.abs(r3.value * r4.value), MIN_POSITIVE);
  return -Math.log(numerator / denominator);
};

export const measurePolyakovLoop = (
  lattice: GaugeLattice,
  axis: GaugeLinkAxis,
  extentOverride?: number,
): PolyakovLoopMeasurement => {
  assertAxisActive(lattice, axis);
  const extent = extentOverride ?? axisExtent(lattice, axis);
  const coercedExtent = coerceExtent('temporalExtent', extent);

  let sumRe = 0;
  let sumIm = 0;
  let sampleCount = 0;

  forEachSite(lattice, (coord) => {
    let cursor = { ...coord };
    let product = createIdentityMatrix();
    for (let step = 0; step < coercedExtent; step++) {
      product = su3_mul(product, getLink(lattice, cursor, axis));
      cursor = lattice.shiftCoordinate(cursor, axis, 1);
    }
    const trace = traceComplex(product);
    sumRe += trace.re * NORMALIZATION;
    sumIm += trace.im * NORMALIZATION;
    sampleCount += 1;
  });

  const average: Complex =
    sampleCount > 0 ? { re: sumRe / sampleCount, im: sumIm / sampleCount } : { re: 0, im: 0 };
  return {
    axis,
    extent: coercedExtent,
    average,
    magnitude: Math.hypot(average.re, average.im),
    sampleCount,
  };
};

export class RunningEstimate {
  private count = 0;
  private mean = 0;
  private m2 = 0;

  push(value: number): RunningEstimateSnapshot {
    if (!Number.isFinite(value)) {
      throw new TypeError(`Running estimate requires finite values (received ${value})`);
    }
    this.count += 1;
    const delta = value - this.mean;
    this.mean += delta / this.count;
    const delta2 = value - this.mean;
    this.m2 += delta * delta2;
    return this.snapshot();
  }

  snapshot(): RunningEstimateSnapshot {
    const variance = this.count > 1 ? this.m2 / (this.count - 1) : 0;
    const standardDeviation = Math.sqrt(Math.max(variance, 0));
    const standardError = this.count > 0 ? standardDeviation / Math.sqrt(this.count) : 0;
    return {
      count: this.count,
      mean: this.mean,
      variance,
      standardDeviation,
      standardError,
    };
  }

  reset(): void {
    this.count = 0;
    this.mean = 0;
    this.m2 = 0;
  }
}

export const binSamples = (samples: readonly number[], binSize: number): number[] => {
  if (!Number.isFinite(binSize)) {
    throw new TypeError(`Bin size must be finite (received ${binSize})`);
  }
  const size = Math.floor(binSize);
  if (size <= 0) {
    throw new RangeError(`Bin size must be positive (received ${binSize})`);
  }
  const bins: number[] = [];
  for (let index = 0; index + size <= samples.length; index += size) {
    let sum = 0;
    for (let offset = 0; offset < size; offset++) {
      sum += samples[index + offset]!;
    }
    bins.push(sum / size);
  }
  return bins;
};

export const jackknife = (
  samples: readonly number[],
  estimator: (values: readonly number[]) => number,
): JackknifeResult => {
  if (samples.length < 2) {
    throw new RangeError('Jackknife requires at least two samples');
  }
  const estimate = estimator(samples);
  const pseudoValues: number[] = [];
  for (let idx = 0; idx < samples.length; idx++) {
    const omitted: number[] = [];
    for (let j = 0; j < samples.length; j++) {
      if (j !== idx) {
        omitted.push(samples[j]!);
      }
    }
    pseudoValues.push(estimator(omitted));
  }
  let pseudoSum = 0;
  for (const value of pseudoValues) {
    pseudoSum += value;
  }
  const jackknifeMean = pseudoSum / pseudoValues.length;
  let varianceAccumulator = 0;
  for (const value of pseudoValues) {
    const delta = value - jackknifeMean;
    varianceAccumulator += delta * delta;
  }
  const error = Math.sqrt(((pseudoValues.length - 1) / pseudoValues.length) * varianceAccumulator);
  const bias = (pseudoValues.length - 1) * (jackknifeMean - estimate);
  return {
    estimate,
    jackknifeMean,
    bias,
    error,
    pseudoValues,
  };
};
