import { createHash } from 'blake3';

import {
  GaugeLattice,
  type GaugeLatticeSnapshot,
  restoreGaugeLattice,
  snapshotGaugeLattice,
} from './lattice.js';
import {
  applyApeSmear,
  computeAveragePlaquette,
  initializeGaugeField,
  mulberry32,
  performSweep,
} from './updateCpu.js';
import { su3_mul } from './su3.js';
import type { Complex3x3 } from './su3.js';
import {
  RunningEstimate,
  computeCreutzRatio,
  measurePolyakovLoop,
  measureWilsonLoopGrid,
  type PolyakovLoopMeasurement,
  type RunningEstimateSnapshot,
  type WilsonLoopMeasurement,
  type WilsonRectangle,
} from './observables.js';
import { computeFluxOverlayState, type FluxOverlayFrameData, type FluxSource } from './overlays.js';
import {
  buildProbeTransportVisualization,
  type ProbeTransportFrameData,
} from './probeTransport.js';
import { hashCanonicalJson } from '../serialization/canonicalJson.js';
import { FLOATS_PER_MATRIX } from './lattice.js';

const EPSILON = 1e-9;
const SEED_AUDIT_INITIAL_HASH = '0000000000000000000000000000000000000000000000000000000000000000';
export const QCD_SNAPSHOT_SCHEMA_VERSION = 2 as const;

const DEFAULT_WILSON_EXTENTS: readonly WilsonRectangle[] = [
  { extentX: 1, extentY: 1, axes: ['x', 'y'] },
  { extentX: 1, extentY: 2, axes: ['x', 'y'] },
  { extentX: 2, extentY: 2, axes: ['x', 'y'] },
];

const QCD_COMPLEX_STRIDE = 2;
const QCD_ROW_STRIDE = 6;
const QCD_MIN_DIMENSION = 16;
const QCD_MAX_DIMENSION = 192;
const QCD_TOTAL_SITE_LIMIT = 256 * 256;
const QCD_CANVAS_SCALE = 8;

const clampDimension = (value: number): number => {
  if (!Number.isFinite(value) || value <= 0) {
    return QCD_MIN_DIMENSION;
  }
  let scaled = Math.round(value / QCD_CANVAS_SCALE);
  if (scaled < QCD_MIN_DIMENSION) {
    scaled = QCD_MIN_DIMENSION;
  } else if (scaled > QCD_MAX_DIMENSION) {
    scaled = QCD_MAX_DIMENSION;
  }
  if (scaled % 2 !== 0) {
    if (scaled + 1 <= QCD_MAX_DIMENSION) {
      scaled += 1;
    } else {
      scaled -= 1;
    }
  }
  return Math.max(QCD_MIN_DIMENSION, scaled);
};

const enforceSiteBudget = (width: number, height: number): { width: number; height: number } => {
  let w = width;
  let h = height;
  const total = w * h;
  if (total <= QCD_TOTAL_SITE_LIMIT) {
    return { width: w, height: h };
  }
  const scale = Math.sqrt(QCD_TOTAL_SITE_LIMIT / total);
  const adjust = (value: number) => Math.max(QCD_MIN_DIMENSION, Math.floor(value * scale));
  w = adjust(w);
  h = adjust(h);
  if (w % 2 !== 0) {
    w = Math.max(QCD_MIN_DIMENSION, w - 1);
  }
  if (h % 2 !== 0) {
    h = Math.max(QCD_MIN_DIMENSION, h - 1);
  }
  if (w * h > QCD_TOTAL_SITE_LIMIT) {
    const secondaryScale = Math.sqrt(QCD_TOTAL_SITE_LIMIT / (w * h));
    w = Math.max(QCD_MIN_DIMENSION, Math.floor(w * secondaryScale) & ~1);
    h = Math.max(QCD_MIN_DIMENSION, Math.floor(h * secondaryScale) & ~1);
  }
  return {
    width: Math.min(Math.max(w, QCD_MIN_DIMENSION), QCD_MAX_DIMENSION),
    height: Math.min(Math.max(h, QCD_MIN_DIMENSION), QCD_MAX_DIMENSION),
  };
};

export const deriveLatticeResolution = (
  canvasWidth: number,
  canvasHeight: number,
): { width: number; height: number } => {
  const width = clampDimension(canvasWidth);
  const height = clampDimension(canvasHeight);
  return enforceSiteBudget(width, height);
};

const clamp = (value: number, min: number, max: number): number =>
  Math.min(Math.max(value, min), max);

const conjugateTranspose = (matrix: Complex3x3): Complex3x3 => {
  const result: Complex3x3 = [
    [
      { re: 0, im: 0 },
      { re: 0, im: 0 },
      { re: 0, im: 0 },
    ],
    [
      { re: 0, im: 0 },
      { re: 0, im: 0 },
      { re: 0, im: 0 },
    ],
    [
      { re: 0, im: 0 },
      { re: 0, im: 0 },
      { re: 0, im: 0 },
    ],
  ];
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const entry = matrix[col][row];
      result[row][col].re = entry.re;
      result[row][col].im = -entry.im;
    }
  }
  return result;
};

const hash32 = (value: number): number => {
  let x = value >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d);
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b);
  x ^= x >>> 16;
  return x >>> 0;
};

const deriveSubstepSeed = (
  baseSeed: number,
  substepIndex: number,
  axis: GaugeLattice['axes'][number],
  parity: 0 | 1,
): number => {
  const axisTag =
    axis === 'x' ? 0x51ed2705 : axis === 'y' ? 0x632beb5 : axis === 'z' ? 0xa511e9b5 : 0x9e3779b9;
  const parityTag = parity === 0 ? 0x7f4a7c15 : 0x45d9f3b;
  const mixed = baseSeed ^ axisTag ^ parityTag ^ (substepIndex * 0x51ed2705);
  return hash32(mixed);
};

const updateSeedAuditHash = (
  prev: string,
  step: {
    seed: number;
    axis: GaugeLattice['axes'][number];
    parity: 0 | 1;
    substepIndex: number;
  },
): string => {
  const hasher = createHash();
  hasher.update(prev);
  hasher.update('|');
  hasher.update(String(step.substepIndex));
  hasher.update(',');
  hasher.update(step.axis);
  hasher.update(',');
  hasher.update(step.parity === 0 ? '0' : '1');
  hasher.update(',');
  hasher.update(String(step.seed >>> 0));
  return hasher.digest('hex');
};

type EnergyField = {
  width: number;
  height: number;
  values: Float32Array;
  max: number;
};

export type QcdSmearingConfig = {
  alpha: number;
  iterations: number;
};

export type QcdAnnealConfig = {
  beta: number;
  overRelaxationSteps: number;
  smearing: QcdSmearingConfig;
  depth: number;
  temporalExtent: number;
  batchLayers: number;
  temperatureSchedule: number[];
};

export type QcdObservables = {
  averagePlaquette: number;
  plaquetteHistory: number[];
  plaquetteEstimate: RunningEstimateSnapshot;
  wilsonLoops: WilsonLoopMeasurement[];
  creutzRatio?: {
    extentX: number;
    extentY: number;
    axes: [GaugeLattice['axes'][number], GaugeLattice['axes'][number]];
    value: number;
  };
  polyakovSamples?: PolyakovLoopMeasurement[];
};

export type QcdSnapshot = {
  schemaVersion: number;
  lattice: GaugeLatticeSnapshot;
  config: QcdAnnealConfig;
  baseSeed: number;
  sweepIndex: number;
  substepIndex: number;
  phaseIndex: number;
  seedAuditHash: string;
  sources: FluxSource[];
  observables: QcdObservables;
  polyakovScan?: PolyakovLoopMeasurement[];
};

export type QcdRuntimeState = {
  lattice: GaugeLattice;
  config: QcdAnnealConfig;
  baseSeed: number;
  sweepIndex: number;
  substepIndex: number;
  phaseIndex: number;
  seedAuditHash: string;
  plaquetteAccumulator: RunningEstimate;
  plaquetteHistory: number[];
  observables: QcdObservables;
  wilsonExtents: readonly WilsonRectangle[];
  polyakovScan: PolyakovLoopMeasurement[];
  gpuPlaneScratch: Float32Array | null;
  planeOrder: PlaneDescriptor[];
  gpuPlaneCursor: number;
};

type PlaneDescriptor = { z: number; t: number };

const createEmptyObservables = (): QcdObservables => ({
  averagePlaquette: 0,
  plaquetteHistory: [],
  plaquetteEstimate: { count: 0, mean: 0, variance: 0, standardDeviation: 0, standardError: 0 },
  wilsonLoops: [],
});

const forEachPlane = (lattice: GaugeLattice, visit: (plane: PlaneDescriptor) => void) => {
  for (let t = 0; t < lattice.temporalExtent; t++) {
    for (let z = 0; z < lattice.depth; z++) {
      visit({ z, t });
    }
  }
};

const computePlaquetteEnergyField = (lattice: GaugeLattice): EnergyField => {
  const { width, height } = lattice;
  const values = new Float32Array(width * height);
  if (!lattice.axes.includes('x') || !lattice.axes.includes('y')) {
    return { width, height, values, max: 0 };
  }

  let max = 0;
  const layerFactor = lattice.depth * lattice.temporalExtent;
  const invLayer = layerFactor > 0 ? 1 / layerFactor : 1;

  forEachPlane(lattice, ({ z, t }) => {
    for (let y = 0; y < height; y++) {
      const yp1 = (y + 1) % height;
      for (let x = 0; x < width; x++) {
        const xp1 = (x + 1) % width;
        const Ux = lattice.getLinkMatrix(x, y, 'x', z, t);
        const Uy = lattice.getLinkMatrix(x, y, 'y', z, t);
        const UxForward = lattice.getLinkMatrix(x, yp1, 'x', z, t);
        const UyForward = lattice.getLinkMatrix(xp1, y, 'y', z, t);
        const plaquette = su3_mul(
          su3_mul(su3_mul(Ux, UyForward), conjugateTranspose(UxForward)),
          conjugateTranspose(Uy),
        );
        const normalized = plaquette[0][0].re + plaquette[1][1].re + plaquette[2][2].re;
        const energy = Math.max(0, 1 - normalized / 3);
        const index = y * width + x;
        values[index] += energy;
      }
    }
  });

  for (let index = 0; index < values.length; index++) {
    values[index] *= invLayer;
    if (values[index] > max) {
      max = values[index];
    }
  }

  return { width, height, values, max };
};

const upsampleEnergyField = (field: EnergyField, width: number, height: number): Float32Array => {
  if (field.width === width && field.height === height) {
    return new Float32Array(field.values);
  }
  const result = new Float32Array(width * height);
  const scaleX = field.width / width;
  const scaleY = field.height / height;
  for (let y = 0; y < height; y++) {
    const sampleY = clamp((y + 0.5) * scaleY - 0.5, 0, field.height - 1);
    const y0 = Math.floor(sampleY);
    const y1 = Math.min(field.height - 1, y0 + 1);
    const ty = sampleY - y0;
    for (let x = 0; x < width; x++) {
      const sampleX = clamp((x + 0.5) * scaleX - 0.5, 0, field.width - 1);
      const x0 = Math.floor(sampleX);
      const x1 = Math.min(field.width - 1, x0 + 1);
      const tx = sampleX - x0;
      const idx00 = y0 * field.width + x0;
      const idx01 = y0 * field.width + x1;
      const idx10 = y1 * field.width + x0;
      const idx11 = y1 * field.width + x1;
      const top = field.values[idx00] * (1 - tx) + field.values[idx01] * tx;
      const bottom = field.values[idx10] * (1 - tx) + field.values[idx11] * tx;
      result[y * width + x] = top * (1 - ty) + bottom * ty;
    }
  }
  return result;
};

const computeEnergyDirections = (
  energy: Float32Array,
  width: number,
  height: number,
): Float32Array => {
  const directions = new Float32Array(width * height * 2);
  for (let y = 0; y < height; y++) {
    const yPrev = y > 0 ? y - 1 : y;
    const yNext = y + 1 < height ? y + 1 : y;
    for (let x = 0; x < width; x++) {
      const xPrev = x > 0 ? x - 1 : x;
      const xNext = x + 1 < width ? x + 1 : x;
      const centerIndex = y * width + x;
      const left = energy[y * width + xPrev];
      const right = energy[y * width + xNext];
      const up = energy[yPrev * width + x];
      const down = energy[yNext * width + x];
      const dx = (right - left) * 0.5;
      const dy = (down - up) * 0.5;
      const mag = Math.hypot(dx, dy);
      const dirIndex = centerIndex * 2;
      if (mag > EPSILON) {
        directions[dirIndex] = dx / mag;
        directions[dirIndex + 1] = dy / mag;
      } else {
        directions[dirIndex] = 0;
        directions[dirIndex + 1] = 0;
      }
    }
  }
  return directions;
};

const blendDirection = (
  baseX: number,
  baseY: number,
  blendX: number,
  blendY: number,
  weightBase: number,
  weightBlend: number,
): [number, number] => {
  const x = baseX * weightBase + blendX * weightBlend;
  const y = baseY * weightBase + blendY * weightBlend;
  const norm = Math.hypot(x, y);
  if (norm > EPSILON) {
    return [x / norm, y / norm];
  }
  return [0, 0];
};

const buildEnergyOverlay = (
  lattice: GaugeLattice,
  width: number,
  height: number,
  sources: readonly FluxSource[],
): FluxOverlayFrameData | null => {
  const energyField = computePlaquetteEnergyField(lattice);
  if (energyField.max <= EPSILON && sources.length < 2) {
    return null;
  }
  const energyUpsampled = upsampleEnergyField(energyField, width, height);
  const directions = computeEnergyDirections(energyUpsampled, width, height);
  let maxEnergy = 0;
  for (let idx = 0; idx < energyUpsampled.length; idx++) {
    if (energyUpsampled[idx] > maxEnergy) {
      maxEnergy = energyUpsampled[idx];
    }
  }
  if (maxEnergy <= EPSILON && sources.length < 2) {
    return null;
  }
  const baseOverlay =
    sources.length >= 2 ? (computeFluxOverlayState({ width, height, sources }) ?? null) : null;

  const resultEnergy = new Float32Array(width * height);
  const resultDirection = new Float32Array(width * height * 2);
  let combinedMax = 0;
  const invEnergy = maxEnergy > 0 ? 1 / maxEnergy : 0;

  for (let idx = 0; idx < resultEnergy.length; idx++) {
    const gaugeNorm = energyUpsampled[idx] * invEnergy;
    const overlayNorm =
      baseOverlay && baseOverlay.energyScale > 0
        ? clamp(baseOverlay.energy[idx] * baseOverlay.energyScale, 0, 1)
        : gaugeNorm;
    const combined = Math.sqrt(Math.max(overlayNorm * gaugeNorm, 0));
    resultEnergy[idx] = combined;
    if (combined > combinedMax) {
      combinedMax = combined;
    }
    const dirIndex = idx * 2;
    const baseDirX = baseOverlay ? baseOverlay.direction[dirIndex] : 0;
    const baseDirY = baseOverlay ? baseOverlay.direction[dirIndex + 1] : 0;
    const gaugeDirX = directions[dirIndex];
    const gaugeDirY = directions[dirIndex + 1];
    const [mixX, mixY] = blendDirection(
      baseDirX,
      baseDirY,
      gaugeDirX,
      gaugeDirY,
      overlayNorm,
      gaugeNorm,
    );
    resultDirection[dirIndex] = mixX;
    resultDirection[dirIndex + 1] = mixY;
  }

  if (combinedMax <= EPSILON) {
    return null;
  }

  return {
    width,
    height,
    energy: resultEnergy,
    direction: resultDirection,
    energyScale: 1 / combinedMax,
    maxEnergy: combinedMax,
  };
};

const createQcdLattice = (shape: {
  width: number;
  height: number;
  depth: number;
  temporalExtent: number;
}) =>
  new GaugeLattice({
    width: shape.width,
    height: shape.height,
    depth: Math.max(1, shape.depth),
    temporalExtent: Math.max(1, shape.temporalExtent),
  });

const getPhaseDescriptor = (
  runtime: QcdRuntimeState,
): { axis: GaugeLattice['axes'][number]; parity: 0 | 1 } => {
  const axisCount = runtime.lattice.axes.length;
  const axisIndex = Math.floor(runtime.phaseIndex / 2) % axisCount;
  const parity = (runtime.phaseIndex % 2) as 0 | 1;
  return { axis: runtime.lattice.axes[axisIndex]!, parity };
};

const advancePhaseIndex = (runtime: QcdRuntimeState) => {
  const axisCount = runtime.lattice.axes.length;
  const totalPhases = Math.max(1, axisCount * 2);
  runtime.phaseIndex = (runtime.phaseIndex + 1) % totalPhases;
};

const axisOffset = (lattice: GaugeLattice, axis: GaugeLattice['axes'][number]): number => {
  const index = lattice.axes.indexOf(axis);
  if (index < 0) {
    throw new RangeError(`Axis ${axis} is inactive`);
  }
  return index * FLOATS_PER_MATRIX;
};

const copyPlaneToScratch = (
  lattice: GaugeLattice,
  plane: PlaneDescriptor,
  scratch: Float32Array,
): void => {
  const width = lattice.width;
  const height = lattice.height;
  const siteStride = lattice.siteStride;
  const xOffset = axisOffset(lattice, 'x');
  const yOffset = axisOffset(lattice, 'y');
  const planeBase =
    (plane.t * lattice.depth + plane.z) * lattice.height * lattice.width * siteStride;

  let scratchCursor = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const siteOffset = planeBase + (y * width + x) * siteStride;
      scratch.set(
        lattice.data.subarray(siteOffset + xOffset, siteOffset + xOffset + FLOATS_PER_MATRIX),
        scratchCursor,
      );
      scratchCursor += FLOATS_PER_MATRIX;
      scratch.set(
        lattice.data.subarray(siteOffset + yOffset, siteOffset + yOffset + FLOATS_PER_MATRIX),
        scratchCursor,
      );
      scratchCursor += FLOATS_PER_MATRIX;
    }
  }
};

const copyScratchToPlane = (
  lattice: GaugeLattice,
  plane: PlaneDescriptor,
  scratch: Float32Array,
): void => {
  const width = lattice.width;
  const height = lattice.height;
  const siteStride = lattice.siteStride;
  const xOffset = axisOffset(lattice, 'x');
  const yOffset = axisOffset(lattice, 'y');
  const planeBase =
    (plane.t * lattice.depth + plane.z) * lattice.height * lattice.width * siteStride;

  let scratchCursor = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const siteOffset = planeBase + (y * width + x) * siteStride;
      lattice.data.set(
        scratch.subarray(scratchCursor, scratchCursor + FLOATS_PER_MATRIX),
        siteOffset + xOffset,
      );
      scratchCursor += FLOATS_PER_MATRIX;
      lattice.data.set(
        scratch.subarray(scratchCursor, scratchCursor + FLOATS_PER_MATRIX),
        siteOffset + yOffset,
      );
      scratchCursor += FLOATS_PER_MATRIX;
    }
  }
};

export const initializeQcdRuntime = (params: {
  latticeSize: { width: number; height: number };
  config: QcdAnnealConfig;
  baseSeed: number;
  startMode?: 'cold' | 'hot';
}): QcdRuntimeState => {
  const { latticeSize, config, baseSeed, startMode = 'cold' } = params;
  const lattice = createQcdLattice({
    width: latticeSize.width,
    height: latticeSize.height,
    depth: Math.max(1, config.depth),
    temporalExtent: Math.max(1, config.temporalExtent),
  });
  const initRng = startMode === 'hot' ? mulberry32(hash32(baseSeed ^ 0x9e3779b9)) : null;
  initializeGaugeField(lattice, startMode, initRng);
  const planeOrder: PlaneDescriptor[] = [];
  forEachPlane(lattice, (plane) => planeOrder.push(plane));

  return {
    lattice,
    config,
    baseSeed,
    sweepIndex: 0,
    substepIndex: 0,
    phaseIndex: 0,
    seedAuditHash: SEED_AUDIT_INITIAL_HASH,
    plaquetteAccumulator: new RunningEstimate(),
    plaquetteHistory: [],
    observables: createEmptyObservables(),
    wilsonExtents: DEFAULT_WILSON_EXTENTS,
    polyakovScan: [],
    gpuPlaneScratch: null,
    planeOrder,
    gpuPlaneCursor: 0,
  };
};

export const restoreQcdRuntime = (snapshot: QcdSnapshot): QcdRuntimeState => {
  const lattice = restoreGaugeLattice(snapshot.lattice);
  const runtime: QcdRuntimeState = {
    lattice,
    config: snapshot.config,
    baseSeed: snapshot.baseSeed,
    sweepIndex: snapshot.sweepIndex,
    substepIndex: snapshot.substepIndex,
    phaseIndex: snapshot.phaseIndex,
    seedAuditHash: snapshot.seedAuditHash,
    plaquetteAccumulator: new RunningEstimate(),
    plaquetteHistory: [...snapshot.observables.plaquetteHistory],
    observables: snapshot.observables,
    wilsonExtents: DEFAULT_WILSON_EXTENTS,
    polyakovScan: snapshot.polyakovScan ? [...snapshot.polyakovScan] : [],
    gpuPlaneScratch: null,
    planeOrder: [],
    gpuPlaneCursor: 0,
  };
  const planeOrder: PlaneDescriptor[] = [];
  forEachPlane(lattice, (plane) => planeOrder.push(plane));
  runtime.planeOrder = planeOrder;
  snapshot.observables.plaquetteHistory.forEach((value) => {
    runtime.plaquetteAccumulator.push(value);
  });
  return runtime;
};

const applySmearingIfNeeded = (runtime: QcdRuntimeState) => {
  const { alpha, iterations } = runtime.config.smearing;
  if (alpha > 0 && iterations > 0) {
    applyApeSmear(runtime.lattice, alpha, iterations);
  }
};

const updateObservables = (runtime: QcdRuntimeState) => {
  const averagePlaquette = computeAveragePlaquette(runtime.lattice);
  runtime.plaquetteHistory.push(averagePlaquette);
  if (runtime.plaquetteHistory.length > 128) {
    runtime.plaquetteHistory.shift();
  }
  const estimate = runtime.plaquetteAccumulator.push(averagePlaquette);
  const grid = measureWilsonLoopGrid(runtime.lattice, 2, 2, ['x', 'y']);
  const wilsonLoops: WilsonLoopMeasurement[] = [];
  runtime.wilsonExtents.forEach((rect) => {
    const axes = rect.axes ?? ['x', 'y'];
    const key = `${rect.extentX}x${rect.extentY}:${axes.join('')}`;
    const measurement = grid.get(key);
    if (measurement) {
      wilsonLoops.push(measurement);
    }
  });
  let creutzRatio: QcdObservables['creutzRatio'] | undefined;
  try {
    const axes: [GaugeLattice['axes'][number], GaugeLattice['axes'][number]] = ['x', 'y'];
    const value = computeCreutzRatio(grid, 1, 1, axes);
    creutzRatio = { extentX: 1, extentY: 1, axes, value };
  } catch {
    creutzRatio = undefined;
  }
  runtime.observables = {
    averagePlaquette,
    plaquetteHistory: [...runtime.plaquetteHistory],
    plaquetteEstimate: estimate,
    wilsonLoops,
    creutzRatio,
    polyakovSamples:
      runtime.polyakovScan.length > 0
        ? [...runtime.polyakovScan]
        : runtime.observables.polyakovSamples,
  };
};

const ensureGpuScratch = (runtime: QcdRuntimeState): Float32Array => {
  const planeSize = runtime.lattice.width * runtime.lattice.height * FLOATS_PER_MATRIX * 2;
  if (!runtime.gpuPlaneScratch || runtime.gpuPlaneScratch.length !== planeSize) {
    runtime.gpuPlaneScratch = new Float32Array(planeSize);
  }
  return runtime.gpuPlaneScratch;
};

export const runGpuSubstep = async (
  runtime: QcdRuntimeState,
  renderer: {
    runQcdHeatbathSweep?: (options: {
      lattice: Float32Array;
      width: number;
      height: number;
      siteStride: number;
      linkStride: number;
      rowStride: number;
      complexStride: number;
      beta: number;
      parity: 0 | 1;
      axis: 'x' | 'y';
      sweepIndex: number;
      overRelaxationSteps: number;
      seed?: number;
      scope?: string | number;
    }) => Promise<boolean>;
  },
  scope: string | number = 'interactive-qcd',
): Promise<boolean> => {
  const phase = getPhaseDescriptor(runtime);
  if (!renderer.runQcdHeatbathSweep) {
    return false;
  }
  if (phase.axis !== 'x' && phase.axis !== 'y') {
    return false;
  }
  if (!runtime.lattice.axes.includes('x') || !runtime.lattice.axes.includes('y')) {
    return false;
  }

  const scratch = ensureGpuScratch(runtime);
  const siteCount = runtime.lattice.width * runtime.lattice.height;
  const floatsPerSite = siteCount > 0 ? scratch.length / siteCount : 0;
  if (!Number.isFinite(floatsPerSite) || Math.floor(floatsPerSite) !== floatsPerSite) {
    console.warn('[qcd] invalid GPU scratch stride', { scratchLength: scratch.length, siteCount });
    return false;
  }
  const siteStride = Math.trunc(floatsPerSite);
  const planeCount = runtime.planeOrder.length;
  if (planeCount === 0) {
    return false;
  }
  const batch = Math.max(1, Math.min(runtime.config.batchLayers, planeCount));
  let processed = 0;
  let success = true;

  while (processed < batch) {
    const plane = runtime.planeOrder[runtime.gpuPlaneCursor];
    runtime.gpuPlaneCursor = (runtime.gpuPlaneCursor + 1) % planeCount;
    copyPlaneToScratch(runtime.lattice, plane, scratch);
    const substepSeed = deriveSubstepSeed(
      runtime.baseSeed,
      runtime.substepIndex,
      phase.axis,
      phase.parity,
    );
    const planeResult = await renderer.runQcdHeatbathSweep({
      lattice: scratch,
      width: runtime.lattice.width,
      height: runtime.lattice.height,
      siteStride,
      linkStride: FLOATS_PER_MATRIX,
      rowStride: QCD_ROW_STRIDE,
      complexStride: QCD_COMPLEX_STRIDE,
      beta: runtime.config.beta,
      parity: phase.parity,
      axis: phase.axis,
      sweepIndex: runtime.substepIndex,
      overRelaxationSteps: runtime.config.overRelaxationSteps,
      seed: substepSeed,
      scope: `${scope}:plane-${plane.z}-${plane.t}`,
    });
    if (!planeResult) {
      success = false;
      break;
    }
    copyScratchToPlane(runtime.lattice, plane, scratch);
    processed += 1;
    if (runtime.gpuPlaneCursor === 0) {
      break;
    }
  }

  if (!success) {
    return false;
  }

  const seedValue = deriveSubstepSeed(
    runtime.baseSeed,
    runtime.substepIndex,
    phase.axis,
    phase.parity,
  );
  runtime.seedAuditHash = updateSeedAuditHash(runtime.seedAuditHash, {
    seed: seedValue,
    axis: phase.axis,
    parity: phase.parity,
    substepIndex: runtime.substepIndex,
  });
  runtime.substepIndex += 1;
  advancePhaseIndex(runtime);
  if (runtime.phaseIndex === 0) {
    runtime.sweepIndex += 1;
    applySmearingIfNeeded(runtime);
    updateObservables(runtime);
  }
  return true;
};

export const runCpuSweep = (runtime: QcdRuntimeState, rng: () => number): void => {
  performSweep(runtime.lattice, runtime.config.beta, rng, runtime.config.overRelaxationSteps);
  runtime.seedAuditHash = updateSeedAuditHash(runtime.seedAuditHash, {
    seed: deriveSubstepSeed(runtime.baseSeed, runtime.substepIndex, 'x', 0),
    axis: 'x',
    parity: 0,
    substepIndex: runtime.substepIndex,
  });
  runtime.substepIndex += runtime.lattice.axes.length * 2;
  runtime.phaseIndex = 0;
  runtime.sweepIndex += 1;
  applySmearingIfNeeded(runtime);
  updateObservables(runtime);
};

export const buildQcdOverlay = (
  runtime: QcdRuntimeState,
  sources: readonly FluxSource[],
  width: number,
  height: number,
): FluxOverlayFrameData | null => buildEnergyOverlay(runtime.lattice, width, height, sources);

export const buildQcdProbeFrame = (
  runtime: QcdRuntimeState,
  sources: readonly FluxSource[],
): ProbeTransportFrameData | null => buildProbeTransportVisualization(runtime.lattice, sources);

export const buildQcdSnapshot = (
  runtime: QcdRuntimeState,
  sources: readonly FluxSource[],
): QcdSnapshot => ({
  schemaVersion: QCD_SNAPSHOT_SCHEMA_VERSION,
  lattice: snapshotGaugeLattice(runtime.lattice),
  config: runtime.config,
  baseSeed: runtime.baseSeed,
  sweepIndex: runtime.sweepIndex,
  substepIndex: runtime.substepIndex,
  phaseIndex: runtime.phaseIndex,
  seedAuditHash: runtime.seedAuditHash,
  sources: sources.map((source) => ({ ...source })),
  observables: runtime.observables,
  polyakovScan: runtime.polyakovScan,
});

export const hashQcdSnapshot = (snapshot: QcdSnapshot): { hash: string; canonicalJson: string } => {
  const { hash, json } = hashCanonicalJson(snapshot, { indent: 2 });
  return { hash, canonicalJson: json };
};

export const runTemperatureScan = (
  runtime: QcdRuntimeState,
  betas: readonly number[],
  axis: GaugeLattice['axes'][number],
): PolyakovLoopMeasurement[] => {
  const samples: PolyakovLoopMeasurement[] = [];
  const latticeClone = new GaugeLattice(
    {
      width: runtime.lattice.width,
      height: runtime.lattice.height,
      depth: runtime.lattice.depth,
      temporalExtent: runtime.lattice.temporalExtent,
    },
    new Float32Array(runtime.lattice.data),
  );
  const rng = mulberry32(hash32(runtime.baseSeed ^ 0x5bf03635));
  betas.forEach((beta) => {
    performSweep(latticeClone, beta, rng, runtime.config.overRelaxationSteps);
    const measurement = measurePolyakovLoop(latticeClone, axis);
    samples.push(measurement);
  });
  runtime.polyakovScan = samples;
  runtime.observables.polyakovSamples = samples;
  return samples;
};
