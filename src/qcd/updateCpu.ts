import { GaugeLattice, type GaugeLinkAxis } from './lattice.js';
import { su3_mul, su3_project, su3_haar, su3_frobNorm, type Complex3x3 } from './su3.js';

type Complex = { re: number; im: number };

export type CpuWilsonUpdateOptions = {
  betaSchedule: readonly number[];
  sweepsPerBeta?: number;
  thermalizationSweeps?: number;
  overRelaxationSteps?: number;
  startMode?: 'cold' | 'hot';
  seed?: number;
  apeSmearing?: {
    alpha: number;
    iterations?: number;
  } | null;
};

export type CpuWilsonUpdateResult = {
  plaquetteHistory: number[];
  totalSweeps: number;
  finalBeta: number;
};

const DEFAULT_SWEEPS_PER_BETA = 4;
const DEFAULT_THERMALIZATION_SWEEPS = 4;
const DEFAULT_OVER_RELAXATION_STEPS = 1;

const createZeroMatrix = (): Complex3x3 =>
  new Array(3)
    .fill(null)
    .map(
      () => new Array(3).fill(null).map(() => ({ re: 0, im: 0 }) as Complex) as Complex[],
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

const matrixLinearCombination = (
  terms: readonly { matrix: Complex3x3; weight: number }[],
): Complex3x3 => {
  const result = createZeroMatrix();
  for (const { matrix, weight } of terms) {
    if (weight === 0 || !Number.isFinite(weight)) continue;
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        const entry = matrix[row][col];
        result[row][col].re += entry.re * weight;
        result[row][col].im += entry.im * weight;
      }
    }
  }
  return result;
};

const matrixScale = (matrix: Complex3x3, scalar: number): Complex3x3 =>
  matrixLinearCombination([{ matrix, weight: scalar }]);

const traceReal = (matrix: Complex3x3): number =>
  matrix[0][0].re + matrix[1][1].re + matrix[2][2].re;

const getStapleMultiplicity = (axisCount: number) => Math.max(1, (axisCount - 1) * 2);

type LatticeCoord = { x: number; y: number; z: number; t: number };

const getLink = (lattice: GaugeLattice, coord: LatticeCoord, axis: GaugeLinkAxis): Complex3x3 =>
  lattice.getLinkMatrix(coord.x, coord.y, axis, coord.z, coord.t);

const computeStaple = (
  lattice: GaugeLattice,
  coord: LatticeCoord,
  axis: GaugeLinkAxis,
): Complex3x3 => {
  const staple = createZeroMatrix();
  const axes = lattice.axes;

  const accumulate = (contribution: Complex3x3) => {
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        staple[row][col].re += contribution[row][col].re;
        staple[row][col].im += contribution[row][col].im;
      }
    }
  };

  for (const otherAxis of axes) {
    if (otherAxis === axis) continue;
    const forwardOther = lattice.shiftCoordinate(coord, otherAxis, 1);
    const forwardAxis = lattice.shiftCoordinate(coord, axis, 1);
    const forwardAxisOther = lattice.shiftCoordinate(forwardAxis, otherAxis, 1);

    const forward = su3_mul(
      su3_mul(getLink(lattice, coord, otherAxis), getLink(lattice, forwardOther, axis)),
      conjugateTranspose(getLink(lattice, forwardAxis, otherAxis)),
    );
    accumulate(forward);

    const backwardOther = lattice.shiftCoordinate(coord, otherAxis, -1);
    const backwardOtherAxis = lattice.shiftCoordinate(backwardOther, axis, 1);

    const backward = su3_mul(
      su3_mul(
        conjugateTranspose(getLink(lattice, backwardOther, otherAxis)),
        getLink(lattice, backwardOther, axis),
      ),
      getLink(lattice, backwardOtherAxis, otherAxis),
    );
    accumulate(backward);
  }

  return staple;
};

const computeNoiseMix = (
  beta: number,
  stapleNorm: number,
): { staple: number; link: number; noise: number } => {
  const scaled = Math.max(0, beta) * stapleNorm;
  const align = Math.tanh(scaled / 12);
  const stapleWeight = align;
  const linkWeight = Math.max(0, 1 - stapleWeight);
  const noiseWeight = Math.sqrt(Math.max(0, (1 - align) * 0.25));
  return { staple: stapleWeight, link: linkWeight, noise: noiseWeight };
};

const applyHeatbathApproximation = (
  current: Complex3x3,
  staple: Complex3x3,
  beta: number,
  rng: () => number,
  overRelaxationSteps: number,
  stapleMultiplicity: number,
): Complex3x3 => {
  const stapleProjected = su3_project(staple);
  const stapleNorm = su3_frobNorm(staple) / stapleMultiplicity;
  const weights = computeNoiseMix(beta, stapleNorm);

  const blended = matrixLinearCombination([
    { matrix: current, weight: weights.link },
    { matrix: stapleProjected, weight: weights.staple },
  ]);

  let proposal = blended;

  if (weights.noise > 1e-6) {
    const noise = su3_haar(1, rng);
    const noisy = matrixLinearCombination([
      { matrix: proposal, weight: 1 },
      { matrix: noise, weight: weights.noise },
    ]);
    proposal = noisy;
  }

  let updated = su3_project(proposal);
  const reflections = Math.max(0, overRelaxationSteps | 0);
  if (reflections > 0) {
    for (let step = 0; step < reflections; step++) {
      const reflected = matrixLinearCombination([
        { matrix: stapleProjected, weight: 2 },
        { matrix: updated, weight: -1 },
      ]);
      updated = su3_project(reflected);
    }
  }
  return updated;
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

export const mulberry32 = (seed: number): (() => number) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let c = Math.imul(t ^ (t >>> 15), 1 | t);
    c ^= c + Math.imul(c ^ (c >>> 7), 61 | c);
    return ((c ^ (c >>> 14)) >>> 0) / 4294967296;
  };
};

export const performSweep = (
  lattice: GaugeLattice,
  beta: number,
  rng: () => number,
  overRelaxationSteps: number,
): void => {
  const axes = lattice.axes;
  const stapleMultiplicity = getStapleMultiplicity(axes.length);
  forEachSite(lattice, (coord) => {
    for (const axis of axes) {
      const current = getLink(lattice, coord, axis);
      const staple = computeStaple(lattice, coord, axis);
      const updated = applyHeatbathApproximation(
        current,
        staple,
        beta,
        rng,
        overRelaxationSteps,
        stapleMultiplicity,
      );
      lattice.setLinkMatrix(coord.x, coord.y, axis, updated, coord.z, coord.t);
    }
  });
};

export const initializeGaugeField = (
  lattice: GaugeLattice,
  startMode: 'cold' | 'hot',
  rng: (() => number) | null = null,
): void => {
  if (startMode === 'cold') {
    lattice.fillIdentity();
    return;
  }

  const sampler = rng ?? Math.random;
  forEachSite(lattice, (coord) => {
    for (const axis of lattice.axes) {
      lattice.setLinkMatrix(coord.x, coord.y, axis, su3_haar(1, sampler), coord.z, coord.t);
    }
  });
};

const computePlaquetteTrace = (
  lattice: GaugeLattice,
  coord: LatticeCoord,
  axisMu: GaugeLinkAxis,
  axisNu: GaugeLinkAxis,
): number => {
  const forwardMu = lattice.shiftCoordinate(coord, axisMu, 1);
  const forwardNu = lattice.shiftCoordinate(coord, axisNu, 1);
  const forwardMuNu = lattice.shiftCoordinate(forwardMu, axisNu, 1);

  const term = su3_mul(
    su3_mul(getLink(lattice, coord, axisMu), getLink(lattice, forwardMu, axisNu)),
    su3_mul(
      conjugateTranspose(getLink(lattice, forwardNu, axisMu)),
      conjugateTranspose(getLink(lattice, coord, axisNu)),
    ),
  );
  return traceReal(term) / 3;
};

export const computeAveragePlaquette = (lattice: GaugeLattice): number => {
  const axes = lattice.axes;
  if (axes.length < 2) {
    return 0;
  }
  let sum = 0;
  let samples = 0;
  forEachSite(lattice, (coord) => {
    for (let i = 0; i < axes.length; i++) {
      for (let j = i + 1; j < axes.length; j++) {
        sum += computePlaquetteTrace(lattice, coord, axes[i]!, axes[j]!);
        samples += 1;
      }
    }
  });
  return samples > 0 ? sum / samples : 0;
};

export const applyApeSmear = (lattice: GaugeLattice, alpha: number, iterations = 1): void => {
  if (!Number.isFinite(alpha) || alpha <= 0) return;
  const clampedAlpha = Math.min(Math.max(alpha, 0), 1);
  const totalIterations = Math.max(0, Math.floor(iterations));
  if (totalIterations === 0) return;
  const axes = lattice.axes;
  const stapleMultiplicity = getStapleMultiplicity(axes.length);

  for (let iter = 0; iter < totalIterations; iter++) {
    const snapshot = new GaugeLattice(
      {
        width: lattice.width,
        height: lattice.height,
        depth: lattice.depth,
        temporalExtent: lattice.temporalExtent,
      },
      new Float32Array(lattice.data),
    );
    forEachSite(lattice, (coord) => {
      for (const axis of axes) {
        const original = getLink(snapshot, coord, axis);
        const staple = computeStaple(snapshot, coord, axis);
        const averagedStaple = su3_project(matrixScale(staple, 1 / stapleMultiplicity));
        const blended = matrixLinearCombination([
          { matrix: original, weight: 1 - clampedAlpha },
          { matrix: averagedStaple, weight: clampedAlpha },
        ]);
        const smeared = su3_project(blended);
        lattice.setLinkMatrix(coord.x, coord.y, axis, smeared, coord.z, coord.t);
      }
    });
  }
};

export const mulberry32Stream = (seed: number): (() => number)[] => {
  const base = mulberry32(seed);
  return Array.from({ length: 4 }, () => {
    const streamSeed = Math.floor(base() * 0xffffffff);
    return mulberry32(streamSeed);
  });
};

export const runWilsonCpuUpdate = (
  lattice: GaugeLattice,
  options: CpuWilsonUpdateOptions,
): CpuWilsonUpdateResult => {
  const betaSchedule = Array.isArray(options.betaSchedule) ? options.betaSchedule : [];
  if (betaSchedule.length === 0) {
    throw new RangeError('CPU Wilson update requires a non-empty beta schedule');
  }
  const sweepsPerBeta = Number.isFinite(options.sweepsPerBeta)
    ? Math.max(1, Math.floor(options.sweepsPerBeta as number))
    : DEFAULT_SWEEPS_PER_BETA;
  const thermalizationSweeps = Number.isFinite(options.thermalizationSweeps)
    ? Math.max(0, Math.floor(options.thermalizationSweeps as number))
    : DEFAULT_THERMALIZATION_SWEEPS;
  const overRelaxationSteps = Number.isFinite(options.overRelaxationSteps)
    ? Math.max(0, Math.floor(options.overRelaxationSteps as number))
    : DEFAULT_OVER_RELAXATION_STEPS;

  const seed = Number.isFinite(options.seed) ? Math.trunc(options.seed as number) : 0;
  const rng = mulberry32(seed);

  initializeGaugeField(lattice, options.startMode === 'hot' ? 'hot' : 'cold', rng);

  if (thermalizationSweeps > 0) {
    const beta = betaSchedule[0];
    for (let sweep = 0; sweep < thermalizationSweeps; sweep++) {
      performSweep(lattice, beta, rng, overRelaxationSteps);
    }
  }

  const plaquetteHistory: number[] = [];
  let totalSweeps = thermalizationSweeps;

  for (const beta of betaSchedule) {
    for (let sweep = 0; sweep < sweepsPerBeta; sweep++) {
      performSweep(lattice, beta, rng, overRelaxationSteps);
      totalSweeps += 1;
      if (options.apeSmearing) {
        applyApeSmear(lattice, options.apeSmearing.alpha, options.apeSmearing.iterations ?? 1);
      }
    }
    plaquetteHistory.push(computeAveragePlaquette(lattice));
  }

  return {
    plaquetteHistory,
    totalSweeps,
    finalBeta: betaSchedule[betaSchedule.length - 1] ?? 0,
  };
};
