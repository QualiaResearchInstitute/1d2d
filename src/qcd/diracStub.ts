/**
 * Diagnostic-only Dirac operator stub.
 *
 * This module provides a lightweight, non-physical Laplacian-like stencil
 * that respects local gauge covariance and supports simple Krylov solvers
 * on tiny lattices. It is intended for visualisation and regression tests
 * only — do not treat the results as physically meaningful.
 */

import { GaugeLattice, type GaugeLinkAxis } from './lattice.js';
import { su3_applyToVector, su3_conjugateTranspose, type Complex3x3 } from './su3.js';

export type DiracStubOptions = {
  mass: number;
  kappa: number;
};

export type DiracStubOperator = {
  lattice: GaugeLattice;
  mass: number;
  kappa: number;
  axes: readonly GaugeLinkAxis[];
};

export type DiracSolveOptions = {
  tolerance?: number;
  maxIterations?: number;
  initialGuess?: Float64Array;
};

export type DiracStubSolution = {
  solution: Float64Array;
  iterations: number;
  residual: number;
  converged: boolean;
};

const DEFAULT_OPTIONS: DiracStubOptions = Object.freeze({
  mass: 4,
  kappa: 0.12,
});

const EPSILON = 1e-12;

const vectorLengthFor = (lattice: GaugeLattice): number => lattice.siteCount * 6;

export const createDiracStubOperator = (
  lattice: GaugeLattice,
  options: Partial<DiracStubOptions> = {},
): DiracStubOperator => {
  const mass = Number.isFinite(options.mass) ? (options.mass as number) : DEFAULT_OPTIONS.mass;
  const kappa = Number.isFinite(options.kappa) ? (options.kappa as number) : DEFAULT_OPTIONS.kappa;
  if (!Number.isFinite(mass) || mass <= 0) {
    throw new RangeError(`Dirac stub mass must be positive (received ${options.mass})`);
  }
  if (!Number.isFinite(kappa) || kappa <= 0) {
    throw new RangeError(`Dirac stub kappa must be positive (received ${options.kappa})`);
  }
  return {
    lattice,
    mass,
    kappa,
    axes: [...lattice.axes],
  };
};

export const createDiracVector = (lattice: GaugeLattice): Float64Array =>
  new Float64Array(vectorLengthFor(lattice));

const assertVectorLength = (operator: DiracStubOperator, vector: Float64Array): void => {
  const expected = vectorLengthFor(operator.lattice);
  if (vector.length !== expected) {
    throw new RangeError(
      `Dirac stub vector length mismatch (expected ${expected}, received ${vector.length})`,
    );
  }
};

const applyContribution = (
  output: Float64Array,
  baseOffset: number,
  contribution: readonly [
    { re: number; im: number },
    { re: number; im: number },
    { re: number; im: number },
  ],
  scale: number,
): void => {
  for (let color = 0; color < 3; color++) {
    const entry = contribution[color];
    const offset = baseOffset + color * 2;
    output[offset] += entry.re * scale;
    output[offset + 1] += entry.im * scale;
  }
};

const readVector = (
  buffer: Float64Array,
  baseOffset: number,
): [{ re: number; im: number }, { re: number; im: number }, { re: number; im: number }] => [
  { re: buffer[baseOffset], im: buffer[baseOffset + 1] },
  { re: buffer[baseOffset + 2], im: buffer[baseOffset + 3] },
  { re: buffer[baseOffset + 4], im: buffer[baseOffset + 5] },
];

const transportNeighbor = (
  link: Complex3x3,
  neighbor: [{ re: number; im: number }, { re: number; im: number }, { re: number; im: number }],
): [{ re: number; im: number }, { re: number; im: number }, { re: number; im: number }] =>
  su3_applyToVector(link, neighbor);

const siteOffset = (
  lattice: GaugeLattice,
  coord: { x: number; y: number; z: number; t: number },
): number => {
  const { width, height, depth, temporalExtent } = lattice;
  const index = (((coord.t * depth + coord.z) * height + coord.y) * width + coord.x) * 6;
  return index;
};

export const applyDiracStub = (
  operator: DiracStubOperator,
  input: Float64Array,
  output: Float64Array,
): void => {
  assertVectorLength(operator, input);
  assertVectorLength(operator, output);
  output.fill(0);
  const { lattice, mass, kappa, axes } = operator;
  const { width, height, depth, temporalExtent } = lattice;

  for (let t = 0; t < temporalExtent; t++) {
    for (let z = 0; z < depth; z++) {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const coord = { x, y, z, t };
          const baseOffset = siteOffset(lattice, coord);
          const psi = readVector(input, baseOffset);
          applyContribution(output, baseOffset, psi, mass);

          for (const axis of axes) {
            const forwardCoord = lattice.shiftCoordinate(coord, axis, 1);
            const forwardOffset = siteOffset(lattice, forwardCoord);
            const forwardVector = readVector(input, forwardOffset);
            const forwardLink = lattice.getLinkMatrix(coord.x, coord.y, axis, coord.z, coord.t);
            const forwardContribution = transportNeighbor(forwardLink, forwardVector);
            applyContribution(output, baseOffset, forwardContribution, -kappa);

            const backwardCoord = lattice.shiftCoordinate(coord, axis, -1);
            const backwardOffset = siteOffset(lattice, backwardCoord);
            const backwardVector = readVector(input, backwardOffset);
            const backwardLink = lattice.getLinkMatrix(
              backwardCoord.x,
              backwardCoord.y,
              axis,
              backwardCoord.z,
              backwardCoord.t,
            );
            const backwardContribution = transportNeighbor(
              su3_conjugateTranspose(backwardLink),
              backwardVector,
            );
            applyContribution(output, baseOffset, backwardContribution, -kappa);
          }
        }
      }
    }
  }
};

const dotReal = (lhs: Float64Array, rhs: Float64Array): number => {
  let sum = 0;
  for (let i = 0; i < lhs.length; i += 2) {
    sum += lhs[i] * rhs[i] + lhs[i + 1] * rhs[i + 1];
  }
  return sum;
};

const axpy = (target: Float64Array, vector: Float64Array, scale: number): void => {
  for (let i = 0; i < target.length; i++) {
    target[i] += vector[i] * scale;
  }
};

export const solveDiracStub = (
  operator: DiracStubOperator,
  source: Float64Array,
  options: DiracSolveOptions = {},
): DiracStubSolution => {
  assertVectorLength(operator, source);
  const tolerance = options.tolerance ?? 1e-8;
  const maxIterations = Math.max(1, options.maxIterations ?? operator.lattice.siteCount * 8);
  const size = source.length;

  const solution = options.initialGuess
    ? Float64Array.from(options.initialGuess)
    : new Float64Array(size);
  const residual = new Float64Array(size);
  const direction = new Float64Array(size);
  const matVec = new Float64Array(size);

  // r = b - A x
  applyDiracStub(operator, solution, matVec);
  for (let i = 0; i < size; i++) {
    residual[i] = source[i] - matVec[i];
    direction[i] = residual[i];
  }
  let residualSq = dotReal(residual, residual);
  let residualNorm = Math.sqrt(Math.max(residualSq, 0));
  if (residualNorm <= tolerance) {
    return {
      solution,
      iterations: 0,
      residual: residualNorm,
      converged: true,
    };
  }

  let iterations = 0;
  let converged = false;

  while (iterations < maxIterations) {
    applyDiracStub(operator, direction, matVec);
    const denom = dotReal(direction, matVec);
    if (Math.abs(denom) <= EPSILON) {
      break;
    }
    const alpha = residualSq / denom;
    axpy(solution, direction, alpha);
    axpy(residual, matVec, -alpha);
    const nextResidualSq = dotReal(residual, residual);
    residualNorm = Math.sqrt(Math.max(nextResidualSq, 0));
    iterations += 1;
    if (residualNorm <= tolerance) {
      converged = true;
      break;
    }
    const beta = nextResidualSq / residualSq;
    for (let i = 0; i < size; i++) {
      direction[i] = residual[i] + direction[i] * beta;
    }
    residualSq = nextResidualSq;
  }

  return {
    solution,
    iterations,
    residual: residualNorm,
    converged,
  };
};
