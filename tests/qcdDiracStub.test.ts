import test from 'node:test';
import assert from 'node:assert/strict';

import { GaugeLattice } from '../src/qcd/lattice.js';
import { initializeGaugeField } from '../src/qcd/updateCpu.js';
import { su3_mul, su3_conjugateTranspose, su3_haar, type Complex3x3 } from '../src/qcd/su3.js';
import {
  createDiracStubOperator,
  applyDiracStub,
  solveDiracStub,
  createDiracVector,
} from '../src/qcd/diracStub.js';

const mulberry32 = (seed: number): (() => number) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let c = Math.imul(t ^ (t >>> 15), 1 | t);
    c ^= c + Math.imul(c ^ (c >>> 7), 61 | c);
    return ((c ^ (c >>> 14)) >>> 0) / 4294967296;
  };
};

const siteKey = (x: number, y: number, z: number, t: number): string => `${x}:${y}:${z}:${t}`;

const identityMatrix = (): Complex3x3 => [
  [
    { re: 1, im: 0 },
    { re: 0, im: 0 },
    { re: 0, im: 0 },
  ],
  [
    { re: 0, im: 0 },
    { re: 1, im: 0 },
    { re: 0, im: 0 },
  ],
  [
    { re: 0, im: 0 },
    { re: 0, im: 0 },
    { re: 1, im: 0 },
  ],
];

const buildGaugeTransforms = (
  lattice: GaugeLattice,
  rng: () => number,
): Map<string, Complex3x3> => {
  const transforms = new Map<string, Complex3x3>();
  for (let t = 0; t < lattice.temporalExtent; t++) {
    for (let z = 0; z < lattice.depth; z++) {
      for (let y = 0; y < lattice.height; y++) {
        for (let x = 0; x < lattice.width; x++) {
          transforms.set(siteKey(x, y, z, t), su3_haar(1, rng));
        }
      }
    }
  }
  return transforms;
};

const applyGaugeToLattice = (lattice: GaugeLattice, transforms: Map<string, Complex3x3>): void => {
  for (let t = 0; t < lattice.temporalExtent; t++) {
    for (let z = 0; z < lattice.depth; z++) {
      for (let y = 0; y < lattice.height; y++) {
        for (let x = 0; x < lattice.width; x++) {
          const coord = { x, y, z, t };
          const gSite = transforms.get(siteKey(x, y, z, t)) ?? identityMatrix();
          for (const axis of lattice.axes) {
            const neighbor = lattice.shiftCoordinate(coord, axis, 1);
            const gNeighbor =
              transforms.get(siteKey(neighbor.x, neighbor.y, neighbor.z, neighbor.t)) ??
              identityMatrix();
            const link = lattice.getLinkMatrix(x, y, axis, z, t);
            const transformed = su3_mul(su3_mul(gSite, link), su3_conjugateTranspose(gNeighbor));
            lattice.setLinkMatrix(x, y, axis, transformed, z, t);
          }
        }
      }
    }
  }
};

const applyGaugeToVector = (
  lattice: GaugeLattice,
  vector: Float64Array,
  transforms: Map<string, Complex3x3>,
): Float64Array => {
  const result = new Float64Array(vector.length);
  for (let t = 0; t < lattice.temporalExtent; t++) {
    for (let z = 0; z < lattice.depth; z++) {
      for (let y = 0; y < lattice.height; y++) {
        for (let x = 0; x < lattice.width; x++) {
          const g = transforms.get(siteKey(x, y, z, t)) ?? identityMatrix();
          const base = (((t * lattice.depth + z) * lattice.height + y) * lattice.width + x) * 6;
          for (let row = 0; row < 3; row++) {
            let re = 0;
            let im = 0;
            for (let col = 0; col < 3; col++) {
              const coeff = g[row][col];
              const srcRe = vector[base + col * 2];
              const srcIm = vector[base + col * 2 + 1];
              re += coeff.re * srcRe - coeff.im * srcIm;
              im += coeff.re * srcIm + coeff.im * srcRe;
            }
            result[base + row * 2] = re;
            result[base + row * 2 + 1] = im;
          }
        }
      }
    }
  }
  return result;
};

const fillRandomDiracVector = (lattice: GaugeLattice, rng: () => number): Float64Array => {
  const vector = createDiracVector(lattice);
  for (let i = 0; i < vector.length; i++) {
    vector[i] = rng() * 2 - 1;
  }
  return vector;
};

const vectorDiffNorm = (lhs: Float64Array, rhs: Float64Array): number => {
  let sum = 0;
  for (let i = 0; i < lhs.length; i++) {
    const diff = lhs[i] - rhs[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
};

test('Dirac stub transport is gauge covariant', () => {
  const lattice = new GaugeLattice({ width: 3, height: 3 });
  initializeGaugeField(lattice, 'hot', mulberry32(2025));
  const operator = createDiracStubOperator(lattice, { mass: 3.8, kappa: 0.15 });
  const source = fillRandomDiracVector(lattice, mulberry32(319));
  const baseline = new Float64Array(source.length);
  applyDiracStub(operator, source, baseline);

  const transforms = buildGaugeTransforms(lattice, mulberry32(991));
  applyGaugeToLattice(lattice, transforms);
  const transformedOperator = createDiracStubOperator(lattice, { mass: 3.8, kappa: 0.15 });
  const sourceGauge = applyGaugeToVector(lattice, source, transforms);
  const transportedBaseline = applyGaugeToVector(lattice, baseline, transforms);
  const result = new Float64Array(source.length);
  applyDiracStub(transformedOperator, sourceGauge, result);

  const diff = vectorDiffNorm(result, transportedBaseline);
  assert.ok(diff <= 1e-9, `gauge covariance violated (‖Δ‖=${diff})`);
});

test('Dirac stub solver recovers known solution on tiny lattice', () => {
  const lattice = new GaugeLattice({ width: 2, height: 2 });
  lattice.fillIdentity();
  const operator = createDiracStubOperator(lattice, { mass: 3.2, kappa: 0.1 });
  const reference = fillRandomDiracVector(lattice, mulberry32(77));
  const source = new Float64Array(reference.length);
  applyDiracStub(operator, reference, source);

  const solution = solveDiracStub(operator, source, { tolerance: 1e-10, maxIterations: 128 });
  assert.ok(solution.converged, 'solver did not report convergence');
  const errorNorm = vectorDiffNorm(reference, solution.solution);
  assert.ok(errorNorm <= 1e-7, `solution mismatch (‖Δ‖=${errorNorm})`);
  assert.ok(solution.residual <= 1e-9, `residual too large (${solution.residual})`);
});
