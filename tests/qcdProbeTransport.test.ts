import test from 'node:test';
import assert from 'node:assert/strict';

import { GaugeLattice } from '../src/qcd/lattice.js';
import { initializeGaugeField } from '../src/qcd/updateCpu.js';
import {
  su3_applyToVector,
  su3_conjugateTranspose,
  su3_mul,
  su3_haar,
  type Complex3x3,
} from '../src/qcd/su3.js';
import {
  transportProbe,
  createPlaquettePath,
  createProbeBasisVector,
  buildProbeTransportVisualization,
  type ProbeColorVector,
} from '../src/qcd/probeTransport.js';

const mulberry32 = (seed: number): (() => number) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let c = Math.imul(t ^ (t >>> 15), 1 | t);
    c ^= c + Math.imul(c ^ (c >>> 7), 61 | c);
    return ((c ^ (c >>> 14)) >>> 0) / 4294967296;
  };
};

const siteKey = (coord: { x: number; y: number; z: number; t: number }): string =>
  `${coord.x}:${coord.y}:${coord.z}:${coord.t}`;

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
          transforms.set(siteKey({ x, y, z, t }), su3_haar(1, rng));
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
          const gSite = transforms.get(siteKey(coord)) ?? identityMatrix();
          for (const axis of lattice.axes) {
            const neighbor = lattice.shiftCoordinate(coord, axis, 1);
            const gNeighbor = transforms.get(siteKey(neighbor)) ?? identityMatrix();
            const link = lattice.getLinkMatrix(x, y, axis, z, t);
            const transformed = su3_mul(su3_mul(gSite, link), su3_conjugateTranspose(gNeighbor));
            lattice.setLinkMatrix(x, y, axis, transformed, z, t);
          }
        }
      }
    }
  }
};

const normalizeVector = (vector: ProbeColorVector): ProbeColorVector => {
  const norm =
    Math.sqrt(vector.reduce((sum, entry) => sum + entry.re * entry.re + entry.im * entry.im, 0)) ||
    1;
  return [
    { re: vector[0].re / norm, im: vector[0].im / norm },
    { re: vector[1].re / norm, im: vector[1].im / norm },
    { re: vector[2].re / norm, im: vector[2].im / norm },
  ];
};

const randomProbeVector = (rng: () => number): ProbeColorVector =>
  normalizeVector([
    { re: rng() * 2 - 1, im: rng() * 2 - 1 },
    { re: rng() * 2 - 1, im: rng() * 2 - 1 },
    { re: rng() * 2 - 1, im: rng() * 2 - 1 },
  ]);

const vectorMaxError = (lhs: ProbeColorVector, rhs: ProbeColorVector): number => {
  let max = 0;
  for (let i = 0; i < 3; i++) {
    const diffRe = lhs[i].re - rhs[i].re;
    const diffIm = lhs[i].im - rhs[i].im;
    const mag = Math.hypot(diffRe, diffIm);
    if (mag > max) {
      max = mag;
    }
  }
  return max;
};

test('probe transport is gauge covariant along closed path', () => {
  const lattice = new GaugeLattice({ width: 4, height: 4 });
  initializeGaugeField(lattice, 'hot', mulberry32(512));
  const origin = { x: 1, y: 2, z: 0, t: 0 };
  const path = [
    { axis: 'x', direction: 1, span: 2 },
    { axis: 'y', direction: 1, span: 1 },
    { axis: 'x', direction: -1, span: 2 },
    { axis: 'y', direction: -1, span: 1 },
  ] as const;
  const probe = randomProbeVector(mulberry32(90));
  const baseline = transportProbe({ lattice, origin, path, vector: probe });

  const transforms = buildGaugeTransforms(lattice, mulberry32(1234));
  const gOrigin = transforms.get(siteKey(origin)) ?? identityMatrix();
  applyGaugeToLattice(lattice, transforms);
  const transformedProbe = su3_applyToVector(gOrigin, probe);
  const transported = transportProbe({ lattice, origin, path, vector: transformedProbe });

  const endKey = siteKey(baseline.end.coord);
  const gEnd = transforms.get(endKey) ?? identityMatrix();
  const expectedEnd = su3_applyToVector(gEnd, baseline.end.vector);
  const error = vectorMaxError(expectedEnd, transported.end.vector);
  assert.ok(error <= 1e-7, `gauge covariance violated at end point (error ${error})`);
});

test('probe visualization returns closed loop for paired sources', () => {
  const lattice = new GaugeLattice({ width: 6, height: 5 });
  lattice.fillIdentity();
  const sources = [
    { x: 1, y: 2, charge: 1 },
    { x: 4, y: 3, charge: -1 },
  ] as const;
  const frame = buildProbeTransportVisualization(lattice, sources);
  assert.ok(frame, 'probe visualization should be available');
  assert.ok(frame!.segments.length > 0, 'expected non-empty segments');
  assert.equal(frame!.nodes.length, frame!.segments.length + 1, 'nodes/segments mismatch');
  assert.equal(frame!.closed, true, 'expected path to close after return segment');
  const basisFrame = buildProbeTransportVisualization(lattice, []);
  assert.ok(basisFrame, 'fallback plaquette should render');
  assert.equal(
    basisFrame!.segments.length,
    createPlaquettePath(lattice.axes).length,
    'plaquette step count mismatch',
  );
});
