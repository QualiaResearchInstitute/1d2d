import test from 'node:test';
import assert from 'node:assert/strict';

import { GaugeLattice, FLOATS_PER_MATRIX, type GaugeLatticeSnapshot } from '../src/qcd/lattice.js';
import type { Complex3x3 } from '../src/qcd/su3.js';
import {
  writeCanonicalJson,
  readCanonicalJson,
  hashCanonicalJson,
} from '../src/serialization/canonicalJson.js';

const f = (value: number) => Math.fround(value);

test('gauge lattice layout exports expected strides', () => {
  const lattice = new GaugeLattice({ width: 4, height: 3 });
  const expectedSiteStride = FLOATS_PER_MATRIX * lattice.axes.length;
  assert.equal(lattice.siteStride, expectedSiteStride);
  assert.equal(lattice.linkStride, FLOATS_PER_MATRIX);
  assert.equal(lattice.rowStride, 6);
  assert.equal(lattice.complexStride, 2);
  assert.equal(lattice.data.length, lattice.siteCount * expectedSiteStride);
  assert.deepEqual(lattice.axes, ['x', 'y']);
});

test('gauge lattice activates z/t axes when extents exceed one', () => {
  const lattice = new GaugeLattice({ width: 2, height: 2, depth: 3, temporalExtent: 2 });
  const expectedAxes = ['x', 'y', 'z', 't'];
  assert.deepEqual(lattice.axes, expectedAxes);
  const expectedStride = FLOATS_PER_MATRIX * expectedAxes.length;
  assert.equal(lattice.siteStride, expectedStride);
  assert.equal(lattice.data.length, lattice.siteCount * expectedStride);
  lattice.fillIdentity();
  const temporalLink = lattice.getLinkMatrix(0, 0, 't');
  assert.equal(temporalLink[0][0].re, 1);
});

test('setLinkMatrix round-trips complex entries through Float32 storage', () => {
  const lattice = new GaugeLattice({ width: 2, height: 1 });
  const matrixX: Complex3x3 = [
    [
      { re: 0.1, im: -0.2 },
      { re: -0.3, im: 0.4 },
      { re: 0.5, im: -0.6 },
    ],
    [
      { re: 0.7, im: 0.8 },
      { re: -0.9, im: 1.0 },
      { re: 1.1, im: -1.2 },
    ],
    [
      { re: -1.3, im: 1.4 },
      { re: 1.5, im: -1.6 },
      { re: -1.7, im: 1.8 },
    ],
  ];
  const matrixY: Complex3x3 = [
    [
      { re: -0.15, im: 0.25 },
      { re: 0.35, im: -0.45 },
      { re: -0.55, im: 0.65 },
    ],
    [
      { re: -0.75, im: 0.85 },
      { re: 0.95, im: -1.05 },
      { re: -1.15, im: 1.25 },
    ],
    [
      { re: 1.35, im: -1.45 },
      { re: -1.55, im: 1.65 },
      { re: 1.75, im: -1.85 },
    ],
  ];

  lattice.setLinkMatrix(0, 0, 'x', matrixX);
  lattice.setLinkMatrix(0, 0, 'y', matrixY);

  const roundTrippedX = lattice.getLinkMatrix(0, 0, 'x');
  const roundTrippedY = lattice.getLinkMatrix(0, 0, 'y');

  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      assert.equal(roundTrippedX[row][col].re, f(matrixX[row][col].re));
      assert.equal(roundTrippedX[row][col].im, f(matrixX[row][col].im));
      assert.equal(roundTrippedY[row][col].re, f(matrixY[row][col].re));
      assert.equal(roundTrippedY[row][col].im, f(matrixY[row][col].im));
    }
  }
});

test('gauge lattice snapshot round-trips through canonical JSON', () => {
  const lattice = new GaugeLattice({ width: 2, height: 2 });
  let value = -3.5;
  for (let y = 0; y < lattice.height; y++) {
    for (let x = 0; x < lattice.width; x++) {
      for (const axis of lattice.axes) {
        const slice = new Float32Array(FLOATS_PER_MATRIX);
        for (let i = 0; i < FLOATS_PER_MATRIX; i++) {
          slice[i] = Math.fround(value);
          value += 0.1375;
        }
        lattice.setLinkSlice(x, y, axis, slice);
      }
    }
  }

  const snapshot = lattice.snapshot();
  const canonicalJson = writeCanonicalJson(snapshot);
  const parsed = readCanonicalJson<GaugeLatticeSnapshot>(canonicalJson);
  const restored = GaugeLattice.restore(parsed);

  assert.deepEqual(parsed, snapshot);
  assert.deepEqual(Array.from(restored.data), snapshot.values);

  const rerendered = writeCanonicalJson(restored.snapshot());
  assert.equal(rerendered, canonicalJson);

  const expectedHash = 'a514db816c41b31c506eb4027c642d564d15671bda062135633f85891c555770';
  const { hash } = hashCanonicalJson(snapshot);
  assert.equal(hash, expectedHash);

  const { hash: restoredHash } = hashCanonicalJson(restored.snapshot());
  assert.equal(restoredHash, expectedHash);
});
