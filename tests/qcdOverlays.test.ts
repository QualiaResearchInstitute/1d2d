import test from 'node:test';
import assert from 'node:assert/strict';

import {
  computeFluxOverlayState,
  buildFluxSu3Block,
  fluxEnergyToOverlayColor,
} from '../src/qcd/overlays.js';
import { computeDeterminant, su3_embed } from '../src/pipeline/su7/math.js';

test('computeFluxOverlayState highlights flux tube between paired sources', () => {
  const state = computeFluxOverlayState({
    width: 32,
    height: 32,
    sources: [
      { x: 8, y: 16, charge: 1 },
      { x: 24, y: 16, charge: -1 },
    ],
  });
  assert.ok(state, 'overlay state missing');
  const { energy, direction, energyScale } = state!;
  assert.equal(energy.length, 32 * 32);
  assert.equal(direction.length, 32 * 32 * 2);
  assert.ok(energyScale > 0);

  const centerIndex = 16 * 32 + 16;
  const edgeIndex = 0;
  assert.ok(
    energy[centerIndex] > energy[edgeIndex],
    'center energy does not exceed boundary energy',
  );
  const dirIndex = centerIndex * 2;
  const dirMagnitude = Math.hypot(direction[dirIndex], direction[dirIndex + 1]);
  assert.ok(dirMagnitude <= 1 + 1e-6, 'direction vectors must be normalized');
});

test('buildFluxSu3Block embeds into SU(7) with negligible determinant drift', () => {
  const block = buildFluxSu3Block(0.75, Math.PI / 3);
  const embedded = su3_embed(block);
  const det = computeDeterminant(embedded);
  const drift = Math.hypot(det.re - 1, det.im);
  assert.ok(drift < 1e-7, `determinant drift ${drift} exceeds tolerance`);
});

test('fluxEnergyToOverlayColor returns bounded RGB values', () => {
  const color = fluxEnergyToOverlayColor(0.65, Math.PI / 4);
  color.forEach((component, index) => {
    assert.ok(component >= 0 && component <= 1, `component ${index} out of range (${component})`);
  });
});
