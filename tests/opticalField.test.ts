import test from 'node:test';
import assert from 'node:assert/strict';

import { OPTICAL_FIELD_SCHEMA_VERSION, OpticalFieldManager } from '../src/fields/opticalField.js';

test('OpticalFieldManager stamps monotonic frame IDs and reuses buffers', () => {
  const manager = new OpticalFieldManager({
    solver: 'unit',
    resolution: { width: 4, height: 4 },
    initialFrameId: 0,
  });

  const frame0 = manager.acquireFrame();
  assert.equal(frame0.getMeta().schemaVersion, OPTICAL_FIELD_SCHEMA_VERSION);
  assert.equal(frame0.getMeta().frameId, -1);
  assert.equal(frame0.componentCount, 1);
  assert.equal(frame0.getMeta().componentCount, 1);

  manager.stampFrame(frame0, { dt: 0.01, timestamp: 1 });
  assert.equal(frame0.getMeta().frameId, 0);
  assert.equal(frame0.getMeta().componentCount, 1);

  manager.releaseFrame(frame0);

  const frame1 = manager.acquireFrame({ timestamp: 2 });
  assert.equal(frame1.buffer, frame0.buffer, 'expected buffer reuse from pool');
  assert.equal(frame1.componentCount, 1);
  assert.equal(frame1.getMeta().componentCount, 1);
  manager.stampFrame(frame1, { dt: 0.01, timestamp: 2 });
  assert.equal(frame1.getMeta().frameId, 1);
});

test('phase alignment rotates complex field and notifies hooks', () => {
  const manager = new OpticalFieldManager({
    solver: 'align',
    resolution: { width: 2, height: 2 },
    initialFrameId: -1,
  });
  const frame = manager.acquireFrame();
  frame.real.fill(0);
  frame.imag.fill(0);
  frame.real[0] = 0;
  frame.imag[0] = 1; // +pi/2
  frame.real[1] = 1;
  frame.imag[1] = 0;
  manager.stampFrame(frame);

  let observedDelta = 0;
  manager.registerPhaseHook(({ phaseDelta }) => {
    observedDelta = phaseDelta;
  });

  const originalDiff = frame.getPhase(1) - frame.getPhase(0);
  const delta = manager.alignPhase(frame, { anchorIndex: 0, referencePhase: 0 });
  assert.ok(Math.abs(delta + Math.PI / 2) < 1e-6);
  assert.ok(Math.abs(observedDelta + Math.PI / 2) < 1e-6);
  assert.ok(Math.abs(frame.getPhase(0)) < 1e-6, 'anchor should align to reference');
  const rotatedDiff = frame.getPhase(1) - frame.getPhase(0);
  const diffError = Math.atan2(
    Math.sin(rotatedDiff - originalDiff),
    Math.cos(rotatedDiff - originalDiff),
  );
  assert.ok(Math.abs(diffError) < 1e-6, 'global rotation preserves relative phase');
});

test('OpticalFieldManager can allocate multi-component frames', () => {
  const manager = new OpticalFieldManager({
    solver: 'polar',
    resolution: { width: 2, height: 2 },
    componentCount: 2,
  });
  const frame = manager.acquireFrame();
  assert.equal(frame.componentCount, 2);
  assert.equal(frame.components.length, 2);
  frame.components[0]!.real[0] = 1;
  frame.components[0]!.imag[0] = 0;
  frame.components[1]!.real[0] = 0;
  frame.components[1]!.imag[0] = 1;
  manager.stampFrame(frame);
  assert.equal(frame.getMeta().componentCount, 2);
  assert.ok(Math.abs(frame.getPhase(0)) < 1e-6);
  assert.ok(Math.abs(frame.getPhase(0, 1) - Math.PI / 2) < 1e-6);
  manager.releaseFrame(frame);
  const reuse = manager.acquireFrame();
  assert.equal(reuse.componentCount, 2);
  assert.equal(reuse.getMeta().componentCount, 2);
  manager.releaseFrame(reuse);
});
