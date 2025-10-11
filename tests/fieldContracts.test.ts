import test from 'node:test';
import assert from 'node:assert/strict';

import {
  assertPhaseField,
  assertRimField,
  assertSurfaceField,
  assertVolumeField,
  makeResolution,
  type PhaseField,
  type RimField,
  type SurfaceField,
  type VolumeField,
} from '../src/fields/contracts.js';
import {
  createInitialStatuses,
  markFieldUnavailable,
  markFieldUpdate,
  refreshFieldStaleness,
} from '../src/fields/state.js';

const makeRimField = (width: number, height: number): RimField => ({
  kind: 'rim',
  resolution: makeResolution(width, height),
  gx: new Float32Array(width * height),
  gy: new Float32Array(width * height),
  mag: new Float32Array(width * height),
});

const makeSurfaceField = (width: number, height: number): SurfaceField => ({
  kind: 'surface',
  resolution: makeResolution(width, height),
  rgba: new Uint8ClampedArray(width * height * 4),
});

const makePhaseField = (width: number, height: number): PhaseField => ({
  kind: 'phase',
  resolution: makeResolution(width, height),
  gradX: new Float32Array(width * height),
  gradY: new Float32Array(width * height),
  vort: new Float32Array(width * height),
  coh: new Float32Array(width * height),
  amp: new Float32Array(width * height),
});

const makeVolumeField = (width: number, height: number): VolumeField => ({
  kind: 'volume',
  resolution: makeResolution(width, height),
  phase: new Float32Array(width * height),
  depth: new Float32Array(width * height),
  intensity: new Float32Array(width * height),
});

test('assertRimField accepts matching buffers', () => {
  const field = makeRimField(4, 4);
  assert.doesNotThrow(() => assertRimField(field, 'test'));
});

test('assertRimField rejects mismatched buffers', () => {
  const field = makeRimField(4, 4);
  field.gx = new Float32Array(3);
  assert.throws(() => assertRimField(field as RimField, 'test'));
});

test('surface field guard enforces rgba length', () => {
  const good = makeSurfaceField(2, 2);
  assert.doesNotThrow(() => assertSurfaceField(good, 'test'));
  const bad = { ...good, rgba: new Uint8ClampedArray(3) } as SurfaceField;
  assert.throws(() => assertSurfaceField(bad, 'bad'));
});

test('phase and volume assertions verify channels', () => {
  const phase = makePhaseField(3, 3);
  assert.doesNotThrow(() => assertPhaseField(phase, 'phase'));
  const brokenPhase = { ...phase, vort: new Float32Array(2) } as PhaseField;
  assert.throws(() => assertPhaseField(brokenPhase, 'broken'));

  const volume = makeVolumeField(2, 2);
  assert.doesNotThrow(() => assertVolumeField(volume, 'vol'));
  const brokenVolume = { ...volume, intensity: new Float32Array(1) } as VolumeField;
  assert.throws(() => assertVolumeField(brokenVolume, 'broken-vol'));
});

test('field staleness lifecycle', () => {
  const statuses = createInitialStatuses();
  const resolution = makeResolution(4, 4);
  const now = 100;
  const fresh = markFieldUpdate(statuses, 'phase', resolution, 'cpu', now);
  assert.equal(fresh.phase.available, true);
  const { next, changes } = refreshFieldStaleness(fresh, now + 500);
  assert.equal(next.phase.stale, true);
  assert.ok(
    changes.some(
      (change: { kind: string; becameStale: boolean }) =>
        change.kind === 'phase' && change.becameStale,
    ),
  );
  const cleared = markFieldUnavailable(next, 'phase', 'cpu', now + 600);
  assert.equal(cleared.phase.available, false);
  const { next: recovered } = refreshFieldStaleness(cleared, now + 650);
  assert.equal(recovered.phase.stale, false);
});
