import test from 'node:test';
import assert from 'node:assert/strict';

import {
  mergeGateAppends,
  sanitizeGateList,
  createDefaultSu7RuntimeParams,
  SU7_MAX_ACTIVE_GATES,
  type Gate,
  type GateList,
  type GatePhaseVector,
} from '../src/pipeline/su7/types.js';
import { createSu7GateList, createSu7GateListSnapshot } from '../src/pipeline/su7/math.js';
import { hashCanonicalJson } from '../src/serialization/canonicalJson.js';

const VECTOR_DIM = 7;
const TAU = Math.PI * 2;

const wrapAngle = (theta: number): number => {
  if (!Number.isFinite(theta)) return 0;
  let t = theta % TAU;
  if (t <= -Math.PI) {
    t += TAU;
  } else if (t > Math.PI) {
    t -= TAU;
  }
  return t;
};

const clampAxis = (axis: number): number => {
  if (!Number.isFinite(axis)) return 0;
  const wrapped = Math.trunc(axis);
  return ((wrapped % VECTOR_DIM) + VECTOR_DIM) % VECTOR_DIM;
};

const normalizePhaseAngles = (angles: number[]): number[] => {
  if (angles.length === 0) return angles;
  const sum = angles.reduce((acc, value) => acc + value, 0);
  const mean = sum / angles.length;
  for (let i = 0; i < angles.length; i++) {
    angles[i] = wrapAngle(angles[i] - mean);
  }
  return angles;
};

const toGatePhaseVector = (values: number[]): GatePhaseVector =>
  values.map((value) => wrapAngle(value)) as GatePhaseVector;

const computeExpectedGains = (base: GateList, appends: readonly Gate[]) => {
  const phases = Array.from(base.gains.phaseAngles);
  const pulses = Array.from(base.gains.pulseAngles);

  let weight = pulses.reduce((acc, value) => acc + Math.abs(value), 0);
  if (!Number.isFinite(weight) || weight <= 1e-6) {
    weight = 1;
  }
  let vecX = Math.cos(base.gains.chiralityPhase) * weight;
  let vecY = Math.sin(base.gains.chiralityPhase) * weight;

  for (const gate of appends) {
    if (gate.kind === 'phase') {
      for (let axis = 0; axis < VECTOR_DIM; axis++) {
        phases[axis] += gate.phases[axis];
      }
    } else {
      const axis = clampAxis(gate.axis);
      pulses[axis] += gate.theta;
      const magnitude = Math.abs(gate.theta);
      if (magnitude > 1e-6) {
        vecX += Math.cos(gate.phase) * magnitude;
        vecY += Math.sin(gate.phase) * magnitude;
        weight += magnitude;
      }
    }
  }

  normalizePhaseAngles(phases);
  const chiralityPhase = Math.atan2(vecY, vecX);

  return {
    baseGain: base.gains.baseGain,
    phaseAngles: toGatePhaseVector(phases),
    pulseAngles: toGatePhaseVector(pulses),
    chiralityPhase: wrapAngle(chiralityPhase),
  };
};

const buildPulseGate = (axis: number, theta: number, phase: number, label: string): Gate => ({
  kind: 'pulse',
  axis,
  theta,
  phase,
  label,
});

const createTestAppends = (count: number): Gate[] => {
  const gates: Gate[] = [];
  for (let i = 0; i < count; i++) {
    const axis = i % VECTOR_DIM;
    const theta = 0.05 + (i % 5) * 0.01;
    const phase = wrapAngle(-Math.PI + ((i * 0.37) % (2 * Math.PI)));
    gates.push(buildPulseGate(axis, theta, phase, `pulse-${i}`));
  }
  return gates;
};

test('mergeGateAppends enforces the active gate limit and reports squashed count', () => {
  const params = createDefaultSu7RuntimeParams();
  const base = createSu7GateList(params);
  const baseGateCount = base.gates.length;
  const capacity = Math.max(SU7_MAX_ACTIVE_GATES - baseGateCount, 0);
  const requested = capacity + 128;
  const appends = createTestAppends(requested);

  const merged = mergeGateAppends(base, appends);

  assert.ok(
    merged.gates.length <= SU7_MAX_ACTIVE_GATES,
    `expected at most ${SU7_MAX_ACTIVE_GATES} gates, got ${merged.gates.length}`,
  );
  const expectedSquashed = Math.max(requested - capacity, 0);
  assert.equal(merged.squashedAppends, expectedSquashed);

  const retained = merged.gates.slice(baseGateCount);
  assert.equal(
    retained.length,
    Math.min(requested, capacity),
    'retained appends length does not match capacity',
  );
  const expectedLabels = appends.slice(-retained.length).map((gate) => gate.label);
  const retainedLabels = retained.map((gate) => gate.label);
  assert.deepEqual(retainedLabels, expectedLabels);
});

test('gate squashing accumulates gains identical to unsquashed evaluation', () => {
  const params = createDefaultSu7RuntimeParams();
  const base = createSu7GateList(params);
  const appends = sanitizeGateList(createTestAppends(384));
  const merged = mergeGateAppends(base, appends);
  const expectedGains = computeExpectedGains(base, appends);

  for (let axis = 0; axis < VECTOR_DIM; axis++) {
    assert.ok(
      Math.abs(merged.gains.phaseAngles[axis] - expectedGains.phaseAngles[axis]) <= 1e-9,
      `phase[${axis}] mismatch`,
    );
    assert.ok(
      Math.abs(merged.gains.pulseAngles[axis] - expectedGains.pulseAngles[axis]) <= 1e-9,
      `pulse[${axis}] mismatch`,
    );
  }
  assert.ok(
    Math.abs(merged.gains.chiralityPhase - expectedGains.chiralityPhase) <= 1e-9,
    'chirality phase mismatch',
  );
});

test('gate squashing leaves canonical gate snapshots unchanged', () => {
  const params = createDefaultSu7RuntimeParams();
  const baselineSnapshot = createSu7GateListSnapshot(params);
  const { hash: baselineHash } = hashCanonicalJson(baselineSnapshot);

  const base = createSu7GateList(params);
  const appends = createTestAppends(320);
  const merged = mergeGateAppends(base, appends);
  assert.ok(merged.squashedAppends >= 0, 'expected squashed appends to be non-negative');

  const afterSnapshot = createSu7GateListSnapshot(params);
  const { hash: afterHash } = hashCanonicalJson(afterSnapshot);
  assert.equal(afterHash, baselineHash, 'canonical snapshot hash should remain stable');
});
