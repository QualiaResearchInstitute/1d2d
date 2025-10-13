import test from 'node:test';
import assert from 'node:assert/strict';

import { createSu7GateList, type Su7ScheduleContext } from '../src/pipeline/su7/math.js';
import { createDefaultSu7RuntimeParams, type Su7Schedule } from '../src/pipeline/su7/types.js';

test('DAW-style lanes yield deterministic gate lists with scoped PRNG', () => {
  const schedule: Su7Schedule = [
    { gain: 0.45, spread: 0.9, lane: 'main', time: 0.1, label: 'phase-main' },
    { gain: 0.32, spread: 1.25, lane: 'texture', time: 1.1, label: 'phase-texture' },
    {
      gain: 0.4,
      spread: 1,
      lane: 'macro',
      macro: true,
      index: 2,
      phase: 0.3,
      time: 0.5,
      label: 'macro-left',
    },
    {
      gain: -0.35,
      spread: 0.85,
      lane: 'macro',
      macro: true,
      index: 5,
      phase: -0.2,
      time: 1.4,
      label: 'macro-right',
    },
  ];
  const params = {
    ...createDefaultSu7RuntimeParams(),
    enabled: true,
    gain: 1.1,
    seed: 8282,
    schedule,
  };
  const first = createSu7GateList(params);
  const second = createSu7GateList(params);
  assert.deepEqual(second, first);

  const reordered = createSu7GateList({
    ...params,
    schedule: [...schedule].reverse(),
  });
  assert.deepEqual(reordered.gains, first.gains);
  assert.deepEqual(reordered.gates, first.gates);

  const macroLeft = first.gates.find(
    (gate) => gate.kind === 'pulse' && gate.label === 'macro-left',
  );
  assert.ok(
    macroLeft && macroLeft.kind === 'pulse',
    'expected macro-left gate in deterministic output',
  );
  assert.equal(first.gains.pulseAngles[2], macroLeft.theta);
  const macroRight = first.gates.find(
    (gate) => gate.kind === 'pulse' && gate.label === 'macro-right',
  );
  assert.ok(
    macroRight && macroRight.kind === 'pulse',
    'expected macro-right gate in deterministic output',
  );
  assert.equal(first.gains.pulseAngles[5], macroRight.theta);
});

test('advanced schedule lanes remain deterministic and respond to macro changes', () => {
  const schedule: Su7Schedule = [
    { gain: 0.38, spread: 0.85, lane: 'main', time: 0.2, label: 'phase-0' },
    { gain: 0.27, spread: 1.05, lane: 'ambient', time: 0.85, label: 'phase-ambient', index: 4 },
    {
      gain: 0.24,
      spread: 0.95,
      lane: 'macro',
      macro: true,
      index: 1,
      phase: 0.15,
      time: 0.55,
      label: 'macro-beta',
    },
  ];
  const params = {
    ...createDefaultSu7RuntimeParams(),
    enabled: true,
    gain: 1.05,
    seed: 5678,
    schedule,
  };
  const context: Su7ScheduleContext = {
    dmt: 0.45,
    arousal: 0.6,
    flow: {
      angle: Math.PI / 6,
      magnitude: 0.82,
      coherence: 0.7,
      axisBias: new Float32Array([1.1, 0.95, 0.85, 1.2, 0.9, 1.05, 0.88]),
      gridSize: 2,
      gridVectors: new Float32Array([0.2, 0.1, -0.12, 0.06, 0.08, -0.04, 0.03, 0.18]),
    },
    curvatureStrength: 0.3,
    parallaxRadial: 0.25,
    volumeCoverage: 0.35,
  };

  const first = createSu7GateList(params, context);
  const second = createSu7GateList(params, context);
  assert.deepEqual(second.gains, first.gains);
  assert.deepEqual(second.gates, first.gates);

  const macroGate = first.gates.find(
    (gate) => gate.kind === 'pulse' && gate.label === 'macro-beta',
  );
  assert.ok(macroGate && macroGate.kind === 'pulse', 'expected macro-beta gate');

  const boostedSchedule = schedule.map((stage) =>
    stage.label === 'macro-beta' ? { ...stage, gain: stage.gain * 1.5 } : stage,
  );
  const boosted = createSu7GateList(
    {
      ...params,
      schedule: boostedSchedule,
    },
    context,
  );
  const boostedGate = boosted.gates.find(
    (gate) => gate.kind === 'pulse' && gate.label === 'macro-beta',
  );
  assert.ok(boostedGate && boostedGate.kind === 'pulse', 'expected boosted macro-beta gate');
  assert.ok(
    Math.abs(boostedGate.theta) > Math.abs(macroGate.theta),
    'macro gate did not respond to gain adjustment',
  );
});
