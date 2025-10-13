import test from 'node:test';
import assert from 'node:assert/strict';

import {
  enforceUnitaryGuardrail,
  detectFlickerGuardrail,
  phase_gate,
} from '../src/pipeline/su7/math.js';
import { projectSu7Vector } from '../src/pipeline/su7/projector.js';
import { cloneComplex7x7, type C7Vector } from '../src/pipeline/su7/types.js';

const buildVector = (entries: number[]): { vector: C7Vector; norm: number } => {
  const normSq = entries.reduce((sum, value) => sum + value * value, 0);
  const norm = Math.sqrt(normSq);
  const inv = norm > 0 ? 1 / norm : 1;
  const vector = entries.map((value) => ({ re: value * inv, im: 0 })) as C7Vector;
  return { vector, norm };
};

test('unitary guardrail re-orthonormalizes matrices above threshold', () => {
  const identity = phase_gate([0, 0, 0, 0, 0, 0, 0]);
  const perturbed = cloneComplex7x7(identity);
  perturbed[0][0].re += 1e-3;
  perturbed[1][0].re += 5e-4;

  const { unitaryError, event } = enforceUnitaryGuardrail(perturbed);
  assert.ok(event, 'expected guardrail event');
  assert.equal(event!.kind, 'autoReorthon');
  assert.ok(event!.before > 1e-6, `expected before > threshold, got ${event!.before}`);
  assert.ok(unitaryError <= 1e-6, `expected corrected error <= threshold, got ${unitaryError}`);
  assert.ok(!event!.forced, 'auto guardrail should not mark forced');
});

test('unitary guardrail honors manual force', () => {
  const identity = phase_gate([0, 0, 0, 0, 0, 0, 0]);
  const { event } = enforceUnitaryGuardrail(identity, { force: true });
  assert.ok(event, 'forced guardrail should emit event');
  assert.equal(event!.kind, 'autoReorthon');
  assert.equal(event!.forced, true);
});

test('projector auto-gain keeps energy near target', () => {
  const { vector, norm } = buildVector([0.6, 0.55, 0.4, 0.1, 0.05, 0.02, 0.03]);
  const baseColor: [number, number, number] = [0.9, 0.9, 0.9];
  const projection = projectSu7Vector({
    vector,
    norm,
    projector: { id: 'identity', weight: 1 },
    gain: 1.8,
    frameGain: 1,
    baseColor,
  });
  assert.ok(projection, 'expected projection result');
  assert.ok(projection.guardrailEvent, 'expected auto-gain guardrail event');
  const event = projection.guardrailEvent!;
  const energy = projection.energy;
  assert.ok(
    Math.abs(energy - event.target) <= event.target * 0.05 + 1e-6,
    `energy ${energy} deviates more than 5% from target ${event.target}`,
  );
  assert.ok(event.after > 0, 'corrected energy should be positive');
});

test('flicker guardrail warns on high-frequency swings', () => {
  const event = detectFlickerGuardrail(0.45, 0.75, 16);
  assert.ok(event, 'expected flicker event');
  assert.equal(event!.kind, 'flicker');
  assert.ok(event!.frequencyHz >= 30, 'expected frequency above 30Hz');
});

test('flicker guardrail ignores low-energy regimes', () => {
  const event = detectFlickerGuardrail(0.01, 0.015, 16, { minEnergy: 0.05 });
  assert.equal(event, null);
});
