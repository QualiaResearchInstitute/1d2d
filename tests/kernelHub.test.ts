import test from 'node:test';
import assert from 'node:assert/strict';

import { KernelSpecHub, type KernelSpecEvent } from '../src/kernel/kernelHub.js';
import { createKernelSpec } from '../src/kernel/kernelSpec.js';

const flushMicrotasks = () => new Promise<void>((resolve) => queueMicrotask(resolve));

test('KernelSpecHub broadcasts identical specs to all subscribers', async () => {
  const hub = new KernelSpecHub();
  const events: Record<string, KernelSpecEvent> = {};
  hub.subscribe((event) => {
    events.angular = event;
  });
  hub.subscribe((event) => {
    events.kuramoto = event;
  });
  hub.subscribe((event) => {
    events.hyperbolic = event;
  });

  const next = createKernelSpec({
    gain: 2.4,
    k0: 0.18,
    Q: 3.6,
    anisotropy: 0.9,
    chirality: 1.1,
    transparency: 0.45,
  });
  const update = hub.replace(next, { source: 'test' });
  assert.notEqual(update, null);

  await flushMicrotasks();

  const solverKeys = ['angular', 'kuramoto', 'hyperbolic'] as const;
  solverKeys.forEach((solver) => {
    const event = events[solver];
    assert.ok(event, `${solver} missing event`);
    assert.deepEqual(event.spec, next, `${solver} spec mismatch`);
    assert.equal(event.version, update?.version, `${solver} version mismatch`);
  });
  const diagnostics = hub.getDiagnostics();
  assert.equal(diagnostics.subscriberCount, solverKeys.length);
  assert.equal(diagnostics.lastVersion, update?.version);
  assert.ok(diagnostics.lastDispatchLatency <= 16);
});

test('KernelSpecHub immediate subscription delivers current snapshot', () => {
  const initial = createKernelSpec({ gain: 1.8 });
  const hub = new KernelSpecHub(initial, 'bootstrap');
  let snapshot: KernelSpecEvent | null = null;
  hub.subscribe((event) => {
    snapshot = event;
  });
  assert.ok(snapshot);
  const event = snapshot!;
  assert.equal(event.source, 'bootstrap');
  assert.deepEqual(event.spec, createKernelSpec(initial));
});
