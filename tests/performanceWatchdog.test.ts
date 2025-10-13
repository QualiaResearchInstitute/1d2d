import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PerformanceWatchdog,
  type PerformanceBudget,
  type PerformanceWatchdogOptions,
} from '../scripts/lib/performanceWatchdog.js';

type ProviderSequences = {
  now: bigint[];
  cpu: NodeJS.CpuUsage[];
  memory: NodeJS.MemoryUsage[];
};

const createProvider = (sequences: ProviderSequences) => ({
  now: () => {
    const value = sequences.now.shift();
    if (value == null) throw new Error('Ran out of now() samples');
    return value;
  },
  cpu: () => {
    const value = sequences.cpu.shift();
    if (value == null) throw new Error('Ran out of cpu() samples');
    return value;
  },
  memory: () => {
    const value = sequences.memory.shift();
    if (value == null) throw new Error('Ran out of memory() samples');
    return value;
  },
});

const makeMemory = (rssMb: number, heapMb: number): NodeJS.MemoryUsage =>
  ({
    rss: rssMb * 1024 * 1024,
    heapTotal: heapMb * 1024 * 1024,
    heapUsed: heapMb * 1024 * 1024,
    external: 0,
    arrayBuffers: 0,
  }) as NodeJS.MemoryUsage;

const createWatchdog = (
  budget: PerformanceBudget,
  providerSequences: ProviderSequences,
  options?: PerformanceWatchdogOptions,
) => new PerformanceWatchdog(budget, options, createProvider(providerSequences));

test('PerformanceWatchdog records samples within budget', () => {
  const sequences: ProviderSequences = {
    now: [0n, 16_000_000n, 16_000_000n, 32_000_000n],
    cpu: [
      { user: 0, system: 0 },
      { user: 4000, system: 1000 },
      { user: 4000, system: 1000 },
      { user: 8000, system: 2000 },
    ],
    memory: [
      makeMemory(220, 150),
      makeMemory(225, 152),
      makeMemory(225, 152),
      makeMemory(230, 155),
    ],
  };
  const watchdog = createWatchdog(
    { frameMs: 20, rssMb: 500, heapMb: 250, cpuPercent: 200 },
    sequences,
  );

  watchdog.beginFrame(0);
  const firstSample = watchdog.endFrame();
  watchdog.beginFrame(1);
  const secondSample = watchdog.endFrame();

  assert.equal(firstSample.frameIndex, 0);
  assert.equal(secondSample.frameIndex, 1);
  assert.ok(firstSample.frameMs > 15.9 && firstSample.frameMs < 16.1);
  assert.ok(firstSample.cpuPercent > 30 && firstSample.cpuPercent < 35);

  const snapshot = watchdog.snapshot();
  assert.equal(snapshot.frames, 2);
  assert.ok(snapshot.frameMsAvg > 15.9 && snapshot.frameMsAvg < 16.1);
  assert.equal(snapshot.frameMsMax, firstSample.frameMs);
  assert.equal(snapshot.rssMbMax, 230);
  assert.equal(snapshot.heapMbMax, 155);
  assert.ok(snapshot.cpuPercentAvg > 30 && snapshot.cpuPercentAvg < 35);
  assert.equal(snapshot.violations.length, 0);
  assert.deepEqual(snapshot.lastSample, secondSample);
  assert.equal(snapshot.history.length, 2);
});

test('PerformanceWatchdog flags budget violations', () => {
  const sequences: ProviderSequences = {
    now: [0n, 20_000_000n],
    cpu: [
      { user: 0, system: 0 },
      { user: 25_000, system: 5_000 },
    ],
    memory: [makeMemory(600, 400), makeMemory(620, 410)],
  };
  const watchdog = createWatchdog(
    { frameMs: 12, rssMb: 500, heapMb: 300, cpuPercent: 150 },
    sequences,
    { tolerance: 0.05 },
  );

  watchdog.beginFrame(0);
  const sample = watchdog.endFrame();

  assert.ok(sample.frameMs > 19.9 && sample.frameMs < 20.1);
  const snapshot = watchdog.snapshot();
  assert.equal(snapshot.frames, 1);
  assert.equal(snapshot.violations.length, 3);
  const types = snapshot.violations.map((v) => v.type).sort();
  assert.deepEqual(types, ['frameMs', 'heapMb', 'rssMb']);
  const frameViolation = snapshot.violations.find((v) => v.type === 'frameMs');
  assert.ok(frameViolation);
  assert.equal(frameViolation?.frameIndex, 0);
});
