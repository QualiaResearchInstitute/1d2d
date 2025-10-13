import test from 'node:test';
import assert from 'node:assert/strict';

import { buildUnitary } from '../src/pipeline/su7/unitary.js';
import type { C7Vector, Complex } from '../src/pipeline/su7/types.js';
import {
  Su7GpuKernel,
  packSu7Unitary,
  packSu7Vectors,
  multiplySu7CpuPacked,
  SU7_GPU_KERNEL_VECTOR_STRIDE,
  Su7GpuKernelWarningEvent,
} from '../src/pipeline/su7/gpuKernel.js';

const VECTOR_COUNT = 256;
const EPSILON = 1e-6;

const buildRandomVector = (seed: number): C7Vector => {
  const rng = (() => {
    let t = seed >>> 0;
    return () => {
      t += 0x6d2b79f5;
      let c = Math.imul(t ^ (t >>> 15), 1 | t);
      c ^= c + Math.imul(c ^ (c >>> 7), 61 | c);
      return ((c ^ (c >>> 14)) >>> 0) / 4294967296;
    };
  })();
  const entries: Complex[] = [];
  for (let i = 0; i < 7; i++) {
    const angle = 2 * Math.PI * rng();
    const radius = Math.sqrt(-Math.log(Math.max(rng(), 1e-9)));
    entries.push({
      re: radius * Math.cos(angle),
      im: radius * Math.sin(angle),
    });
  }
  let normSq = 0;
  for (const entry of entries) {
    normSq += entry.re * entry.re + entry.im * entry.im;
  }
  const norm = Math.sqrt(normSq);
  return entries.map((entry) => ({ re: entry.re / norm, im: entry.im / norm })) as C7Vector;
};

const computeRms = (a: Float32Array, b: Float32Array): number => {
  assert.equal(a.length, b.length, 'vector length mismatch');
  let sumSq = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sumSq += diff * diff;
  }
  return Math.sqrt(sumSq / a.length);
};

test('su7 gpu kernel matches cpu packed multiply', async () => {
  const unitary = buildUnitary({ seed: 404 });
  const unitaryPacked = packSu7Unitary(unitary);
  const vectors: C7Vector[] = new Array(VECTOR_COUNT);
  for (let i = 0; i < VECTOR_COUNT; i++) {
    vectors[i] = buildRandomVector(1000 + i);
  }
  const packedVectors = packSu7Vectors(vectors);
  const cpuBaseline = multiplySu7CpuPacked(unitaryPacked, packedVectors, VECTOR_COUNT);

  const kernel = await Su7GpuKernel.create({ backend: 'auto', label: 'test-su7-kernel' });
  const gpuOutput = await kernel.dispatch({
    unitary: unitaryPacked,
    input: packedVectors,
    vectorCount: VECTOR_COUNT,
    output: new Float32Array(cpuBaseline.length),
  });
  const rms = computeRms(cpuBaseline, gpuOutput);
  assert.ok(rms <= EPSILON, `expected RMS <= ${EPSILON}, got ${rms}`);

  const stats = kernel.getStats();
  assert.ok(stats && stats.sampleCount >= 1, 'expected profiling samples');
  assert.equal(gpuOutput.length, VECTOR_COUNT * SU7_GPU_KERNEL_VECTOR_STRIDE);

  kernel.dispose();
});

test('su7 gpu kernel profiling emits warning on drift', async () => {
  const unitary = buildUnitary({ seed: 17 });
  const unitaryPacked = packSu7Unitary(unitary);
  const vectors: C7Vector[] = [buildRandomVector(2024)];
  const packedVectors = packSu7Vectors(vectors);
  const timeline: number[] = [];
  let current = 0;
  const pushDuration = (delta: number) => {
    timeline.push(current, current + delta);
    current += delta + 0.05;
  };
  const nowFn = () => (timeline.length > 0 ? timeline.shift()! : current);
  let warning: Su7GpuKernelWarningEvent | null = null;
  const kernel = await Su7GpuKernel.create({
    backend: 'cpu-only',
    now: nowFn,
    onWarning: (event) => {
      warning = event;
    },
    profileCapacity: 128,
  });
  for (let i = 0; i < 95; i++) {
    pushDuration(1);
    await kernel.dispatch({ unitary: unitaryPacked, input: packedVectors, vectorCount: 1 });
  }
  const baselineStats = kernel.getStats();
  assert.ok(baselineStats && baselineStats.baselineMs != null, 'baseline not recorded');
  pushDuration(1.6);
  await kernel.dispatch({ unitary: unitaryPacked, input: packedVectors, vectorCount: 1 });
  const updatedStats = kernel.getStats();
  assert.ok(updatedStats && updatedStats.warning, 'expected warning flag after drift');
  assert.ok(warning, 'expected onWarning callback');
  assert.ok(warning!.drift > 0.1, `expected drift > 10%, got ${warning!.drift}`);
  kernel.dispose();
});
