import test from 'node:test';
import assert from 'node:assert/strict';

import { GaugeLattice } from '../src/qcd/lattice.js';
import {
  measureWilsonRectangle,
  measureWilsonLoops,
  measureWilsonLoopGrid,
  getWilsonLoopValue,
  computeCreutzRatio,
  measurePolyakovLoop,
  binSamples,
  jackknife,
  RunningEstimate,
} from '../src/qcd/observables.js';
import { runWilsonCpuUpdate, initializeGaugeField } from '../src/qcd/updateCpu.js';
import { su3_mul, su3_haar, type Complex3x3 } from '../src/qcd/su3.js';
import { initializeQcdRuntime, runTemperatureScan } from '../src/qcd/runtime.js';

type Complex = { re: number; im: number };

const mulberry32 = (seed: number): (() => number) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let c = Math.imul(t ^ (t >>> 15), 1 | t);
    c ^= c + Math.imul(c ^ (c >>> 7), 61 | c);
    return ((c ^ (c >>> 14)) >>> 0) / 4294967296;
  };
};

const createZeroMatrix = (): Complex3x3 =>
  new Array(3).fill(null).map(
    () =>
      [
        { re: 0, im: 0 },
        { re: 0, im: 0 },
        { re: 0, im: 0 },
      ] as [Complex, Complex, Complex],
  ) as Complex3x3;

const conjugateTranspose = (matrix: Complex3x3): Complex3x3 => {
  const result = createZeroMatrix();
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const entry = matrix[col][row];
      result[row][col] = { re: entry.re, im: -entry.im };
    }
  }
  return result;
};

const applyGaugeTransformation = (lattice: GaugeLattice, siteTransforms: Complex3x3[][]): void => {
  const width = lattice.width;
  const height = lattice.height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const gSite = siteTransforms[y][x];
      for (const axis of lattice.axes) {
        const nx = axis === 'x' ? (x + 1) % width : x;
        const ny = axis === 'y' ? (y + 1) % height : y;
        const gNeighbor = siteTransforms[ny][nx];
        const link = lattice.getLinkMatrix(x, y, axis);
        const transformed = su3_mul(su3_mul(gSite, link), conjugateTranspose(gNeighbor));
        lattice.setLinkMatrix(x, y, axis, transformed);
      }
    }
  }
};

const average = (values: readonly number[]): number =>
  values.reduce((sum, value) => sum + value, 0) / values.length;

test('identity gauge field yields unity Wilson and Polyakov loops', () => {
  const lattice = new GaugeLattice({ width: 4, height: 4 });
  lattice.fillIdentity();

  const wilson = measureWilsonRectangle(lattice, 2, 1);
  assert.equal(wilson.value, 1);
  assert.equal(wilson.normalized.re, 1);
  assert.equal(wilson.normalized.im, 0);

  const polyakovX = measurePolyakovLoop(lattice, 'x');
  const polyakovY = measurePolyakovLoop(lattice, 'y');
  assert.equal(polyakovX.average.re, 1);
  assert.equal(polyakovX.average.im, 0);
  assert.equal(polyakovY.average.re, 1);
  assert.equal(polyakovY.average.im, 0);
});

test('identity gauge field yields unity Wilson loops across z/t planes', () => {
  const lattice2p1 = new GaugeLattice({ width: 3, height: 3, depth: 2 });
  lattice2p1.fillIdentity();
  const xz = measureWilsonRectangle(lattice2p1, 1, 1, ['x', 'z']);
  const yz = measureWilsonRectangle(lattice2p1, 1, 1, ['y', 'z']);
  assert.equal(xz.value, 1);
  assert.equal(yz.value, 1);

  const lattice3p1 = new GaugeLattice({ width: 2, height: 2, depth: 2, temporalExtent: 3 });
  lattice3p1.fillIdentity();
  const xt = measureWilsonRectangle(lattice3p1, 1, 1, ['x', 't']);
  const yt = measureWilsonRectangle(lattice3p1, 1, 1, ['y', 't']);
  assert.equal(xt.value, 1);
  assert.equal(yt.value, 1);
});

test('Polyakov loop along temporal axis is unity for cold field', () => {
  const lattice = new GaugeLattice({ width: 2, height: 2, temporalExtent: 4 });
  lattice.fillIdentity();
  const loop = measurePolyakovLoop(lattice, 't');
  assert.equal(loop.average.re, 1);
  assert.equal(loop.average.im, 0);
  assert.equal(loop.magnitude, 1);
  assert.equal(loop.sampleCount, lattice.siteCount);
});

test('Wilson loops are invariant under local gauge transformations', () => {
  const lattice = new GaugeLattice({ width: 4, height: 4 });
  initializeGaugeField(lattice, 'hot', mulberry32(1337));

  const rectangles = [
    { extentX: 1, extentY: 1 },
    { extentX: 1, extentY: 2 },
    { extentX: 2, extentY: 1 },
    { extentX: 2, extentY: 2 },
  ] as const;
  const baseline = measureWilsonLoops(lattice, rectangles);
  const loopMap = new Map(
    baseline.map((entry) => [`${entry.extentX}:${entry.extentY}`, entry.value]),
  );

  const rng = mulberry32(4545);
  const siteTransforms: Complex3x3[][] = [];
  for (let y = 0; y < lattice.height; y++) {
    const row: Complex3x3[] = [];
    for (let x = 0; x < lattice.width; x++) {
      row.push(su3_haar(1, rng));
    }
    siteTransforms.push(row);
  }

  applyGaugeTransformation(lattice, siteTransforms);
  const transformed = measureWilsonLoops(lattice, rectangles);

  for (const entry of transformed) {
    const key = `${entry.extentX}:${entry.extentY}`;
    const reference = loopMap.get(key);
    assert.ok(reference != null, `missing reference for ${key}`);
    assert.ok(
      Math.abs(reference - entry.value) <= 1e-8,
      `gauge invariance violated for ${key}: ${reference} vs ${entry.value}`,
    );
  }
});

test('Wilson observables show area law trend and stable Creutz ratios', () => {
  const samples = 8;
  const latticeSize = 6;
  const maxExtent = 3;
  const loopSamples: Record<'1x1' | '1x2' | '2x2', number[]> = {
    '1x1': [],
    '1x2': [],
    '2x2': [],
  };
  const creutzSamples: Record<number, number[]> = {
    1: [],
    2: [],
  };
  const running = new RunningEstimate();

  for (let sample = 0; sample < samples; sample++) {
    const lattice = new GaugeLattice({ width: latticeSize, height: latticeSize });
    runWilsonCpuUpdate(lattice, {
      betaSchedule: [1.65],
      sweepsPerBeta: 4,
      thermalizationSweeps: 4,
      overRelaxationSteps: 1,
      startMode: 'hot',
      seed: 9000 + sample,
    });

    const table = measureWilsonLoopGrid(lattice, maxExtent, maxExtent, ['x', 'y']);
    const w11 = getWilsonLoopValue(table, 1, 1, ['x', 'y']);
    const w12 = getWilsonLoopValue(table, 1, 2, ['x', 'y']);
    const w22 = getWilsonLoopValue(table, 2, 2, ['x', 'y']);
    assert.ok(w11 && w12 && w22, 'failed to sample Wilson loops in x-y plane');

    loopSamples['1x1'].push(w11.value);
    loopSamples['1x2'].push(w12.value);
    loopSamples['2x2'].push(w22.value);

    const chi1 = computeCreutzRatio(table, 1, 1, ['x', 'y']);
    const chi2 = computeCreutzRatio(table, 2, 2, ['x', 'y']);
    creutzSamples[1].push(chi1);
    creutzSamples[2].push(chi2);

    running.push(loopSamples['1x1'][loopSamples['1x1'].length - 1]!);
  }

  const w11 = average(loopSamples['1x1']);
  const w12 = average(loopSamples['1x2']);
  const w22 = average(loopSamples['2x2']);

  assert.ok(w11 > w12, `area law violated: W(1,1)=${w11} <= W(1,2)=${w12}`);
  assert.ok(w12 > w22, `area law violated: W(1,2)=${w12} <= W(2,2)=${w22}`);

  const chi1Binned = binSamples(creutzSamples[1], 2);
  const chi2Binned = binSamples(creutzSamples[2], 2);
  assert.ok(chi1Binned.length >= 2);
  assert.ok(chi2Binned.length >= 2);

  const estimator = (values: readonly number[]) => average(values);
  const chi1Stats = jackknife(chi1Binned, estimator);
  const chi2Stats = jackknife(chi2Binned, estimator);

  const diff = Math.abs(chi2Stats.estimate - chi1Stats.estimate);
  const combinedError = Math.sqrt(chi1Stats.error ** 2 + chi2Stats.error ** 2);
  assert.ok(
    diff <= combinedError + 0.02,
    `Creutz ratios not consistent within errors: Δ=${diff}, σ=${combinedError}`,
  );

  const finalRunning = running.snapshot();
  assert.equal(finalRunning.count, samples);
  assert.ok(Math.abs(finalRunning.mean - w11) <= 1e-9);
  assert.ok(finalRunning.standardDeviation > 0);
  assert.ok(finalRunning.standardError > 0);
});

test('temperature scan collects Polyakov samples without mutating lattice', () => {
  const runtime = initializeQcdRuntime({
    latticeSize: { width: 2, height: 2 },
    config: {
      beta: 5.0,
      overRelaxationSteps: 1,
      smearing: { alpha: 0, iterations: 0 },
      depth: 1,
      temporalExtent: 3,
      batchLayers: 1,
      temperatureSchedule: [],
    },
    baseSeed: 4242,
    startMode: 'cold',
  });
  runtime.lattice.fillIdentity();
  const schedule = [4.6, 4.8, 5.0];
  const before = new Float32Array(runtime.lattice.data);
  const samples = runTemperatureScan(runtime, schedule, 't');
  assert.equal(samples.length, schedule.length);
  assert.equal(runtime.polyakovScan.length, schedule.length);
  assert.equal(runtime.observables.polyakovSamples?.length, schedule.length);
  samples.forEach((sample) => {
    assert.equal(sample.axis, 't');
    assert.equal(sample.extent, runtime.lattice.temporalExtent);
    assert.ok(Number.isFinite(sample.magnitude));
    assert.equal(sample.sampleCount, runtime.lattice.siteCount);
  });
  assert.deepEqual(Array.from(runtime.lattice.data), Array.from(before));
});

test('jackknife requires sufficient samples', () => {
  assert.throws(() => jackknife([1], (values) => values[0]!), /at least two samples/i);
});
