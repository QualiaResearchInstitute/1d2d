import assert from 'node:assert/strict';
import test from 'node:test';
import { performance } from 'node:perf_hooks';
import {
  createHyperbolicAtlas,
  mapHyperbolicPolarToPixel,
  packageHyperbolicAtlasForGpu,
} from '../src/hyperbolic/atlas.js';

const safeAtanh = (value: number) => 0.5 * Math.log((1 + value) / (1 - value));

test('hyperbolic atlas radial coordinate is monotonic along principal axes', () => {
  const width = 256;
  const height = 256;
  const atlas = createHyperbolicAtlas({
    width,
    height,
    curvatureStrength: 0.6,
    mode: 'poincare',
  });
  const { polar } = atlas;
  const cx = Math.round(atlas.metadata.centerX);
  const cy = Math.round(atlas.metadata.centerY);
  const centerIdx = cy * width + cx;
  const baseRadius = polar[centerIdx * 2];
  assert.ok(Math.abs(baseRadius) < 1e-4, `center pixel radius expected ≈0, got ${baseRadius}`);

  // Scan along +X and +Y directions from the centre.
  const directions: Array<{ dx: number; dy: number }> = [
    { dx: 1, dy: 0 },
    { dx: 0, dy: 1 },
    { dx: 1, dy: 1 },
  ];
  for (const dir of directions) {
    let lastRadius = 0;
    for (let step = 1; step < Math.min(width, height) / 2; step++) {
      const x = cx + dir.dx * step;
      const y = cy + dir.dy * step;
      if (x < 0 || x >= width || y < 0 || y >= height) break;
      const idx = y * width + x;
      const r = polar[idx * 2];
      assert.ok(
        r >= lastRadius - 1e-4,
        `radius decreased at step=${step} dir=(${dir.dx},${dir.dy})`,
      );
      lastRadius = r;
    }
  }
});

test('hyperbolic atlas area weights exhibit exponential growth', () => {
  const width = 320;
  const height = 320;
  const atlas = createHyperbolicAtlas({
    width,
    height,
    curvatureStrength: 0.75,
    mode: 'poincare',
  });
  const { polar, areaWeights, metadata } = atlas;
  const maxHyper = 2 * metadata.curvatureScale * safeAtanh(metadata.diskLimit);
  // Sample a handful of radii well inside the disk limit to avoid clipping.
  const radii = [0.5, 1.0, 1.5, 2.0, 2.5].filter((r) => r < maxHyper * 0.9);
  assert.ok(radii.length >= 3, 'expected at least three usable radius samples');

  const cumulative = radii.map(() => 0);
  for (let idx = 0; idx < atlas.resolution.texels; idx++) {
    const radius = polar[idx * 2];
    const weight = areaWeights[idx];
    for (let i = 0; i < radii.length; i++) {
      if (radius <= radii[i]) cumulative[i] += weight;
    }
  }

  const expected = radii.map((r) => 2 * Math.PI * (Math.cosh(r) - 1));
  for (let i = 1; i < radii.length; i++) {
    const actualRatio = cumulative[i] / cumulative[i - 1];
    const expectedRatio = expected[i] / expected[i - 1];
    const relError = Math.abs(actualRatio - expectedRatio) / expectedRatio;
    assert.ok(
      relError < 0.18,
      `area growth deviates from exponential baseline at r=${radii[i]} (relative error ${relError.toFixed(
        3,
      )})`,
    );
  }
});

test('hyperbolic atlas GPU package re-expresses atlas fields faithfully', () => {
  const atlas = createHyperbolicAtlas({
    width: 32,
    height: 32,
    curvatureStrength: 0.5,
    mode: 'klein',
  });
  const pkg = packageHyperbolicAtlasForGpu(atlas);
  const { resolution, coords, polar, jacobians, areaWeights } = atlas;
  assert.equal(pkg.layout.stride, 9);
  assert.equal(pkg.buffer.length, resolution.texels * pkg.layout.stride);

  const first = pkg.buffer.slice(0, pkg.layout.stride);
  assert.equal(first[0], coords[0]);
  assert.equal(first[1], coords[1]);
  assert.equal(first[2], polar[0]);
  assert.equal(first[3], polar[1]);
  assert.equal(first[4], jacobians[0]);
  assert.equal(first[5], jacobians[1]);
  assert.equal(first[8], areaWeights[0]);

  // Spot check a polar coordinate reconstruction via helper mapping.
  const theta = Math.PI / 3;
  const radius = 1.25;
  const [px, py] = mapHyperbolicPolarToPixel(atlas, radius, theta);
  assert.ok(Number.isFinite(px) && Number.isFinite(py));
});

test('hyperbolic atlas generation stays within 5 s for 1080p', { timeout: 10_000 }, () => {
  const start = performance.now();
  const atlas = createHyperbolicAtlas({
    width: 1920,
    height: 1080,
    curvatureStrength: 0.7,
    mode: 'poincare',
  });
  const elapsed = performance.now() - start;
  assert.ok(
    elapsed < 5000,
    `1080p atlas generation exceeded 5s budget (took ${(elapsed / 1000).toFixed(2)}s)`,
  );
  assert.equal(atlas.coords.length, 1920 * 1080 * 2);
  assert.equal(atlas.polar.length, 1920 * 1080 * 2);
});
