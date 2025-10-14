import test from 'node:test';
import assert from 'node:assert/strict';

import { runMediaPipeline } from '../src/media/mediaPipeline.js';
import type { ImageBuffer } from '../src/pipeline/edgeDetection.js';

const makeGradientImage = (width: number, height: number): ImageBuffer => {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const value = Math.round((x / Math.max(1, width - 1)) * 255);
      data[idx] = value;
      data[idx + 1] = value;
      data[idx + 2] = value;
      data[idx + 3] = 255;
    }
  }
  return { data, width, height };
};

test('media pipeline derives edge and phase fields', () => {
  const image = makeGradientImage(8, 6);
  const result = runMediaPipeline(image, { kuramoto: { enabled: false } });
  assert.equal(result.rim.resolution.width, image.width);
  assert.equal(result.rim.resolution.height, image.height);
  assert.equal(result.phase.field.resolution.width, image.width);
  assert.equal(result.phase.field.resolution.height, image.height);
  assert.ok(result.phase.metrics.amplitudeMean > 0);
  assert.ok(result.phase.metrics.coherenceMean > 0);
  assert.ok(result.telemetry.metrics.edgePixelCount !== undefined);
  assert.ok(result.telemetry.durations.edgeMs !== undefined);
  assert.ok(result.telemetry.durations.phaseMs !== undefined);
});

test('media pipeline Kuramoto stage is deterministic with fixed seed', () => {
  const image = makeGradientImage(10, 10);
  const options = { kuramoto: { seed: 1337, steps: 4, dt: 0.01 } };
  const a = runMediaPipeline(image, options);
  const b = runMediaPipeline(image, options);
  assert.ok(a.kuramoto, 'first run should include Kuramoto output');
  assert.ok(b.kuramoto, 'second run should include Kuramoto output');
  const phaseA = a.kuramoto!.phase;
  const phaseB = b.kuramoto!.phase;
  const tolerance = 1e-6;
  for (let i = 0; i < phaseA.gradX.length; i++) {
    assert.ok(Math.abs(phaseA.gradX[i] - phaseB.gradX[i]) < tolerance);
    assert.ok(Math.abs(phaseA.gradY[i] - phaseB.gradY[i]) < tolerance);
    assert.ok(Math.abs(phaseA.vort[i] - phaseB.vort[i]) < tolerance);
    assert.ok(Math.abs(phaseA.coh[i] - phaseB.coh[i]) < tolerance);
    assert.ok(Math.abs(phaseA.amp[i] - phaseB.amp[i]) < tolerance);
  }
  assert.ok(a.kuramoto?.deterministic?.verified);
});
