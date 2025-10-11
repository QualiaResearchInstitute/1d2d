import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_TRACER_CONFIG,
  applyTracerFeedback,
  mapTracerConfigToRuntime,
} from '../src/pipeline/tracerFeedback.js';

test('tracer default parameters yield 1/e half-life within expected window', () => {
  const config = { ...DEFAULT_TRACER_CONFIG, enabled: true };
  const runtime = mapTracerConfigToRuntime(config);
  assert.ok(runtime.enabled, 'runtime should be enabled for tracer test');

  const width = 1;
  const height = 1;
  const out = new Uint8ClampedArray([255, 255, 255, 255]);
  const state = new Float32Array(width * height * 3);

  const dt = 1 / 60;

  // Initial impulse frame (base image present).
  applyTracerFeedback({
    out,
    state,
    width,
    height,
    runtime,
    dt,
    timeSeconds: 0,
  });

  const targetAmplitude = runtime.gain / Math.E;
  let accumulatedTime = 0;
  let amplitude = runtime.gain;
  let iteration = 0;

  while (amplitude > targetAmplitude && iteration < 2000) {
    iteration += 1;
    accumulatedTime = iteration * dt;
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 255;
    applyTracerFeedback({
      out,
      state,
      width,
      height,
      runtime,
      dt,
      timeSeconds: accumulatedTime,
    });
    amplitude = out[0] / 255;
  }

  assert.ok(iteration > 0, 'tracer loop should decay below target amplitude');
  assert.ok(
    accumulatedTime >= 0.45,
    `half-life ${accumulatedTime.toFixed(3)}s shorter than expected floor`,
  );
  assert.ok(
    accumulatedTime <= 2.05,
    `half-life ${accumulatedTime.toFixed(3)}s longer than expected ceiling`,
  );
});
