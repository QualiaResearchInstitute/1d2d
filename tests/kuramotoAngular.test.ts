import test from 'node:test';
import assert from 'node:assert/strict';

import {
  createKuramotoState,
  createDerivedViews,
  derivedBufferSize,
  deriveKuramotoFields,
  initKuramotoState,
  stepKuramotoState,
  createIrradianceFrameBuffer,
  createTelemetrySnapshot,
  type KuramotoParams,
  type KuramotoState,
  type PhaseField,
} from '../src/kuramotoCore.js';
import { AngularSpectrumSolver } from '../src/optics/angularSpectrum.js';
import { KERNEL_SPEC_DEFAULT } from '../src/kernel/kernelSpec.js';

const defaultParams: KuramotoParams = {
  alphaKur: 0.2,
  gammaKur: 0.15,
  omega0: 0,
  K0: 0.6,
  epsKur: 0.001,
  fluxX: 0,
  fluxY: 0,
  smallWorldWeight: 0,
  p_sw: 0,
  smallWorldEnabled: false,
};

const makePhaseField = (width: number, height: number): PhaseField => {
  const buffer = new ArrayBuffer(derivedBufferSize(width, height));
  return createDerivedViews(buffer, width, height);
};

const computeMaxDiff = (a: Float32Array, b: Float32Array) => {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = Math.abs(a[i] - b[i]);
    if (diff > max) max = diff;
  }
  return max;
};

test('Kuramoto and angular-spectrum phases agree after alignment', () => {
  const width = 12;
  const height = 10;
  const steps = 4;
  const dt = 0.016;
  const kurState = createKuramotoState(width, height);
  const kurPhase = makePhaseField(width, height);
  initKuramotoState(kurState, 1, kurPhase);

  for (let i = 0; i < steps; i++) {
    stepKuramotoState(kurState, defaultParams, dt, () => 0, dt * (i + 1));
  }
  deriveKuramotoFields(kurState, kurPhase);
  assert.equal(kurState.field.getMeta().frameId, steps - 1);

  const solver = new AngularSpectrumSolver({
    width,
    height,
    wavelengthNm: 550,
    pixelPitchMeters: 1e-6,
    dzMeters: 0.004,
    componentCount: kurState.field.componentCount,
  });

  const angularFrame = solver.propagate(kurState.field, { dzMeters: 0.004, timestamp: 0.25 });
  solver.alignPhase(angularFrame, 0, kurState.field.getPhase(0));
  const angularTelemetry = createTelemetrySnapshot();
  const angularIrradiance = createIrradianceFrameBuffer(width, height, angularFrame.getMeta());
  const angularState: KuramotoState = {
    width,
    height,
    manager: solver.getManager(),
    field: angularFrame,
    componentCount: angularFrame.componentCount,
    components: angularFrame.components,
    Zr: angularFrame.real,
    Zi: angularFrame.imag,
    telemetry: angularTelemetry,
    irradiance: angularIrradiance,
  };
  const angularPhase = makePhaseField(width, height);
  deriveKuramotoFields(angularState, angularPhase);

  const metrics = {
    gradX: computeMaxDiff(kurPhase.gradX, angularPhase.gradX),
    gradY: computeMaxDiff(kurPhase.gradY, angularPhase.gradY),
    vort: computeMaxDiff(kurPhase.vort, angularPhase.vort),
    coh: computeMaxDiff(kurPhase.coh, angularPhase.coh),
    amp: computeMaxDiff(kurPhase.amp, angularPhase.amp),
  };

  const tolerance = 1e-5;
  assert.ok(metrics.gradX < tolerance, `gradX diff ${metrics.gradX}`);
  assert.ok(metrics.gradY < tolerance, `gradY diff ${metrics.gradY}`);
  assert.ok(metrics.vort < tolerance, `vort diff ${metrics.vort}`);
  assert.ok(metrics.coh < tolerance, `coh diff ${metrics.coh}`);
  assert.ok(metrics.amp < tolerance, `amp diff ${metrics.amp}`);
  assert.ok(angularFrame.getMeta().frameId >= 0, 'angular solver should stamp frame IDs');
});

test('Kuramoto telemetry updates order parameter and irradiance metadata', () => {
  const width = 2;
  const height = 1;
  const state = createKuramotoState(width, height);
  initKuramotoState(state, 0);
  const params: KuramotoParams = {
    alphaKur: 0,
    gammaKur: 0,
    omega0: 0,
    K0: 0,
    epsKur: 0,
    fluxX: 0,
    fluxY: 0,
    smallWorldWeight: 0,
    p_sw: 0,
    smallWorldEnabled: false,
  };
  const result = stepKuramotoState(state, params, 0, () => 0, 1, {
    telemetry: { kernelVersion: 7 },
  });
  const meta = state.field.getMeta();
  assert.equal(result.telemetry.frameId, meta.frameId);
  assert.equal(state.irradiance.opticalMeta.frameId, meta.frameId);
  assert.equal(result.telemetry.kernelVersion, 7);
  assert.equal(state.irradiance.kernelVersion, 7);
  assert.equal(result.telemetry.kernel.gain, KERNEL_SPEC_DEFAULT.gain);
  assert.equal(state.irradiance.kernel.gain, KERNEL_SPEC_DEFAULT.gain);
  assert.equal(result.telemetry.orderParameter.sampleCount, width * height);
  assert.ok(
    result.telemetry.orderParameter.magnitude > 0.99,
    `expected high coherence, got ${result.telemetry.orderParameter.magnitude}`,
  );
  for (let i = 0; i < state.irradiance.L.length; i++) {
    assert.ok(Math.abs(state.irradiance.L[i] - 1) < 1e-6, `irradiance L[${i}]`);
    assert.ok(Math.abs(state.irradiance.M[i] - 1) < 1e-6, `irradiance M[${i}]`);
    assert.ok(Math.abs(state.irradiance.S[i] - 1) < 1e-6, `irradiance S[${i}]`);
  }
  assert.equal(state.irradiance.opticalMeta.timestamp, meta.timestamp);
});
