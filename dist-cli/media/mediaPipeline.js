import {
  createKuramotoState,
  createNormalGenerator,
  createDerivedViews,
  derivedBufferSize,
  deriveKuramotoFields,
  initKuramotoState,
  stepKuramotoState,
} from '../kuramotoCore.js';
import { computeEdgeField } from '../pipeline/edgeDetection.js';
import { computePhaseField } from '../pipeline/phaseField.js';
const now = () =>
  typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
const DEFAULT_KURAMOTO_PARAMS = {
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
const simulateKuramotoPhase = (baseParams, resolution, options) => {
  const params = { ...baseParams, ...options.params };
  const steps = Math.max(1, Math.floor(options.steps ?? 8));
  const dt = options.dt ?? 0.016;
  const qInit = options.qInit ?? 1;
  const buffer = new ArrayBuffer(derivedBufferSize(resolution.width, resolution.height));
  const phase = createDerivedViews(buffer, resolution.width, resolution.height);
  const state = createKuramotoState(resolution.width, resolution.height);
  initKuramotoState(state, qInit, phase);
  const randn = createNormalGenerator(options.seed);
  for (let step = 0; step < steps; step++) {
    const timestamp = dt * (step + 1);
    stepKuramotoState(state, params, dt, randn, timestamp, { params });
  }
  deriveKuramotoFields(state, phase, { params });
  return phase;
};
const verifyDeterminism = (params, resolution, options, baseline) => {
  const texels = resolution.width * resolution.height;
  if (!options.determinismSample && texels > 128 * 128) {
    return { verified: true, maxDelta: undefined };
  }
  const comparison = simulateKuramotoPhase(params, resolution, options);
  let maxDelta = 0;
  for (let i = 0; i < texels; i++) {
    const delta =
      Math.abs(baseline.gradX[i] - comparison.gradX[i]) +
      Math.abs(baseline.gradY[i] - comparison.gradY[i]) +
      Math.abs(baseline.vort[i] - comparison.vort[i]) +
      Math.abs(baseline.coh[i] - comparison.coh[i]) +
      Math.abs(baseline.amp[i] - comparison.amp[i]);
    if (delta > maxDelta) maxDelta = delta;
  }
  const tolerance = 1e-4;
  return { verified: maxDelta < tolerance, maxDelta };
};
export const runMediaPipeline = (image, options = {}) => {
  options.onStage?.('edge');
  const edgeStart = now();
  const rim = computeEdgeField(image);
  const edgeEnd = now();
  const durations = {
    edgeMs: edgeEnd - edgeStart,
  };
  options.onStage?.('phase');
  const phaseStart = now();
  const phase = computePhaseField(rim, options.phase);
  const phaseEnd = now();
  durations.phaseMs = phaseEnd - phaseStart;
  let edgePixelCount = 0;
  let edgeMagnitudeSum = 0;
  for (let i = 0; i < rim.mag.length; i++) {
    const value = rim.mag[i];
    if (value > 0.25) edgePixelCount++;
    edgeMagnitudeSum += value;
  }
  const edgeMagnitudeMean = edgeMagnitudeSum / rim.mag.length;
  const phaseVariance = phase.metrics.amplitudeStd * phase.metrics.amplitudeStd;
  const coherenceMean = phase.metrics.coherenceMean;
  const metrics = {
    edgePixelCount,
    edgeMagnitudeMean,
    phaseVariance,
    coherenceMean,
  };
  let kuramoto;
  const kurOptions = options.kuramoto;
  if (!kurOptions || kurOptions.enabled !== false) {
    options.onStage?.('kuramoto');
    const kurStart = now();
    const params = { ...DEFAULT_KURAMOTO_PARAMS, ...kurOptions?.params };
    const phaseField = simulateKuramotoPhase(params, rim.resolution, kurOptions ?? {});
    const kurEnd = now();
    durations.kuramotoMs = kurEnd - kurStart;
    let deterministic;
    try {
      deterministic = verifyDeterminism(params, rim.resolution, kurOptions ?? {}, phaseField);
    } catch (error) {
      console.warn('[mediaPipeline] Determinism check failed', error);
      deterministic = { verified: false };
    }
    kuramoto = {
      phase: phaseField,
      deterministic,
    };
  }
  return {
    rim,
    phase,
    kuramoto,
    telemetry: {
      durations,
      metrics,
    },
  };
};
