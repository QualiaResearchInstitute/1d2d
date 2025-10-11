const TAU = Math.PI * 2;

export type TracerConfig = {
  enabled: boolean;
  decay: number;
  gain: number;
  modulationEnabled: boolean;
  modulationFrequency: number;
  modulationDepth: number;
  modulationPhase: number;
};

export type TracerRuntimeParams = {
  enabled: boolean;
  gain: number;
  tau: number;
  modulationDepth: number;
  modulationFrequency: number;
  modulationPhase: number;
};

export const DEFAULT_TRACER_CONFIG: TracerConfig = {
  enabled: false,
  decay: 0.18,
  gain: 0.64,
  modulationEnabled: false,
  modulationFrequency: 12,
  modulationDepth: 0.35,
  modulationPhase: 0,
};

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

export const sanitizeTracerConfig = (config: TracerConfig | undefined): TracerConfig => {
  const source = config ?? DEFAULT_TRACER_CONFIG;
  const decay = clamp01(source.decay);
  return {
    enabled: Boolean(source.enabled),
    decay,
    gain: clamp01(Number.isFinite(source.gain) ? source.gain : DEFAULT_TRACER_CONFIG.gain),
    modulationEnabled: Boolean(source.modulationEnabled),
    modulationFrequency: Math.max(
      0,
      Math.min(
        48,
        Number.isFinite(source.modulationFrequency)
          ? source.modulationFrequency
          : DEFAULT_TRACER_CONFIG.modulationFrequency,
      ),
    ),
    modulationDepth: clamp01(
      Number.isFinite(source.modulationDepth)
        ? source.modulationDepth
        : DEFAULT_TRACER_CONFIG.modulationDepth,
    ),
    modulationPhase: Number.isFinite(source.modulationPhase)
      ? source.modulationPhase
      : DEFAULT_TRACER_CONFIG.modulationPhase,
  };
};

export const computeTracerTau = (decay: number): number => {
  const decayNorm = clamp01(decay);
  const MIN_TAU = 0.35;
  const MAX_TAU = 2.0;
  return MIN_TAU + (MAX_TAU - MIN_TAU) * (1 - decayNorm);
};

export const mapTracerConfigToRuntime = (config: TracerConfig): TracerRuntimeParams => {
  const sanitized = sanitizeTracerConfig(config);
  return {
    enabled: sanitized.enabled,
    gain: sanitized.gain,
    tau: computeTracerTau(sanitized.decay),
    modulationDepth: sanitized.modulationEnabled ? sanitized.modulationDepth : 0,
    modulationFrequency: sanitized.modulationEnabled ? sanitized.modulationFrequency : 0,
    modulationPhase: sanitized.modulationEnabled ? sanitized.modulationPhase : 0,
  };
};

export const applyTracerFeedback = (params: {
  out: Uint8ClampedArray;
  state: Float32Array;
  width: number;
  height: number;
  runtime: TracerRuntimeParams;
  dt: number;
  timeSeconds: number;
}): void => {
  const { out, state, width, height, runtime } = params;
  if (!runtime.enabled) {
    return;
  }
  const pixelCount = width * height;
  const expectedStateLength = pixelCount * 3;
  if (state.length !== expectedStateLength) {
    throw new Error(
      `[tracer] state buffer length ${state.length} does not match ${expectedStateLength}`,
    );
  }
  const dt = Math.max(1 / 480, Math.min(params.dt, 0.25));
  const tau = Math.max(0.05, runtime.tau);
  const decayFactor = Math.exp(-dt / tau);
  const sineComponent =
    runtime.modulationFrequency > 0 && runtime.modulationDepth > 1e-6
      ? runtime.modulationDepth *
        Math.sin(params.timeSeconds * TAU * runtime.modulationFrequency + runtime.modulationPhase)
      : 0;
  const modulatedGain = Math.max(0, Math.min(1.2, runtime.gain * (1 + sineComponent)));

  for (let i = 0; i < pixelCount; i++) {
    const outIndex = i * 4;
    const stateIndex = i * 3;

    const baseR = out[outIndex] / 255;
    const baseG = out[outIndex + 1] / 255;
    const baseB = out[outIndex + 2] / 255;

    const prevR = state[stateIndex];
    const prevG = state[stateIndex + 1];
    const prevB = state[stateIndex + 2];

    const tailR = Math.max(prevR - baseR, 0);
    const tailG = Math.max(prevG - baseG, 0);
    const tailB = Math.max(prevB - baseB, 0);

    const finalR = Math.max(0, Math.min(1, baseR + tailR * modulatedGain));
    const finalG = Math.max(0, Math.min(1, baseG + tailG * modulatedGain));
    const finalB = Math.max(0, Math.min(1, baseB + tailB * modulatedGain));

    out[outIndex] = Math.round(finalR * 255);
    out[outIndex + 1] = Math.round(finalG * 255);
    out[outIndex + 2] = Math.round(finalB * 255);

    const decayedR = prevR * decayFactor;
    const decayedG = prevG * decayFactor;
    const decayedB = prevB * decayFactor;

    state[stateIndex] = Math.max(baseR, decayedR);
    state[stateIndex + 1] = Math.max(baseG, decayedG);
    state[stateIndex + 2] = Math.max(baseB, decayedB);
  }
};
