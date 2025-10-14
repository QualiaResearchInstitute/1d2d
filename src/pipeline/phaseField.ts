import { makeResolution, type PhaseField, type RimField } from '../fields/contracts.js';

export interface PhaseFieldOptions {
  readonly amplitudeScale?: number;
  readonly coherenceFloor?: number;
  readonly normalizeGradients?: boolean;
}

export interface PhaseFieldMetrics {
  readonly amplitudeMean: number;
  readonly amplitudeStd: number;
  readonly coherenceMean: number;
  readonly vorticityMean: number;
}

export interface PhaseFieldResult {
  readonly field: PhaseField;
  readonly metrics: PhaseFieldMetrics;
}

const DEFAULT_OPTIONS: Required<PhaseFieldOptions> = {
  amplitudeScale: 1,
  coherenceFloor: 0.15,
  normalizeGradients: true,
};

export const computePhaseField = (
  rim: RimField,
  options: PhaseFieldOptions = {},
): PhaseFieldResult => {
  const { amplitudeScale, coherenceFloor, normalizeGradients } = { ...DEFAULT_OPTIONS, ...options };
  const { resolution } = rim;
  const total = resolution.texels;
  const gradX = new Float32Array(total);
  const gradY = new Float32Array(total);
  const vort = new Float32Array(total);
  const coh = new Float32Array(total);
  const amp = new Float32Array(total);

  let maxGrad = 0;
  for (let i = 0; i < total; i++) {
    const gx = rim.gx[i];
    const gy = rim.gy[i];
    gradX[i] = gx;
    gradY[i] = gy;
    const gradMag = Math.hypot(gx, gy);
    if (gradMag > maxGrad) {
      maxGrad = gradMag;
    }
  }

  const invMaxGrad = normalizeGradients && maxGrad > 1e-8 ? 1 / maxGrad : 1;
  let ampSum = 0;
  let ampSqSum = 0;
  let cohSum = 0;
  let vortSum = 0;
  for (let i = 0; i < total; i++) {
    const gx = gradX[i] * invMaxGrad;
    const gy = gradY[i] * invMaxGrad;
    gradX[i] = gx;
    gradY[i] = gy;
    const magnitude = Math.hypot(gx, gy);
    const amplitude = Math.min(1, Math.max(0, rim.mag[i] * amplitudeScale));
    const coherence = Math.max(coherenceFloor, 1 - magnitude);
    amp[i] = amplitude;
    coh[i] = coherence;
    ampSum += amplitude;
    ampSqSum += amplitude * amplitude;
    cohSum += coherence;
  }

  const { width, height } = resolution;
  const idx = (x: number, y: number) => y * width + x;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = idx(x, y);
      if (x === 0 || y === 0 || x === width - 1 || y === height - 1) {
        vort[i] = 0;
        continue;
      }
      const dvx = gradY[idx(x + 1, y)] - gradY[idx(x - 1, y)];
      const dvy = gradX[idx(x, y + 1)] - gradX[idx(x, y - 1)];
      const curl = 0.5 * (dvx - dvy);
      vort[i] = curl;
      vortSum += curl;
    }
  }

  const ampMean = ampSum / total;
  const ampVariance = Math.max(ampSqSum / total - ampMean * ampMean, 0);
  const amplitudeStd = Math.sqrt(ampVariance);
  const coherenceMean = cohSum / total;
  const vorticityMean = vortSum / total;

  const field: PhaseField = {
    kind: 'phase',
    resolution: makeResolution(width, height),
    gradX,
    gradY,
    vort,
    coh,
    amp,
  };

  return {
    field,
    metrics: {
      amplitudeMean: ampMean,
      amplitudeStd,
      coherenceMean,
      vorticityMean,
    },
  };
};
