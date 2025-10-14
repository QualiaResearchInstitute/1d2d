import { rgbToLms, srgbToLinear } from '../colorSpaces.js';
const EPSILON = 1e-9;
const DEFAULT_PHASE_OFFSETS = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3];
const DEFAULT_FALLBACK_VECTOR = [
  { re: 1, im: 0 },
  { re: 0, im: 0 },
  { re: 0, im: 0 },
  { re: 0, im: 0 },
  { re: 0, im: 0 },
  { re: 0, im: 0 },
  { re: 0, im: 0 },
];
const ensureFinite = (value) => (Number.isFinite(value) ? value : 0);
const cloneFallbackVector = () => DEFAULT_FALLBACK_VECTOR.map((entry) => ({ ...entry }));
const resolveDimensions = (params) => {
  if (Number.isFinite(params.width) && Number.isFinite(params.height)) {
    return {
      width: Math.max(0, Math.trunc(params.width ?? 0)),
      height: Math.max(0, Math.trunc(params.height ?? 0)),
    };
  }
  const field = params.surface ?? params.rim ?? params.phase ?? params.volume ?? null;
  if (field) {
    return {
      width: field.resolution.width,
      height: field.resolution.height,
    };
  }
  return { width: 0, height: 0 };
};
const computeGaugeAngle = (mode, index, rim, phase) => {
  const pickPhaseAngle = () => {
    if (!phase || !phase.gradX || !phase.gradY) return null;
    const gx = ensureFinite(phase.gradX[index]);
    const gy = ensureFinite(phase.gradY[index]);
    const mag = Math.hypot(gx, gy);
    return mag > EPSILON ? Math.atan2(gy, gx) : null;
  };
  const pickRimAngle = () => {
    if (!rim || !rim.gx || !rim.gy) return null;
    const gx = ensureFinite(rim.gx[index]);
    const gy = ensureFinite(rim.gy[index]);
    const mag = Math.hypot(gx, gy);
    return mag > EPSILON ? Math.atan2(gy, gx) : null;
  };
  if (mode === 'none') {
    return 0;
  }
  if (mode === 'rim') {
    const rimAngle = pickRimAngle();
    if (rimAngle != null) return rimAngle;
    const phaseAngle = pickPhaseAngle();
    return phaseAngle ?? 0;
  }
  if (mode === 'phase') {
    const phaseAngle = pickPhaseAngle();
    if (phaseAngle != null) return phaseAngle;
    const rimAngle = pickRimAngle();
    return rimAngle ?? 0;
  }
  return 0;
};
const rotateByGauge = (re, im, cosGauge, sinGauge) => [
  re * cosGauge + im * sinGauge,
  im * cosGauge - re * sinGauge,
];
const buildComplex = (re, im) => ({
  re,
  im,
});
export const embedToC7 = (params) => {
  const { width, height } = resolveDimensions(params);
  if (width <= 0 || height <= 0) {
    return { vectors: [], norms: new Float32Array(0), width, height };
  }
  const rim = params.rim ?? null;
  const phase = params.phase ?? null;
  const volume = params.volume ?? null;
  const surface = params.surface ?? null;
  const gaugeMode = params.gauge ?? 'rim';
  const phaseOffsets = params.rimPhaseOffsets ?? DEFAULT_PHASE_OFFSETS;
  const rimMag = rim?.mag ?? null;
  const rimGx = rim?.gx ?? null;
  const rimGy = rim?.gy ?? null;
  const phaseGradX = phase?.gradX ?? null;
  const phaseGradY = phase?.gradY ?? null;
  const vort = phase?.vort ?? null;
  const coh = phase?.coh ?? null;
  const amp = phase?.amp ?? null;
  const depth = volume?.depth ?? null;
  const surfaceData = surface?.rgba ?? null;
  const texels = width * height;
  const vectors = new Array(texels);
  const norms = new Float32Array(texels);
  const cx = (width - 1) * 0.5;
  const cy = (height - 1) * 0.5;
  for (let idx = 0; idx < texels; idx++) {
    const y = Math.floor(idx / width);
    const x = idx - y * width;
    const gaugeAngle = computeGaugeAngle(gaugeMode, idx, rim, phase);
    const applyGauge = Math.abs(gaugeAngle) > EPSILON;
    const cosGauge = applyGauge ? Math.cos(gaugeAngle) : 1;
    const sinGauge = applyGauge ? Math.sin(gaugeAngle) : 0;
    let re0 = 0;
    let im0 = 0;
    let re1 = 0;
    let im1 = 0;
    let re2 = 0;
    let im2 = 0;
    let re3 = 0;
    let im3 = 0;
    let re4 = 0;
    let im4 = 0;
    let re5 = 0;
    let im5 = 0;
    let re6 = 0;
    let im6 = 0;
    if (surfaceData) {
      const base = idx * 4;
      const R = srgbToLinear((surfaceData[base] ?? 0) / 255);
      const G = srgbToLinear((surfaceData[base + 1] ?? 0) / 255);
      const B = srgbToLinear((surfaceData[base + 2] ?? 0) / 255);
      const [L, M, S] = rgbToLms(R, G, B);
      const rimWeight = rimMag ? Math.max(ensureFinite(rimMag[idx]), 0) : 1;
      const phase0 = phaseOffsets[0];
      const phase1 = phaseOffsets[1];
      const phase2 = phaseOffsets[2];
      const ampL = ensureFinite(L * rimWeight);
      const ampM = ensureFinite(M * rimWeight);
      const ampS = ensureFinite(S * rimWeight);
      if (ampL > EPSILON) {
        re0 = ampL * Math.cos(phase0);
        im0 = ampL * Math.sin(phase0);
      }
      if (ampM > EPSILON) {
        re1 = ampM * Math.cos(phase1);
        im1 = ampM * Math.sin(phase1);
      }
      if (ampS > EPSILON) {
        re2 = ampS * Math.cos(phase2);
        im2 = ampS * Math.sin(phase2);
      }
    } else if (rimMag) {
      const rimWeight = Math.max(ensureFinite(rimMag[idx]), 0);
      if (rimWeight > EPSILON) {
        const phase0 = phaseOffsets[0];
        const phase1 = phaseOffsets[1];
        const phase2 = phaseOffsets[2];
        re0 = rimWeight * Math.cos(phase0);
        im0 = rimWeight * Math.sin(phase0);
        re1 = rimWeight * Math.cos(phase1);
        im1 = rimWeight * Math.sin(phase1);
        re2 = rimWeight * Math.cos(phase2);
        im2 = rimWeight * Math.sin(phase2);
      }
    }
    if (phaseGradX && phaseGradY) {
      const gx = ensureFinite(phaseGradX[idx]);
      const gy = ensureFinite(phaseGradY[idx]);
      re3 = gx;
      im3 = gy;
    } else if (rimGx && rimGy) {
      const gx = ensureFinite(rimGx[idx]);
      const gy = ensureFinite(rimGy[idx]);
      re3 = gx;
      im3 = gy;
    }
    if (vort) {
      const vortVal = ensureFinite(vort[idx]);
      const amp4 = Math.abs(vortVal);
      if (amp4 > EPSILON) {
        im4 = vortVal >= 0 ? amp4 : -amp4;
      }
    }
    if (amp || coh) {
      const ampVal = amp ? ensureFinite(amp[idx]) : 0;
      const cohVal = coh ? ensureFinite(coh[idx]) : 0;
      const amplitude = Math.max(ampVal, cohVal, 0);
      if (amplitude > EPSILON) {
        const theta = Math.atan2(im3, re3) || 0;
        re5 = amplitude * Math.cos(theta);
        im5 = amplitude * Math.sin(theta);
      }
    }
    if (depth) {
      const wx = x;
      const wy = y;
      const leftIdx = wy * width + Math.max(0, wx - 1);
      const rightIdx = wy * width + Math.min(width - 1, wx + 1);
      const upIdx = Math.max(0, wy - 1) * width + wx;
      const downIdx = Math.min(height - 1, wy + 1) * width + wx;
      const left = ensureFinite(depth[leftIdx]);
      const right = ensureFinite(depth[rightIdx]);
      const up = ensureFinite(depth[upIdx]);
      const down = ensureFinite(depth[downIdx]);
      const gx = (right - left) * 0.5;
      const gy = (down - up) * 0.5;
      const amp6 = Math.hypot(gx, gy);
      if (amp6 > EPSILON) {
        const radialAngle = Math.atan2(wy - cy, wx - cx);
        re6 = amp6 * Math.cos(radialAngle);
        im6 = amp6 * Math.sin(radialAngle);
      }
    }
    if (applyGauge) {
      [re0, im0] = rotateByGauge(re0, im0, cosGauge, sinGauge);
      [re1, im1] = rotateByGauge(re1, im1, cosGauge, sinGauge);
      [re2, im2] = rotateByGauge(re2, im2, cosGauge, sinGauge);
      [re3, im3] = rotateByGauge(re3, im3, cosGauge, sinGauge);
      [re4, im4] = rotateByGauge(re4, im4, cosGauge, sinGauge);
      [re5, im5] = rotateByGauge(re5, im5, cosGauge, sinGauge);
      [re6, im6] = rotateByGauge(re6, im6, cosGauge, sinGauge);
    }
    const normSq =
      re0 * re0 +
      im0 * im0 +
      re1 * re1 +
      im1 * im1 +
      re2 * re2 +
      im2 * im2 +
      re3 * re3 +
      im3 * im3 +
      re4 * re4 +
      im4 * im4 +
      re5 * re5 +
      im5 * im5 +
      re6 * re6 +
      im6 * im6;
    if (normSq > EPSILON) {
      const norm = Math.sqrt(normSq);
      norms[idx] = norm;
      const invNorm = 1 / norm;
      re0 *= invNorm;
      im0 *= invNorm;
      re1 *= invNorm;
      im1 *= invNorm;
      re2 *= invNorm;
      im2 *= invNorm;
      re3 *= invNorm;
      im3 *= invNorm;
      re4 *= invNorm;
      im4 *= invNorm;
      re5 *= invNorm;
      im5 *= invNorm;
      re6 *= invNorm;
      im6 *= invNorm;
      vectors[idx] = [
        buildComplex(re0, im0),
        buildComplex(re1, im1),
        buildComplex(re2, im2),
        buildComplex(re3, im3),
        buildComplex(re4, im4),
        buildComplex(re5, im5),
        buildComplex(re6, im6),
      ];
    } else {
      norms[idx] = 0;
      vectors[idx] = cloneFallbackVector();
    }
  }
  return { vectors, norms, width, height };
};
