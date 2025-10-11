import type { PhaseField, RimField, SurfaceField, VolumeField } from '../fields/contracts.js';
import { createHyperbolicAtlas, type HyperbolicAtlas } from '../hyperbolic/atlas.js';
export type { RimField, PhaseField, SurfaceField, VolumeField } from '../fields/contracts.js';
import { clampKernelSpec, cloneKernelSpec, type KernelSpec } from '../kernel/kernelSpec.js';
import type { KuramotoTelemetrySnapshot } from '../kuramotoCore.js';
export type { KernelSpec } from '../kernel/kernelSpec.js';
import { computeTextureDiagnostics } from './textureDiagnostics.js';
import { embedToC7 } from './su7/embed.js';
import { buildScheduledUnitary, computeSu7Telemetry, type Su7ScheduleContext } from './su7/math.js';
import { projectSu7Vector, type Su7ProjectionResult } from './su7/projector.js';
import { applySU7 } from './su7/unitary.js';
import {
  createDefaultSu7RuntimeParams,
  type C7Vector,
  type Complex7x7,
  type Su7RuntimeParams,
  type Su7Telemetry,
} from './su7/types.js';

const DMT_SENS = {
  g1: 0.6,
  k1: 0.35,
  q1: 0.5,
  a1: 0.7,
  c1: 0.6,
  t1: 0.5,
} as const;

export type SurfaceRegion = 'surfaces' | 'edges' | 'both';
export type WallpaperGroup =
  | 'off'
  | 'p1'
  | 'p2'
  | 'pm'
  | 'pg'
  | 'cm'
  | 'pmm'
  | 'pmg'
  | 'pgg'
  | 'cmm'
  | 'p4'
  | 'p4m'
  | 'p4g'
  | 'p3'
  | 'p3m1'
  | 'p31m'
  | 'p6'
  | 'p6m';
export type DisplayMode =
  | 'color'
  | 'grayBaseColorRims'
  | 'grayBaseGrayRims'
  | 'colorBaseGrayRims'
  | 'colorBaseBlendedRims';

export type CurvatureMode = 'poincare' | 'klein';

export type CouplingConfig = {
  rimToSurfaceBlend: number;
  rimToSurfaceAlign: number;
  surfaceToRimOffset: number;
  surfaceToRimSigma: number;
  surfaceToRimHue: number;
  kurToTransparency: number;
  kurToOrientation: number;
  kurToChirality: number;
  volumePhaseToHue: number;
  volumeDepthToWarp: number;
};

export type ComposerFieldId = 'surface' | 'rim' | 'kur' | 'volume';

export type ComposerFieldConfig = {
  exposure: number;
  gamma: number;
  weight: number;
};

export type DmtRoutingMode = 'auto' | 'rimBias' | 'surfaceBias';

export type SolverRegime = 'balanced' | 'rimLocked' | 'surfaceLocked';

export type ComposerConfig = {
  fields: Record<ComposerFieldId, ComposerFieldConfig>;
  dmtRouting: DmtRoutingMode;
  solverRegime: SolverRegime;
};

export type ComposerFieldMetrics = {
  energy: number;
  share: number;
  weight: number;
  blend: number;
};

export type ComposerTelemetry = {
  fields: Record<ComposerFieldId, ComposerFieldMetrics>;
  dmtRouting: DmtRoutingMode;
  solverRegime: SolverRegime;
  coupling: {
    base: CouplingConfig;
    effective: CouplingConfig;
    scale: number;
  };
};

export const COMPOSER_FIELD_LIST: ComposerFieldId[] = ['surface', 'rim', 'kur', 'volume'];

export const createDefaultComposerConfig = (): ComposerConfig => ({
  fields: {
    surface: { exposure: 1.0, gamma: 1.0, weight: 1.0 },
    rim: { exposure: 1.0, gamma: 1.0, weight: 1.0 },
    kur: { exposure: 1.0, gamma: 1.0, weight: 1.0 },
    volume: { exposure: 1.0, gamma: 1.0, weight: 1.0 },
  },
  dmtRouting: 'auto',
  solverRegime: 'balanced',
});

export const computeComposerBlendGain = (config: ComposerConfig): [number, number] => {
  const dmtRoutingMode = config.dmtRouting;
  const solverMode = config.solverRegime;
  const rimRoutingFactor =
    dmtRoutingMode === 'rimBias' ? 1.1 : dmtRoutingMode === 'surfaceBias' ? 0.9 : 1;
  const surfaceRoutingFactor =
    dmtRoutingMode === 'surfaceBias' ? 1.1 : dmtRoutingMode === 'rimBias' ? 0.9 : 1;
  const rimSolverFactor =
    solverMode === 'rimLocked' ? 1.1 : solverMode === 'surfaceLocked' ? 0.9 : 1;
  const surfaceSolverFactor =
    solverMode === 'surfaceLocked' ? 1.1 : solverMode === 'rimLocked' ? 0.9 : 1;
  return [rimRoutingFactor * rimSolverFactor, surfaceRoutingFactor * surfaceSolverFactor];
};

const clampComposerValue = (value: number, min: number, max: number, fallback: number) => {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return clamp(value, min, max);
};

const sanitizeComposerConfig = (config: ComposerConfig | undefined): ComposerConfig => {
  const defaults = createDefaultComposerConfig();
  const source = config ?? defaults;
  const result: ComposerConfig = {
    fields: {
      surface: { ...defaults.fields.surface },
      rim: { ...defaults.fields.rim },
      kur: { ...defaults.fields.kur },
      volume: { ...defaults.fields.volume },
    },
    dmtRouting: source.dmtRouting ?? defaults.dmtRouting,
    solverRegime: source.solverRegime ?? defaults.solverRegime,
  };
  COMPOSER_FIELD_LIST.forEach((field) => {
    const incoming = source.fields?.[field];
    const fallback = defaults.fields[field];
    result.fields[field] = {
      exposure: clampComposerValue(
        incoming?.exposure ?? fallback.exposure,
        0,
        8,
        fallback.exposure,
      ),
      gamma: clampComposerValue(incoming?.gamma ?? fallback.gamma, 0.2, 5, fallback.gamma),
      weight: clampComposerValue(incoming?.weight ?? fallback.weight, 0, 2.5, fallback.weight),
    };
  });
  return result;
};

type OpBase = {
  tx?: number;
  ty?: number;
};

export type Op =
  | (OpBase & { kind: 'rot'; angle: number })
  | (OpBase & { kind: 'mirrorX' })
  | (OpBase & { kind: 'mirrorY' })
  | (OpBase & { kind: 'diag1' })
  | (OpBase & { kind: 'diag2' });

export type TextureChannelStats = {
  wallpapericity: number;
  wallpaperStd: number;
  beatEnergy: number;
  beatStd: number;
  resonanceRate: number;
  sampleCount: number;
};

export type TextureEarlyVisionMetrics = TextureChannelStats & {
  dogMean: number;
  dogStd: number;
  orientationMean: number;
  orientationStd: number;
  divisiveMean: number;
  divisiveStd: number;
};

export type TextureMetrics = TextureChannelStats & {
  earlyVision: TextureEarlyVisionMetrics;
  crystallizer: TextureChannelStats;
};

export type ParallaxMetrics = {
  radialSlope: number;
  eccentricityMean: number;
  kMean: number;
  kStd: number;
  tagConsistency: number;
  sampleCount: number;
};

export type MotionEnergyMetrics = {
  branchCount: number;
  phaseShiftMean: number;
  phaseShiftStd: number;
  parallaxMean: number;
  parallaxStd: number;
  parallaxAbsMean: number;
  source: 'orientation' | 'wallpaper' | 'none';
};

export type BranchAttentionEntry = {
  index: number;
  sampleCount: number;
  meanPhase: number;
  coherence: number;
  meanParallax: number;
  hyperbolicGain: number;
  magnitude: number;
};

export type BranchAttentionSummary = {
  kind: 'orientation' | 'wallpaper';
  entries: BranchAttentionEntry[];
};

export type BranchAttentionHooks = {
  onOrientation?(summary: BranchAttentionSummary): void;
  onWallpaper?(summary: BranchAttentionSummary): void;
};

export type RainbowFrameMetrics = {
  rim: {
    mean: number;
    max: number;
    std: number;
    count: number;
  };
  warp: {
    mean: number;
    std: number;
    dominantAngle: number;
    count: number;
  };
  gradient: {
    gradMean: number;
    gradStd: number;
    vortMean: number;
    vortStd: number;
    cohMean: number;
    cohStd: number;
    sampleCount: number;
  };
  compositor: {
    effectiveBlend: number;
    surfaceMean: number;
    surfaceMax: number;
    surfaceCount: number;
  };
  volume: {
    phaseMean: number;
    phaseStd: number;
    depthMean: number;
    depthStd: number;
    depthGradMean: number;
    depthGradStd: number;
    intensityMean: number;
    intensityStd: number;
    sampleCount: number;
  };
  parallax: ParallaxMetrics;
  motionEnergy: MotionEnergyMetrics;
  composer: ComposerTelemetry;
  kuramoto: {
    orderParameter: {
      magnitude: number;
      phase: number;
      real: number;
      imag: number;
      sampleCount: number;
    };
    interference: {
      mean: number;
      variance: number;
      max: number;
    };
    frameId: number;
    timestamp: number;
    dt: number;
    kernelVersion: number;
    kernel: KernelSpec;
  };
  texture: TextureMetrics;
  su7: Su7Telemetry;
};

export type RimDebugRequest = {
  energy: Float32Array;
  hue: Float32Array;
  energyHist: Uint32Array;
  hueHist: Uint32Array;
};

export type RimDebugResult = {
  energyMin: number;
  energyMax: number;
  hueMin: number;
  hueMax: number;
};

export type SurfaceDebugRequest = {
  phases: Float32Array[];
  magnitudes: Float32Array[];
  magnitudeHist: Float32Array;
  orientationCount: number;
  flowVectors?: {
    x: Float32Array;
    y: Float32Array;
    hyperbolicScale?: Float32Array;
  };
};

export type SurfaceDebugResult = {
  magnitudeMax: number;
  orientationCount: number;
};

export type RainbowFrameInput = {
  width: number;
  height: number;
  timeSeconds: number;
  out: Uint8ClampedArray;
  surface: SurfaceField | null;
  rim: RimField | null;
  phase: PhaseField | null;
  volume: VolumeField | null;
  kernel: KernelSpec;
  dmt: number;
  arousal: number;
  blend: number;
  normPin: boolean;
  normTarget: number;
  lastObs: number;
  lambdaRef: number;
  lambdas: { L: number; M: number; S: number };
  beta2: number;
  microsaccade: boolean;
  alive: boolean;
  phasePin: boolean;
  edgeThreshold: number;
  wallpaperGroup: WallpaperGroup;
  surfEnabled: boolean;
  orientationAngles: number[];
  thetaMode: 'gradient' | 'global';
  thetaGlobal: number;
  polBins: number;
  jitter: number;
  coupling: CouplingConfig;
  couplingBase?: CouplingConfig;
  sigma: number;
  contrast: number;
  rimAlpha: number;
  rimEnabled: boolean;
  displayMode: DisplayMode;
  surfaceBlend: number;
  surfaceRegion: SurfaceRegion;
  warpAmp: number;
  curvatureStrength: number;
  curvatureMode: CurvatureMode;
  hyperbolicAtlas?: HyperbolicAtlas | null;
  kurEnabled: boolean;
  kurTelemetry?: KuramotoTelemetrySnapshot;
  debug?: {
    rim?: RimDebugRequest;
    surface?: SurfaceDebugRequest;
  };
  su7: Su7RuntimeParams;
  composer?: ComposerConfig;
  attentionHooks?: BranchAttentionHooks;
};

export type RainbowFrameResult = {
  metrics: RainbowFrameMetrics;
  obsAverage: number | null;
  debug?: {
    rim?: RimDebugResult;
    surface?: SurfaceDebugResult;
  };
};

export const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

const clamp01 = (v: number) => clamp(v, 0, 1);

const gauss = (x: number, s: number) => Math.exp(-(x * x) / (2 * s * s + 1e-9));

const luma01 = (r: number, g: number, b: number) => clamp01(0.2126 * r + 0.7152 * g + 0.0722 * b);

const mixScalar = (a: number, b: number, t: number) => a * (1 - t) + b * t;

const wrapPi = (theta: number) => {
  let t = theta;
  const twoPi = 2 * Math.PI;
  t = t - Math.floor((t + Math.PI) / twoPi) * twoPi;
  return t - Math.PI;
};

const computeC7Norm = (vector: C7Vector): number => {
  let sum = 0;
  for (let i = 0; i < vector.length; i++) {
    const cell = vector[i];
    sum += cell.re * cell.re + cell.im * cell.im;
  }
  return Math.sqrt(sum);
};

const responseSoft = (value: number, shaping: number) => {
  const v = clamp01(value);
  const exp = mixScalar(1.2, 3.5, clamp01(shaping));
  return 1 - Math.pow(1 - v, exp);
};

const responsePow = (value: number, shaping: number) => {
  const v = clamp01(value);
  const exp = mixScalar(1.65, 0.55, clamp01(shaping));
  return Math.pow(v, exp);
};

const FLOW_FIELD_GRID_SIZE = 4;

export const deriveSu7ScheduleContext = (params: {
  width: number;
  height: number;
  phase: PhaseField | null;
  rim: RimField | null;
  volume: VolumeField | null;
  dmt: number;
  arousal: number;
  curvatureStrength: number;
}): Su7ScheduleContext => {
  const { width, height, phase, rim, volume, dmt, arousal, curvatureStrength } = params;
  const gradX = phase?.gradX ?? rim?.gx ?? null;
  const gradY = phase?.gradY ?? rim?.gy ?? null;
  const gridSize = FLOW_FIELD_GRID_SIZE;
  const gridVectors = new Float32Array(gridSize * gridSize * 2);
  const gridCounts = new Uint32Array(gridSize * gridSize);
  const axisBias = new Float32Array(7);
  axisBias.fill(1e-5);
  const stride = Math.max(1, Math.floor(Math.sqrt(Math.max(width * height, 1)) / 48));
  let sumGx = 0;
  let sumGy = 0;
  let magSum = 0;
  let magSqSum = 0;
  let samples = 0;
  let radialSum = 0;
  let radialAbsSum = 0;
  const cx = (width - 1) * 0.5;
  const cy = (height - 1) * 0.5;

  if (gradX && gradY) {
    for (let y = 0; y < height; y += stride) {
      for (let x = 0; x < width; x += stride) {
        const idx = y * width + x;
        const gx = Number.isFinite(gradX[idx]) ? gradX[idx] : 0;
        const gy = Number.isFinite(gradY[idx]) ? gradY[idx] : 0;
        const mag = Math.hypot(gx, gy);
        if (mag <= 1e-6) continue;
        sumGx += gx;
        sumGy += gy;
        magSum += mag;
        magSqSum += mag * mag;
        samples++;
        const normX = width > 1 ? x / (width - 1) : 0;
        const normY = height > 1 ? y / (height - 1) : 0;
        const cellX = Math.min(gridSize - 1, Math.floor(normX * gridSize));
        const cellY = Math.min(gridSize - 1, Math.floor(normY * gridSize));
        const cell = cellY * gridSize + cellX;
        gridVectors[cell * 2] += gx;
        gridVectors[cell * 2 + 1] += gy;
        gridCounts[cell] += 1;
        const angle = Math.atan2(gy, gx);
        const axis = ((Math.round(((((angle % TAU) + TAU) % TAU) / TAU) * 7) % 7) + 7) % 7;
        axisBias[axis] += mag;
        const dx = x - cx;
        const dy = y - cy;
        const radius = Math.hypot(dx, dy);
        if (radius > 1e-6) {
          const radialX = dx / radius;
          const radialY = dy / radius;
          const radialComponent = gx * radialX + gy * radialY;
          radialSum += radialComponent;
          radialAbsSum += Math.abs(radialComponent);
        }
      }
    }
  }

  for (let i = 0; i < gridCounts.length; i++) {
    const count = gridCounts[i];
    if (count > 0) {
      const inv = 1 / count;
      gridVectors[i * 2] *= inv;
      gridVectors[i * 2 + 1] *= inv;
    }
  }

  let flowContext: Su7ScheduleContext['flow'] = null;
  if (samples > 0) {
    const meanMag = magSum / samples;
    const variance = Math.max(0, magSqSum / samples - meanMag * meanMag);
    const coherence = meanMag > 1e-6 ? Math.exp(-variance / (meanMag * meanMag + 1e-6)) : 0;
    const axisBiasNorm = new Float32Array(7);
    let maxBias = 0;
    for (let i = 0; i < 7; i++) {
      if (axisBias[i] > maxBias) {
        maxBias = axisBias[i];
      }
    }
    const denom = maxBias > 1e-6 ? maxBias : 1;
    for (let i = 0; i < 7; i++) {
      axisBiasNorm[i] = axisBias[i] / denom;
    }
    flowContext = {
      angle: Math.atan2(sumGy, sumGx),
      magnitude: meanMag,
      coherence,
      axisBias: axisBiasNorm,
      gridSize,
      gridVectors,
    };
  }

  let volumeCoverage = 0;
  if (volume?.depth) {
    const depth = volume.depth;
    let sum = 0;
    let count = 0;
    for (let y = 0; y < height; y += stride) {
      for (let x = 0; x < width; x += stride) {
        const idx = y * width + x;
        const value = Number.isFinite(depth[idx]) ? Math.abs(depth[idx]) : 0;
        sum += value;
        count++;
      }
    }
    volumeCoverage = count > 0 ? Math.tanh(sum / count) : 0;
  }

  const parallaxRadial = radialAbsSum > 1e-6 ? clamp(radialSum / radialAbsSum, -1, 1) : 0;

  return {
    dmt: clamp01(dmt),
    arousal: clamp01(arousal),
    flow: flowContext,
    curvatureStrength,
    parallaxRadial,
    volumeCoverage,
  };
};

type BranchAccum = {
  phaseCos: number;
  phaseSin: number;
  parallaxSum: number;
  magnitudeSum: number;
  hyperbolicSum: number;
  count: number;
};

const createBranchAccumulators = (count: number): BranchAccum[] =>
  Array.from({ length: count }, () => ({
    phaseCos: 0,
    phaseSin: 0,
    parallaxSum: 0,
    magnitudeSum: 0,
    hyperbolicSum: 0,
    count: 0,
  }));

const computeMotionEnergyFromBranches = (
  branches: BranchAccum[] | null,
  source: MotionEnergyMetrics['source'],
): MotionEnergyMetrics | null => {
  if (!branches) return null;
  const phases: number[] = [];
  const tags: number[] = [];
  for (const branch of branches) {
    if (branch.count <= 0) continue;
    const magnitude = Math.hypot(branch.phaseCos, branch.phaseSin);
    const angle = magnitude > 1e-9 ? Math.atan2(branch.phaseSin, branch.phaseCos) : 0;
    const tag = clamp(branch.parallaxSum / branch.count, -1, 1);
    phases.push(angle);
    tags.push(tag);
  }
  const branchCount = phases.length;
  if (branchCount === 0) {
    return null;
  }
  if (branchCount === 1) {
    const tag = tags[0];
    return {
      branchCount,
      phaseShiftMean: 0,
      phaseShiftStd: 0,
      parallaxMean: tag,
      parallaxStd: 0,
      parallaxAbsMean: Math.abs(tag),
      source,
    };
  }
  let meanCos = 0;
  let meanSin = 0;
  for (const phase of phases) {
    meanCos += Math.cos(phase);
    meanSin += Math.sin(phase);
  }
  meanCos /= branchCount;
  meanSin /= branchCount;
  const meanAngle = Math.atan2(meanSin, meanCos);

  let shiftSum = 0;
  let shiftSumSq = 0;
  for (const phase of phases) {
    const delta = Math.abs(wrapPi(phase - meanAngle));
    shiftSum += delta;
    shiftSumSq += delta * delta;
  }
  const phaseShiftMean = shiftSum / branchCount;
  const phaseShiftVar = Math.max(0, shiftSumSq / branchCount - phaseShiftMean * phaseShiftMean);
  const phaseShiftStd = Math.sqrt(phaseShiftVar);

  let tagMean = 0;
  let tagAbsSum = 0;
  for (const tag of tags) {
    tagMean += tag;
    tagAbsSum += Math.abs(tag);
  }
  tagMean /= branchCount;
  const parallaxAbsMean = tagAbsSum / branchCount;

  let tagVar = 0;
  for (const tag of tags) {
    const diff = tag - tagMean;
    tagVar += diff * diff;
  }
  const parallaxStd = Math.sqrt(Math.max(0, tagVar / branchCount));

  return {
    branchCount,
    phaseShiftMean,
    phaseShiftStd,
    parallaxMean: tagMean,
    parallaxStd,
    parallaxAbsMean,
    source,
  };
};

const finalizeBranchAttentionSummary = (
  branches: BranchAccum[] | null,
  kind: BranchAttentionSummary['kind'],
): BranchAttentionSummary | null => {
  if (!branches) return null;
  const entries: BranchAttentionEntry[] = [];
  for (let index = 0; index < branches.length; index++) {
    const branch = branches[index];
    if (branch.count <= 0) continue;
    const inv = 1 / branch.count;
    const meanCos = branch.phaseCos * inv;
    const meanSin = branch.phaseSin * inv;
    const coherence = Math.min(1, Math.hypot(meanCos, meanSin));
    const meanPhase = Math.atan2(meanSin, meanCos);
    const meanParallax = clamp(branch.parallaxSum * inv, -1, 1);
    const hyperbolicGain = branch.hyperbolicSum > 0 ? Math.max(branch.hyperbolicSum * inv, 0) : 1;
    const magnitude = Math.max(branch.magnitudeSum * inv, 0);
    entries.push({
      index,
      sampleCount: branch.count,
      meanPhase,
      coherence,
      meanParallax,
      hyperbolicGain,
      magnitude,
    });
  }
  if (!entries.length) return null;
  return {
    kind,
    entries,
  };
};

export const computeHyperbolicFlowScale = (hyperRadius: number, curvatureStrength: number) => {
  const strength = clamp01(curvatureStrength);
  if (strength <= 1e-6) return 1;
  const limitedRadius = Math.min(Math.abs(hyperRadius), 6);
  if (limitedRadius <= 1e-6) return 1;
  const base = Math.sinh(limitedRadius) / limitedRadius;
  if (!Number.isFinite(base)) return 1;
  return Math.max(1, 1 + (base - 1) * strength);
};

const computeHyperbolicTransparencyBoost = (
  hyperScale: number,
  radiusNorm: number,
  tagAbs: number,
  kNorm: number,
  transparency: number,
) => {
  if (hyperScale <= 1 + 1e-6 || kNorm <= 1e-6) return 1;
  const strength = clamp01(0.45 + 0.4 * transparency);
  if (strength <= 1e-6) return 1;
  const weight = clamp01(radiusNorm) * clamp01(kNorm) * clamp01(tagAbs);
  const boost = 1 + (hyperScale - 1) * strength * weight;
  return clamp(boost, 0.4, 2.8);
};

export const hash2 = (x: number, y: number) => {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
};

export const kEff = (k: KernelSpec, d: number): KernelSpec => ({
  gain: k.gain * (1 + DMT_SENS.g1 * d),
  k0: k.k0 * (1 + DMT_SENS.k1 * d),
  Q: k.Q * (1 + DMT_SENS.q1 * d),
  anisotropy: k.anisotropy + DMT_SENS.a1 * d,
  chirality: k.chirality + DMT_SENS.c1 * d,
  transparency: k.transparency + DMT_SENS.t1 * d,
  couplingPreset: k.couplingPreset,
});

export const groupOps = (kind: WallpaperGroup): Op[] => {
  const HALF = 0.5;
  const THIRD = 1 / 3;
  const rot = (angle: number, tx = 0, ty = 0): Op => ({
    kind: 'rot',
    angle,
    tx,
    ty,
  });
  const id = (tx = 0, ty = 0): Op => rot(0, tx, ty);
  const mirrorX = (tx = 0, ty = 0): Op => ({ kind: 'mirrorX', tx, ty });
  const mirrorY = (tx = 0, ty = 0): Op => ({ kind: 'mirrorY', tx, ty });
  const diag1 = (tx = 0, ty = 0): Op => ({ kind: 'diag1', tx, ty });
  const diag2 = (tx = 0, ty = 0): Op => ({ kind: 'diag2', tx, ty });

  switch (kind) {
    case 'p1':
      return [id(), id(HALF, 0), id(0, HALF), id(HALF, HALF)];
    case 'p2':
      return [rot(0), rot(Math.PI), rot(0, HALF, HALF), rot(Math.PI, HALF, HALF)];
    case 'pm':
      return [id(), mirrorX(), id(HALF, 0), mirrorX(HALF, 0)];
    case 'pg':
      return [id(), mirrorX(HALF, 0), id(0, HALF), mirrorX(HALF, HALF)];
    case 'cm':
      return [id(), mirrorY(0, HALF), mirrorX(HALF, 0), mirrorY(HALF, HALF)];
    case 'pmm':
      return [rot(0), rot(Math.PI), mirrorX(), mirrorY()];
    case 'pmg':
      return [rot(0), mirrorX(), mirrorY(HALF, 0), rot(Math.PI, HALF, 0)];
    case 'pgg':
      return [rot(0), rot(Math.PI), mirrorX(HALF, 0), mirrorY(0, HALF)];
    case 'cmm':
      return [rot(0), mirrorX(), mirrorY(), diag1(), diag2(), rot(Math.PI, HALF, HALF)];
    case 'p4':
      return [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2].map((angle) => rot(angle));
    case 'p4m':
      return [
        rot(0),
        rot(Math.PI / 2),
        rot(Math.PI),
        rot((3 * Math.PI) / 2),
        mirrorX(),
        mirrorY(),
        diag1(),
        diag2(),
      ];
    case 'p4g':
      return [
        rot(0),
        rot(Math.PI / 2),
        rot(Math.PI),
        rot((3 * Math.PI) / 2),
        mirrorX(HALF, 0),
        mirrorY(0, HALF),
        diag1(HALF, 0),
        diag2(0, HALF),
      ];
    case 'p3':
      return [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3].map((angle) => rot(angle));
    case 'p3m1':
      return [
        rot(0),
        rot((2 * Math.PI) / 3),
        rot((4 * Math.PI) / 3),
        mirrorX(),
        diag1(THIRD, 0),
        diag2(0, THIRD),
      ];
    case 'p31m':
      return [
        rot(0),
        rot((2 * Math.PI) / 3),
        rot((4 * Math.PI) / 3),
        mirrorY(),
        diag1(2 * THIRD, THIRD),
        diag2(THIRD, 2 * THIRD),
      ];
    case 'p6':
      return Array.from({ length: 6 }, (_, j) => rot((j * Math.PI) / 3));
    case 'p6m':
      return [...Array.from({ length: 6 }, (_, j) => rot((j * Math.PI) / 3)), mirrorX(), mirrorY()];
    default:
      return [rot(0)];
  }
};

export type GpuWallpaperOp = {
  kind: number;
  angle: number;
  tx: number;
  ty: number;
};

export const toGpuOps = (ops: Op[]): GpuWallpaperOp[] =>
  ops.map((op) => {
    const tx = op.tx ?? 0;
    const ty = op.ty ?? 0;
    switch (op.kind) {
      case 'rot':
        return { kind: 0, angle: op.angle, tx, ty };
      case 'mirrorX':
        return { kind: 1, angle: 0, tx, ty };
      case 'mirrorY':
        return { kind: 2, angle: 0, tx, ty };
      case 'diag1':
        return { kind: 3, angle: 0, tx, ty };
      case 'diag2':
        return { kind: 4, angle: 0, tx, ty };
      default:
        return { kind: 0, angle: 0, tx, ty };
    }
  });

export const applyOp = (op: Op, x: number, y: number, cx: number, cy: number) => {
  const dx = x - cx;
  const dy = y - cy;
  let px = dx;
  let py = dy;
  switch (op.kind) {
    case 'rot': {
      const c = Math.cos(op.angle);
      const s = Math.sin(op.angle);
      px = c * dx - s * dy;
      py = s * dx + c * dy;
      break;
    }
    case 'mirrorX':
      px = -dx;
      py = dy;
      break;
    case 'mirrorY':
      px = dx;
      py = -dy;
      break;
    case 'diag1':
      px = dy;
      py = dx;
      break;
    case 'diag2':
      px = -dy;
      py = -dx;
      break;
    default:
      break;
  }
  const width = cx * 2;
  const height = cy * 2;
  const tx = (op.tx ?? 0) * width;
  const ty = (op.ty ?? 0) * height;
  return { x: cx + px + tx, y: cy + py + ty };
};

export const sampleScalar = (arr: Float32Array, x: number, y: number, W: number, H: number) => {
  const xx = clamp(x, 0, W - 1.001);
  const yy = clamp(y, 0, H - 1.001);
  const x0 = Math.floor(xx);
  const y0 = Math.floor(yy);
  const x1 = x0 + 1;
  const y1 = y0 + 1;
  const fx = xx - x0;
  const fy = yy - y0;
  const idx = (ix: number, iy: number) => iy * W + ix;
  const v00 = arr[idx(x0, y0)];
  const v10 = arr[idx(Math.min(x1, W - 1), y0)];
  const v01 = arr[idx(x0, Math.min(y1, H - 1))];
  const v11 = arr[idx(Math.min(x1, W - 1), Math.min(y1, H - 1))];
  const v0 = v00 * (1 - fx) + v10 * fx;
  const v1 = v01 * (1 - fx) + v11 * fx;
  return v0 * (1 - fy) + v1 * fy;
};

export const sampleRGB = (data: Uint8ClampedArray, x: number, y: number, W: number, H: number) => {
  const xx = clamp(x, 0, W - 1.001);
  const yy = clamp(y, 0, H - 1.001);
  const x0 = Math.floor(xx);
  const y0 = Math.floor(yy);
  const x1 = Math.min(x0 + 1, W - 1);
  const y1 = Math.min(y0 + 1, H - 1);
  const fx = xx - x0;
  const fy = yy - y0;
  const idx = (ix: number, iy: number) => (iy * W + ix) * 4;
  const mix = (a: number, b: number, t: number) => a * (1 - t) + b * t;
  const i00 = idx(x0, y0);
  const i10 = idx(x1, y0);
  const i01 = idx(x0, y1);
  const i11 = idx(x1, y1);
  const r0 = mix(data[i00], data[i10], fx);
  const g0 = mix(data[i00 + 1], data[i10 + 1], fx);
  const b0 = mix(data[i00 + 2], data[i10 + 2], fx);
  const r1 = mix(data[i01], data[i11], fx);
  const g1 = mix(data[i01 + 1], data[i11 + 1], fx);
  const b1 = mix(data[i01 + 2], data[i11 + 2], fx);
  return {
    R: mix(r0, r1, fy),
    G: mix(g0, g1, fy),
    B: mix(b0, b1, fy),
  };
};

const wallpaperAt = (
  xp: number,
  yp: number,
  cosA: number[],
  sinA: number[],
  ke: KernelSpec,
  tSeconds: number,
  alive: boolean,
) => {
  const N = cosA.length;
  const twoPI = 2 * Math.PI;
  let gx = 0;
  let gy = 0;
  for (let j = 0; j < N; j++) {
    const phase =
      ke.chirality * (j / Math.max(1, N)) +
      (alive ? 0.2 * Math.sin(twoPI * 0.3 * tSeconds + j) : 0);
    const s = xp * cosA[j] + yp * sinA[j];
    const arg = twoPI * ke.k0 * s + phase;
    const d = -twoPI * ke.k0 * Math.sin(arg);
    gx += d * cosA[j];
    gy += d * sinA[j];
  }
  const inv = N > 0 ? 1 / N : 1;
  return { gx: gx * inv, gy: gy * inv };
};

const finalizeStats = (sum: number, sumSq: number, count: number) => {
  if (count <= 0) {
    return { mean: 0, std: 0 };
  }
  const mean = sum / count;
  const variance = clamp(sumSq / count - mean * mean, 0, Infinity);
  return { mean, std: Math.sqrt(variance) };
};

const TAU = 2 * Math.PI;
const CRYSTAL_BEAT_SCALE = 12;
const RIM_ENERGY_HIST_SCALE = 4;
const SURFACE_HIST_BINS = 32;

export const renderRainbowFrame = (input: RainbowFrameInput): RainbowFrameResult => {
  const {
    width,
    height,
    timeSeconds,
    out,
    surface,
    rim,
    phase,
    volume,
    kernel,
    dmt,
    arousal,
    blend,
    normPin,
    normTarget,
    lastObs,
    lambdaRef,
    lambdas,
    beta2,
    microsaccade,
    alive,
    phasePin,
    edgeThreshold,
    wallpaperGroup,
    surfEnabled,
    orientationAngles,
    thetaMode,
    thetaGlobal,
    polBins,
    jitter,
    coupling,
    sigma,
    contrast,
    rimAlpha,
    rimEnabled,
    displayMode,
    surfaceBlend,
    surfaceRegion,
    warpAmp,
    curvatureStrength,
    curvatureMode,
    kurEnabled,
    debug,
    su7,
    composer,
    attentionHooks,
  } = input;
  const rimDebug = debug?.rim ?? null;
  const surfaceDebug = debug?.surface ?? null;
  if (rimDebug) {
    rimDebug.energyHist.fill(0);
    rimDebug.hueHist.fill(0);
  }
  if (surfaceDebug) {
    surfaceDebug.magnitudeHist.fill(0);
  }

  const kernelSpec = clampKernelSpec(kernel);

  const composerConfig = sanitizeComposerConfig(composer);
  const composerFields = composerConfig.fields;
  const initialFieldMetrics = (field: ComposerFieldId): ComposerFieldMetrics => ({
    energy: 0,
    share: 0,
    weight: composerFields[field].weight,
    blend: 0,
  });
  const baseCoupling: CouplingConfig = input.couplingBase
    ? { ...input.couplingBase }
    : { ...coupling };
  const composerTelemetry: ComposerTelemetry = {
    fields: {
      surface: initialFieldMetrics('surface'),
      rim: initialFieldMetrics('rim'),
      kur: initialFieldMetrics('kur'),
      volume: initialFieldMetrics('volume'),
    },
    dmtRouting: composerConfig.dmtRouting,
    solverRegime: composerConfig.solverRegime,
    coupling: {
      base: baseCoupling,
      effective: { ...coupling },
      scale: 1,
    },
  };

  const su7Params: Su7RuntimeParams = su7 ?? createDefaultSu7RuntimeParams();
  let surfaceFieldActive = surface;
  const surfaceData = surface?.rgba ?? null;
  const rimField = rim ?? null;
  const phaseField = phase ?? null;
  const volumeField = volume ?? null;
  let volumePhase = volumeField?.phase ?? null;
  let volumeDepth = volumeField?.depth ?? null;
  let volumeIntensity = volumeField?.intensity ?? null;
  let volumeDepthGrad: Float32Array | null = null;
  let volPhaseSum = 0;
  let volPhaseSumSq = 0;
  let volDepthSum = 0;
  let volDepthSumSq = 0;
  let volIntensitySum = 0;
  let volIntensitySumSq = 0;
  let volGradSum = 0;
  let volGradSumSq = 0;
  let volSampleCount = 0;
  const su7Context: Su7ScheduleContext | null =
    su7Params.enabled && su7Params.gain > 1e-4
      ? deriveSu7ScheduleContext({
          width,
          height,
          phase: phaseField,
          rim: rimField,
          volume: volumeField,
          dmt,
          arousal,
          curvatureStrength,
        })
      : null;
  const su7Telemetry = computeSu7Telemetry(su7Params, su7Context ?? undefined);

  const metrics: RainbowFrameMetrics = {
    rim: { mean: 0, max: 0, std: 0, count: 0 },
    warp: { mean: 0, std: 0, dominantAngle: 0, count: 0 },
    gradient: {
      gradMean: 0,
      gradStd: 0,
      vortMean: 0,
      vortStd: 0,
      cohMean: 0,
      cohStd: 0,
      sampleCount: 0,
    },
    compositor: {
      effectiveBlend: 0,
      surfaceMean: 0,
      surfaceMax: 0,
      surfaceCount: 0,
    },
    volume: {
      phaseMean: 0,
      phaseStd: 0,
      depthMean: 0,
      depthStd: 0,
      depthGradMean: 0,
      depthGradStd: 0,
      intensityMean: 0,
      intensityStd: 0,
      sampleCount: 0,
    },
    parallax: {
      radialSlope: 0,
      eccentricityMean: 0,
      kMean: 0,
      kStd: 0,
      tagConsistency: 0,
      sampleCount: 0,
    },
    motionEnergy: {
      branchCount: 0,
      phaseShiftMean: 0,
      phaseShiftStd: 0,
      parallaxMean: 0,
      parallaxStd: 0,
      parallaxAbsMean: 0,
      source: 'none',
    },
    composer: composerTelemetry,
    kuramoto: {
      orderParameter: {
        magnitude: 0,
        phase: 0,
        real: 0,
        imag: 0,
        sampleCount: 0,
      },
      interference: {
        mean: 0,
        variance: 0,
        max: 0,
      },
      frameId: -1,
      timestamp: 0,
      dt: 0,
      kernelVersion: 0,
      kernel: cloneKernelSpec(kernelSpec),
    },
    texture: {
      wallpapericity: 0,
      wallpaperStd: 0,
      beatEnergy: 0,
      beatStd: 0,
      resonanceRate: 0,
      sampleCount: 0,
      earlyVision: {
        wallpapericity: 0,
        wallpaperStd: 0,
        beatEnergy: 0,
        beatStd: 0,
        resonanceRate: 0,
        sampleCount: 0,
        dogMean: 0,
        dogStd: 0,
        orientationMean: 0,
        orientationStd: 0,
        divisiveMean: 0,
        divisiveStd: 0,
      },
      crystallizer: {
        wallpapericity: 0,
        wallpaperStd: 0,
        beatEnergy: 0,
        beatStd: 0,
        resonanceRate: 0,
        sampleCount: 0,
      },
    },
    su7: su7Telemetry,
  };

  if (input.kurTelemetry) {
    const tele = input.kurTelemetry;
    metrics.kuramoto.orderParameter.magnitude = tele.orderParameter.magnitude;
    metrics.kuramoto.orderParameter.phase = tele.orderParameter.phase;
    metrics.kuramoto.orderParameter.real = tele.orderParameter.real;
    metrics.kuramoto.orderParameter.imag = tele.orderParameter.imag;
    metrics.kuramoto.orderParameter.sampleCount = tele.orderParameter.sampleCount;
    metrics.kuramoto.interference.mean = tele.interference.mean;
    metrics.kuramoto.interference.variance = tele.interference.variance;
    metrics.kuramoto.interference.max = tele.interference.max;
    metrics.kuramoto.frameId = tele.frameId;
    metrics.kuramoto.timestamp = tele.timestamp;
    metrics.kuramoto.dt = tele.dt;
    metrics.kuramoto.kernelVersion = tele.kernelVersion;
    metrics.kuramoto.kernel = cloneKernelSpec(tele.kernel);
  }

  const [rimComposerGain, surfaceComposerGain] = computeComposerBlendGain(composerConfig);

  const applyComposerScalar = (value: number, config: ComposerFieldConfig) => {
    const scaled = clamp01(value * config.exposure);
    const gammaApplied = Math.pow(scaled, config.gamma);
    return clamp01(gammaApplied);
  };

  const applyComposerColor = (value: number, config: ComposerFieldConfig) =>
    clamp01(Math.pow(clamp01(value * config.exposure), config.gamma));

  const applyComposerVec3 = (
    rgb: [number, number, number],
    config: ComposerFieldConfig,
  ): [number, number, number] => [
    applyComposerColor(rgb[0], config),
    applyComposerColor(rgb[1], config),
    applyComposerColor(rgb[2], config),
  ];

  const composerEnergy: Record<ComposerFieldId, number> = {
    surface: 0,
    rim: 0,
    kur: 0,
    volume: 0,
  };
  const composerBlend: Record<ComposerFieldId, number> = {
    surface: 0,
    rim: 0,
    kur: 0,
    volume: 0,
  };
  const composerSamples: Record<ComposerFieldId, number> = {
    surface: 0,
    rim: 0,
    kur: 0,
    volume: 0,
  };

  const {
    rimToSurfaceBlend,
    rimToSurfaceAlign,
    surfaceToRimOffset,
    surfaceToRimSigma,
    surfaceToRimHue,
    kurToTransparency,
    kurToOrientation,
    kurToChirality,
    volumePhaseToHue,
    volumeDepthToWarp,
  } = coupling;

  const coupleVolumePhase = volumePhase != null && volumePhaseToHue > 1e-4;
  let coupleVolumeWarp = false;

  const coupleRimSurface = rimToSurfaceBlend > 1e-4 || rimToSurfaceAlign > 1e-4;
  const coupleSurfaceRim =
    surfaceToRimOffset > 1e-4 || surfaceToRimSigma > 1e-4 || surfaceToRimHue > 1e-4;
  const coupleKurTransparency = kurEnabled && kurToTransparency > 1e-4;
  const coupleKurOrientation = kurEnabled && kurToOrientation > 1e-4;
  const coupleKurChirality = kurEnabled && kurToChirality > 1e-4;
  const couplingDmtScale = 1 + 0.65 * dmt;
  composerTelemetry.coupling.scale = couplingDmtScale;
  composerTelemetry.coupling.effective = {
    rimToSurfaceBlend: coupling.rimToSurfaceBlend * couplingDmtScale,
    rimToSurfaceAlign: coupling.rimToSurfaceAlign * couplingDmtScale,
    surfaceToRimOffset: coupling.surfaceToRimOffset * couplingDmtScale,
    surfaceToRimSigma: coupling.surfaceToRimSigma * couplingDmtScale,
    surfaceToRimHue: coupling.surfaceToRimHue * couplingDmtScale,
    kurToTransparency: coupling.kurToTransparency * couplingDmtScale,
    kurToOrientation: coupling.kurToOrientation * couplingDmtScale,
    kurToChirality: coupling.kurToChirality * couplingDmtScale,
    volumePhaseToHue: coupling.volumePhaseToHue * couplingDmtScale,
    volumeDepthToWarp: coupling.volumeDepthToWarp * couplingDmtScale,
  };
  const sigmaFloorBase = Math.max(0.25, sigma * 0.25);

  const hasImage = Boolean(surfaceData && rimField);
  if (!hasImage) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        const n = Math.sin((x / width) * Math.PI * 6 + timeSeconds);
        const m = Math.sin((y / height) * Math.PI * 6 - timeSeconds * 0.7);
        const v = clamp01(0.5 + 0.5 * n * m);
        out[i + 0] = Math.floor(v * 255);
        out[i + 1] = Math.floor((0.4 + 0.6 * v) * 180);
        out[i + 2] = Math.floor((1 - v) * 255);
        out[i + 3] = 255;
      }
    }
    return { metrics, obsAverage: null };
  }

  let baseData = surfaceData!;
  let gx = rimField!.gx;
  let gy = rimField!.gy;
  let mag = rimField!.mag;
  let gradX = phaseField?.gradX ?? null;
  let gradY = phaseField?.gradY ?? null;
  let vort = phaseField?.vort ?? null;
  let coh = phaseField?.coh ?? null;
  let amp = phaseField?.amp ?? null;
  let su7Vectors: C7Vector[] | null = null;
  let su7Norms: Float32Array | null = null;
  let su7Unitary: Complex7x7 | null = null;
  const su7Projector = su7Params.projector;
  const su7Active = su7Params.enabled && su7Params.gain > 1e-4;
  if (su7Active) {
    const embedResult = embedToC7({
      surface,
      rim: rimField,
      phase: phaseField,
      volume,
      width,
      height,
      gauge: 'rim',
    });
    if (embedResult.vectors.length === width * height) {
      su7Vectors = embedResult.vectors;
      su7Norms = embedResult.norms;
      su7Unitary = buildScheduledUnitary(su7Params, su7Context ?? undefined);
    }
  }
  let su7DecimationStride = 1;
  let su7DecimationMode: 'hybrid' | 'stride' | 'edges' = 'hybrid';
  if (su7Active) {
    const rawParams = su7Params as Record<string, unknown>;
    const strideCandidate = rawParams['decimationStride'];
    if (
      typeof strideCandidate === 'number' &&
      Number.isFinite(strideCandidate) &&
      strideCandidate >= 1
    ) {
      su7DecimationStride = Math.max(1, Math.floor(strideCandidate));
    } else {
      su7DecimationStride = 2;
    }
    const modeCandidate = rawParams['decimationMode'];
    if (modeCandidate === 'stride' || modeCandidate === 'edges' || modeCandidate === 'hybrid') {
      su7DecimationMode = modeCandidate;
    } else {
      su7DecimationMode = 'hybrid';
    }
  }
  let su7NormDeltaSum = 0;
  let su7NormDeltaMax = 0;
  let su7NormSampleCount = 0;
  let su7EnergySum = 0;
  let su7EnergySampleCount = 0;
  const ke = clampKernelSpec(kEff(kernelSpec, dmt));
  const effectiveBlend = clamp01(blend + ke.transparency * 0.5);
  const eps = 1e-6;
  const frameGain = normPin ? Math.pow((normTarget + eps) / (lastObs + eps), 0.5) : 1.0;

  metrics.compositor.effectiveBlend = effectiveBlend;

  const baseOffsets = {
    L: beta2 * (lambdaRef / lambdas.L - 1),
    M: beta2 * (lambdaRef / lambdas.M - 1),
    S: beta2 * (lambdaRef / lambdas.S - 1),
  } as const;

  const jitterPhase = microsaccade ? timeSeconds * 6.0 : 0.0;
  const breath = alive ? 0.15 * Math.sin(2 * Math.PI * 0.55 * timeSeconds) : 0.0;

  const ops = wallpaperGroup === 'off' ? [] : groupOps(wallpaperGroup);
  const opsCount = ops.length;
  const useWallpaper = surfEnabled || coupleSurfaceRim;
  const cx = width * 0.5;
  const cy = height * 0.5;
  const maxRadius = Math.hypot(cx, cy) || 1;
  const cosA = orientationAngles.map((a) => Math.cos(a));
  const sinA = orientationAngles.map((a) => Math.sin(a));
  const orientationCount = cosA.length;
  const orientationMagnitudes = orientationCount > 0 ? new Float32Array(orientationCount) : null;
  const wallpaperMagnitudes = opsCount > 0 ? new Float32Array(opsCount) : null;
  let crystalWallpaperSum = 0;
  let crystalWallpaperSumSq = 0;
  let crystalBeatSum = 0;
  let crystalBeatSumSq = 0;
  let crystalResonance = 0;
  let crystalSamples = 0;

  const curvatureStrengthClamped = clamp(Math.abs(curvatureStrength), 0, 0.95);
  const curvatureActive = curvatureStrengthClamped > 1e-4;
  const atlas = curvatureActive
    ? (input.hyperbolicAtlas ??
      createHyperbolicAtlas({
        width,
        height,
        curvatureStrength: curvatureStrengthClamped,
        mode: curvatureMode,
      }))
    : null;
  const atlasCoords = atlas?.coords ?? null;
  const atlasJacobians = atlas?.jacobians ?? null;
  const atlasAreaWeights = atlas?.areaWeights ?? null;
  const atlasPolar = atlas?.polar ?? null;
  const atlasMetadata = atlas?.metadata ?? null;
  const atlasSampleScale = atlasMetadata?.maxRadius ?? 1;
  const maxHyperRadius =
    curvatureActive && atlasMetadata
      ? 2 * atlasMetadata.curvatureScale * Math.atanh(Math.min(0.999999, atlasMetadata.diskLimit))
      : maxRadius;

  if (curvatureActive && atlas && atlasCoords) {
    const sampleScalar = (buffer: Float32Array, sx: number, sy: number): number => {
      const xClamped = Math.min(Math.max(sx, 0), width - 1);
      const yClamped = Math.min(Math.max(sy, 0), height - 1);
      const x0 = Math.floor(xClamped);
      const y0 = Math.floor(yClamped);
      const x1 = Math.min(x0 + 1, width - 1);
      const y1 = Math.min(y0 + 1, height - 1);
      const tx = xClamped - x0;
      const ty = yClamped - y0;
      const idx00 = y0 * width + x0;
      const idx10 = y0 * width + x1;
      const idx01 = y1 * width + x0;
      const idx11 = y1 * width + x1;
      const v00 = buffer[idx00];
      const v10 = buffer[idx10];
      const v01 = buffer[idx01];
      const v11 = buffer[idx11];
      const v0 = v00 + (v10 - v00) * tx;
      const v1 = v01 + (v11 - v01) * tx;
      return v0 + (v1 - v0) * ty;
    };
    const sampleRgba = (
      buffer: Uint8ClampedArray,
      sx: number,
      sy: number,
      target: Uint8ClampedArray,
      offset: number,
    ) => {
      const xClamped = Math.min(Math.max(sx, 0), width - 1);
      const yClamped = Math.min(Math.max(sy, 0), height - 1);
      const x0 = Math.floor(xClamped);
      const y0 = Math.floor(yClamped);
      const x1 = Math.min(x0 + 1, width - 1);
      const y1 = Math.min(y0 + 1, height - 1);
      const tx = xClamped - x0;
      const ty = yClamped - y0;
      const idx00 = (y0 * width + x0) * 4;
      const idx10 = (y0 * width + x1) * 4;
      const idx01 = (y1 * width + x0) * 4;
      const idx11 = (y1 * width + x1) * 4;
      for (let c = 0; c < 4; c++) {
        const v00 = buffer[idx00 + c];
        const v10 = buffer[idx10 + c];
        const v01 = buffer[idx01 + c];
        const v11 = buffer[idx11 + c];
        const v0 = v00 + (v10 - v00) * tx;
        const v1 = v01 + (v11 - v01) * tx;
        const value = v0 + (v1 - v0) * ty;
        target[offset + c] = Math.round(Math.min(Math.max(value, 0), 255));
      }
    };

    const baseResampled = new Uint8ClampedArray(baseData.length);
    const gxResampled = new Float32Array(gx.length);
    const gyResampled = new Float32Array(gy.length);
    const magResampled = new Float32Array(mag.length);
    const gradXResampled = gradX ? new Float32Array(gradX.length) : null;
    const gradYResampled = gradY ? new Float32Array(gradY.length) : null;
    const vortResampled = vort ? new Float32Array(vort.length) : null;
    const cohResampled = coh ? new Float32Array(coh.length) : null;
    const ampResampled = amp ? new Float32Array(amp.length) : null;
    const volumePhaseResampled = volumePhase ? new Float32Array(volumePhase.length) : null;
    const volumeDepthResampled = volumeDepth ? new Float32Array(volumeDepth.length) : null;
    const volumeIntensityResampled = volumeIntensity
      ? new Float32Array(volumeIntensity.length)
      : null;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        const coordOffset = idx * 2;
        const sx = atlasCoords[coordOffset];
        const sy = atlasCoords[coordOffset + 1];
        const rgbaIdx = idx * 4;
        sampleRgba(baseData, sx, sy, baseResampled, rgbaIdx);
        gxResampled[idx] = sampleScalar(gx, sx, sy);
        gyResampled[idx] = sampleScalar(gy, sx, sy);
        magResampled[idx] = sampleScalar(mag, sx, sy);
        if (gradXResampled && gradX) {
          gradXResampled[idx] = sampleScalar(gradX, sx, sy);
        }
        if (gradYResampled && gradY) {
          gradYResampled[idx] = sampleScalar(gradY, sx, sy);
        }
        if (vortResampled && vort) {
          vortResampled[idx] = sampleScalar(vort, sx, sy);
        }
        if (cohResampled && coh) {
          cohResampled[idx] = sampleScalar(coh, sx, sy);
        }
        if (ampResampled && amp) {
          ampResampled[idx] = sampleScalar(amp, sx, sy);
        }
        if (volumePhaseResampled && volumePhase) {
          volumePhaseResampled[idx] = sampleScalar(volumePhase, sx, sy);
        }
        if (volumeDepthResampled && volumeDepth) {
          volumeDepthResampled[idx] = sampleScalar(volumeDepth, sx, sy);
        }
        if (volumeIntensityResampled && volumeIntensity) {
          volumeIntensityResampled[idx] = sampleScalar(volumeIntensity, sx, sy);
        }
      }
    }

    if (atlasJacobians) {
      const texels = width * height;
      const jacobians = atlasJacobians;
      for (let idx = 0; idx < texels; idx++) {
        const jacOffset = idx * 4;
        const j11 = jacobians[jacOffset] * atlasSampleScale;
        const j12 = jacobians[jacOffset + 1] * atlasSampleScale;
        const j21 = jacobians[jacOffset + 2] * atlasSampleScale;
        const j22 = jacobians[jacOffset + 3] * atlasSampleScale;
        const gradSampleX = gxResampled[idx];
        const gradSampleY = gyResampled[idx];
        const pullbackX = gradSampleX * j11 + gradSampleY * j21;
        const pullbackY = gradSampleX * j12 + gradSampleY * j22;
        gxResampled[idx] = pullbackX;
        gyResampled[idx] = pullbackY;
        magResampled[idx] = Math.hypot(pullbackX, pullbackY);
        if (gradXResampled && gradYResampled) {
          const phaseSampleX = gradXResampled[idx];
          const phaseSampleY = gradYResampled[idx];
          gradXResampled[idx] = phaseSampleX * j11 + phaseSampleY * j21;
          gradYResampled[idx] = phaseSampleX * j12 + phaseSampleY * j22;
        }
      }
    }

    baseData = baseResampled;
    gx = gxResampled;
    gy = gyResampled;
    mag = magResampled;
    if (gradXResampled) gradX = gradXResampled;
    if (gradYResampled) gradY = gradYResampled;
    if (vortResampled) vort = vortResampled;
    if (cohResampled) coh = cohResampled;
    if (ampResampled) amp = ampResampled;
    if (volumePhaseResampled) volumePhase = volumePhaseResampled;
    if (volumeDepthResampled) volumeDepth = volumeDepthResampled;
    if (volumeIntensityResampled) volumeIntensity = volumeIntensityResampled;
    surfaceFieldActive = surface
      ? { kind: 'surface', resolution: surface.resolution, rgba: baseResampled }
      : surfaceFieldActive;
  }

  const textureDiag = computeTextureDiagnostics(surfaceFieldActive ?? surface, {
    orientations: orientationAngles,
  });
  metrics.texture.earlyVision = {
    wallpapericity: textureDiag.wallpapericityMean,
    wallpaperStd: textureDiag.wallpapericityStd,
    beatEnergy: textureDiag.beatEnergyMean,
    beatStd: textureDiag.beatEnergyStd,
    resonanceRate: textureDiag.resonanceRate,
    sampleCount: textureDiag.sampleCount,
    dogMean: textureDiag.dogMean,
    dogStd: textureDiag.dogStd,
    orientationMean: textureDiag.orientationMean,
    orientationStd: textureDiag.orientationStd,
    divisiveMean: textureDiag.divisiveMean,
    divisiveStd: textureDiag.divisiveStd,
  };

  const crystalBeatWeight = Math.min(textureDiag.beatEnergyMean * CRYSTAL_BEAT_SCALE, 1);
  if (volumePhase && volumeDepth && volumeIntensity) {
    volumeDepthGrad = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      const upY = y > 0 ? y - 1 : y;
      const downY = y + 1 < height ? y + 1 : y;
      for (let x = 0; x < width; x++) {
        const leftX = x > 0 ? x - 1 : x;
        const rightX = x + 1 < width ? x + 1 : x;
        const idx = y * width + x;
        const idxLeft = y * width + leftX;
        const idxRight = y * width + rightX;
        const idxUp = upY * width + x;
        const idxDown = downY * width + x;
        const phaseVal = volumePhase[idx];
        const depthVal = volumeDepth[idx];
        const intensityVal = volumeIntensity[idx];
        volPhaseSum += phaseVal;
        volPhaseSumSq += phaseVal * phaseVal;
        volDepthSum += depthVal;
        volDepthSumSq += depthVal * depthVal;
        volIntensitySum += intensityVal;
        volIntensitySumSq += intensityVal * intensityVal;
        let basisX = 1;
        let basisY = 1;
        if (curvatureActive && atlasJacobians) {
          const jacOffset = idx * 4;
          const j11 = atlasJacobians[jacOffset] * atlasSampleScale;
          const j12 = atlasJacobians[jacOffset + 1] * atlasSampleScale;
          const j21 = atlasJacobians[jacOffset + 2] * atlasSampleScale;
          const j22 = atlasJacobians[jacOffset + 3] * atlasSampleScale;
          basisX = Math.hypot(j11, j21);
          basisY = Math.hypot(j12, j22);
        }
        const scaleX = basisX > 1e-6 ? basisX : 1;
        const scaleY = basisY > 1e-6 ? basisY : 1;
        const dx = ((volumeDepth[idxRight] - volumeDepth[idxLeft]) * 0.5) / scaleX;
        const dy = ((volumeDepth[idxDown] - volumeDepth[idxUp]) * 0.5) / scaleY;
        const gradVal = Math.hypot(dx, dy);
        volumeDepthGrad[idx] = gradVal;
        volGradSum += gradVal;
        volGradSumSq += gradVal * gradVal;
        volSampleCount++;
      }
    }
  }
  coupleVolumeWarp = volumeDepthGrad != null && volumeDepthToWarp > 1e-4;
  const orientationBranches =
    orientationCount > 1 ? createBranchAccumulators(orientationCount) : null;
  const wallpaperBranches = opsCount > 1 ? createBranchAccumulators(opsCount) : null;
  let parallaxRadiusSum = 0;
  let parallaxRadiusSqSum = 0;
  let parallaxKSum = 0;
  let parallaxKSumSq = 0;
  let parallaxRadiusKSum = 0;
  let parallaxTagAbsSum = 0;
  let parallaxSamples = 0;
  const accumulateCrystalMetrics = (magnitudes: Float32Array | null, count: number) => {
    if (!magnitudes || count <= 0) return;
    let total = 0;
    let maxVal = 0;
    let secondVal = 0;
    for (let i = 0; i < count; i++) {
      const m = magnitudes[i];
      total += m;
      if (m > maxVal) {
        secondVal = maxVal;
        maxVal = m;
      } else if (m > secondVal) {
        secondVal = m;
      }
    }
    const mean = count > 0 ? total / count : 0;
    const ratio = maxVal > eps ? secondVal / (maxVal + eps) : 0;
    const beatLocal = ratio * mean * crystalBeatWeight;
    crystalWallpaperSum += mean;
    crystalWallpaperSumSq += mean * mean;
    crystalBeatSum += beatLocal;
    crystalBeatSumSq += beatLocal * beatLocal;
    if (crystalBeatWeight > 0 && ratio > 0.6 && mean > 0.1) {
      crystalResonance++;
    }
    crystalSamples++;
  };

  // ∇θ statistics
  if (gradX && gradY) {
    let gradSum = 0;
    let gradSumSq = 0;
    let vortSum = 0;
    let vortSumSq = 0;
    let cohSum = 0;
    let cohSumSq = 0;
    const total = gradX.length;
    let weightSum = 0;
    for (let i = 0; i < total; i++) {
      const weight = curvatureActive && atlasAreaWeights ? atlasAreaWeights[i] : 1;
      const gMag = Math.hypot(gradX[i], gradY[i]);
      gradSum += gMag * weight;
      gradSumSq += gMag * gMag * weight;
      const v = vort ? vort[i] : 0;
      vortSum += v * weight;
      vortSumSq += v * v * weight;
      const c = coh ? coh[i] : 0;
      cohSum += c * weight;
      cohSumSq += c * c * weight;
      weightSum += weight;
    }
    const effectiveCount = weightSum > 1e-9 ? weightSum : total;
    metrics.gradient.gradMean = gradSum / effectiveCount;
    metrics.gradient.gradStd = finalizeStats(gradSum, gradSumSq, effectiveCount).std;
    metrics.gradient.vortMean = vortSum / effectiveCount;
    metrics.gradient.vortStd = finalizeStats(vortSum, vortSumSq, effectiveCount).std;
    metrics.gradient.cohMean = cohSum / effectiveCount;
    metrics.gradient.cohStd = finalizeStats(cohSum, cohSumSq, effectiveCount).std;
    metrics.gradient.sampleCount = effectiveCount;
  }

  let muJ = 0;
  let cnt = 0;
  if (phasePin && microsaccade) {
    const stride = 8;
    for (let yy = 0; yy < height; yy += stride) {
      for (let xx = 0; xx < width; xx += stride) {
        const idx = yy * width + xx;
        if (mag[idx] >= edgeThreshold) {
          const coordOffset = idx * 2;
          const seedX = curvatureActive && atlasCoords ? atlasCoords[coordOffset] : xx;
          const seedY = curvatureActive && atlasCoords ? atlasCoords[coordOffset + 1] : yy;
          muJ += Math.sin(jitterPhase + hash2(seedX, seedY) * Math.PI * 2);
          cnt++;
        }
      }
    }
    muJ = cnt ? muJ / cnt : 0;
  }

  let obsSum = 0;
  let obsCount = 0;

  let rimSum = 0;
  let rimSumSq = 0;
  let rimCount = 0;
  let rimMax = 0;

  let warpSum = 0;
  let warpSumSq = 0;
  let warpCount = 0;
  let warpCos = 0;
  let warpSin = 0;

  let surfaceSum = 0;
  let surfaceMax = 0;
  let surfaceCount = 0;

  let debugEnergyMin = Number.POSITIVE_INFINITY;
  let debugEnergyMax = 0;
  let debugHueMin = Number.POSITIVE_INFINITY;
  let debugHueMax = 0;
  let debugMagnitudeMax = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = y * width + x;
      const i = p * 4;
      const coordOffset = p * 2;
      const posX = curvatureActive && atlasCoords ? atlasCoords[coordOffset] : x;
      const posY = curvatureActive && atlasCoords ? atlasCoords[coordOffset + 1] : y;
      const sampleDx = posX - cx;
      const sampleDy = posY - cy;
      const sampleRadius = Math.hypot(sampleDx, sampleDy);
      const sampleRadiusNorm = maxRadius > 0 ? sampleRadius / maxRadius : 0;
      const hyperRadius = curvatureActive && atlasPolar ? atlasPolar[coordOffset] : sampleRadius;
      const hyperTheta =
        curvatureActive && atlasPolar
          ? atlasPolar[coordOffset + 1]
          : Math.atan2(sampleDy, sampleDx);
      const radiusNorm =
        maxHyperRadius > 1e-6 ? clamp(hyperRadius / maxHyperRadius, 0, 0.999999) : sampleRadiusNorm;
      const hyperFlowScale =
        curvatureActive && curvatureStrengthClamped > 1e-6
          ? computeHyperbolicFlowScale(hyperRadius, curvatureStrengthClamped)
          : 1;
      const radialX = Math.cos(hyperTheta);
      const radialY = Math.sin(hyperTheta);
      out[i + 0] = baseData[i + 0];
      out[i + 1] = baseData[i + 1];
      out[i + 2] = baseData[i + 2];
      out[i + 3] = 255;

      if (displayMode === 'grayBaseColorRims' || displayMode === 'grayBaseGrayRims') {
        const yb = Math.floor(0.2126 * out[i + 0] + 0.7152 * out[i + 1] + 0.0722 * out[i + 2]);
        out[i + 0] = yb;
        out[i + 1] = yb;
        out[i + 2] = yb;
      }

      const baseColorSrgb: [number, number, number] = [
        clamp01(baseData[i + 0] / 255),
        clamp01(baseData[i + 1] / 255),
        clamp01(baseData[i + 2] / 255),
      ];
      const magVal = mag[p];
      let composerWeightRim = composerFields.rim.weight;
      let composerWeightSurface = composerFields.surface.weight;
      let composerWeightKur = composerFields.kur.weight;
      let composerWeightVolume = composerFields.volume.weight;
      let su7Projection: Su7ProjectionResult | null = null;
      if (su7Unitary && su7Vectors && su7Norms) {
        const baseVector = su7Vectors[p];
        const baseNorm = su7Norms[p] ?? 0;
        if (baseVector && baseNorm > 0) {
          const strideEligible =
            su7DecimationStride <= 1 ||
            (x % su7DecimationStride === 0 && y % su7DecimationStride === 0);
          const rimEligible = magVal >= edgeThreshold;
          let shouldProject = false;
          if (su7DecimationStride <= 1) {
            shouldProject = true;
          } else if (su7DecimationMode === 'stride') {
            shouldProject = strideEligible;
          } else if (su7DecimationMode === 'edges') {
            shouldProject = rimEligible;
          } else {
            shouldProject = strideEligible || rimEligible;
          }
          if (shouldProject) {
            const transformed = applySU7(su7Unitary, baseVector);
            const transformedNorm = computeC7Norm(transformed);
            const normDelta = Math.abs(transformedNorm - baseNorm);
            su7NormDeltaSum += normDelta;
            if (normDelta > su7NormDeltaMax) {
              su7NormDeltaMax = normDelta;
            }
            su7NormSampleCount += 1;
            su7Projection = projectSu7Vector({
              vector: transformed,
              norm: baseNorm,
              projector: su7Projector,
              gain: su7Params.gain,
              frameGain,
              baseColor: baseColorSrgb,
            });
            if (su7Projection) {
              su7EnergySum += su7Projection.energy;
              su7EnergySampleCount += 1;
            }
          }
          if (su7Projection?.composerWeights) {
            const weights = su7Projection.composerWeights;
            if (weights.rim != null) {
              composerWeightRim = clamp(composerWeightRim * weights.rim, 0.05, 3);
            }
            if (weights.surface != null) {
              composerWeightSurface = clamp(composerWeightSurface * weights.surface, 0.05, 3);
            }
            if (weights.kur != null) {
              composerWeightKur = clamp(composerWeightKur * weights.kur, 0.05, 3);
            }
            if (weights.volume != null) {
              composerWeightVolume = clamp(composerWeightVolume * weights.volume, 0.05, 3);
            }
          }
        }
      }

      let baseGradX = 0;
      let baseGradY = 0;
      if (kurEnabled && gradX && gradY) {
        baseGradX = gradX[p];
        baseGradY = gradY[p];
      }

      let flowX = 0;
      let flowY = 0;
      let projectionStretch = 1;
      const debugOrientationCap = surfaceDebug
        ? Math.min(surfaceDebug.orientationCount, orientationCount)
        : 0;
      if (orientationCount > 0) {
        const latticeFreq = 0.015 + 0.004 * orientationCount;
        const timeShift = timeSeconds * 0.6 + dmt * 0.3;
        for (let k = 0; k < orientationCount; k++) {
          const proj = sampleDx * cosA[k] + sampleDy * sinA[k];
          const phaseVal = proj * latticeFreq + timeShift + k * 0.35;
          const magnitude = 0.5 + 0.5 * Math.sin(phaseVal + beta2 * 0.25);
          const wrappedPhase = ((phaseVal % TAU) + TAU) % TAU;
          if (orientationBranches) {
            const branch = orientationBranches[k];
            const branchVecX = cosA[k] * magnitude;
            const branchVecY = sinA[k] * magnitude;
            const branchMag = Math.hypot(branchVecX, branchVecY);
            if (branchMag > 1e-6) {
              const invMag = 1 / branchMag;
              branch.phaseCos += branchVecX * invMag;
              branch.phaseSin += branchVecY * invMag;
              branch.magnitudeSum += magnitude;
              branch.hyperbolicSum += hyperFlowScale;
              const tag = clamp((branchVecX * radialX + branchVecY * radialY) * invMag, -1, 1);
              branch.parallaxSum += tag;
              branch.count += 1;
            }
          }
          flowX += cosA[k] * magnitude;
          flowY += sinA[k] * magnitude;
          if (orientationMagnitudes) {
            orientationMagnitudes[k] = clamp01(magnitude);
          }
          if (surfaceDebug && k < debugOrientationCap) {
            surfaceDebug.phases[k][p] = wrappedPhase;
            surfaceDebug.magnitudes[k][p] = magnitude;
            const histBase = k * SURFACE_HIST_BINS;
            const idx = Math.min(
              SURFACE_HIST_BINS - 1,
              Math.floor(clamp(magnitude, 0, 0.999999) * SURFACE_HIST_BINS),
            );
            surfaceDebug.magnitudeHist[histBase + idx] += 1;
            if (magnitude > debugMagnitudeMax) debugMagnitudeMax = magnitude;
          }
        }
        const inv = 1 / Math.max(orientationCount, 1);
        flowX *= inv;
        flowY *= inv;
        if (orientationMagnitudes) {
          accumulateCrystalMetrics(orientationMagnitudes, orientationCount);
        }
      } else if (useWallpaper) {
        for (let k = 0; k < opsCount; k++) {
          const pt = applyOp(ops[k], posX, posY, cx, cy);
          const r = wallpaperAt(pt.x - cx, pt.y - cy, cosA, sinA, ke, timeSeconds, alive);
          if (wallpaperBranches) {
            const branch = wallpaperBranches[k];
            const branchVecX = r.gx;
            const branchVecY = r.gy;
            const branchMag = Math.hypot(branchVecX, branchVecY);
            if (branchMag > 1e-6) {
              const invMag = 1 / branchMag;
              branch.phaseCos += branchVecX * invMag;
              branch.phaseSin += branchVecY * invMag;
              branch.magnitudeSum += branchMag;
              branch.hyperbolicSum += hyperFlowScale;
              const tag = clamp((branchVecX * radialX + branchVecY * radialY) * invMag, -1, 1);
              branch.parallaxSum += tag;
              branch.count += 1;
            }
          }
          flowX += r.gx;
          flowY += r.gy;
          if (wallpaperMagnitudes) {
            const gradMag = Math.hypot(r.gx, r.gy);
            const normalized = Math.min(gradMag / (1 + 0.75 * ke.k0), 1);
            wallpaperMagnitudes[k] = normalized;
          }
        }
        const inv = opsCount > 0 ? 1 / opsCount : 1;
        flowX *= inv;
        flowY *= inv;
        if (wallpaperMagnitudes) {
          accumulateCrystalMetrics(wallpaperMagnitudes, opsCount);
        }
      }

      if (coupleVolumeWarp && volumeDepthGrad) {
        const gradVal = volumeDepthGrad[p];
        const gradNorm = Math.min(gradVal / (0.18 + 0.12 * ke.k0 + 1e-6), 1);
        const gradResponse = responseSoft(gradNorm, volumeDepthToWarp);
        const gradSignal = applyComposerScalar(gradResponse, composerFields.volume);
        composerEnergy.volume += gradSignal;
        composerBlend.volume += gradSignal;
        composerSamples.volume += 1;
        const boost = 1 + couplingDmtScale * volumeDepthToWarp * gradSignal * composerWeightVolume;
        flowX *= boost;
        flowY *= boost;
      }

      if (curvatureActive && atlasJacobians) {
        const jacOffset = p * 4;
        const j11 = atlasJacobians[jacOffset] * atlasSampleScale;
        const j12 = atlasJacobians[jacOffset + 1] * atlasSampleScale;
        const j21 = atlasJacobians[jacOffset + 2] * atlasSampleScale;
        const j22 = atlasJacobians[jacOffset + 3] * atlasSampleScale;
        const stretchX = Math.hypot(j11, j21);
        const stretchY = Math.hypot(j12, j22);
        projectionStretch = (stretchX + stretchY) * 0.5;
        if (!Number.isFinite(projectionStretch) || projectionStretch <= 0) {
          projectionStretch = 1;
        }
        const det = j11 * j22 - j12 * j21;
        if (Math.abs(det) > 1e-9) {
          const invDet = 1 / det;
          const flowHyperX = (j22 * flowX - j12 * flowY) * invDet;
          const flowHyperY = (-j21 * flowX + j11 * flowY) * invDet;
          const gradHyperX = (j22 * baseGradX - j12 * baseGradY) * invDet;
          const gradHyperY = (-j21 * baseGradX + j11 * baseGradY) * invDet;
          flowX = flowHyperX;
          flowY = flowHyperY;
          baseGradX = gradHyperX;
          baseGradY = gradHyperY;
        } else {
          flowX = 0;
          flowY = 0;
          baseGradX = 0;
          baseGradY = 0;
          projectionStretch = 0;
        }
      }

      if (hyperFlowScale !== 1) {
        flowX *= hyperFlowScale;
        flowY *= hyperFlowScale;
        baseGradX *= hyperFlowScale;
        baseGradY *= hyperFlowScale;
      }

      const sharedGX = baseGradX + flowX;
      const sharedGY = baseGradY + flowY;
      const sharedMag = Math.hypot(sharedGX, sharedGY);
      let parallaxTransparencyBoost = 1;
      const normDenom = Math.max(0.12 + 0.85 * ke.k0, 1e-3);
      const kNorm = sharedMag > 1e-6 ? Math.min(sharedMag / normDenom, 1) : 0;
      if (sharedMag > 1e-6) {
        const tag = clamp((sharedGX * radialX + sharedGY * radialY) / sharedMag, -1, 1);
        parallaxTransparencyBoost = computeHyperbolicTransparencyBoost(
          hyperFlowScale,
          radiusNorm,
          Math.abs(tag),
          kNorm,
          ke.transparency,
        );
        parallaxRadiusSum += radiusNorm;
        parallaxRadiusSqSum += radiusNorm * radiusNorm;
        parallaxKSum += sharedMag;
        parallaxKSumSq += sharedMag * sharedMag;
        parallaxRadiusKSum += radiusNorm * sharedMag;
        parallaxTagAbsSum += Math.abs(tag);
        parallaxSamples++;
      }

      if (surfaceDebug?.flowVectors) {
        surfaceDebug.flowVectors.x[p] = sharedGX;
        surfaceDebug.flowVectors.y[p] = sharedGY;
        if (surfaceDebug.flowVectors.hyperbolicScale) {
          const hyperDebugScale =
            curvatureActive && curvatureMode === 'klein' && atlasJacobians
              ? hyperFlowScale * projectionStretch
              : curvatureActive
                ? hyperFlowScale
                : 0;
          surfaceDebug.flowVectors.hyperbolicScale[p] = hyperDebugScale;
        }
      }

      const gxT0 = sharedGX;
      const gyT0 = sharedGY;

      const warpMag = sharedMag;
      if (warpMag > 0) {
        warpSum += warpMag;
        warpSumSq += warpMag * warpMag;
        warpCos += gxT0 / warpMag;
        warpSin += gyT0 / warpMag;
        warpCount++;
      }

      let rimEnergy = 0;

      if (rimEnabled && magVal >= edgeThreshold) {
        let nx = gx[p];
        let ny = gy[p];
        const nlen = Math.hypot(nx, ny) + 1e-8;
        nx /= nlen;
        ny /= nlen;
        const tx = -ny;
        const ty = nx;
        const TAU = Math.PI * 2;
        const thetaRaw = Math.atan2(ny, nx);
        const thetaEdge =
          thetaMode === 'gradient'
            ? polBins > 0
              ? Math.round((thetaRaw / TAU) * polBins) * (TAU / polBins)
              : thetaRaw
            : thetaGlobal;

        let thetaUse = thetaEdge;
        if (coupleKurOrientation) {
          const kurNorm = Math.hypot(baseGradX, baseGradY);
          if (kurNorm > 1e-6) {
            const ex = Math.cos(thetaUse);
            const ey = Math.sin(thetaUse);
            const kx = baseGradX / kurNorm;
            const ky = baseGradY / kurNorm;
            const mixW = clamp01(kurToOrientation * couplingDmtScale * composerWeightKur);
            const vx = mixScalar(ex, kx, mixW);
            const vy = mixScalar(ey, ky, mixW);
            thetaUse = Math.atan2(vy, vx);
            if (polBins > 0) {
              thetaUse = Math.round((thetaUse / TAU) * polBins) * (TAU / polBins);
            }
          }
        }

        const delta = ke.anisotropy * 0.9;
        const rho = ke.chirality * 0.75;
        const thetaEff = thetaUse + rho * timeSeconds;
        const polL = 0.5 * (1 + Math.cos(delta) * Math.cos(2 * thetaEff));
        const polM = 0.5 * (1 + Math.cos(delta) * Math.cos(2 * (thetaEff + 0.3)));
        const polS = 0.5 * (1 + Math.cos(delta) * Math.cos(2 * (thetaEff + 0.6)));

        const rawJ = Math.sin(jitterPhase + hash2(posX, posY) * Math.PI * 2);
        const localJ = jitter * (microsaccade ? (phasePin ? rawJ - muJ : rawJ) : 0);

        const warpNorm = Math.hypot(flowX, flowY);
        let bias = 0;
        if (surfaceToRimOffset > 1e-4 && warpNorm > 1e-6) {
          const proj = (flowX * nx + flowY * ny) / (warpNorm + 1e-9);
          const magnitude = responseSoft(warpNorm / (1.05 + 0.55 * ke.k0), surfaceToRimOffset);
          const signed = clamp(proj, -1, 1) * magnitude;
          bias = clamp(
            0.65 * couplingDmtScale * surfaceToRimOffset * composerWeightSurface * signed,
            -0.9,
            0.9,
          );
        }
        let sigmaEff = sigma;
        if (surfaceToRimSigma > 1e-4 && warpNorm > 1e-6) {
          const sharpness = responsePow(warpNorm / (1.15 + 0.6 * ke.k0), surfaceToRimSigma);
          const drop = clamp01(
            0.75 * couplingDmtScale * surfaceToRimSigma * composerWeightSurface * sharpness,
          );
          const sigmaFloor = Math.min(sigma, sigmaFloorBase);
          sigmaEff = clamp(sigma * (1 - drop), sigmaFloor, sigma);
        }

        const offL = baseOffsets.L + localJ * 0.35 + bias;
        const offM = baseOffsets.M + localJ * 0.5 + bias;
        const offS = baseOffsets.S + localJ * 0.8 + bias;

        let hueShift = 0;
        if (coupleVolumePhase && volumePhase) {
          const phaseVal = volumePhase[p];
          const phaseNorm = clamp(phaseVal / Math.PI, -1, 1);
          const phaseMag = applyComposerScalar(Math.abs(phaseNorm), composerFields.volume);
          const signedPhase = phaseNorm < 0 ? -phaseMag : phaseMag;
          const volHue = clamp(
            couplingDmtScale * volumePhaseToHue * composerWeightVolume * signedPhase * 0.6,
            -1.5,
            1.5,
          );
          hueShift += volHue;
          composerEnergy.volume += Math.abs(signedPhase);
          composerBlend.volume += Math.abs(signedPhase);
          composerSamples.volume += 1;
        }
        if (surfaceToRimHue > 1e-4 && warpNorm > 1e-6) {
          const latticeAngle = Math.atan2(flowY, flowX);
          const tangentAngle = Math.atan2(ty, tx);
          const deltaAngle = wrapPi(latticeAngle - tangentAngle);
          const hueAmplitude = responseSoft(
            Math.min(Math.abs(deltaAngle) / Math.PI, 1),
            surfaceToRimHue,
          );
          const warpWeight = responsePow(warpNorm / (1.1 + 0.6 * ke.k0), surfaceToRimHue);
          const hueSignal = applyComposerScalar(
            clamp01(hueAmplitude * warpWeight),
            composerFields.surface,
          );
          const signed = (deltaAngle === 0 ? 0 : Math.sign(deltaAngle)) * hueSignal;
          hueShift = clamp(
            couplingDmtScale * surfaceToRimHue * composerWeightSurface * signed * 0.9,
            -1.4,
            1.4,
          );
        }

        const pL = sampleScalar(
          mag,
          x + (offL + breath) * nx,
          y + (offL + breath) * ny,
          width,
          height,
        );
        const pM = sampleScalar(
          mag,
          x + (offM + breath) * nx,
          y + (offM + breath) * ny,
          width,
          height,
        );
        const pS = sampleScalar(
          mag,
          x + (offS + breath) * nx,
          y + (offS + breath) * ny,
          width,
          height,
        );

        const gL = gauss(offL, sigmaEff) * ke.gain;
        const gM = gauss(offM, sigmaEff) * ke.gain;
        const gS = gauss(offS, sigmaEff) * ke.gain;

        const QQ = 1 + 0.5 * ke.Q;
        const modL = Math.pow(0.5 * (1 + Math.cos(2 * Math.PI * ke.k0 * offL)), QQ);
        const modM = Math.pow(0.5 * (1 + Math.cos(2 * Math.PI * ke.k0 * offM)), QQ);
        const modS = Math.pow(0.5 * (1 + Math.cos(2 * Math.PI * ke.k0 * offS)), QQ);

        const chiPhase = 2 * Math.PI * ke.k0 * (x * tx + y * ty) * 0.002 + hueShift;
        let chBase = ke.chirality;
        if (coupleKurChirality && vort) {
          const vortVal = clamp(vort[p], -1, 1);
          const vortScaled = 0.5 * couplingDmtScale * kurToChirality * composerWeightKur * vortVal;
          chBase = clamp(chBase + vortScaled, -3, 3);
        }
        const chiL = 0.5 + 0.5 * Math.sin(chiPhase) * chBase;
        const chiM = 0.5 + 0.5 * Math.sin(chiPhase + 0.8) * chBase;
        const chiS = 0.5 + 0.5 * Math.sin(chiPhase + 1.6) * chBase;

        const cont = contrast * frameGain;
        const Lc = pL * gL * modL * chiL * polL * cont;
        const Mc = pM * gM * modM * chiM * polM * cont;
        const Sc = pS * gS * modS * chiS * polS * cont;

        rimEnergy = (Lc + Mc + Sc) / Math.max(1e-6, cont);

        let R = 4.4679 * Lc + -3.5873 * Mc + 0.1193 * Sc;
        let G = -1.2186 * Lc + 2.3809 * Mc + -0.1624 * Sc;
        let B = 0.0497 * Lc + -0.2439 * Mc + 1.2045 * Sc;
        R = clamp01(R);
        G = clamp01(G);
        B = clamp01(B);
        if (displayMode === 'grayBaseGrayRims' || displayMode === 'colorBaseGrayRims') {
          const yr = luma01(R, G, B);
          R = yr;
          G = yr;
          B = yr;
        } else if (displayMode === 'colorBaseBlendedRims') {
          const baseR = clamp01(baseData[i + 0] / 255);
          const baseG = clamp01(baseData[i + 1] / 255);
          const baseB = clamp01(baseData[i + 2] / 255);
          const rimSum = R + G + B;
          const baseSum = baseR + baseG + baseB;
          const invBaseSum = baseSum > 1e-6 ? 1 / baseSum : 1 / 3;
          const baseW0 = baseSum > 1e-6 ? baseR * invBaseSum : 1 / 3;
          const baseW1 = baseSum > 1e-6 ? baseG * invBaseSum : 1 / 3;
          const baseW2 = baseSum > 1e-6 ? baseB * invBaseSum : 1 / 3;
          const naturalR = clamp01(rimSum * baseW0);
          const naturalG = clamp01(rimSum * baseW1);
          const naturalB = clamp01(rimSum * baseW2);
          const hueBlend = 0.7;
          const baseBlend = 0.25;
          R = clamp01(R * (1 - hueBlend) + naturalR * hueBlend);
          G = clamp01(G * (1 - hueBlend) + naturalG * hueBlend);
          B = clamp01(B * (1 - hueBlend) + naturalB * hueBlend);
          R = clamp01(R * (1 - baseBlend) + baseR * baseBlend);
          G = clamp01(G * (1 - baseBlend) + baseG * baseBlend);
          B = clamp01(B * (1 - baseBlend) + baseB * baseBlend);
        }
        R *= rimAlpha;
        G *= rimAlpha;
        B *= rimAlpha;

        const rimAdjusted = applyComposerVec3([R, G, B], composerFields.rim);
        R = rimAdjusted[0];
        G = rimAdjusted[1];
        B = rimAdjusted[2];

        if (rimDebug) {
          const eNorm = clamp(rimEnergy / RIM_ENERGY_HIST_SCALE, 0, 0.999999);
          const eIdx = Math.min(
            rimDebug.energyHist.length - 1,
            Math.floor(eNorm * rimDebug.energyHist.length),
          );
          rimDebug.energy[p] = rimEnergy;
          rimDebug.energyHist[eIdx] += 1;

          const hueNumerator = Math.sqrt(3) * (G - B);
          const hueDenominator = 2 * R - G - B + 1e-9;
          let hue = Math.atan2(hueNumerator, hueDenominator) / TAU;
          if (hue < 0) hue += 1;
          rimDebug.hue[p] = hue;
          const hIdx = Math.min(
            rimDebug.hueHist.length - 1,
            Math.floor(hue * rimDebug.hueHist.length),
          );
          rimDebug.hueHist[hIdx] += 1;
          if (rimEnergy < debugEnergyMin) debugEnergyMin = rimEnergy;
          if (rimEnergy > debugEnergyMax) debugEnergyMax = rimEnergy;
          if (hue < debugHueMin) debugHueMin = hue;
          if (hue > debugHueMax) debugHueMax = hue;
        }

        let pixelBlend = clamp01(effectiveBlend * rimComposerGain);
        pixelBlend = clamp01(pixelBlend * parallaxTransparencyBoost);
        if (coupleKurTransparency && coh) {
          const cohVal = clamp01(coh[p]);
          const ampVal = amp ? clamp01(amp[p]) : cohVal;
          let kurMeasure = 0.5 * (cohVal + ampVal);
          kurMeasure = applyComposerScalar(kurMeasure, composerFields.kur);
          const boost = clamp(
            1 + couplingDmtScale * kurToTransparency * composerWeightKur * (kurMeasure - 0.5) * 1.5,
            0.1,
            2.5,
          );
          pixelBlend = clamp01(pixelBlend * boost);
          composerEnergy.kur += kurMeasure;
          composerBlend.kur += kurMeasure;
          composerSamples.kur += 1;
        }
        pixelBlend = clamp01(pixelBlend * composerWeightRim);

        out[i + 0] = Math.floor(out[i + 0] * (1 - pixelBlend) + R * 255 * pixelBlend);
        out[i + 1] = Math.floor(out[i + 1] * (1 - pixelBlend) + G * 255 * pixelBlend);
        out[i + 2] = Math.floor(out[i + 2] * (1 - pixelBlend) + B * 255 * pixelBlend);

        composerEnergy.rim += rimEnergy * pixelBlend;
        composerBlend.rim += pixelBlend;
        composerSamples.rim += 1;

        rimSum += rimEnergy;
        rimSumSq += rimEnergy * rimEnergy;
        rimCount++;
        if (rimEnergy > rimMax) rimMax = rimEnergy;

        if ((x & 7) === 0 && (y & 7) === 0) {
          obsSum += (pL + pM + pS) / 3;
          obsCount++;
        }
      } else if (rimDebug) {
        rimDebug.energy[p] = 0;
        rimDebug.hue[p] = 0;
      }

      if (surfEnabled) {
        let mask = 1.0;
        if (surfaceRegion === 'surfaces') {
          if (edgeThreshold <= 1e-6) {
            mask = 1;
          } else {
            mask = clamp01((edgeThreshold - magVal) / edgeThreshold);
          }
        } else if (surfaceRegion === 'edges') {
          mask = clamp01((magVal - edgeThreshold) / Math.max(1e-6, 1 - edgeThreshold));
        }
        if (mask > 1e-3) {
          let gxSurf = gxT0;
          let gySurf = gyT0;
          if (coupleRimSurface && magVal >= edgeThreshold) {
            const denom = Math.hypot(gx[p], gy[p]) + 1e-8;
            const tx = -gy[p] / denom;
            const ty = gx[p] / denom;
            const surfLen = Math.hypot(gxSurf, gySurf);
            if (surfLen > 1e-6) {
              const dot = (gxSurf * tx + gySurf * ty) / (surfLen + 1e-9);
              const alignment = responsePow((dot + 1) * 0.5, rimToSurfaceAlign);
              const alignWeight = clamp01(
                rimToSurfaceAlign * couplingDmtScale * composerWeightSurface * alignment,
              );
              const targetX = tx * surfLen;
              const targetY = ty * surfLen;
              gxSurf = mixScalar(gxSurf, targetX, alignWeight);
              gySurf = mixScalar(gySurf, targetY, alignWeight);
            }
          }
          if (coupleKurOrientation) {
            const kurNorm = Math.hypot(baseGradX, baseGradY);
            if (kurNorm > 1e-6) {
              const surfLen = Math.hypot(gxSurf, gySurf);
              if (surfLen > 1e-6) {
                const targetX = (baseGradX / kurNorm) * surfLen;
                const targetY = (baseGradY / kurNorm) * surfLen;
                const weight = clamp01(
                  kurToOrientation * couplingDmtScale * composerWeightKur * 0.75,
                );
                gxSurf = mixScalar(gxSurf, targetX, weight);
                gySurf = mixScalar(gySurf, targetY, weight);
              }
            }
          }
          const dirNorm = Math.hypot(gxSurf, gySurf) + 1e-6;
          const dirAngle = Math.atan2(gySurf, gxSurf);
          const dirW = 1 + 0.5 * ke.anisotropy * Math.cos(2 * dirAngle);
          const amplitude = clamp01(warpAmp * dirNorm * dirW);
          const phaseShift = dirAngle + timeSeconds * 0.45;
          let rW = clamp01(0.5 + 0.5 * Math.sin(phaseShift));
          let gW = clamp01(0.5 + 0.5 * Math.sin(phaseShift + (2 * Math.PI) / 3));
          let bW = clamp01(0.5 + 0.5 * Math.sin(phaseShift + (4 * Math.PI) / 3));
          rW *= amplitude;
          gW *= amplitude;
          bW *= amplitude;
          if (displayMode === 'grayBaseGrayRims') {
            const yy = luma01(rW, gW, bW);
            rW = yy;
            gW = yy;
            bW = yy;
          }
          const surfaceAdjusted = applyComposerVec3([rW, gW, bW], composerFields.surface);
          rW = surfaceAdjusted[0];
          gW = surfaceAdjusted[1];
          bW = surfaceAdjusted[2];
          let sb = surfaceBlend * mask * surfaceComposerGain;
          sb *= parallaxTransparencyBoost;
          if (coupleRimSurface) {
            const energy = responseSoft(rimEnergy / (0.75 + 0.25 * ke.gain), rimToSurfaceBlend);
            const rimBoost = clamp(
              1 + couplingDmtScale * rimToSurfaceBlend * composerWeightRim * energy,
              0.2,
              3,
            );
            sb *= rimBoost;
          }
          if (coupleKurTransparency && coh) {
            const cohVal = clamp01(coh[p]);
            const ampVal = amp ? clamp01(amp[p]) : cohVal;
            let kurMeasure = 0.5 * (cohVal + ampVal);
            kurMeasure = applyComposerScalar(kurMeasure, composerFields.kur);
            const boost = clamp(
              1 +
                couplingDmtScale * kurToTransparency * composerWeightKur * (kurMeasure - 0.5) * 1.5,
              0.1,
              3,
            );
            sb *= boost;
            composerEnergy.kur += kurMeasure;
            composerBlend.kur += kurMeasure;
            composerSamples.kur += 1;
          }
          sb = clamp01(sb * composerWeightSurface);
          out[i + 0] = Math.floor(out[i + 0] * (1 - sb) + rW * 255 * sb);
          out[i + 1] = Math.floor(out[i + 1] * (1 - sb) + gW * 255 * sb);
          out[i + 2] = Math.floor(out[i + 2] * (1 - sb) + bW * 255 * sb);

          const surfaceLuma = luma01(rW, gW, bW);
          composerEnergy.surface += surfaceLuma * sb;
          composerBlend.surface += sb;
          composerSamples.surface += 1;

          surfaceSum += sb;
          if (sb > surfaceMax) surfaceMax = sb;
          surfaceCount++;
        }
      }

      if (su7Projection) {
        const mix = clamp01(su7Projection.mix);
        if (mix > 1e-6) {
          const currentR = out[i + 0] / 255;
          const currentG = out[i + 1] / 255;
          const currentB = out[i + 2] / 255;
          const su7Rgb = su7Projection.rgb;
          const blendedR = clamp01(currentR * (1 - mix) + su7Rgb[0] * mix);
          const blendedG = clamp01(currentG * (1 - mix) + su7Rgb[1] * mix);
          const blendedB = clamp01(currentB * (1 - mix) + su7Rgb[2] * mix);
          out[i + 0] = Math.round(blendedR * 255);
          out[i + 1] = Math.round(blendedG * 255);
          out[i + 2] = Math.round(blendedB * 255);
        }
        if (su7Projection.overlays) {
          for (const overlay of su7Projection.overlays) {
            const overlayMix = clamp01(overlay.mix);
            if (overlayMix <= 1e-6) continue;
            const currentR = out[i + 0] / 255;
            const currentG = out[i + 1] / 255;
            const currentB = out[i + 2] / 255;
            const overlayRgb = overlay.rgb;
            const blendedR = clamp01(currentR * (1 - overlayMix) + overlayRgb[0] * overlayMix);
            const blendedG = clamp01(currentG * (1 - overlayMix) + overlayRgb[1] * overlayMix);
            const blendedB = clamp01(currentB * (1 - overlayMix) + overlayRgb[2] * overlayMix);
            out[i + 0] = Math.round(blendedR * 255);
            out[i + 1] = Math.round(blendedG * 255);
            out[i + 2] = Math.round(blendedB * 255);
          }
        }
      }
    }
  }

  const crystalWallpaperStats = finalizeStats(
    crystalWallpaperSum,
    crystalWallpaperSumSq,
    crystalSamples,
  );
  const crystalBeatStats = finalizeStats(crystalBeatSum, crystalBeatSumSq, crystalSamples);
  const crystalResonanceRate = crystalSamples > 0 ? crystalResonance / crystalSamples : 0;
  metrics.texture.crystallizer = {
    wallpapericity: crystalWallpaperStats.mean,
    wallpaperStd: crystalWallpaperStats.std,
    beatEnergy: crystalBeatStats.mean,
    beatStd: crystalBeatStats.std,
    resonanceRate: crystalResonanceRate,
    sampleCount: crystalSamples,
  };
  const earlyVisionTexture = metrics.texture.earlyVision;
  metrics.texture.wallpapericity =
    crystalSamples > 0
      ? 0.5 * (earlyVisionTexture.wallpapericity + crystalWallpaperStats.mean)
      : earlyVisionTexture.wallpapericity;
  metrics.texture.wallpaperStd = Math.max(
    earlyVisionTexture.wallpaperStd,
    crystalWallpaperStats.std,
  );
  metrics.texture.beatEnergy = Math.max(earlyVisionTexture.beatEnergy, crystalBeatStats.mean);
  metrics.texture.beatStd = Math.max(earlyVisionTexture.beatStd, crystalBeatStats.std);
  metrics.texture.resonanceRate = Math.max(earlyVisionTexture.resonanceRate, crystalResonanceRate);
  metrics.texture.sampleCount = Math.max(earlyVisionTexture.sampleCount, crystalSamples);

  const rimStats = finalizeStats(rimSum, rimSumSq, rimCount);
  metrics.rim = {
    mean: rimStats.mean,
    std: rimStats.std,
    max: rimMax,
    count: rimCount,
  };

  const warpStats = finalizeStats(warpSum, warpSumSq, warpCount);
  const warpAngle = warpCount > 0 ? Math.atan2(warpSin, warpCos) : 0;
  metrics.warp = {
    mean: warpStats.mean,
    std: warpStats.std,
    dominantAngle: warpAngle,
    count: warpCount,
  };

  const parallaxStats = finalizeStats(parallaxKSum, parallaxKSumSq, parallaxSamples);
  let radialSlope = 0;
  if (parallaxSamples > 1) {
    const denom = parallaxSamples * parallaxRadiusSqSum - parallaxRadiusSum * parallaxRadiusSum;
    if (Math.abs(denom) > 1e-9) {
      radialSlope =
        (parallaxSamples * parallaxRadiusKSum - parallaxRadiusSum * parallaxKSum) / denom;
    }
  }
  const eccentricityMean = parallaxSamples > 0 ? parallaxRadiusSum / parallaxSamples : 0;
  const tagConsistency = parallaxSamples > 0 ? parallaxTagAbsSum / parallaxSamples : 0;
  metrics.parallax = {
    radialSlope,
    eccentricityMean,
    kMean: parallaxStats.mean,
    kStd: parallaxStats.std,
    tagConsistency,
    sampleCount: parallaxSamples,
  };

  const motionOrientation = computeMotionEnergyFromBranches(orientationBranches, 'orientation');
  const motionWallpaper = computeMotionEnergyFromBranches(wallpaperBranches, 'wallpaper');

  if (attentionHooks) {
    const orientationSummary = finalizeBranchAttentionSummary(orientationBranches, 'orientation');
    if (orientationSummary) {
      attentionHooks.onOrientation?.(orientationSummary);
    }
    const wallpaperSummary = finalizeBranchAttentionSummary(wallpaperBranches, 'wallpaper');
    if (wallpaperSummary) {
      attentionHooks.onWallpaper?.(wallpaperSummary);
    }
  }

  metrics.motionEnergy = motionOrientation ?? motionWallpaper ?? metrics.motionEnergy;

  metrics.compositor.surfaceMean = surfaceCount ? surfaceSum / surfaceCount : 0;
  metrics.compositor.surfaceMax = surfaceMax;
  metrics.compositor.surfaceCount = surfaceCount;

  if (volSampleCount > 0) {
    const phaseStats = finalizeStats(volPhaseSum, volPhaseSumSq, volSampleCount);
    const depthStats = finalizeStats(volDepthSum, volDepthSumSq, volSampleCount);
    const intensityStats = finalizeStats(volIntensitySum, volIntensitySumSq, volSampleCount);
    const gradStats = finalizeStats(volGradSum, volGradSumSq, volSampleCount);
    metrics.volume.phaseMean = phaseStats.mean;
    metrics.volume.phaseStd = phaseStats.std;
    metrics.volume.depthMean = depthStats.mean;
    metrics.volume.depthStd = depthStats.std;
    metrics.volume.intensityMean = intensityStats.mean;
    metrics.volume.intensityStd = intensityStats.std;
    metrics.volume.depthGradMean = gradStats.mean;
    metrics.volume.depthGradStd = gradStats.std;
    metrics.volume.sampleCount = volSampleCount;
  } else {
    metrics.volume.phaseMean = 0;
    metrics.volume.phaseStd = 0;
    metrics.volume.depthMean = 0;
    metrics.volume.depthStd = 0;
    metrics.volume.depthGradMean = 0;
    metrics.volume.depthGradStd = 0;
    metrics.volume.intensityMean = 0;
    metrics.volume.intensityStd = 0;
    metrics.volume.sampleCount = 0;
  }

  const totalComposerEnergy =
    composerEnergy.surface + composerEnergy.rim + composerEnergy.kur + composerEnergy.volume;
  const composerFieldMetrics = metrics.composer.fields;
  COMPOSER_FIELD_LIST.forEach((field) => {
    const samples = composerSamples[field];
    const energyTotal = composerEnergy[field];
    composerFieldMetrics[field].energy = samples > 0 ? energyTotal / samples : 0;
    composerFieldMetrics[field].blend = samples > 0 ? composerBlend[field] / samples : 0;
    composerFieldMetrics[field].share =
      totalComposerEnergy > 1e-6 ? energyTotal / totalComposerEnergy : 0;
  });
  metrics.compositor.effectiveBlend = clamp01(
    effectiveBlend * rimComposerGain * composerFields.rim.weight,
  );

  if (su7Active && su7Unitary && su7Vectors && su7Norms) {
    metrics.su7.normDeltaMean =
      su7NormSampleCount > 0 ? su7NormDeltaSum / su7NormSampleCount : su7Telemetry.normDeltaMean;
    metrics.su7.normDeltaMax = su7NormSampleCount > 0 ? su7NormDeltaMax : su7Telemetry.normDeltaMax;
    metrics.su7.projectorEnergy =
      su7EnergySampleCount > 0 ? su7EnergySum / su7EnergySampleCount : su7Telemetry.projectorEnergy;
  } else {
    metrics.su7.unitaryError = 0;
    metrics.su7.determinantDrift = 0;
    metrics.su7.normDeltaMean = 0;
    metrics.su7.normDeltaMax = 0;
    metrics.su7.projectorEnergy = 0;
  }

  const obs = obsCount > 0 ? clamp(obsSum / obsCount, 0.001, 10) : clamp(lastObs, 0.001, 10);
  let debugResult: RainbowFrameResult['debug'];
  if (rimDebug || surfaceDebug) {
    debugResult = {};
    if (rimDebug) {
      const energyMin = Number.isFinite(debugEnergyMin) ? debugEnergyMin : 0;
      const energyMax = debugEnergyMax;
      const hueMin = Number.isFinite(debugHueMin) ? debugHueMin : 0;
      const hueMax = debugHueMax;
      debugResult.rim = {
        energyMin,
        energyMax,
        hueMin,
        hueMax,
      };
    }
    if (surfaceDebug) {
      debugResult.surface = {
        magnitudeMax: debugMagnitudeMax,
        orientationCount: surfaceDebug.orientationCount,
      };
    }
  }

  return { metrics, obsAverage: obs, debug: debugResult };
};
