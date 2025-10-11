import React, { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  createDerivedViews,
  createKuramotoState,
  createNormalGenerator,
  derivedBufferSize,
  deriveKuramotoFields as deriveKuramotoFieldsCore,
  initKuramotoState,
  stepKuramotoState,
  type PhaseField,
  type KuramotoParams,
  type KuramotoState,
  type KuramotoTelemetrySnapshot,
  type IrradianceFrameBuffer,
  type KuramotoInstrumentationSnapshot,
} from './kuramotoCore';
import {
  createGpuRenderer,
  type GpuRenderer,
  type Su7TexturePayload,
  type Su7Uniforms,
  type Su7ProjectorMode,
} from './gpuRenderer';
import {
  applyOp,
  clamp,
  computeComposerBlendGain,
  createDefaultComposerConfig,
  groupOps,
  hash2,
  kEff,
  deriveSu7ScheduleContext,
  renderRainbowFrame,
  toGpuOps,
  COMPOSER_FIELD_LIST,
  type RainbowFrameMetrics,
  type SurfaceRegion,
  type WallpaperGroup,
  type DisplayMode,
  type CouplingConfig,
  type ComposerConfig,
  type ComposerFieldId,
  type ComposerTelemetry,
  type DmtRoutingMode,
  type SolverRegime,
  type CurvatureMode,
} from './pipeline/rainbowFrame';
import {
  cloneSu7RuntimeParams,
  createDefaultSu7RuntimeParams,
  sanitizeSu7RuntimeParams,
  type Su7ProjectorDescriptor,
  type Su7RuntimeParams,
  type Su7Schedule,
  type Su7Telemetry,
} from './pipeline/su7/types.js';
import { embedToC7 } from './pipeline/su7/embed.js';
import { buildScheduledUnitary } from './pipeline/su7/math.js';
import {
  ensureSu7TileBuffer,
  ensureSu7VectorBuffers,
  fillSu7TileBuffer,
  fillSu7VectorBuffers,
  SU7_TILE_SIZE,
  SU7_TILE_TEXTURE_ROWS_PER_TILE,
  SU7_TILE_TEXTURE_WIDTH,
  type Su7VectorBuffers,
} from './pipeline/su7/gpuPacking.js';
import {
  createHyperbolicAtlas,
  mapHyperbolicPolarToPixel,
  packageHyperbolicAtlasForGpu,
  type HyperbolicAtlas,
} from './hyperbolic/atlas';
import {
  COUPLING_KERNEL_PRESETS,
  clampKernelSpec,
  createKernelSpec,
  getDefaultKernelSpec,
  kernelSpecToJSON,
  type CouplingKernelPreset,
  type KernelSpec,
} from './kernel/kernelSpec';
import { getKernelSpecHub } from './kernel/kernelHub';
import { computeEdgeField } from './pipeline/edgeDetection';
import {
  FIELD_CONTRACTS,
  FIELD_KINDS,
  assertPhaseField,
  assertRimField,
  assertSurfaceField,
  assertVolumeField,
  describeImageData,
  type FieldKind,
  type FieldResolution,
  type RimField,
  type SurfaceField,
  type VolumeField,
} from './fields/contracts';
import type { OpticalFieldMetadata } from './fields/opticalField.js';
import {
  createInitialStatuses,
  markFieldUnavailable as setFieldUnavailable,
  markFieldUpdate as setFieldAvailable,
  refreshFieldStaleness,
  type FieldStatusMap,
} from './fields/state';
import {
  createVolumeStubState,
  snapshotVolumeStub,
  stepVolumeStub,
  type VolumeStubState,
} from './volumeStub';
import {
  DEFAULT_TRACER_CONFIG,
  applyTracerFeedback,
  mapTracerConfigToRuntime,
  sanitizeTracerConfig,
  type TracerConfig,
} from './pipeline/tracerFeedback';
import { DEFAULT_SYNTHETIC_SIZE, SYNTHETIC_CASES, type SyntheticCaseId } from './dev/syntheticDeck';

type KurRegime = 'locked' | 'highEnergy' | 'chaotic' | 'custom';

type PresetParams = {
  edgeThreshold: number;
  blend: number;
  kernel: KernelSpec;
  dmt: number;
  arousal: number;
  thetaMode: 'gradient' | 'global';
  thetaGlobal: number;
  beta2: number;
  jitter: number;
  sigma: number;
  microsaccade: boolean;
  speed: number;
  contrast: number;
  phasePin: boolean;
  alive: boolean;
  rimAlpha: number;
  su7Enabled: boolean;
  su7Gain: number;
  su7Preset: string;
  su7Seed: number;
  su7Schedule: Su7Schedule;
  su7Projector: Su7ProjectorDescriptor;
  su7ScheduleStrength: number;
};

type PresetSurface = {
  surfEnabled: boolean;
  surfaceBlend: number;
  warpAmp: number;
  nOrient: number;
  wallGroup: WallpaperGroup;
  surfaceRegion: SurfaceRegion;
};

type PresetDisplay = {
  displayMode: DisplayMode;
  polBins: number;
  normPin: boolean;
  curvatureStrength: number;
  curvatureMode: CurvatureMode;
  hyperbolicGuideSpacing: number;
};

const MAX_CURVATURE_STRENGTH = 0.95;
const HYPERBOLIC_GUIDE_SPACING_MIN = 0.25;
const HYPERBOLIC_GUIDE_SPACING_MAX = 2.5;
const DEFAULT_HYPERBOLIC_GUIDE_SPACING = 0.75;

type PresetRuntime = {
  renderBackend: 'gpu' | 'cpu';
  rimEnabled: boolean;
  showRimDebug: boolean;
  showHyperbolicGrid: boolean;
  showHyperbolicGuide: boolean;
  showSurfaceDebug: boolean;
  showPhaseDebug: boolean;
  phaseHeatmapEnabled: boolean;
  volumeEnabled: boolean;
  telemetryEnabled: boolean;
  telemetryOverlayEnabled: boolean;
  frameLoggingEnabled: boolean;
};

type PresetKuramoto = {
  kurEnabled: boolean;
  kurSync: boolean;
  kurRegime: KurRegime;
  K0: number;
  alphaKur: number;
  gammaKur: number;
  omega0: number;
  epsKur: number;
  fluxX: number;
  fluxY: number;
  qInit: number;
  smallWorldEnabled: boolean;
  smallWorldWeight: number;
  p_sw: number;
  smallWorldSeed: number;
  smallWorldDegree: number;
};

type PresetDeveloper = {
  selectedSyntheticCase: SyntheticCaseId;
};

type PresetMedia = {
  imagePath: string;
  imageName: string | null;
};

type ImageAsset = {
  path: string;
  name: string | null;
  width: number;
  height: number;
  mimeType: string | null;
};

type Preset = {
  name: string;
  params: PresetParams;
  surface: PresetSurface;
  display: PresetDisplay;
  tracer: TracerConfig;
  runtime: PresetRuntime;
  kuramoto: PresetKuramoto;
  developer: PresetDeveloper;
  media: PresetMedia | null;
  coupling: CouplingConfig;
  composer: ComposerConfig;
  couplingToggles: CouplingToggleState;
};

type CouplingToggleState = {
  rimToSurface: boolean;
  surfaceToRim: boolean;
};

type RenderFrameOptions = {
  toggles?: CouplingToggleState;
  volumeFieldOverride?: VolumeField | null;
  applyTracer?: boolean;
};

const RECORDING_FPS = 60;
const RECORDING_MIN_BITRATE = 8_000_000;
type RecordingPresetId = 'balanced' | 'high' | 'extreme';

const RECORDING_PRESETS: Record<
  RecordingPresetId,
  { label: string; bitsPerPixel: number; maxBitrate: number }
> = {
  balanced: { label: 'Balanced · ≈2.4 bpp', bitsPerPixel: 2.4, maxBitrate: 160_000_000 },
  high: { label: 'High · ≈3.2 bpp', bitsPerPixel: 3.2, maxBitrate: 220_000_000 },
  extreme: { label: 'Extreme · ≈4.5 bpp', bitsPerPixel: 4.5, maxBitrate: 320_000_000 },
};
const RECORDING_PRESET_ORDER: RecordingPresetId[] = ['balanced', 'high', 'extreme'];
const RECORDING_DEFAULT_PRESET: RecordingPresetId = 'balanced';
const RECORDING_FALLBACK_PRESET: Record<RecordingPresetId, RecordingPresetId | null> = {
  balanced: null,
  high: 'balanced',
  extreme: 'high',
};

type CaptureFormatConfig = {
  id: string;
  label: string;
  mimeType: string;
  container: 'mp4' | 'webm';
  quality: number;
};

const CAPTURE_FORMAT_CANDIDATES = [
  {
    id: 'webm-vp9',
    label: 'WebM · VP9',
    mimeType: 'video/webm;codecs=vp9',
    container: 'webm',
    quality: 3,
  },
  {
    id: 'webm-vp8',
    label: 'WebM · VP8',
    mimeType: 'video/webm;codecs=vp8',
    container: 'webm',
    quality: 2.5,
  },
  {
    id: 'webm-generic',
    label: 'WebM · Automatic',
    mimeType: 'video/webm',
    container: 'webm',
    quality: 2.4,
  },
  {
    id: 'mp4-h264-high',
    label: 'MP4 · H.264 (High)',
    mimeType: 'video/mp4;codecs=avc1.4d402a',
    container: 'mp4',
    quality: 2.2,
  },
  {
    id: 'mp4-h264-main',
    label: 'MP4 · H.264 (Main)',
    mimeType: 'video/mp4;codecs=avc1.4d401e',
    container: 'mp4',
    quality: 2.0,
  },
  {
    id: 'mp4-h264',
    label: 'MP4 · H.264',
    mimeType: 'video/mp4;codecs=h264',
    container: 'mp4',
    quality: 1.8,
  },
  {
    id: 'mp4-generic',
    label: 'MP4 · Automatic',
    mimeType: 'video/mp4',
    container: 'mp4',
    quality: 1.6,
  },
] satisfies Readonly<CaptureFormatConfig[]>;

type CaptureFormatId = (typeof CAPTURE_FORMAT_CANDIDATES)[number]['id'];

const CAPTURE_FORMAT_BY_ID = CAPTURE_FORMAT_CANDIDATES.reduce<
  Record<CaptureFormatId, CaptureFormatConfig>
>(
  (acc, candidate) => {
    acc[candidate.id] = candidate;
    return acc;
  },
  {} as Record<CaptureFormatId, CaptureFormatConfig>,
);

const DEFAULT_COUPLING_TOGGLES: CouplingToggleState = {
  rimToSurface: true,
  surfaceToRim: true,
};

const SU7_SCHEDULE_STRENGTH_MIN = 0;
const SU7_SCHEDULE_STRENGTH_MAX = 3;
const SU7_SCHEDULE_STRENGTH_DEFAULT = 1;

type Su7UiPresetId =
  | 'identity'
  | 'subtleChromaticSwirl'
  | 'orientationVortexCoupling'
  | 'volumeDepthChromaWarp';

const SU7_PRESET_DEFINITIONS: Record<
  Su7UiPresetId,
  {
    label: string;
    description: string;
    schedule: Su7Schedule;
    defaultStrength: number;
    defaultGain: number;
    projectorId: string;
    projectorWeight?: number;
  }
> = {
  identity: {
    label: 'Identity',
    description: 'No SU7 modulation; projectors mirror the base image.',
    schedule: Object.freeze([]) as Su7Schedule,
    defaultStrength: 0,
    defaultGain: 0,
    projectorId: 'identity',
  },
  subtleChromaticSwirl: {
    label: 'Subtle chromatic swirl',
    description: 'Applies low-amplitude stochastic offsets for gentle color drift.',
    schedule: Object.freeze([
      { gain: 0.38, spread: 1.15, label: 'swirl-a' },
      { gain: 0.24, spread: 0.75, label: 'swirl-b' },
    ]) as Su7Schedule,
    defaultStrength: 1.1,
    defaultGain: 1.0,
    projectorId: 'composerweights',
    projectorWeight: 0.85,
  },
  orientationVortexCoupling: {
    label: 'Orientation-vortex coupling',
    description: 'Binds rim orientation energy into paired SU7 axes for spin couplings.',
    schedule: Object.freeze([
      { gain: 0.46, index: 1, spread: 0.55, label: 'orientation-ring' },
      { gain: 0.34, index: 4, spread: 0.65, label: 'vortex-core' },
      { gain: 0.26, spread: 0.45, label: 'sustain' },
    ]) as Su7Schedule,
    defaultStrength: 1.25,
    defaultGain: 1.2,
    projectorId: 'overlaysplit',
    projectorWeight: 1.0,
  },
  volumeDepthChromaWarp: {
    label: 'Volume-depth chroma warp',
    description: 'Maps synthetic depth oscillations onto chroma swings and rim bias.',
    schedule: Object.freeze([
      { gain: 0.52, spread: 1.6, label: 'depth-breath' },
      { gain: 0.33, spread: 2.3, label: 'chroma-tilt' },
      { gain: 0.28, index: 6, spread: 0.85, label: 'rim-anchor' },
    ]) as Su7Schedule,
    defaultStrength: 1.35,
    defaultGain: 1.4,
    projectorId: 'directrgb',
    projectorWeight: 0.9,
  },
};

const SU7_PRESET_KEYS: Su7UiPresetId[] = [
  'identity',
  'subtleChromaticSwirl',
  'orientationVortexCoupling',
  'volumeDepthChromaWarp',
];

const SU7_PRESET_OPTIONS = SU7_PRESET_KEYS.map((key) => ({
  value: key,
  label: SU7_PRESET_DEFINITIONS[key].label,
}));

const isSu7PresetId = (value: string): value is Su7UiPresetId =>
  (SU7_PRESET_KEYS as readonly Su7UiPresetId[]).includes(value as Su7UiPresetId);

const SU7_PROJECTOR_OPTIONS: Array<{ value: string; label: string }> = [
  { value: 'identity', label: 'Identity blend' },
  { value: 'composerweights', label: 'Composer weights' },
  { value: 'overlaysplit', label: 'Overlay split overlays' },
  { value: 'directrgb', label: 'Direct RGB primaries' },
];

const cloneSu7Schedule = (schedule: Su7Schedule): Su7Schedule =>
  schedule.map((stage) => ({ ...stage }));

const scaleSu7Schedule = (schedule: Su7Schedule, strength: number): Su7Schedule =>
  schedule.map((stage) => ({
    ...stage,
    gain: stage.gain * strength,
  }));

const clampSu7ScheduleStrength = (value: number | undefined): number =>
  clamp(
    Number.isFinite(value) ? (value as number) : SU7_SCHEDULE_STRENGTH_DEFAULT,
    SU7_SCHEDULE_STRENGTH_MIN,
    SU7_SCHEDULE_STRENGTH_MAX,
  );

const deriveSu7BaseSchedule = (schedule: Su7Schedule, strength: number): Su7Schedule => {
  if (!schedule.length) {
    return [];
  }
  const normalizer = Math.abs(strength) > 1e-6 ? strength : 1;
  return schedule.map((stage) => ({
    ...stage,
    gain: stage.gain / normalizer,
  }));
};

const cloneTracerConfig = (config: TracerConfig): TracerConfig =>
  sanitizeTracerConfig({ ...config });

const COUPLING_PRESET_LABELS: Record<CouplingKernelPreset, string> = {
  dmt: 'DMT - Mexican hat',
  '5meo': '5-MeO - Uniform',
};

const COUPLING_PRESET_OPTIONS = (
  Object.keys(COUPLING_KERNEL_PRESETS) as CouplingKernelPreset[]
).map((value) => ({
  value,
  label: COUPLING_PRESET_LABELS[value] ?? value,
}));

const WALLPAPER_GROUP_VALUES: WallpaperGroup[] = [
  'off',
  'p1',
  'p2',
  'pm',
  'pg',
  'cm',
  'pmm',
  'pmg',
  'pgg',
  'cmm',
  'p4',
  'p4g',
  'p4m',
  'p3',
  'p31m',
  'p3m1',
  'p6',
  'p6m',
];

const DISPLAY_MODE_VALUES: DisplayMode[] = [
  'color',
  'grayBaseColorRims',
  'grayBaseGrayRims',
  'colorBaseGrayRims',
  'colorBaseBlendedRims',
];

const CURVATURE_MODE_VALUES: CurvatureMode[] = ['poincare', 'klein'];

const SURFACE_REGION_VALUES: SurfaceRegion[] = ['surfaces', 'edges', 'both'];

const RENDER_BACKEND_VALUES: Array<'gpu' | 'cpu'> = ['gpu', 'cpu'];

const KUR_REGIME_VALUES: KurRegime[] = ['locked', 'highEnergy', 'chaotic', 'custom'];

const THETA_MODE_VALUES = ['gradient', 'global'] as const;
const COMPOSER_ROUTING_VALUES = ['auto', 'rimBias', 'surfaceBias'] as const;
const COMPOSER_SOLVER_VALUES = ['balanced', 'rimLocked', 'surfaceLocked'] as const;

const SYNTHETIC_CASE_IDS = SYNTHETIC_CASES.map((entry) => entry.id);

const cloneCouplingToggles = (value: CouplingToggleState): CouplingToggleState => ({
  rimToSurface: value.rimToSurface,
  surfaceToRim: value.surfaceToRim,
});

const applyCouplingToggles = (
  config: CouplingConfig,
  toggles: CouplingToggleState,
): CouplingConfig => {
  const next = cloneCouplingConfig(config);
  if (!toggles.rimToSurface) {
    next.rimToSurfaceBlend = 0;
    next.rimToSurfaceAlign = 0;
  }
  if (!toggles.surfaceToRim) {
    next.surfaceToRimOffset = 0;
    next.surfaceToRimSigma = 0;
    next.surfaceToRimHue = 0;
  }
  return next;
};

const PRESETS: Preset[] = [
  {
    name: 'Rainbow Rims + DMT Kernel Effects',
    params: {
      edgeThreshold: 0.08,
      blend: 0.39,
      kernel: createKernelSpec({
        gain: 3.0,
        k0: 0.2,
        Q: 4.6,
        anisotropy: 0.95,
        chirality: 1.46,
        transparency: 0.28,
      }),
      dmt: 0.2,
      arousal: 0.35,
      thetaMode: 'gradient',
      thetaGlobal: 0,
      beta2: 1.9,
      jitter: 1.16,
      sigma: 4.0,
      microsaccade: true,
      speed: 1.32,
      contrast: 1.62,
      phasePin: true,
      alive: false,
      rimAlpha: 1.0,
      su7Enabled: false,
      su7Gain: 1,
      su7Preset: 'identity',
      su7Seed: 0,
      su7Schedule: [],
      su7Projector: { id: 'identity' },
      su7ScheduleStrength: 1,
    },
    surface: {
      surfEnabled: false,
      surfaceBlend: 0.35,
      warpAmp: 1.0,
      nOrient: 4,
      wallGroup: 'p4',
      surfaceRegion: 'surfaces',
    },
    display: {
      displayMode: 'color',
      polBins: 16,
      normPin: true,
      curvatureStrength: 0,
      curvatureMode: 'poincare',
      hyperbolicGuideSpacing: DEFAULT_HYPERBOLIC_GUIDE_SPACING,
    },
    tracer: cloneTracerConfig(DEFAULT_TRACER_CONFIG),
    runtime: {
      renderBackend: 'gpu',
      rimEnabled: true,
      showRimDebug: false,
      showHyperbolicGrid: false,
      showHyperbolicGuide: false,
      showSurfaceDebug: false,
      showPhaseDebug: false,
      phaseHeatmapEnabled: false,
      volumeEnabled: false,
      telemetryEnabled: false,
      telemetryOverlayEnabled: false,
      frameLoggingEnabled: true,
    },
    kuramoto: {
      kurEnabled: false,
      kurSync: false,
      kurRegime: 'locked',
      K0: 0.6,
      alphaKur: 0.2,
      gammaKur: 0.15,
      omega0: 0,
      epsKur: 0.002,
      fluxX: 0,
      fluxY: 0,
      qInit: 1,
      smallWorldEnabled: false,
      smallWorldWeight: 0.75,
      p_sw: 0.05,
      smallWorldSeed: 1337,
      smallWorldDegree: 12,
    },
    developer: {
      selectedSyntheticCase: 'circles',
    },
    media: null,
    coupling: {
      rimToSurfaceBlend: 0.45,
      rimToSurfaceAlign: 0.55,
      surfaceToRimOffset: 0.4,
      surfaceToRimSigma: 0.6,
      surfaceToRimHue: 0.5,
      kurToTransparency: 0.35,
      kurToOrientation: 0.35,
      kurToChirality: 0.6,
      volumePhaseToHue: 0.35,
      volumeDepthToWarp: 0.3,
    },
    composer: createDefaultComposerConfig(),
    couplingToggles: DEFAULT_COUPLING_TOGGLES,
  },
];

const KUR_REGIME_PRESETS: Record<
  Exclude<KurRegime, 'custom'>,
  {
    label: string;
    description: string;
    params: {
      K0: number;
      alphaKur: number;
      gammaKur: number;
      omega0: number;
      epsKur: number;
    };
  }
> = {
  locked: {
    label: 'Locked coherence',
    description: 'Low-noise lattice with stable phase alignment.',
    params: {
      K0: 0.85,
      alphaKur: 0.18,
      gammaKur: 0.16,
      omega0: 0.0,
      epsKur: 0.0025,
    },
  },
  highEnergy: {
    label: 'High-energy flux',
    description: 'Strong coupling and drive yielding intense wavefronts.',
    params: {
      K0: 1.35,
      alphaKur: 0.12,
      gammaKur: 0.1,
      omega0: 0.55,
      epsKur: 0.006,
    },
  },
  chaotic: {
    label: 'Chaotic drift',
    description: 'Loose locking with broadband oscillations and noise.',
    params: {
      K0: 1.1,
      alphaKur: 0.28,
      gammaKur: 0.12,
      omega0: 0.35,
      epsKur: 0.012,
    },
  },
};

type FrameProfilerState = {
  enabled: boolean;
  samples: number[];
  maxSamples: number;
  label: string;
};

const RIM_HIST_BINS = 64;
const SURFACE_HIST_BINS = 32;
const PHASE_HIST_BINS = 64;
const PHASE_HIST_SCALE = 2.5;

const COMPOSER_FIELD_LABELS: Record<ComposerFieldId, string> = {
  surface: 'Surface',
  rim: 'Rim',
  kur: 'Kuramoto',
  volume: 'Volume',
};

const cloneCouplingConfig = (value: CouplingConfig): CouplingConfig => ({
  rimToSurfaceBlend: value.rimToSurfaceBlend,
  rimToSurfaceAlign: value.rimToSurfaceAlign,
  surfaceToRimOffset: value.surfaceToRimOffset,
  surfaceToRimSigma: value.surfaceToRimSigma,
  surfaceToRimHue: value.surfaceToRimHue,
  kurToTransparency: value.kurToTransparency,
  kurToOrientation: value.kurToOrientation,
  kurToChirality: value.kurToChirality,
  volumePhaseToHue: value.volumePhaseToHue,
  volumeDepthToWarp: value.volumeDepthToWarp,
});

const sanitizeNumber = (value: unknown, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

const sanitizeBoolean = (value: unknown, fallback: boolean): boolean =>
  typeof value === 'boolean' ? value : fallback;

const sanitizeEnum = <T extends string>(value: unknown, allowed: readonly T[], fallback: T): T => {
  if (typeof value === 'string' && (allowed as readonly string[]).includes(value)) {
    return value as T;
  }
  return fallback;
};

const sanitizeCouplingConfig = (
  value: Partial<CouplingConfig> | null | undefined,
  fallback: CouplingConfig,
): CouplingConfig => {
  const source = value ?? {};
  return {
    rimToSurfaceBlend:
      typeof source.rimToSurfaceBlend === 'number'
        ? source.rimToSurfaceBlend
        : fallback.rimToSurfaceBlend,
    rimToSurfaceAlign:
      typeof source.rimToSurfaceAlign === 'number'
        ? source.rimToSurfaceAlign
        : fallback.rimToSurfaceAlign,
    surfaceToRimOffset:
      typeof source.surfaceToRimOffset === 'number'
        ? source.surfaceToRimOffset
        : fallback.surfaceToRimOffset,
    surfaceToRimSigma:
      typeof source.surfaceToRimSigma === 'number'
        ? source.surfaceToRimSigma
        : fallback.surfaceToRimSigma,
    surfaceToRimHue:
      typeof source.surfaceToRimHue === 'number'
        ? source.surfaceToRimHue
        : fallback.surfaceToRimHue,
    kurToTransparency:
      typeof source.kurToTransparency === 'number'
        ? source.kurToTransparency
        : fallback.kurToTransparency,
    kurToOrientation:
      typeof source.kurToOrientation === 'number'
        ? source.kurToOrientation
        : fallback.kurToOrientation,
    kurToChirality:
      typeof source.kurToChirality === 'number' ? source.kurToChirality : fallback.kurToChirality,
    volumePhaseToHue:
      typeof source.volumePhaseToHue === 'number'
        ? source.volumePhaseToHue
        : fallback.volumePhaseToHue,
    volumeDepthToWarp:
      typeof source.volumeDepthToWarp === 'number'
        ? source.volumeDepthToWarp
        : fallback.volumeDepthToWarp,
  };
};

const clampComposerValue = (value: number, min: number, max: number, fallback: number) => {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return clamp(value, min, max);
};

const cloneComposerConfig = (config: ComposerConfig): ComposerConfig => ({
  fields: {
    surface: { ...config.fields.surface },
    rim: { ...config.fields.rim },
    kur: { ...config.fields.kur },
    volume: { ...config.fields.volume },
  },
  dmtRouting: config.dmtRouting,
  solverRegime: config.solverRegime,
});

const sanitizeComposerImport = (
  value: Partial<ComposerConfig> | null | undefined,
  fallback: ComposerConfig,
): ComposerConfig => {
  const defaults = createDefaultComposerConfig();
  const base = fallback ?? defaults;
  const source = value ?? {};
  const result = cloneComposerConfig(base);
  result.dmtRouting = sanitizeEnum(source.dmtRouting, COMPOSER_ROUTING_VALUES, base.dmtRouting);
  result.solverRegime = sanitizeEnum(
    source.solverRegime,
    COMPOSER_SOLVER_VALUES,
    base.solverRegime,
  );
  COMPOSER_FIELD_LIST.forEach((field) => {
    const incoming = source.fields?.[field];
    const fallbackField = base.fields[field];
    const defaultField = defaults.fields[field];
    result.fields[field] = {
      exposure: clampComposerValue(
        typeof incoming?.exposure === 'number' ? incoming.exposure : fallbackField.exposure,
        0,
        8,
        defaultField.exposure,
      ),
      gamma: clampComposerValue(
        typeof incoming?.gamma === 'number' ? incoming.gamma : fallbackField.gamma,
        0.2,
        5,
        defaultField.gamma,
      ),
      weight: clampComposerValue(
        typeof incoming?.weight === 'number' ? incoming.weight : fallbackField.weight,
        0,
        2.5,
        defaultField.weight,
      ),
    };
  });
  return result;
};

const clampInt = (value: number, min: number, max: number): number =>
  Math.min(Math.max(Math.round(value), min), max);

const sanitizePresetMedia = (media: unknown, fallback: PresetMedia | null): PresetMedia | null => {
  if (!media || typeof media !== 'object') return null;
  const maybePath = (media as { imagePath?: unknown }).imagePath;
  if (typeof maybePath !== 'string' || maybePath.length === 0) {
    return null;
  }
  const maybeName = (media as { imageName?: unknown }).imageName;
  return {
    imagePath: maybePath,
    imageName: typeof maybeName === 'string' ? maybeName : (fallback?.imageName ?? null),
  };
};

const sanitizePresetPayload = (raw: unknown, fallback: Preset): Preset | null => {
  if (!raw || typeof raw !== 'object') return null;
  const source = raw as Record<string, unknown>;
  const paramsSource = (source.params as Record<string, unknown>) ?? {};
  const kernelSource = (paramsSource.kernel as Record<string, unknown>) ?? {};
  const surfaceSource = (source.surface as Record<string, unknown>) ?? {};
  const displaySource = (source.display as Record<string, unknown>) ?? {};
  const runtimeSource = (source.runtime as Record<string, unknown>) ?? {};
  const kuramotoSource = (source.kuramoto as Record<string, unknown>) ?? {};
  const developerSource = (source.developer as Record<string, unknown>) ?? {};
  const togglesSource = (source.couplingToggles as Record<string, unknown>) ?? {};

  const fallbackParams = fallback.params;
  const fallbackSurface = fallback.surface;
  const fallbackDisplay = fallback.display;
  const fallbackRuntime = fallback.runtime;
  const fallbackKuramoto = fallback.kuramoto;
  const fallbackDeveloper = fallback.developer;
  const fallbackToggles = fallback.couplingToggles;
  const fallbackSu7: Su7RuntimeParams = {
    enabled: fallbackParams.su7Enabled,
    gain: fallbackParams.su7Gain,
    preset: fallbackParams.su7Preset,
    seed: fallbackParams.su7Seed,
    schedule: fallbackParams.su7Schedule,
    projector: fallbackParams.su7Projector,
  };
  const sanitizedSu7 = sanitizeSu7RuntimeParams(
    {
      enabled: paramsSource.su7Enabled,
      gain: paramsSource.su7Gain,
      preset: paramsSource.su7Preset,
      seed: paramsSource.su7Seed,
      schedule: paramsSource.su7Schedule as unknown,
      projector: paramsSource.su7Projector,
    },
    fallbackSu7,
  );

  const sanitized: Preset = {
    name: typeof source.name === 'string' ? (source.name as string) : fallback.name,
    params: {
      edgeThreshold: sanitizeNumber(paramsSource.edgeThreshold, fallbackParams.edgeThreshold),
      blend: sanitizeNumber(paramsSource.blend, fallbackParams.blend),
      kernel: createKernelSpec({
        gain: sanitizeNumber(kernelSource.gain, fallbackParams.kernel.gain),
        k0: sanitizeNumber(kernelSource.k0, fallbackParams.kernel.k0),
        Q: sanitizeNumber(kernelSource.Q, fallbackParams.kernel.Q),
        anisotropy: sanitizeNumber(kernelSource.anisotropy, fallbackParams.kernel.anisotropy),
        chirality: sanitizeNumber(kernelSource.chirality, fallbackParams.kernel.chirality),
        transparency: sanitizeNumber(kernelSource.transparency, fallbackParams.kernel.transparency),
        couplingPreset:
          typeof kernelSource.couplingPreset === 'string'
            ? (kernelSource.couplingPreset as CouplingKernelPreset)
            : fallbackParams.kernel.couplingPreset,
      }),
      dmt: sanitizeNumber(paramsSource.dmt, fallbackParams.dmt),
      arousal: sanitizeNumber(paramsSource.arousal, fallbackParams.arousal),
      thetaMode: sanitizeEnum(paramsSource.thetaMode, THETA_MODE_VALUES, fallbackParams.thetaMode),
      thetaGlobal: sanitizeNumber(paramsSource.thetaGlobal, fallbackParams.thetaGlobal),
      beta2: sanitizeNumber(paramsSource.beta2, fallbackParams.beta2),
      jitter: sanitizeNumber(paramsSource.jitter, fallbackParams.jitter),
      sigma: sanitizeNumber(paramsSource.sigma, fallbackParams.sigma),
      microsaccade: sanitizeBoolean(paramsSource.microsaccade, fallbackParams.microsaccade),
      speed: sanitizeNumber(paramsSource.speed, fallbackParams.speed),
      contrast: sanitizeNumber(paramsSource.contrast, fallbackParams.contrast),
      phasePin: sanitizeBoolean(paramsSource.phasePin, fallbackParams.phasePin),
      alive: sanitizeBoolean(paramsSource.alive, fallbackParams.alive),
      rimAlpha: Math.min(
        Math.max(sanitizeNumber(paramsSource.rimAlpha, fallbackParams.rimAlpha), 0),
        1,
      ),
      su7Enabled: sanitizedSu7.enabled,
      su7Gain: sanitizedSu7.gain,
      su7Preset: sanitizedSu7.preset,
      su7Seed: sanitizedSu7.seed,
      su7Schedule: sanitizedSu7.schedule,
      su7Projector: sanitizedSu7.projector,
      su7ScheduleStrength: clampSu7ScheduleStrength(
        sanitizeNumber(
          paramsSource.su7ScheduleStrength,
          fallbackParams.su7ScheduleStrength ?? SU7_SCHEDULE_STRENGTH_DEFAULT,
        ),
      ),
    },
    surface: {
      surfEnabled: sanitizeBoolean(surfaceSource.surfEnabled, fallbackSurface.surfEnabled),
      surfaceBlend: sanitizeNumber(surfaceSource.surfaceBlend, fallbackSurface.surfaceBlend),
      warpAmp: sanitizeNumber(surfaceSource.warpAmp, fallbackSurface.warpAmp),
      nOrient: clampInt(sanitizeNumber(surfaceSource.nOrient, fallbackSurface.nOrient), 2, 8),
      wallGroup: sanitizeEnum(
        surfaceSource.wallGroup,
        WALLPAPER_GROUP_VALUES,
        fallbackSurface.wallGroup,
      ),
      surfaceRegion: sanitizeEnum(
        surfaceSource.surfaceRegion,
        SURFACE_REGION_VALUES,
        fallbackSurface.surfaceRegion,
      ),
    },
    display: {
      displayMode: sanitizeEnum(
        displaySource.displayMode,
        DISPLAY_MODE_VALUES,
        fallbackDisplay.displayMode,
      ),
      polBins: clampInt(sanitizeNumber(displaySource.polBins, fallbackDisplay.polBins), 0, 32),
      normPin: sanitizeBoolean(displaySource.normPin, fallbackDisplay.normPin),
      curvatureStrength: clamp(
        sanitizeNumber(displaySource.curvatureStrength, fallbackDisplay.curvatureStrength),
        0,
        MAX_CURVATURE_STRENGTH,
      ),
      curvatureMode: sanitizeEnum(
        displaySource.curvatureMode,
        CURVATURE_MODE_VALUES,
        fallbackDisplay.curvatureMode,
      ),
      hyperbolicGuideSpacing: clamp(
        sanitizeNumber(
          displaySource.hyperbolicGuideSpacing,
          fallbackDisplay.hyperbolicGuideSpacing ?? DEFAULT_HYPERBOLIC_GUIDE_SPACING,
        ),
        HYPERBOLIC_GUIDE_SPACING_MIN,
        HYPERBOLIC_GUIDE_SPACING_MAX,
      ),
    },
    runtime: {
      renderBackend: sanitizeEnum(
        runtimeSource.renderBackend,
        RENDER_BACKEND_VALUES,
        fallbackRuntime.renderBackend,
      ),
      rimEnabled: sanitizeBoolean(runtimeSource.rimEnabled, fallbackRuntime.rimEnabled),
      showRimDebug: sanitizeBoolean(runtimeSource.showRimDebug, fallbackRuntime.showRimDebug),
      showHyperbolicGrid: sanitizeBoolean(
        runtimeSource.showHyperbolicGrid,
        fallbackRuntime.showHyperbolicGrid,
      ),
      showHyperbolicGuide: sanitizeBoolean(
        runtimeSource.showHyperbolicGuide,
        fallbackRuntime.showHyperbolicGuide,
      ),
      showSurfaceDebug: sanitizeBoolean(
        runtimeSource.showSurfaceDebug,
        fallbackRuntime.showSurfaceDebug,
      ),
      showPhaseDebug: sanitizeBoolean(runtimeSource.showPhaseDebug, fallbackRuntime.showPhaseDebug),
      phaseHeatmapEnabled: sanitizeBoolean(
        runtimeSource.phaseHeatmapEnabled,
        fallbackRuntime.phaseHeatmapEnabled,
      ),
      volumeEnabled: sanitizeBoolean(runtimeSource.volumeEnabled, fallbackRuntime.volumeEnabled),
      telemetryEnabled: sanitizeBoolean(
        runtimeSource.telemetryEnabled,
        fallbackRuntime.telemetryEnabled,
      ),
      telemetryOverlayEnabled: sanitizeBoolean(
        runtimeSource.telemetryOverlayEnabled,
        fallbackRuntime.telemetryOverlayEnabled,
      ),
      frameLoggingEnabled: sanitizeBoolean(
        runtimeSource.frameLoggingEnabled,
        fallbackRuntime.frameLoggingEnabled,
      ),
    },
    kuramoto: {
      kurEnabled: sanitizeBoolean(kuramotoSource.kurEnabled, fallbackKuramoto.kurEnabled),
      kurSync: sanitizeBoolean(kuramotoSource.kurSync, fallbackKuramoto.kurSync),
      kurRegime: sanitizeEnum(
        kuramotoSource.kurRegime,
        KUR_REGIME_VALUES,
        fallbackKuramoto.kurRegime,
      ),
      K0: sanitizeNumber(kuramotoSource.K0, fallbackKuramoto.K0),
      alphaKur: sanitizeNumber(kuramotoSource.alphaKur, fallbackKuramoto.alphaKur),
      gammaKur: sanitizeNumber(kuramotoSource.gammaKur, fallbackKuramoto.gammaKur),
      omega0: sanitizeNumber(kuramotoSource.omega0, fallbackKuramoto.omega0),
      epsKur: sanitizeNumber(kuramotoSource.epsKur, fallbackKuramoto.epsKur),
      fluxX: sanitizeNumber(kuramotoSource.fluxX, fallbackKuramoto.fluxX),
      fluxY: sanitizeNumber(kuramotoSource.fluxY, fallbackKuramoto.fluxY),
      qInit: clampInt(sanitizeNumber(kuramotoSource.qInit, fallbackKuramoto.qInit), 0, 999),
      smallWorldEnabled: sanitizeBoolean(
        kuramotoSource.smallWorldEnabled,
        fallbackKuramoto.smallWorldEnabled,
      ),
      smallWorldWeight: sanitizeNumber(
        kuramotoSource.smallWorldWeight,
        fallbackKuramoto.smallWorldWeight,
      ),
      p_sw: sanitizeNumber(kuramotoSource.p_sw, fallbackKuramoto.p_sw),
      smallWorldSeed: clampInt(
        sanitizeNumber(kuramotoSource.smallWorldSeed, fallbackKuramoto.smallWorldSeed),
        0,
        Number.MAX_SAFE_INTEGER,
      ),
      smallWorldDegree: clampInt(
        sanitizeNumber(kuramotoSource.smallWorldDegree, fallbackKuramoto.smallWorldDegree),
        0,
        64,
      ),
    },
    developer: {
      selectedSyntheticCase:
        typeof developerSource.selectedSyntheticCase === 'string' &&
        SYNTHETIC_CASE_IDS.includes(developerSource.selectedSyntheticCase as SyntheticCaseId)
          ? (developerSource.selectedSyntheticCase as SyntheticCaseId)
          : fallbackDeveloper.selectedSyntheticCase,
    },
    media: sanitizePresetMedia(source.media, fallback.media),
    coupling: sanitizeCouplingConfig(source.coupling as Partial<CouplingConfig>, fallback.coupling),
    composer: sanitizeComposerImport(source.composer as Partial<ComposerConfig>, fallback.composer),
    couplingToggles: {
      rimToSurface: sanitizeBoolean(togglesSource.rimToSurface, fallbackToggles.rimToSurface),
      surfaceToRim: sanitizeBoolean(togglesSource.surfaceToRim, fallbackToggles.surfaceToRim),
    },
  };

  return sanitized;
};

const arrayBufferToBase64 = (buffer: ArrayBuffer): string => {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.length;
  for (let i = 0; i < len; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
};

const textEncoder = new TextEncoder();

type TarEntry = {
  name: string;
  content: Uint8Array;
  mode?: number;
  mtime?: number;
};

const TAR_BLOCK_SIZE = 512;

const writeString = (buffer: Uint8Array, offset: number, value: string) => {
  const bytes = textEncoder.encode(value);
  const length = Math.min(bytes.length, buffer.length - offset);
  buffer.set(bytes.subarray(0, length), offset);
};

const writeOctal = (buffer: Uint8Array, offset: number, length: number, value: number) => {
  const octal = value.toString(8).padStart(length - 1, '0');
  for (let i = 0; i < length - 1; i += 1) {
    buffer[offset + i] = octal.charCodeAt(i);
  }
  buffer[offset + length - 1] = 0;
};

const createTarBlob = (entries: TarEntry[]): Blob => {
  const parts: Uint8Array[] = [];
  const nowSeconds = Math.floor(Date.now() / 1000);

  entries.forEach(({ name, content, mode = 0o644, mtime = nowSeconds }) => {
    const header = new Uint8Array(TAR_BLOCK_SIZE);
    writeString(header, 0, name);
    writeOctal(header, 100, 8, mode);
    writeOctal(header, 108, 8, 0);
    writeOctal(header, 116, 8, 0);
    writeOctal(header, 124, 12, content.length);
    writeOctal(header, 136, 12, mtime);
    header[156] = '0'.charCodeAt(0);
    writeString(header, 257, 'ustar');
    header[262] = 0;
    writeString(header, 263, '00');
    writeString(header, 265, 'user');
    writeString(header, 297, 'group');
    for (let i = 148; i < 156; i += 1) {
      header[i] = 0x20;
    }
    let checksum = 0;
    for (let i = 0; i < TAR_BLOCK_SIZE; i += 1) {
      checksum += header[i];
    }
    const checksumOctal = checksum.toString(8).padStart(6, '0');
    for (let i = 0; i < 6; i += 1) {
      header[148 + i] = checksumOctal.charCodeAt(i);
    }
    header[154] = 0;
    header[155] = 0x20;
    parts.push(header);
    parts.push(content);
    const remainder = content.length % TAR_BLOCK_SIZE;
    if (remainder !== 0) {
      parts.push(new Uint8Array(TAR_BLOCK_SIZE - remainder));
    }
  });

  parts.push(new Uint8Array(TAR_BLOCK_SIZE));
  parts.push(new Uint8Array(TAR_BLOCK_SIZE));

  return new Blob(parts, { type: 'application/x-tar' });
};

const compressBlobGzip = async (blob: Blob): Promise<Blob | null> => {
  const CompressionStreamCtor = (
    window as typeof window & {
      CompressionStream?: typeof CompressionStream;
    }
  ).CompressionStream;
  if (!CompressionStreamCtor) {
    return null;
  }
  try {
    const stream = blob.stream().pipeThrough(new CompressionStreamCtor('gzip'));
    const compressed = await new Response(stream).blob();
    return compressed;
  } catch {
    return null;
  }
};

const downloadBlob = (blob: Blob, filename: string) => {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
};

const formatCouplingKey = (key: keyof CouplingConfig) =>
  key.replace(/([A-Z])/g, ' $1').replace(/^./, (ch) => ch.toUpperCase());

type KurFrameView = {
  buffer: ArrayBuffer;
  gradX: Float32Array;
  gradY: Float32Array;
  vort: Float32Array;
  coh: Float32Array;
  amp: Float32Array;
  timestamp: number;
  frameId: number;
  kernelVersion: number;
  meta: OpticalFieldMetadata;
  instrumentation: KuramotoInstrumentationSnapshot;
};

type WorkerFrameMessage = {
  kind: 'frame';
  buffer: ArrayBuffer;
  timestamp: number;
  frameId: number;
  queueDepth: number;
  kernelVersion?: number;
  meta: OpticalFieldMetadata;
  instrumentation: KuramotoInstrumentationSnapshot;
};

type WorkerReadyMessage = { kind: 'ready'; width: number; height: number };
type WorkerLogMessage = { kind: 'log'; message: string };
type WorkerSimulateResultMessage = {
  kind: 'simulateResult';
  buffers: ArrayBuffer[];
  width: number;
  height: number;
  frameCount: number;
};

type WorkerIncomingMessage =
  | WorkerFrameMessage
  | WorkerReadyMessage
  | WorkerLogMessage
  | WorkerSimulateResultMessage;

type TelemetryPhase = 'frame' | 'renderGpu' | 'renderCpu' | 'kuramoto';

type TelemetryRecord = {
  phase: TelemetryPhase;
  ms: number;
  ts: number;
};

type ParitySceneSummary = {
  label: string;
  mismatched: number;
  percent: number;
  maxDelta: number;
  maxCoord: [number, number];
  cpuColor: [number, number, number];
  gpuColor: [number, number, number];
};

type ParitySummary = {
  scenes: ParitySceneSummary[];
  tolerancePercent: number;
  timestamp: number;
};

type PerformanceSnapshot = {
  frameCount: number;
  cpuMs: number;
  gpuMs: number;
  cpuFps: number;
  gpuFps: number;
  throughputGain: number;
  timestamp: number;
};

type FrameMetricsEntry = {
  backend: 'cpu' | 'gpu';
  ts: number;
  metrics: RainbowFrameMetrics;
  kernelVersion: number;
};

const printRgb = (values: [number, number, number]) =>
  `(${values.map((value) => Math.round(value)).join(', ')})`;

declare global {
  interface Window {
    __setFrameProfiler?: (enabled: boolean, sampleCount?: number, label?: string) => void;
    __runFrameRegression?: (frameCount?: number) => { maxDelta: number; perFrameMax: number[] };
    __setRenderBackend?: (backend: 'gpu' | 'cpu') => void;
    __runGpuParityCheck?: () => Promise<{
      scenes: {
        label: string;
        mismatched: number;
        percent: number;
        maxDelta: number;
        maxCoord: [number, number];
        cpuColor: [number, number, number];
        gpuColor: [number, number, number];
      }[];
      tolerancePercent: number;
      timestamp: number;
    } | null>;
    __measureRenderPerformance?: (frameCount?: number) => {
      frameCount: number;
      cpuMs: number;
      gpuMs: number;
      cpuFps: number;
      gpuFps: number;
      throughputGain: number;
    } | null;
    __setTelemetryEnabled?: (enabled: boolean) => void;
    __getTelemetryHistory?: () => TelemetryRecord[];
    __getFrameMetrics?: () => FrameMetricsEntry[];
  }
}

const formatBytes = (bytes: number) => {
  if (bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const idx = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / Math.pow(1024, idx);
  const precision = value >= 100 || idx === 0 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(precision)} ${units[idx]}`;
};
const clamp01 = (v: number) => clamp(v, 0, 1);
const useRandN = () => {
  const spareRef = useRef<number | null>(null);
  return useCallback(() => {
    if (spareRef.current != null) {
      const value = spareRef.current;
      spareRef.current = null;
      return value;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    const mag = Math.sqrt(-2.0 * Math.log(u));
    const z0 = mag * Math.cos(2 * Math.PI * v);
    const z1 = mag * Math.sin(2 * Math.PI * v);
    spareRef.current = z1;
    return z0;
  }, []);
};

const displayModeToEnum = (mode: DisplayMode) => {
  switch (mode) {
    case 'grayBaseColorRims':
      return 1;
    case 'grayBaseGrayRims':
      return 2;
    case 'colorBaseGrayRims':
      return 3;
    case 'colorBaseBlendedRims':
      return 4;
    default:
      return 0;
  }
};

const FIELD_STATUS_STYLES = {
  ok: {
    border: 'rgba(34,197,94,0.35)',
    background: 'rgba(34,197,94,0.16)',
    color: '#bbf7d0',
  },
  warn: {
    border: 'rgba(251,191,36,0.4)',
    background: 'rgba(251,191,36,0.18)',
    color: '#fde68a',
  },
  stale: {
    border: 'rgba(248,113,113,0.45)',
    background: 'rgba(248,113,113,0.2)',
    color: '#fecaca',
  },
  missing: {
    border: 'rgba(148,163,184,0.35)',
    background: 'rgba(148,163,184,0.16)',
    color: '#e2e8f0',
  },
} as const;

const FIELD_STATUS_LABELS = {
  ok: 'fresh',
  warn: 'lagging',
  stale: 'stale',
  missing: 'missing',
} as const;

const surfaceRegionToEnum = (region: SurfaceRegion) => {
  switch (region) {
    case 'surfaces':
      return 0;
    case 'edges':
      return 1;
    default:
      return 2;
  }
};

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const hyperbolicGridCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const presetFileInputRef = useRef<HTMLInputElement | null>(null);
  const gpuStateRef = useRef<{ gl: WebGL2RenderingContext; renderer: GpuRenderer } | null>(null);
  const pendingStaticUploadRef = useRef(true);
  const su7VectorBuffersRef = useRef<Su7VectorBuffers | null>(null);
  const su7TileBufferRef = useRef<Float32Array | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const captureStreamRef = useRef<MediaStream | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const recordedUrlRef = useRef<string | null>(null);
  const recordingMimeTypeRef = useRef<string | null>(null);

  const kernelHub = useMemo(() => getKernelSpecHub(), []);
  const kernelEventRef = useRef(kernelHub.getSnapshot());
  const kurKernelVersionRef = useRef(kernelEventRef.current.version);
  const kurAppliedKernelVersionRef = useRef(kurKernelVersionRef.current);

  const [width, setWidth] = useState(720);
  const [height, setHeight] = useState(480);
  const [renderBackend, setRenderBackend] = useState<'gpu' | 'cpu'>('gpu');

  const [imgBitmap, setImgBitmap] = useState<ImageBitmap | null>(null);
  const [imageAsset, setImageAsset] = useState<ImageAsset | null>(null);
  const [includeImageInPreset, setIncludeImageInPreset] = useState(true);
  const [shareStatus, setShareStatus] = useState<{ message: string; url?: string } | null>(null);
  const [frameExportDuration, setFrameExportDuration] = useState(10);
  const [frameExporting, setFrameExporting] = useState(false);
  const [frameExportProgress, setFrameExportProgress] = useState<null | {
    current: number;
    total: number;
    message: string;
  }>(null);
  const [frameExportError, setFrameExportError] = useState<string | null>(null);
  const shareStatusTimeoutRef = useRef<number | null>(null);
  const frameExportAbortRef = useRef<{ canceled: boolean } | null>(null);
  const basePixelsRef = useRef<ImageData | null>(null);
  const surfaceFieldRef = useRef<SurfaceField | null>(null);
  const rimFieldRef = useRef<RimField | null>(null);

  const [edgeThreshold, setEdgeThreshold] = useState(0.22);
  const [blend, setBlend] = useState(0.65);
  const [rimAlpha, setRimAlpha] = useState(1.0);
  const [rimEnabled, setRimEnabled] = useState(true);
  const [showRimDebug, setShowRimDebug] = useState(false);
  const [showSurfaceDebug, setShowSurfaceDebug] = useState(false);
  const [showPhaseDebug, setShowPhaseDebug] = useState(false);
  const [showHyperbolicGrid, setShowHyperbolicGrid] = useState(false);
  const [showHyperbolicGuide, setShowHyperbolicGuide] = useState(false);
  const [hyperbolicGuideSpacing, setHyperbolicGuideSpacing] = useState(
    DEFAULT_HYPERBOLIC_GUIDE_SPACING,
  );

  const [beta2, setBeta2] = useState(1.1);
  const [jitter, setJitter] = useState(0.5);
  const [sigma, setSigma] = useState(1.4);
  const [microsaccade, setMicrosaccade] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [contrast, setContrast] = useState(1.0);
  const [su7Params, setSu7Params] = useState<Su7RuntimeParams>(() =>
    createDefaultSu7RuntimeParams(),
  );
  const su7BaseScheduleRef = useRef<Su7Schedule>(
    cloneSu7Schedule(SU7_PRESET_DEFINITIONS.identity.schedule),
  );
  const [su7ScheduleStrength, setSu7ScheduleStrength] = useState<number>(
    SU7_SCHEDULE_STRENGTH_DEFAULT,
  );
  const handleSu7EnabledChange = useCallback(
    (value: boolean) => {
      setSu7Params((prev) => ({ ...prev, enabled: value }));
    },
    [setSu7Params],
  );
  const handleSu7GainChange = useCallback(
    (value: number) => {
      setSu7Params((prev) => ({
        ...prev,
        gain: Number.isFinite(value) ? Math.max(0, value) : prev.gain,
      }));
    },
    [setSu7Params],
  );
  const handleSu7SeedChange = useCallback(
    (value: number) => {
      setSu7Params((prev) => ({
        ...prev,
        seed: Number.isFinite(value) ? Math.trunc(value) : prev.seed,
      }));
    },
    [setSu7Params],
  );
  const handleSu7ProjectorChange = useCallback(
    (value: string) => {
      setSu7Params((prev) => ({
        ...prev,
        projector: {
          ...prev.projector,
          id: value.toLowerCase(),
        },
      }));
    },
    [setSu7Params],
  );
  const handleSu7PresetChange = useCallback(
    (value: string) => {
      const presetId = isSu7PresetId(value) ? value : 'identity';
      const definition = SU7_PRESET_DEFINITIONS[presetId];
      const baseSchedule = cloneSu7Schedule(definition.schedule);
      su7BaseScheduleRef.current = baseSchedule;
      const strength = clampSu7ScheduleStrength(definition.defaultStrength);
      const schedule = scaleSu7Schedule(baseSchedule, strength);
      setSu7ScheduleStrength(strength);
      setSu7Params((prev) => ({
        ...prev,
        preset: presetId,
        gain: definition.defaultGain,
        schedule,
        projector: {
          ...prev.projector,
          id: definition.projectorId,
          weight:
            typeof definition.projectorWeight === 'number'
              ? definition.projectorWeight
              : prev.projector.weight,
        },
      }));
    },
    [setSu7Params, setSu7ScheduleStrength],
  );
  const handleSu7ScheduleStrengthChange = useCallback(
    (value: number) => {
      const strength = clampSu7ScheduleStrength(value);
      const base = su7BaseScheduleRef.current;
      const schedule = scaleSu7Schedule(base, strength);
      setSu7ScheduleStrength(strength);
      setSu7Params((prev) => ({
        ...prev,
        schedule,
      }));
    },
    [setSu7Params, setSu7ScheduleStrength],
  );
  const su7PresetOptions = useMemo(() => {
    if (su7Params.preset && !isSu7PresetId(su7Params.preset)) {
      const label = su7Params.preset.length > 0 ? su7Params.preset : 'Custom preset';
      return [...SU7_PRESET_OPTIONS, { value: su7Params.preset, label }];
    }
    return SU7_PRESET_OPTIONS;
  }, [su7Params.preset]);
  const su7PresetSelectValue =
    typeof su7Params.preset === 'string' && su7Params.preset.length > 0
      ? su7Params.preset
      : 'identity';
  const su7PresetMeta = useMemo(
    () => (isSu7PresetId(su7Params.preset) ? SU7_PRESET_DEFINITIONS[su7Params.preset] : null),
    [su7Params.preset],
  );
  const su7ProjectorId =
    typeof su7Params.projector?.id === 'string' && su7Params.projector.id.length > 0
      ? su7Params.projector.id.toLowerCase()
      : undefined;
  const su7ProjectorOptions = useMemo(() => {
    if (
      su7ProjectorId &&
      !SU7_PROJECTOR_OPTIONS.some((option) => option.value === su7ProjectorId)
    ) {
      return [...SU7_PROJECTOR_OPTIONS, { value: su7ProjectorId, label: su7ProjectorId }];
    }
    return SU7_PROJECTOR_OPTIONS;
  }, [su7ProjectorId]);
  const su7ProjectorSelectValue = su7ProjectorId ?? SU7_PROJECTOR_OPTIONS[0].value;

  const [kernel, setKernel] = useState<KernelSpec>(() => getDefaultKernelSpec());
  const updateKernel = useCallback(
    (patch: Partial<KernelSpec>) => setKernel((prev) => clampKernelSpec({ ...prev, ...patch })),
    [],
  );
  useEffect(() => {
    return kernelHub.subscribe((event) => {
      kernelEventRef.current = event;
      kurKernelVersionRef.current = event.version;
      if (!kurSyncRef.current && workerRef.current && workerReadyRef.current) {
        workerRef.current.postMessage({
          kind: 'kernelSpec',
          spec: event.spec,
          version: event.version,
        });
      }
    });
  }, [kernelHub]);
  useEffect(() => {
    kernelHub.replace(kernel, { source: 'ui', force: true });
  }, [kernel, kernelHub]);
  const [dmt, setDmt] = useState(0.0);
  const [arousal, setArousal] = useState(0.3);

  const [thetaMode, setThetaMode] = useState<'gradient' | 'global'>('gradient');
  const [thetaGlobal, setThetaGlobal] = useState(0);

  const [displayMode, setDisplayMode] = useState<DisplayMode>('color');
  const [curvatureStrength, setCurvatureStrength] = useState(0);
  const [curvatureMode, setCurvatureMode] = useState<CurvatureMode>('poincare');
  const hyperbolicAtlas = useMemo<HyperbolicAtlas | null>(() => {
    const strength = Math.abs(curvatureStrength);
    if (strength <= 1e-4) {
      return null;
    }
    return createHyperbolicAtlas({
      width,
      height,
      curvatureStrength: strength,
      mode: curvatureMode,
    });
  }, [width, height, curvatureStrength, curvatureMode]);
  const hyperbolicAtlasGpu = useMemo(
    () => (hyperbolicAtlas ? packageHyperbolicAtlasForGpu(hyperbolicAtlas) : null),
    [hyperbolicAtlas],
  );

  useEffect(() => {
    if (hyperbolicAtlas) {
      return;
    }
    if (showHyperbolicGrid) {
      setShowHyperbolicGrid(false);
    }
    if (showHyperbolicGuide) {
      setShowHyperbolicGuide(false);
    }
  }, [hyperbolicAtlas, showHyperbolicGrid, showHyperbolicGuide]);

  useEffect(() => {
    const canvas = hyperbolicGridCanvasRef.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    if ((!showHyperbolicGrid && !showHyperbolicGuide) || !hyperbolicAtlas) {
      return;
    }
    const {
      metadata: { curvatureScale, diskLimit, centerX, centerY },
    } = hyperbolicAtlas;
    const safeAtanh = (value: number) => 0.5 * Math.log((1 + value) / (1 - value));
    const maxHyper = 2 * curvatureScale * safeAtanh(Math.min(0.999999, diskLimit));
    if (!Number.isFinite(maxHyper) || maxHyper <= 0) {
      return;
    }
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (showHyperbolicGrid) {
      const radialLimit = maxHyper * 0.97;
      const radialCount = Math.max(3, Math.min(8, Math.round(radialLimit / 0.6) || 4));
      const angularCount = 12;
      const circleSamples = Math.max(64, Math.min(360, Math.round(width * 0.9)));

      const radialIncrement = radialLimit / radialCount;
      ctx.globalAlpha = 0.9;
      ctx.setLineDash([]);
      ctx.lineWidth = Math.max(0.75, Math.min(1.5, Math.sqrt(width * height) / 900));
      ctx.strokeStyle = 'rgba(94, 234, 212, 0.7)';
      for (let i = 1; i <= radialCount; i++) {
        const radius = i * radialIncrement;
        ctx.beginPath();
        let started = false;
        for (let s = 0; s <= circleSamples; s++) {
          const theta = (s / circleSamples) * Math.PI * 2;
          const [px, py] = mapHyperbolicPolarToPixel(hyperbolicAtlas, radius, theta);
          if (!Number.isFinite(px) || !Number.isFinite(py)) {
            continue;
          }
          if (!started) {
            ctx.moveTo(px, py);
            started = true;
          } else {
            ctx.lineTo(px, py);
          }
        }
        if (started) {
          ctx.closePath();
          ctx.stroke();
        }
      }

      ctx.setLineDash([6, 10]);
      ctx.strokeStyle = 'rgba(165, 180, 252, 0.65)';
      for (let a = 0; a < angularCount; a++) {
        const theta = (a / angularCount) * Math.PI * 2;
        ctx.beginPath();
        let started = false;
        for (let s = 0; s <= circleSamples; s++) {
          const radius = (s / circleSamples) * radialLimit;
          const [px, py] = mapHyperbolicPolarToPixel(hyperbolicAtlas, radius, theta);
          if (!Number.isFinite(px) || !Number.isFinite(py)) {
            continue;
          }
          if (!started) {
            ctx.moveTo(px, py);
            started = true;
          } else {
            ctx.lineTo(px, py);
          }
        }
        if (started) {
          ctx.stroke();
        }
      }

      ctx.setLineDash([]);
      ctx.strokeStyle = 'rgba(248, 250, 252, 0.6)';
      ctx.lineWidth = Math.max(0.9, Math.min(1.8, Math.sqrt(width * height) / 850));
      ctx.beginPath();
      ctx.moveTo(centerX - 12, centerY);
      ctx.lineTo(centerX + 12, centerY);
      ctx.moveTo(centerX, centerY - 12);
      ctx.lineTo(centerX, centerY + 12);
      ctx.stroke();
    }

    if (showHyperbolicGuide) {
      ctx.globalAlpha = 1;
      ctx.setLineDash([]);
      const spacing = clamp(
        hyperbolicGuideSpacing,
        HYPERBOLIC_GUIDE_SPACING_MIN,
        HYPERBOLIC_GUIDE_SPACING_MAX,
      );
      const tickCount = Math.min(
        8,
        Math.max(1, Math.floor((maxHyper * 0.95) / Math.max(spacing, 1e-4))),
      );
      const axisTheta = -Math.PI / 4;
      const normalTheta = axisTheta + Math.PI / 2;
      const axisThickness = Math.max(1.05, Math.min(2.2, Math.sqrt(width * height) / 650));
      const labelPadding = Math.max(6, Math.min(12, Math.sqrt(width * height) / 110));
      const tickLength = Math.max(10, Math.min(18, Math.sqrt(width * height) / 90));
      const axisExtentRadius = spacing * (tickCount + 0.65);

      ctx.strokeStyle = 'rgba(226, 232, 240, 0.95)';
      ctx.lineWidth = axisThickness;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      const [axisEndX, axisEndY] = mapHyperbolicPolarToPixel(
        hyperbolicAtlas,
        Math.min(axisExtentRadius, maxHyper * 0.98),
        axisTheta,
      );
      if (Number.isFinite(axisEndX) && Number.isFinite(axisEndY)) {
        ctx.lineTo(axisEndX, axisEndY);
        ctx.stroke();
      }

      ctx.fillStyle = 'rgba(226, 232, 240, 0.88)';
      const fontSize = Math.max(11, Math.min(15, Math.sqrt(width * height) / 105));
      ctx.font = `${fontSize}px/1.3 "Inter", "Segoe UI", sans-serif`;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';

      for (let t = 1; t <= tickCount; t++) {
        const radius = spacing * t;
        const sampleRadius = Math.min(radius, maxHyper * 0.985);
        const [px, py] = mapHyperbolicPolarToPixel(hyperbolicAtlas, sampleRadius, axisTheta);
        if (!Number.isFinite(px) || !Number.isFinite(py)) {
          continue;
        }
        const tickOffsetX = Math.cos(normalTheta) * tickLength;
        const tickOffsetY = Math.sin(normalTheta) * tickLength;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + tickOffsetX, py + tickOffsetY);
        ctx.stroke();

        const hyperLabel = radius.toFixed(2);
        const euclideanDistance = Math.hypot(px - centerX, py - centerY);
        const linearLabel = euclideanDistance.toFixed(0);
        const label = `${hyperLabel}σ / ${linearLabel}px`;
        const textX = px + tickOffsetX + Math.cos(normalTheta) * labelPadding;
        const textY = py + tickOffsetY + Math.sin(normalTheta) * labelPadding;
        ctx.fillText(label, textX, textY);
      }

      const originRadius = Math.max(3, Math.min(6, Math.sqrt(width * height) / 180));
      ctx.beginPath();
      ctx.arc(centerX, centerY, originRadius, 0, Math.PI * 2);
      ctx.fill();

      const captionRadius = Math.min(spacing * 0.45, maxHyper * 0.2);
      const [captionX, captionY] = mapHyperbolicPolarToPixel(
        hyperbolicAtlas,
        captionRadius,
        axisTheta,
      );
      if (Number.isFinite(captionX) && Number.isFinite(captionY)) {
        const captionText = `Hyperbolic ruler · Δσ=${spacing.toFixed(2)}`;
        ctx.fillText(
          captionText,
          captionX + Math.cos(normalTheta) * (tickLength * 0.4),
          captionY + Math.sin(normalTheta) * (tickLength * 0.4),
        );
      }
    }

    ctx.restore();
  }, [
    showHyperbolicGrid,
    showHyperbolicGuide,
    hyperbolicAtlas,
    width,
    height,
    hyperbolicGuideSpacing,
  ]);

  useEffect(() => {
    const state = gpuStateRef.current;
    if (state) {
      state.renderer.setHyperbolicAtlas(hyperbolicAtlasGpu);
    }
  }, [hyperbolicAtlasGpu]);

  const [tracerConfig, setTracerConfig] = useState<TracerConfig>(() =>
    cloneTracerConfig(DEFAULT_TRACER_CONFIG),
  );
  const tracerRuntime = useMemo(() => mapTracerConfigToRuntime(tracerConfig), [tracerConfig]);
  const tracerBufferRef = useRef<Float32Array | null>(null);
  const tracerLastTimeRef = useRef<number | null>(null);
  const gpuTracerRef = useRef<{ lastTime: number | null; needsReset: boolean }>({
    lastTime: null,
    needsReset: true,
  });
  const resetTracerState = useCallback(() => {
    tracerBufferRef.current = null;
    tracerLastTimeRef.current = null;
    gpuTracerRef.current.lastTime = null;
    gpuTracerRef.current.needsReset = true;
  }, []);
  const setTracerValue = useCallback(
    <K extends keyof TracerConfig>(key: K) =>
      (value: TracerConfig[K]) => {
        setTracerConfig((prev) =>
          sanitizeTracerConfig({
            ...prev,
            [key]: value,
          }),
        );
      },
    [],
  );

  const [phasePin, setPhasePin] = useState(true);
  const [alive, setAlive] = useState(false);
  const [polBins, setPolBins] = useState(16);
  const [normPin, setNormPin] = useState(true);
  const [kurRegime, setKurRegime] = useState<KurRegime>('locked');

  const [surfEnabled, setSurfEnabled] = useState(false);
  const [wallGroup, setWallGroup] = useState<WallpaperGroup>('p4');
  const [nOrient, setNOrient] = useState(4);
  const [surfaceBlend, setSurfaceBlend] = useState(0.35);
  const [warpAmp, setWarpAmp] = useState(1.0);
  const [surfaceRegion, setSurfaceRegion] = useState<SurfaceRegion>('surfaces');

  const [coupling, setCoupling] = useState<CouplingConfig>({
    rimToSurfaceBlend: 0.45,
    rimToSurfaceAlign: 0.55,
    surfaceToRimOffset: 0.4,
    surfaceToRimSigma: 0.6,
    surfaceToRimHue: 0.5,
    kurToTransparency: 0.35,
    kurToOrientation: 0.35,
    kurToChirality: 0.6,
    volumePhaseToHue: 0.35,
    volumeDepthToWarp: 0.3,
  });
  const setCouplingValue = useCallback(
    (key: keyof CouplingConfig) => (value: number) =>
      setCoupling((prev) => ({ ...prev, [key]: value })),
    [],
  );
  const [couplingToggles, setCouplingToggles] =
    useState<CouplingToggleState>(DEFAULT_COUPLING_TOGGLES);
  const setCouplingToggle = useCallback(
    (key: keyof CouplingToggleState) => (value: boolean) =>
      setCouplingToggles((prev) => ({ ...prev, [key]: value })),
    [],
  );
  const computeCouplingPair = useCallback(
    (override?: CouplingToggleState) => {
      const toggles = override ?? couplingToggles;
      return {
        base: cloneCouplingConfig(coupling),
        effective: applyCouplingToggles(coupling, toggles),
      };
    },
    [coupling, couplingToggles],
  );

  const [composer, setComposer] = useState<ComposerConfig>(() => createDefaultComposerConfig());
  const setComposerFieldValue = useCallback(
    (field: ComposerFieldId, key: 'exposure' | 'gamma' | 'weight') => (value: number) =>
      setComposer((prev) => ({
        ...prev,
        fields: {
          ...prev.fields,
          [field]: {
            ...prev.fields[field],
            [key]: value,
          },
        },
      })),
    [],
  );
  const handleComposerRouting = useCallback((routing: DmtRoutingMode) => {
    setComposer((prev) => ({ ...prev, dmtRouting: routing }));
  }, []);
  const handleComposerSolver = useCallback((solver: SolverRegime) => {
    setComposer((prev) => ({ ...prev, solverRegime: solver }));
  }, []);

  const composerUniforms = useMemo(() => {
    const exposure = new Float32Array(COMPOSER_FIELD_LIST.length);
    const gamma = new Float32Array(COMPOSER_FIELD_LIST.length);
    const weight = new Float32Array(COMPOSER_FIELD_LIST.length);
    COMPOSER_FIELD_LIST.forEach((field, idx) => {
      const cfg = composer.fields[field];
      exposure[idx] = cfg.exposure;
      gamma[idx] = cfg.gamma;
      weight[idx] = cfg.weight;
    });
    const blendGain = computeComposerBlendGain(composer);
    return { exposure, gamma, weight, blendGain };
  }, [composer]);

  const [volumeEnabled, setVolumeEnabled] = useState(false);
  const [kurEnabled, setKurEnabled] = useState(false);
  const [kurSync, setKurSync] = useState(false);
  const [K0, setK0] = useState(0.6);
  const [alphaKur, setAlphaKur] = useState(0.2);
  const [gammaKur, setGammaKur] = useState(0.15);
  const [omega0, setOmega0] = useState(0.0);
  const [epsKur, setEpsKur] = useState(0.002);
  const [fluxX, setFluxX] = useState(0);
  const [fluxY, setFluxY] = useState(0);
  const [qInit, setQInit] = useState(1);
  const [smallWorldEnabled, setSmallWorldEnabled] = useState(false);
  const [smallWorldWeight, setSmallWorldWeight] = useState(0.75);
  const [pSw, setPSw] = useState(0.05);
  const [smallWorldSeed, setSmallWorldSeed] = useState(1337);
  const [smallWorldDegree, setSmallWorldDegree] = useState(12);
  const [presetIndex, setPresetIndex] = useState(0);
  const [telemetryEnabled, setTelemetryEnabled] = useState(false);
  const [telemetryOverlayEnabled, setTelemetryOverlayEnabled] = useState(false);
  const [telemetrySnapshot, setTelemetrySnapshot] = useState<TelemetrySnapshot | null>(null);
  const [frameLoggingEnabled, setFrameLoggingEnabled] = useState(true);
  const [lastParityResult, setLastParityResult] = useState<ParitySummary | null>(null);
  const [lastPerfResult, setLastPerfResult] = useState<PerformanceSnapshot | null>(null);
  const [recordingStatus, setRecordingStatus] = useState<'idle' | 'recording' | 'finalizing'>(
    'idle',
  );
  const [recordingError, setRecordingError] = useState<string | null>(null);
  const [recordingDownload, setRecordingDownload] = useState<{
    url: string;
    size: number;
    mimeType: string;
    filename: string;
  } | null>(null);
  const [recordingPreset, setRecordingPreset] =
    useState<RecordingPresetId>(RECORDING_DEFAULT_PRESET);
  const [captureSupport, setCaptureSupport] = useState<{
    checked: boolean;
    supported: CaptureFormatConfig[];
  }>({ checked: false, supported: [] });
  const [recordingFormatId, setRecordingFormatId] = useState<CaptureFormatId | null>(null);
  const [fieldStatuses, setFieldStatuses] = useState<FieldStatusMap>(() => createInitialStatuses());
  const [rimDebugSnapshot, setRimDebugSnapshot] = useState<RimDebugSnapshot | null>(null);
  const [surfaceDebugSnapshot, setSurfaceDebugSnapshot] = useState<SurfaceDebugSnapshot | null>(
    null,
  );
  const [phaseDebugSnapshot, setPhaseDebugSnapshot] = useState<PhaseDebugSnapshot | null>(null);
  const [phaseHeatmapEnabled, setPhaseHeatmapEnabled] = useState(false);
  const [phaseHeatmapSnapshot, setPhaseHeatmapSnapshot] = useState<PhaseHeatmapSnapshot | null>(
    null,
  );
  const [selectedSyntheticCase, setSelectedSyntheticCase] = useState<SyntheticCaseId>('circles');
  const [syntheticBaselines, setSyntheticBaselines] = useState<
    Partial<Record<SyntheticCaseId, { metrics: RainbowFrameMetrics; timestamp: number }>>
  >({});
  const markFieldFresh = useCallback(
    (kind: FieldKind, resolution: FieldResolution, source: string) => {
      const now = performance.now();
      setFieldStatuses((prev) => {
        const next = setFieldAvailable(prev, kind, resolution, source, now);
        if (!prev[kind].available) {
          const contract = FIELD_CONTRACTS[kind];
          console.info(
            `[fields] ${contract.label} available ${resolution.width}x${resolution.height} via ${source}`,
          );
        }
        return next;
      });
    },
    [],
  );

  const markFieldGone = useCallback((kind: FieldKind, source: string) => {
    const now = performance.now();
    setFieldStatuses((prev) => {
      if (!prev[kind].available) return prev;
      const contract = FIELD_CONTRACTS[kind];
      console.warn(`[fields] ${contract.label} unavailable via ${source}`);
      return setFieldUnavailable(prev, kind, source, now);
    });
  }, []);

  const stopCaptureStream = useCallback(() => {
    if (captureStreamRef.current) {
      captureStreamRef.current.getTracks().forEach((track) => track.stop());
      captureStreamRef.current = null;
    }
  }, []);

  const recordingBitrate = useMemo(() => {
    const pixelsPerFrame = width * height;
    if (pixelsPerFrame === 0) {
      return RECORDING_MIN_BITRATE;
    }
    const preset = RECORDING_PRESETS[recordingPreset];
    const target = pixelsPerFrame * RECORDING_FPS * preset.bitsPerPixel;
    const clamped = Math.min(target, preset.maxBitrate);
    return Math.max(Math.round(clamped), RECORDING_MIN_BITRATE);
  }, [width, height, recordingPreset]);

  const handleRecordingPresetChange = useCallback((event: ChangeEvent<HTMLSelectElement>) => {
    const value = event.target.value as RecordingPresetId;
    setRecordingPreset(value);
  }, []);

  const handleRecordingFormatChange = useCallback((event: ChangeEvent<HTMLSelectElement>) => {
    const value = event.target.value as CaptureFormatId;
    setRecordingFormatId(value);
  }, []);

  const recordingPresetRef = useRef(recordingPreset);
  useEffect(() => {
    recordingPresetRef.current = recordingPreset;
  }, [recordingPreset]);

  const recordingFormatRef = useRef<CaptureFormatId | null>(recordingFormatId);
  useEffect(() => {
    recordingFormatRef.current = recordingFormatId;
  }, [recordingFormatId]);

  useEffect(() => {
    if (!tracerRuntime.enabled) {
      resetTracerState();
    }
  }, [tracerRuntime.enabled, resetTracerState]);

  useEffect(() => {
    resetTracerState();
  }, [width, height, resetTracerState]);

  const normTargetRef = useRef(0.6);
  const lastObsRef = useRef(0.6);

  const kurSyncRef = useRef(false);
  const kurStateRef = useRef<KuramotoState | null>(null);
  const kurTelemetryRef = useRef<KuramotoTelemetrySnapshot | null>(null);
  const kurIrradianceRef = useRef<IrradianceFrameBuffer | null>(null);
  const kurLogRef = useRef<{ kernelVersion: number; frameId: number }>({
    kernelVersion: -1,
    frameId: -1,
  });
  const cpuDerivedRef = useRef<PhaseField | null>(null);
  const cpuDerivedBufferRef = useRef<ArrayBuffer | null>(null);
  const gradXRef = useRef<Float32Array | null>(null);
  const gradYRef = useRef<Float32Array | null>(null);
  const vortRef = useRef<Float32Array | null>(null);
  const cohRef = useRef<Float32Array | null>(null);
  const ampRef = useRef<Float32Array | null>(null);
  const volumeStubRef = useRef<VolumeStubState | null>(null);
  const volumeFieldRef = useRef<VolumeField | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const workerReadyRef = useRef(false);
  const workerInflightRef = useRef(0);
  const workerNextFrameIdRef = useRef(0);
  const workerPendingFramesRef = useRef<KurFrameView[]>([]);
  const workerActiveFrameRef = useRef<KurFrameView | null>(null);
  const workerLastFrameIdRef = useRef(-1);
  const skipNextPresetApplyRef = useRef(false);

  const frameBufferRef = useRef<{
    image: ImageData;
    data: Uint8ClampedArray;
    width: number;
    height: number;
  } | null>(null);

  const metricsScratchRef = useRef<Uint8ClampedArray | null>(null);
  const rimDebugRef = useRef<{
    energy: Float32Array;
    hue: Float32Array;
    energyHist: Uint32Array;
    hueHist: Uint32Array;
  } | null>(null);
  const surfaceDebugRef = useRef<{
    phases: Float32Array[];
    magnitudes: Float32Array[];
    magnitudeHist: Float32Array;
    orientationCount: number;
  } | null>(null);
  const phaseDebugRef = useRef<Float32Array | null>(null);
  const phaseHeatmapRef = useRef<{ width: number; height: number; data: Float32Array } | null>(
    null,
  );

  const orientationCacheRef = useRef<{
    count: number;
    cos: Float32Array;
    sin: Float32Array;
  }>({
    count: 0,
    cos: new Float32Array(0),
    sin: new Float32Array(0),
  });

  const frameProfilerRef = useRef<FrameProfilerState>({
    enabled: false,
    samples: [],
    maxSamples: 120,
    label: 'frame-profiler',
  });

  useEffect(() => {
    const interval = window.setInterval(() => {
      const now = performance.now();
      setFieldStatuses((prev) => {
        const { next, changes } = refreshFieldStaleness(prev, now);
        if (changes.length === 0) return prev;
        for (const change of changes) {
          const contract = FIELD_CONTRACTS[change.kind];
          if (change.becameStale) {
            console.warn(`[fields] ${contract.label} stale ${change.stalenessMs.toFixed(0)}ms`);
          } else if (change.recovered) {
            console.info(`[fields] ${contract.label} recovered`);
          }
        }
        return next;
      });
    }, 300);
    return () => window.clearInterval(interval);
  }, []);

  const metricsRef = useRef<FrameMetricsEntry[]>([]);
  const telemetryOverlayLogRef = useRef(0);

  const telemetryRef = useRef<{
    enabled: boolean;
    thresholds: Record<TelemetryPhase, number>;
    history: TelemetryRecord[];
    lastLogTs: number;
  }>({
    enabled: false,
    thresholds: {
      frame: 28,
      renderGpu: 10,
      renderCpu: 20,
      kuramoto: 8,
    },
    history: [],
    lastLogTs: 0,
  });

  const frameLogRef = useRef<{
    windowStart: number;
    frames: number;
  }>({
    windowStart: performance.now(),
    frames: 0,
  });

  const randn = useRandN();

  const logKurTelemetry = useCallback((telemetry: KuramotoTelemetrySnapshot | null) => {
    if (!telemetry) return;
    const last = kurLogRef.current;
    if (telemetry.frameId === last.frameId) {
      return;
    }
    const baseMessage = `[kur-telemetry] frame ${telemetry.frameId} kernel v${telemetry.kernelVersion} R=${telemetry.orderParameter.magnitude.toFixed(4)} phase=${telemetry.orderParameter.phase.toFixed(3)} meanE=${telemetry.interference.mean.toFixed(4)}`;
    if (telemetry.kernelVersion !== last.kernelVersion) {
      console.info(
        `${baseMessage} gain=${telemetry.kernel.gain.toFixed(3)} k0=${telemetry.kernel.k0.toFixed(
          3,
        )} anis=${telemetry.kernel.anisotropy.toFixed(3)} chir=${telemetry.kernel.chirality.toFixed(3)}`,
      );
    } else {
      console.debug(baseMessage);
    }
    kurLogRef.current = { kernelVersion: telemetry.kernelVersion, frameId: telemetry.frameId };
  }, []);

  const recordTelemetry = useCallback((phase: TelemetryPhase, ms: number) => {
    const tele = telemetryRef.current;
    if (!tele.enabled) return;
    const threshold = tele.thresholds[phase] ?? tele.thresholds.frame;
    if (ms > threshold) {
      const now = performance.now();
      if (now - tele.lastLogTs > 1000) {
        console.warn(`[telemetry] ${phase} ${ms.toFixed(2)}ms`);
        tele.lastLogTs = now;
      }
    }
    tele.history.push({ phase, ms, ts: performance.now() });
    if (tele.history.length > 360) {
      tele.history.shift();
    }
  }, []);

  useEffect(() => {
    telemetryRef.current.enabled = telemetryEnabled;
    if (!telemetryEnabled) {
      telemetryRef.current.history = [];
      telemetryRef.current.lastLogTs = 0;
      metricsRef.current = [];
      setTelemetryOverlayEnabled(false);
    }
  }, [telemetryEnabled]);

  useEffect(() => {
    if (!frameLoggingEnabled) {
      frameLogRef.current.frames = 0;
      frameLogRef.current.windowStart = performance.now();
    } else {
      frameLogRef.current.windowStart = performance.now();
      frameLogRef.current.frames = 0;
    }
  }, [frameLoggingEnabled]);

  useEffect(() => {
    if (!telemetryEnabled) {
      setTelemetrySnapshot(null);
      return;
    }
    const interval = window.setInterval(() => {
      const history = metricsRef.current.slice(-120);
      if (history.length === 0) return;
      const accum = COMPOSER_FIELD_LIST.reduce<
        Record<ComposerFieldId, { energy: number; blend: number; share: number; weight: number }>
      >(
        (acc, field) => {
          acc[field] = { energy: 0, blend: 0, share: 0, weight: 0 };
          return acc;
        },
        {} as Record<
          ComposerFieldId,
          { energy: number; blend: number; share: number; weight: number }
        >,
      );
      const su7Totals = {
        unitaryError: 0,
        determinantDrift: 0,
        normDeltaMax: 0,
        normDeltaMean: 0,
        projectorEnergy: 0,
      };
      const su7UnitaryValues: number[] = [];
      const su7NormMeanValues: number[] = [];
      const su7NormMaxValues: number[] = [];
      const su7ProjectorEnergyValues: number[] = [];
      history.forEach((entry) => {
        const telemetry = entry.metrics.composer;
        COMPOSER_FIELD_LIST.forEach((field) => {
          accum[field].energy += telemetry.fields[field].energy;
          accum[field].blend += telemetry.fields[field].blend;
          accum[field].share += telemetry.fields[field].share;
          accum[field].weight = telemetry.fields[field].weight;
        });
        const su7 = entry.metrics.su7;
        su7Totals.unitaryError += su7.unitaryError;
        su7Totals.determinantDrift += su7.determinantDrift;
        su7Totals.normDeltaMax += su7.normDeltaMax;
        su7Totals.normDeltaMean += su7.normDeltaMean;
        su7Totals.projectorEnergy += su7.projectorEnergy;
        if (Number.isFinite(su7.unitaryError)) {
          su7UnitaryValues.push(su7.unitaryError);
        }
        if (Number.isFinite(su7.normDeltaMean)) {
          su7NormMeanValues.push(su7.normDeltaMean);
        }
        if (Number.isFinite(su7.normDeltaMax)) {
          su7NormMaxValues.push(su7.normDeltaMax);
        }
        if (Number.isFinite(su7.projectorEnergy)) {
          su7ProjectorEnergyValues.push(su7.projectorEnergy);
        }
      });
      const lastComposer = history[history.length - 1].metrics.composer;
      const count = history.length;
      const unitaryLatest = su7UnitaryValues[su7UnitaryValues.length - 1] ?? 0;
      const unitaryMean =
        su7UnitaryValues.length > 0
          ? su7UnitaryValues.reduce((acc, value) => acc + value, 0) / su7UnitaryValues.length
          : 0;
      const unitaryMax =
        su7UnitaryValues.length > 0
          ? su7UnitaryValues.reduce(
              (acc, value) => (value > acc ? value : acc),
              Number.NEGATIVE_INFINITY,
            )
          : 0;
      const snapshot: TelemetrySnapshot = {
        fields: {} as Record<ComposerFieldId, TelemetryFieldSnapshot>,
        coupling: {
          scale: lastComposer.coupling.scale,
          base: cloneCouplingConfig(lastComposer.coupling.base),
          effective: cloneCouplingConfig(lastComposer.coupling.effective),
        },
        su7: {
          unitaryError: su7Totals.unitaryError / count,
          determinantDrift: su7Totals.determinantDrift / count,
          normDeltaMax: su7Totals.normDeltaMax / count,
          normDeltaMean: su7Totals.normDeltaMean / count,
          projectorEnergy: su7Totals.projectorEnergy / count,
        },
        su7Histograms: {
          normDeltaMean: computeTelemetryHistogram(su7NormMeanValues, 16),
          normDeltaMax: computeTelemetryHistogram(su7NormMaxValues, 16),
          projectorEnergy: computeTelemetryHistogram(su7ProjectorEnergyValues, 16),
        },
        su7Unitary: {
          latest: unitaryLatest,
          mean: unitaryMean,
          max: unitaryMax === Number.NEGATIVE_INFINITY ? 0 : unitaryMax,
        },
        frameSamples: history.length,
        updatedAt: Date.now(),
      };
      COMPOSER_FIELD_LIST.forEach((field) => {
        snapshot.fields[field] = {
          energy: accum[field].energy / count,
          blend: accum[field].blend / count,
          share: accum[field].share / count,
          weight: accum[field].weight,
        };
      });
      setTelemetrySnapshot(snapshot);
    }, 500);
    return () => window.clearInterval(interval);
  }, [telemetryEnabled]);

  useEffect(() => {
    if (!telemetryOverlayEnabled || !telemetryEnabled) return;
    if (!telemetrySnapshot || telemetrySnapshot.frameSamples === 0) return;
    const { su7Histograms, su7Unitary } = telemetrySnapshot;
    if (
      !hasTelemetryHistogramSamples(su7Histograms.normDeltaMean) &&
      !hasTelemetryHistogramSamples(su7Histograms.normDeltaMax) &&
      !hasTelemetryHistogramSamples(su7Histograms.projectorEnergy)
    ) {
      return;
    }
    const now = performance.now();
    if (now - telemetryOverlayLogRef.current < 1000) {
      return;
    }
    telemetryOverlayLogRef.current = now;
    const formatRange = (hist: TelemetryHistogramSnapshot) => formatTelemetryRange(hist, 2);
    console.info(
      `[telemetry-overlay] unitary=${su7Unitary.latest.toExponential(2)} avg=${su7Unitary.mean.toExponential(
        2,
      )} max=${su7Unitary.max.toExponential(2)} normΔμ=${formatRange(
        su7Histograms.normDeltaMean,
      )} normΔmax=${formatRange(su7Histograms.normDeltaMax)} energy=${formatRange(
        su7Histograms.projectorEnergy,
      )}`,
    );
  }, [telemetryOverlayEnabled, telemetryEnabled, telemetrySnapshot]);

  useEffect(() => {
    window.__getFrameMetrics = () => [...metricsRef.current];
    return () => {
      window.__getFrameMetrics = undefined;
    };
  }, []);

  const orientations = useMemo(() => {
    const N = clamp(Math.round(nOrient), 2, 8);
    return Array.from({ length: N }, (_, j) => (j * 2 * Math.PI) / N);
  }, [nOrient]);

  const lambdas = useMemo(() => ({ L: 560, M: 530, S: 420 }), []);
  const lambdaRef = 520;
  const fieldStatusEntries = useMemo(
    () =>
      FIELD_KINDS.map((kind) => {
        const status = fieldStatuses[kind];
        const contract = FIELD_CONTRACTS[kind];
        const staleness = status.stalenessMs;
        let state: 'ok' | 'warn' | 'stale' | 'missing';
        if (!status.available) {
          state = 'missing';
        } else if (status.stale) {
          state = 'stale';
        } else if (
          contract.lifetime.expectedMs !== Number.POSITIVE_INFINITY &&
          staleness > contract.lifetime.expectedMs * 1.5
        ) {
          state = 'warn';
        } else {
          state = 'ok';
        }
        return {
          kind,
          label: contract.label,
          state,
          stalenessMs: staleness,
          resolution: status.resolution,
        };
      }),
    [fieldStatuses],
  );

  const ensureKurCpuState = useCallback(() => {
    if (
      !kurStateRef.current ||
      kurStateRef.current.width !== width ||
      kurStateRef.current.height !== height
    ) {
      kurStateRef.current = createKuramotoState(width, height);
      if (kurStateRef.current) {
        kurTelemetryRef.current = kurStateRef.current.telemetry;
        kurIrradianceRef.current = kurStateRef.current.irradiance;
      }
    }
    if (kurStateRef.current) {
      kurTelemetryRef.current = kurStateRef.current.telemetry;
      kurIrradianceRef.current = kurStateRef.current.irradiance;
    }
    const expected = width * height;
    if (
      !cpuDerivedRef.current ||
      cpuDerivedRef.current.gradX.length !== expected ||
      cpuDerivedRef.current.resolution.width !== width ||
      cpuDerivedRef.current.resolution.height !== height
    ) {
      const buffer = new ArrayBuffer(derivedBufferSize(width, height));
      cpuDerivedBufferRef.current = buffer;
      const derived = createDerivedViews(buffer, width, height);
      assertPhaseField(derived, 'cpu:init');
      cpuDerivedRef.current = derived;
      gradXRef.current = derived.gradX;
      gradYRef.current = derived.gradY;
      vortRef.current = derived.vort;
      cohRef.current = derived.coh;
      ampRef.current = derived.amp;
      return true;
    }
    assertPhaseField(cpuDerivedRef.current, 'cpu:reuse');
    gradXRef.current = cpuDerivedRef.current.gradX;
    gradYRef.current = cpuDerivedRef.current.gradY;
    vortRef.current = cpuDerivedRef.current.vort;
    cohRef.current = cpuDerivedRef.current.coh;
    ampRef.current = cpuDerivedRef.current.amp;
    return false;
  }, [width, height]);

  const ensureVolumeState = useCallback(() => {
    if (width <= 0 || height <= 0) {
      volumeStubRef.current = null;
      volumeFieldRef.current = null;
      return;
    }
    const stub = volumeStubRef.current;
    if (!stub || stub.width !== width || stub.height !== height) {
      volumeStubRef.current = createVolumeStubState(width, height);
    }
  }, [width, height]);

  const ensureTracerBuffer = useCallback((): Float32Array | null => {
    if (width <= 0 || height <= 0) {
      tracerBufferRef.current = null;
      tracerLastTimeRef.current = null;
      return null;
    }
    const expected = width * height * 3;
    let buffer = tracerBufferRef.current;
    if (!buffer || buffer.length !== expected) {
      buffer = new Float32Array(expected);
      tracerBufferRef.current = buffer;
    }
    return buffer;
  }, [width, height]);

  const ensureFrameBuffer = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      let buffer = frameBufferRef.current;
      if (!buffer || buffer.width !== width || buffer.height !== height) {
        const image = ctx.createImageData(width, height);
        buffer = {
          image,
          data: image.data,
          width,
          height,
        };
        frameBufferRef.current = buffer;
      }
      return buffer;
    },
    [width, height],
  );

  const ensureRimDebugBuffers = useCallback(() => {
    const total = width * height;
    let buffers = rimDebugRef.current;
    if (!buffers || buffers.energy.length !== total) {
      buffers = {
        energy: new Float32Array(total),
        hue: new Float32Array(total),
        energyHist: new Uint32Array(RIM_HIST_BINS),
        hueHist: new Uint32Array(RIM_HIST_BINS),
      };
      rimDebugRef.current = buffers;
    } else {
      buffers.energyHist.fill(0);
      buffers.hueHist.fill(0);
    }
    return buffers;
  }, [width, height]);

  const ensureSurfaceDebugBuffers = useCallback(
    (orientationCount: number) => {
      if (orientationCount <= 0) return null;
      const total = width * height;
      let buffers = surfaceDebugRef.current;
      const needsResize =
        !buffers ||
        buffers.orientationCount !== orientationCount ||
        buffers.magnitudes.length !== orientationCount ||
        buffers.magnitudes[0]?.length !== total;
      if (needsResize) {
        const phases = Array.from({ length: orientationCount }, () => new Float32Array(total));
        const magnitudes = Array.from({ length: orientationCount }, () => new Float32Array(total));
        const flowVectors = {
          x: new Float32Array(total),
          y: new Float32Array(total),
          hyperbolicScale: new Float32Array(total),
        };
        buffers = {
          phases,
          magnitudes,
          magnitudeHist: new Float32Array(orientationCount * SURFACE_HIST_BINS),
          orientationCount,
          flowVectors,
        };
        surfaceDebugRef.current = buffers;
      } else if (buffers) {
        buffers.magnitudeHist.fill(0);
      }
      return buffers ?? null;
    },
    [width, height],
  );

  const updateDebugSnapshots = useCallback(
    (
      commit: boolean,
      rimBuffers: ReturnType<typeof ensureRimDebugBuffers> | null,
      surfaceBuffers: ReturnType<typeof ensureSurfaceDebugBuffers> | null,
      debug: ReturnType<typeof renderRainbowFrame>['debug'],
    ) => {
      if (!commit) return;
      if (showRimDebug && rimBuffers && debug?.rim) {
        setRimDebugSnapshot({
          energyRange: [debug.rim.energyMin, debug.rim.energyMax],
          hueRange: [debug.rim.hueMin, debug.rim.hueMax],
          energyHist: Array.from(rimBuffers.energyHist),
          hueHist: Array.from(rimBuffers.hueHist),
        });
      } else if (!showRimDebug) {
        setRimDebugSnapshot(null);
      }
      if (showSurfaceDebug && surfaceBuffers && debug?.surface) {
        const hist: number[][] = [];
        for (let k = 0; k < surfaceBuffers.orientationCount; k++) {
          const start = k * SURFACE_HIST_BINS;
          const slice = surfaceBuffers.magnitudeHist.slice(start, start + SURFACE_HIST_BINS);
          hist.push(Array.from(slice));
        }
        setSurfaceDebugSnapshot({
          orientationCount: surfaceBuffers.orientationCount,
          magnitudeMax: debug.surface.magnitudeMax,
          magnitudeHist: hist,
        });
      } else if (!showSurfaceDebug) {
        setSurfaceDebugSnapshot(null);
      }
    },
    [showRimDebug, showSurfaceDebug, setRimDebugSnapshot, setSurfaceDebugSnapshot],
  );

  const updatePhaseDebug = useCallback(
    (commit: boolean, phaseField: PhaseField | null | undefined) => {
      if (!commit) {
        if (!showPhaseDebug) {
          setPhaseDebugSnapshot(null);
        }
        if (!phaseHeatmapEnabled) {
          setPhaseHeatmapSnapshot(null);
        }
        return;
      }
      if (showPhaseDebug && phaseField && phaseField.amp) {
        let hist = phaseDebugRef.current;
        if (!hist || hist.length !== PHASE_HIST_BINS) {
          hist = new Float32Array(PHASE_HIST_BINS);
          phaseDebugRef.current = hist;
        } else {
          hist.fill(0);
        }
        let minAmp = Number.POSITIVE_INFINITY;
        let maxAmp = 0;
        const amps = phaseField.amp;
        for (let i = 0; i < amps.length; i++) {
          const value = amps[i];
          if (value < minAmp) minAmp = value;
          if (value > maxAmp) maxAmp = value;
          const norm = Math.min(0.999999, value / PHASE_HIST_SCALE);
          const bin = Math.min(PHASE_HIST_BINS - 1, Math.floor(norm * PHASE_HIST_BINS));
          hist[bin] += 1;
        }
        setPhaseDebugSnapshot({
          ampRange: [Number.isFinite(minAmp) ? minAmp : 0, maxAmp],
          ampHist: Array.from(hist),
        });
      } else if (!showPhaseDebug || !phaseField) {
        setPhaseDebugSnapshot(null);
      }

      if (phaseHeatmapEnabled && phaseField && phaseField.coh) {
        const { width: srcW, height: srcH } = phaseField.resolution;
        const targetW = Math.max(1, Math.min(96, srcW));
        const targetH = Math.max(1, Math.min(96, srcH));
        const scaleX = srcW / targetW;
        const scaleY = srcH / targetH;
        let store = phaseHeatmapRef.current;
        if (!store || store.width !== targetW || store.height !== targetH) {
          store = {
            width: targetW,
            height: targetH,
            data: new Float32Array(targetW * targetH),
          };
          phaseHeatmapRef.current = store;
        }
        const data = store.data;
        let minVal = Number.POSITIVE_INFINITY;
        let maxVal = 0;
        const cohValues = phaseField.coh;
        for (let y = 0; y < targetH; y++) {
          const srcY = Math.min(Math.floor((y + 0.5) * scaleY), srcH - 1);
          for (let x = 0; x < targetW; x++) {
            const srcX = Math.min(Math.floor((x + 0.5) * scaleX), srcW - 1);
            const value = cohValues[srcY * srcW + srcX];
            data[y * targetW + x] = value;
            if (value < minVal) minVal = value;
            if (value > maxVal) maxVal = value;
          }
        }
        if (!Number.isFinite(minVal)) {
          minVal = 0;
        }
        if (!Number.isFinite(maxVal) || maxVal < minVal) {
          maxVal = minVal;
        }
        setPhaseHeatmapSnapshot({
          width: targetW,
          height: targetH,
          values: Float32Array.from(data),
          min: minVal,
          max: maxVal,
        });
      } else if (!phaseHeatmapEnabled || !phaseField) {
        setPhaseHeatmapSnapshot(null);
      }
    },
    [showPhaseDebug, phaseHeatmapEnabled, setPhaseDebugSnapshot, setPhaseHeatmapSnapshot],
  );

  useEffect(() => {
    if (!phaseHeatmapEnabled) {
      setPhaseHeatmapSnapshot(null);
      phaseHeatmapRef.current = null;
    }
  }, [phaseHeatmapEnabled]);

  const getOrientationCache = useCallback((count: number) => {
    if (orientationCacheRef.current.count !== count) {
      orientationCacheRef.current = {
        count,
        cos: new Float32Array(count),
        sin: new Float32Array(count),
      };
    }
    return orientationCacheRef.current;
  }, []);

  const refreshGpuStaticTextures = useCallback(() => {
    const state = gpuStateRef.current;
    if (!state) return;
    const { renderer } = state;
    if (basePixelsRef.current) {
      renderer.uploadBase(basePixelsRef.current);
    }
    if (rimFieldRef.current) {
      renderer.uploadRim(rimFieldRef.current);
    }
    pendingStaticUploadRef.current = false;
  }, []);

  const ensureGpuRenderer = useCallback(() => {
    if (renderBackend !== 'gpu') return null;
    const canvas = canvasRef.current;
    if (!canvas) return null;
    canvas.width = width;
    canvas.height = height;
    let state = gpuStateRef.current;
    if (!state || state.gl.canvas !== canvas) {
      const gl = canvas.getContext('webgl2', {
        alpha: false,
        antialias: false,
        premultipliedAlpha: false,
        preserveDrawingBuffer: true,
      });
      if (!gl) {
        console.warn('[gpu] WebGL2 unavailable; falling back to CPU renderer.');
        setRenderBackend('cpu');
        return null;
      }
      if (state) {
        state.renderer.dispose();
      }
      const renderer = createGpuRenderer(gl);
      state = { gl, renderer };
      gpuStateRef.current = state;
      pendingStaticUploadRef.current = true;
      gpuTracerRef.current.lastTime = null;
      gpuTracerRef.current.needsReset = true;
      renderer.setHyperbolicAtlas(hyperbolicAtlasGpu);
    }
    state.renderer.resize(width, height);
    if (pendingStaticUploadRef.current) {
      refreshGpuStaticTextures();
    }
    return state;
  }, [renderBackend, width, height, refreshGpuStaticTextures, hyperbolicAtlasGpu]);

  const setFrameProfiler = useCallback(
    (enabled: boolean, sampleCount = 120, label = 'frame-profiler') => {
      const profiler = frameProfilerRef.current;
      profiler.enabled = enabled;
      profiler.maxSamples = sampleCount;
      profiler.samples = [];
      profiler.label = label;
      console.log(
        `[frame-profiler] ${enabled ? `collecting ${sampleCount} samples (${label})` : 'disabled'}`,
      );
    },
    [],
  );

  const initKuramotoCpu = useCallback(
    (q: number) => {
      ensureKurCpuState();
      if (!kurStateRef.current || !cpuDerivedRef.current) return;
      initKuramotoState(kurStateRef.current, q, cpuDerivedRef.current);
    },
    [ensureKurCpuState],
  );

  const getKurParams = useCallback((): KuramotoParams => {
    return {
      alphaKur,
      gammaKur,
      omega0,
      K0,
      epsKur,
      fluxX,
      fluxY,
      smallWorldWeight: smallWorldEnabled ? smallWorldWeight : 0,
      p_sw: pSw,
      smallWorldEnabled,
      smallWorldSeed,
      smallWorldDegree,
    };
  }, [
    alphaKur,
    gammaKur,
    omega0,
    K0,
    epsKur,
    fluxX,
    fluxY,
    smallWorldEnabled,
    smallWorldWeight,
    pSw,
    smallWorldSeed,
    smallWorldDegree,
  ]);

  // Re-initialize the Kuramoto field whenever the canvas size or twist changes.
  useEffect(() => {
    if (kurEnabled) {
      initKuramotoCpu(qInit);
    } else {
      ensureKurCpuState();
      if (cpuDerivedRef.current) {
        cpuDerivedRef.current.gradX.fill(0);
        cpuDerivedRef.current.gradY.fill(0);
        cpuDerivedRef.current.vort.fill(0);
        cpuDerivedRef.current.coh.fill(0.5);
        cpuDerivedRef.current.amp.fill(0);
        ampRef.current = cpuDerivedRef.current.amp;
        markFieldGone('phase', 'cpu-reset');
      }
      markFieldGone('volume', 'cpu-reset');
    }
  }, [kurEnabled, qInit, initKuramotoCpu, ensureKurCpuState]);

  useEffect(() => {
    if (!volumeEnabled) {
      volumeFieldRef.current = null;
      volumeStubRef.current = null;
      markFieldGone('volume', 'volume-disabled');
      return;
    }
    ensureVolumeState();
    if (volumeStubRef.current) {
      const field = snapshotVolumeStub(volumeStubRef.current);
      assertVolumeField(field, 'volume:init');
      volumeFieldRef.current = field;
      markFieldFresh('volume', field.resolution, 'volume:stub');
    }
  }, [volumeEnabled, ensureVolumeState, markFieldFresh, markFieldGone]);

  const stepKuramotoCpu = useCallback(
    (dt: number) => {
      if (!kurEnabled) return;
      ensureKurCpuState();
      if (!kurStateRef.current) return;
      const kernelSnapshot = kernelEventRef.current;
      kurAppliedKernelVersionRef.current = kernelSnapshot.version;
      const timestamp = typeof performance !== 'undefined' ? performance.now() : Date.now();
      const result = stepKuramotoState(kurStateRef.current, getKurParams(), dt, randn, timestamp, {
        kernel: kernelSnapshot.spec,
        controls: { dmt },
        telemetry: { kernelVersion: kernelSnapshot.version },
      });
      kurTelemetryRef.current = result.telemetry;
      kurIrradianceRef.current = result.irradiance;
      logKurTelemetry(result.telemetry);
    },
    [kurEnabled, ensureKurCpuState, getKurParams, randn, dmt, logKurTelemetry],
  );

  const deriveKurFieldsCpu = useCallback(() => {
    if (!kurEnabled) return;
    ensureKurCpuState();
    if (!kurStateRef.current || !cpuDerivedRef.current) return;
    const kernelSnapshot = kernelEventRef.current;
    deriveKuramotoFieldsCore(kurStateRef.current, cpuDerivedRef.current, {
      kernel: kernelSnapshot.spec,
      controls: { dmt },
    });
    markFieldFresh('phase', cpuDerivedRef.current.resolution, 'cpu');
  }, [kurEnabled, ensureKurCpuState, dmt, markFieldFresh]);

  const resetKuramotoField = useCallback(() => {
    initKuramotoCpu(qInit);
    if (!kurSyncRef.current && workerRef.current && workerReadyRef.current) {
      workerRef.current.postMessage({
        kind: 'reset',
        qInit,
      });
    }
  }, [initKuramotoCpu, qInit]);

  const markKurCustom = useCallback(() => {
    setKurRegime('custom');
  }, [setKurRegime]);

  const applyKurRegime = useCallback(
    (regime: Exclude<KurRegime, 'custom'>) => {
      const preset = KUR_REGIME_PRESETS[regime];
      setK0(preset.params.K0);
      setAlphaKur(preset.params.alphaKur);
      setGammaKur(preset.params.gammaKur);
      setOmega0(preset.params.omega0);
      setEpsKur(preset.params.epsKur);
      setKurRegime(regime);
      resetKuramotoField();
    },
    [setK0, setAlphaKur, setGammaKur, setOmega0, setEpsKur, resetKuramotoField],
  );

  useEffect(() => {
    kurSyncRef.current = kurSync;
  }, [kurSync]);

  const clearWorkerData = useCallback(() => {
    workerPendingFramesRef.current = [];
    workerActiveFrameRef.current = null;
    workerInflightRef.current = 0;
    workerNextFrameIdRef.current = 0;
    workerLastFrameIdRef.current = -1;
    gradXRef.current = null;
    gradYRef.current = null;
    vortRef.current = null;
    cohRef.current = null;
    ampRef.current = null;
    phaseDebugRef.current = null;
    phaseHeatmapRef.current = null;
  }, []);

  const releaseFrameToWorker = useCallback((frame: KurFrameView) => {
    const worker = workerRef.current;
    if (!worker) return;
    try {
      worker.postMessage({ kind: 'returnBuffer', buffer: frame.buffer }, [frame.buffer]);
    } catch (err) {
      console.debug('[kur-worker] buffer return skipped', err);
    }
  }, []);

  const swapWorkerFrame = useCallback(() => {
    const worker = workerRef.current;
    if (!worker) return;
    const queue = workerPendingFramesRef.current;
    if (queue.length === 0) return;
    const next = queue.shift()!;
    const prev = workerActiveFrameRef.current;
    workerActiveFrameRef.current = next;
    workerLastFrameIdRef.current = next.frameId;
    gradXRef.current = next.gradX;
    gradYRef.current = next.gradY;
    vortRef.current = next.vort;
    cohRef.current = next.coh;
    ampRef.current = next.amp;
    kurTelemetryRef.current = next.instrumentation.telemetry;
    kurIrradianceRef.current = null;
    logKurTelemetry(next.instrumentation.telemetry);
    if (prev) {
      releaseFrameToWorker(prev);
    }
  }, [releaseFrameToWorker, logKurTelemetry]);

  const handleWorkerFrame = useCallback(
    (msg: WorkerFrameMessage) => {
      if (!workerRef.current) return;
      workerInflightRef.current = Math.max(workerInflightRef.current - 1, 0);
      const derived = createDerivedViews(msg.buffer, width, height);
      assertPhaseField(derived, 'worker:frame');
      markFieldFresh('phase', derived.resolution, 'worker');
      const frame: KurFrameView = {
        buffer: msg.buffer,
        gradX: derived.gradX,
        gradY: derived.gradY,
        vort: derived.vort,
        coh: derived.coh,
        amp: derived.amp,
        timestamp: msg.timestamp,
        frameId: msg.meta?.frameId ?? msg.frameId,
        kernelVersion: msg.kernelVersion ?? 0,
        meta: msg.meta,
        instrumentation: msg.instrumentation,
      };
      if (msg.meta && msg.meta.frameId !== msg.frameId) {
        console.warn(
          `[kur-worker] frameId mismatch meta=${msg.meta.frameId} payload=${msg.frameId}`,
        );
      }
      if (frame.kernelVersion !== kernelEventRef.current.version) {
        console.debug(
          `[kur-worker] kernel version drift: worker=${frame.kernelVersion} ui=${kernelEventRef.current.version}`,
        );
      }
      const lastFrameId = workerLastFrameIdRef.current;
      if (frame.frameId <= lastFrameId) {
        console.debug(`[kur-worker] dropping stale frame ${frame.frameId} (last=${lastFrameId})`);
        releaseFrameToWorker(frame);
        return;
      }
      const queue = workerPendingFramesRef.current;
      let inserted = false;
      for (let i = 0; i < queue.length; i++) {
        const existing = queue[i];
        if (frame.frameId === existing.frameId) {
          console.debug(`[kur-worker] skipping duplicate frame ${frame.frameId}`);
          releaseFrameToWorker(frame);
          return;
        }
        if (frame.frameId < existing.frameId) {
          queue.splice(i, 0, frame);
          inserted = true;
          break;
        }
      }
      if (!inserted) {
        queue.push(frame);
      }
      if (!kurSyncRef.current) {
        swapWorkerFrame();
      }
    },
    [height, width, swapWorkerFrame, releaseFrameToWorker],
  );

  const stopKurWorker = useCallback(() => {
    const worker = workerRef.current;
    if (worker) {
      worker.terminate();
    }
    workerRef.current = null;
    workerReadyRef.current = false;
    workerInflightRef.current = 0;
    workerNextFrameIdRef.current = 0;
    clearWorkerData();
  }, [clearWorkerData]);

  const handleWorkerMessage = useCallback(
    (msg: WorkerIncomingMessage) => {
      switch (msg.kind) {
        case 'ready':
          workerReadyRef.current = true;
          if (!kurSyncRef.current && workerRef.current) {
            const snapshot = kernelEventRef.current;
            workerRef.current.postMessage({
              kind: 'kernelSpec',
              spec: snapshot.spec,
              version: snapshot.version,
            });
          }
          break;
        case 'frame':
          handleWorkerFrame(msg);
          break;
        case 'log':
          console.log(msg.message);
          break;
        default:
          console.warn('[kur-worker] unknown message', msg);
          break;
      }
    },
    [handleWorkerFrame],
  );

  const startKurWorker = useCallback(() => {
    if (!kurEnabled || kurSyncRef.current) return;
    stopKurWorker();
    clearWorkerData();
    const bufferSize = derivedBufferSize(width, height);
    const buffers = [new ArrayBuffer(bufferSize), new ArrayBuffer(bufferSize)];
    const worker = new Worker(new URL('./kuramotoWorker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;
    workerReadyRef.current = false;
    workerInflightRef.current = 0;
    workerNextFrameIdRef.current = 0;
    workerPendingFramesRef.current = [];
    workerActiveFrameRef.current = null;
    worker.onmessage = (event: MessageEvent<WorkerIncomingMessage>) => {
      handleWorkerMessage(event.data);
    };
    worker.postMessage(
      {
        kind: 'init',
        width,
        height,
        params: getKurParams(),
        qInit,
        buffers,
        seed: Math.floor(Math.random() * 1e9),
      },
      buffers,
    );
  }, [
    clearWorkerData,
    getKurParams,
    handleWorkerMessage,
    height,
    kurEnabled,
    qInit,
    stopKurWorker,
    width,
  ]);

  useEffect(() => {
    if (!kurEnabled || kurSync) {
      stopKurWorker();
      clearWorkerData();
      if (kurEnabled && kurSync) {
        ensureKurCpuState();
        initKuramotoCpu(qInit);
      }
      return;
    }
    startKurWorker();
    return () => {
      stopKurWorker();
      clearWorkerData();
    };
  }, [
    kurEnabled,
    kurSync,
    startKurWorker,
    stopKurWorker,
    clearWorkerData,
    ensureKurCpuState,
    initKuramotoCpu,
    qInit,
  ]);

  useEffect(() => {
    if (!kurEnabled || kurSyncRef.current) return;
    const worker = workerRef.current;
    if (!worker || !workerReadyRef.current) return;
    worker.postMessage({
      kind: 'updateParams',
      params: getKurParams(),
    });
  }, [getKurParams, kurEnabled, kurSync]);

  const renderFrameCore = useCallback(
    (
      out: Uint8ClampedArray,
      tSeconds: number,
      commitObs = true,
      fieldsOverride?: Pick<KurFrameView, 'gradX' | 'gradY' | 'vort' | 'coh' | 'amp'>,
      options?: RenderFrameOptions,
    ) => {
      const kernelSnapshot = kernelEventRef.current;
      const kernelSpec = kernelSnapshot.spec;
      const surfaceField = surfaceFieldRef.current;
      if (surfaceField) {
        assertSurfaceField(surfaceField, 'cpu:surface');
      }
      const rimField = rimFieldRef.current;
      if (rimField) {
        assertRimField(rimField, 'cpu:rim');
      }
      const volumeField = options?.volumeFieldOverride ?? volumeFieldRef.current;
      if (volumeField) {
        assertVolumeField(volumeField, 'cpu:volume');
      }
      let phaseField: PhaseField | null = null;
      const phaseSource = cpuDerivedRef.current;
      if (phaseSource) {
        const resolution = phaseSource.resolution;
        const gradX = fieldsOverride?.gradX ?? gradXRef.current;
        const gradY = fieldsOverride?.gradY ?? gradYRef.current;
        const vort = fieldsOverride?.vort ?? vortRef.current;
        const coh = fieldsOverride?.coh ?? cohRef.current;
        const amp = fieldsOverride?.amp ?? ampRef.current;
        if (gradX && gradY && vort && coh && amp) {
          phaseField = {
            kind: 'phase',
            resolution,
            gradX,
            gradY,
            vort,
            coh,
            amp,
          };
          assertPhaseField(phaseField, fieldsOverride ? 'phase:override' : 'phase:cpu');
          markFieldFresh('phase', resolution, fieldsOverride ? 'worker' : 'cpu');
        }
      }

      const rimDebugRequest = showRimDebug ? ensureRimDebugBuffers() : null;
      const surfaceDebugRequest =
        showSurfaceDebug && orientations.length > 0
          ? ensureSurfaceDebugBuffers(orientations.length)
          : null;

      const { base: couplingBase, effective: couplingEffective } = computeCouplingPair(
        options?.toggles,
      );

      const result = renderRainbowFrame({
        width,
        height,
        timeSeconds: tSeconds,
        out,
        surface: surfaceField,
        rim: rimField,
        phase: phaseField,
        volume: volumeField,
        kernel: kernelSpec,
        dmt,
        arousal,
        blend,
        normPin,
        normTarget: normTargetRef.current,
        lastObs: lastObsRef.current,
        lambdaRef,
        lambdas,
        beta2,
        microsaccade,
        alive,
        phasePin,
        edgeThreshold,
        wallpaperGroup: wallGroup,
        surfEnabled,
        orientationAngles: orientations,
        thetaMode,
        thetaGlobal,
        polBins,
        jitter,
        coupling: couplingEffective,
        couplingBase,
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
        hyperbolicAtlas,
        kurEnabled,
        debug:
          rimDebugRequest || surfaceDebugRequest
            ? {
                rim: rimDebugRequest ?? undefined,
                surface: surfaceDebugRequest ?? undefined,
              }
            : undefined,
        su7: su7Params,
        composer,
        kurTelemetry: kurTelemetryRef.current ?? undefined,
      });
      const shouldApplyTracer = tracerRuntime.enabled && (options?.applyTracer ?? true);
      if (shouldApplyTracer) {
        const tracerBuffer = ensureTracerBuffer();
        if (tracerBuffer) {
          const lastTime = tracerLastTimeRef.current;
          const dt =
            lastTime != null ? Math.max(1 / 480, Math.min(tSeconds - lastTime, 0.25)) : 1 / 60;
          applyTracerFeedback({
            out,
            state: tracerBuffer,
            width,
            height,
            runtime: tracerRuntime,
            dt,
            timeSeconds: tSeconds,
          });
          tracerLastTimeRef.current = tSeconds;
        }
      } else if (tracerRuntime.enabled) {
        tracerLastTimeRef.current = tSeconds;
      }
      if (commitObs && result.obsAverage != null) {
        lastObsRef.current = result.obsAverage;
      }
      if (commitObs && result.metrics) {
        metricsRef.current.push({
          backend: 'cpu',
          ts: performance.now(),
          metrics: result.metrics,
          kernelVersion: kernelSnapshot.version,
        });
        if (metricsRef.current.length > 240) {
          metricsRef.current.shift();
        }
      }
      updateDebugSnapshots(commitObs, rimDebugRequest, surfaceDebugRequest, result.debug);
      updatePhaseDebug(commitObs, phaseField);
      return result.metrics;
    },
    [
      dmt,
      arousal,
      blend,
      normPin,
      lambdas,
      lambdaRef,
      beta2,
      microsaccade,
      alive,
      surfaceFieldRef,
      rimFieldRef,
      phasePin,
      height,
      width,
      edgeThreshold,
      wallGroup,
      surfEnabled,
      orientations,
      thetaMode,
      thetaGlobal,
      polBins,
      jitter,
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
      hyperbolicAtlas,
      kurEnabled,
      tracerRuntime,
      ensureTracerBuffer,
      volumeEnabled,
      composer,
      computeCouplingPair,
      ensureRimDebugBuffers,
      ensureSurfaceDebugBuffers,
      updateDebugSnapshots,
      updatePhaseDebug,
      basePixelsRef,
    ],
  );

  const logFrameMetrics = useCallback(
    (tSeconds: number) => {
      if (!frameLoggingEnabled) {
        return;
      }
      const state = frameLogRef.current;
      const now = performance.now();
      if (state.frames === 0) {
        state.windowStart = now;
      }
      state.frames += 1;
      const elapsed = now - state.windowStart;
      if (elapsed < 1000) {
        return;
      }
      const fps = state.frames > 0 ? (state.frames * 1000) / elapsed : 0;
      state.windowStart = now;
      state.frames = 0;

      if (!metricsScratchRef.current || metricsScratchRef.current.length !== width * height * 4) {
        metricsScratchRef.current = new Uint8ClampedArray(width * height * 4);
      }
      const scratch = metricsScratchRef.current;
      let metrics: RainbowFrameMetrics | null = null;
      try {
        metrics = renderFrameCore(scratch, tSeconds, false, undefined, { applyTracer: false });
      } catch (error) {
        console.warn('[frame-log] failed to sample metrics', error);
      }
      if (!metrics) {
        console.log(
          `[frame-log] fps=${fps.toFixed(1)} metrics=unavailable heatmap=${phaseHeatmapEnabled ? 'on' : 'off'}`,
        );
        return;
      }
      const rimEnergy = metrics.composer.fields.rim.energy;
      const surfaceEnergy = metrics.composer.fields.surface.energy;
      const cohMean = metrics.gradient.cohMean ?? 0;
      const cohStd = metrics.gradient.cohStd ?? 0;
      console.log(
        `[frame-log] fps=${fps.toFixed(1)} rim=${rimEnergy.toFixed(3)} surface=${surfaceEnergy.toFixed(
          3,
        )} |Z|=${cohMean.toFixed(3)}±${cohStd.toFixed(3)} heatmap=${phaseHeatmapEnabled ? 'on' : 'off'}`,
      );
    },
    [frameLoggingEnabled, renderFrameCore, phaseHeatmapEnabled, width, height],
  );

  const drawFrameGpu = useCallback(
    (
      state: { gl: WebGL2RenderingContext; renderer: GpuRenderer },
      tSeconds: number,
      commitObs: boolean,
    ) => {
      const kernelSnapshot = kernelEventRef.current;
      const kernelSpec = kernelSnapshot.spec;
      const { gl, renderer } = state;
      if (!rimFieldRef.current || !basePixelsRef.current) {
        gl.clear(gl.COLOR_BUFFER_BIT);
        return;
      }

      const rimDebugRequest = showRimDebug ? ensureRimDebugBuffers() : null;
      const surfaceDebugRequest =
        showSurfaceDebug && orientations.length > 0
          ? ensureSurfaceDebugBuffers(orientations.length)
          : null;

      const { base: couplingBase, effective: couplingEffective } = computeCouplingPair();

      const telemetryActive = telemetryRef.current.enabled && commitObs;
      const needsCpuCompositor =
        commitObs && (telemetryActive || rimDebugRequest != null || surfaceDebugRequest != null);
      const renderStart = telemetryActive ? performance.now() : 0;

      const ke = kEff(kernelSpec, dmt);
      const effectiveBlend = clamp01(blend + ke.transparency * 0.5);
      const eps = 1e-6;
      const frameGain = normPin
        ? Math.pow((normTargetRef.current + eps) / (lastObsRef.current + eps), 0.5)
        : 1.0;

      const baseOffsets = {
        L: beta2 * (lambdaRef / lambdas.L - 1),
        M: beta2 * (lambdaRef / lambdas.M - 1),
        S: beta2 * (lambdaRef / lambdas.S - 1),
      } as const;

      const jitterPhase = microsaccade ? tSeconds * 6.0 : 0.0;
      const breath = alive ? 0.15 * Math.sin(2 * Math.PI * 0.55 * tSeconds) : 0.0;

      const rimField = rimFieldRef.current!;
      const { gx, gy, mag } = rimField;
      const gradX = gradXRef.current;
      const gradY = gradYRef.current;
      const vort = vortRef.current;
      const coh = cohRef.current;
      const amp = ampRef.current;
      const volumeField = volumeFieldRef.current;
      if (volumeField) {
        assertVolumeField(volumeField, 'gpu:volume-active');
      }
      const surfaceField = surfaceFieldRef.current;
      if (surfaceField) {
        assertSurfaceField(surfaceField, 'gpu:surface');
      }
      assertRimField(rimField, 'gpu:rim-active');
      const phaseSource = cpuDerivedRef.current;
      const phaseField =
        phaseSource && gradX && gradY && vort && coh && amp
          ? {
              kind: 'phase' as const,
              resolution: phaseSource.resolution,
              gradX,
              gradY,
              vort,
              coh,
              amp,
            }
          : null;
      if (phaseField) {
        assertPhaseField(phaseField, 'gpu:phase-active');
      }

      const ops = groupOps(wallGroup);
      const gpuOps = toGpuOps(ops);
      const useWallpaper =
        surfEnabled ||
        couplingEffective.surfaceToRimOffset > 1e-4 ||
        couplingEffective.surfaceToRimSigma > 1e-4 ||
        couplingEffective.surfaceToRimHue > 1e-4;
      const N = orientations.length;
      const orientationCache = getOrientationCache(N);
      const cosA = orientationCache.cos;
      const sinA = orientationCache.sin;
      for (let j = 0; j < N; j++) {
        cosA[j] = Math.cos(orientations[j]);
        sinA[j] = Math.sin(orientations[j]);
      }

      const cx = width * 0.5;
      const cy = height * 0.5;
      let muJ = 0;
      if (phasePin && microsaccade) {
        let muSum = 0;
        let muCount = 0;
        for (let yy = 0; yy < height; yy += 8) {
          for (let xx = 0; xx < width; xx += 8) {
            const idx = yy * width + xx;
            if (mag[idx] >= edgeThreshold) {
              muSum += Math.sin(jitterPhase + hash2(xx, yy) * Math.PI * 2);
              muCount++;
            }
          }
        }
        muJ = muCount ? muSum / muCount : 0;
      }

      let metricDebug: ReturnType<typeof renderRainbowFrame>['debug'] | null = null;
      if (needsCpuCompositor) {
        if (!metricsScratchRef.current || metricsScratchRef.current.length !== width * height * 4) {
          metricsScratchRef.current = new Uint8ClampedArray(width * height * 4);
        }
        const scratch = metricsScratchRef.current;
        const metricsResult = renderRainbowFrame({
          width,
          height,
          timeSeconds: tSeconds,
          out: scratch,
          surface: surfaceField,
          rim: rimField,
          phase: phaseField,
          volume: volumeField,
          kernel: kernelSpec,
          dmt,
          arousal,
          blend,
          normPin,
          normTarget: normTargetRef.current,
          lastObs: lastObsRef.current,
          lambdaRef,
          lambdas,
          beta2,
          microsaccade,
          alive,
          phasePin,
          edgeThreshold,
          wallpaperGroup: wallGroup,
          surfEnabled,
          orientationAngles: orientations,
          thetaMode,
          thetaGlobal,
          polBins,
          jitter,
          coupling: couplingEffective,
          couplingBase,
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
          hyperbolicAtlas,
          kurEnabled,
          debug:
            rimDebugRequest || surfaceDebugRequest
              ? {
                  rim: rimDebugRequest ?? undefined,
                  surface: surfaceDebugRequest ?? undefined,
                }
              : undefined,
          su7: su7Params,
          composer,
          kurTelemetry: kurTelemetryRef.current ?? undefined,
        });
        metricDebug = metricsResult.debug;
        if (telemetryActive) {
          metricsRef.current.push({
            backend: 'gpu',
            ts: performance.now(),
            metrics: metricsResult.metrics,
            kernelVersion: kernelSnapshot.version,
          });
          if (metricsRef.current.length > 240) {
            metricsRef.current.shift();
          }
          if (metricsResult.obsAverage != null) {
            lastObsRef.current = metricsResult.obsAverage;
          }
        }
      }
      updateDebugSnapshots(
        commitObs,
        rimDebugRequest,
        surfaceDebugRequest,
        metricDebug ?? undefined,
      );
      updatePhaseDebug(commitObs, phaseField);

      renderer.uploadPhase(phaseField);
      renderer.uploadVolume(volumeField ?? null);

      let su7Uniforms: Su7Uniforms = {
        enabled: false,
        gain: su7Params.gain,
        decimationStride: 1,
        decimationMode: 'hybrid',
        projectorMode: 'identity',
        projectorWeight: 0,
        projectorMatrix: null,
      };
      let su7Payload: Su7TexturePayload | null = null;
      const texelCount = width * height;
      if (su7Params.enabled && su7Params.gain > 1e-4 && texelCount > 0) {
        const embedResult = embedToC7({
          surface: surfaceField,
          rim: rimField,
          phase: phaseField ?? undefined,
          volume: volumeField ?? undefined,
          width,
          height,
          gauge: 'rim',
        });
        if (embedResult.vectors.length === texelCount) {
          const vectorBuffers = ensureSu7VectorBuffers(su7VectorBuffersRef.current, texelCount);
          fillSu7VectorBuffers(embedResult.vectors, embedResult.norms, texelCount, vectorBuffers);
          su7VectorBuffersRef.current = vectorBuffers;
          const su7ContextGpu =
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
          const unitary = buildScheduledUnitary(su7Params, su7ContextGpu ?? undefined);
          const tileCols = Math.max(1, Math.ceil(width / SU7_TILE_SIZE));
          const tileRows = Math.max(1, Math.ceil(height / SU7_TILE_SIZE));
          const tileBuffer = ensureSu7TileBuffer(su7TileBufferRef.current, tileCols, tileRows);
          fillSu7TileBuffer(unitary, tileCols, tileRows, tileBuffer);
          su7TileBufferRef.current = tileBuffer;
          const rawParams = su7Params as Record<string, unknown>;
          let decimationStride = 2;
          const strideCandidate = rawParams['decimationStride'];
          if (
            typeof strideCandidate === 'number' &&
            Number.isFinite(strideCandidate) &&
            strideCandidate >= 1
          ) {
            decimationStride = Math.max(1, Math.floor(strideCandidate));
          }
          let decimationMode: 'hybrid' | 'stride' | 'edges' = 'hybrid';
          const modeCandidate = rawParams['decimationMode'];
          if (
            modeCandidate === 'stride' ||
            modeCandidate === 'edges' ||
            modeCandidate === 'hybrid'
          ) {
            decimationMode = modeCandidate;
          }
          const projectorId =
            typeof su7Params.projector?.id === 'string'
              ? su7Params.projector.id.toLowerCase()
              : 'identity';
          let projectorMode: Su7ProjectorMode = 'identity';
          if (projectorId === 'composerweights') {
            projectorMode = 'composerWeights';
          } else if (projectorId === 'overlaysplit') {
            projectorMode = 'overlaySplit';
          } else if (projectorId === 'directrgb') {
            projectorMode = 'directRgb';
          }
          const projectorWeight = Math.min(
            1,
            Math.max(0, Math.abs(su7Params.projector?.weight ?? 1)),
          );
          su7Uniforms = {
            enabled: true,
            gain: su7Params.gain,
            decimationStride,
            decimationMode,
            projectorMode,
            projectorWeight,
            projectorMatrix: null,
          };
          su7Payload = {
            width,
            height,
            vectors: [
              vectorBuffers.tex0,
              vectorBuffers.tex1,
              vectorBuffers.tex2,
              vectorBuffers.tex3,
            ],
            tileData: tileBuffer,
            tileCols,
            tileRows,
            tileSize: SU7_TILE_SIZE,
            tileTexWidth: SU7_TILE_TEXTURE_WIDTH,
            tileTexRowsPerTile: SU7_TILE_TEXTURE_ROWS_PER_TILE,
          };
        }
      }

      renderer.uploadSu7(su7Payload);

      const couplingScale = 1 + 0.65 * dmt;

      const lastGpuTracerTime = gpuTracerRef.current.lastTime;
      const tracerDelta = lastGpuTracerTime != null ? Math.max(0, tSeconds - lastGpuTracerTime) : 0;
      const tracerDt =
        lastGpuTracerTime != null ? Math.max(1 / 480, Math.min(tracerDelta, 0.25)) : 1 / 60;
      const tracerEnabled = tracerRuntime.enabled;
      const tracerReset = gpuTracerRef.current.needsReset || !tracerEnabled;

      renderer.render({
        time: tSeconds,
        edgeThreshold,
        effectiveBlend,
        displayMode: displayModeToEnum(displayMode),
        baseOffsets: [baseOffsets.L, baseOffsets.M, baseOffsets.S],
        sigma,
        jitter,
        jitterPhase,
        breath,
        muJ,
        phasePin,
        microsaccade,
        polBins,
        thetaMode: thetaMode === 'gradient' ? 0 : 1,
        thetaGlobal,
        contrast,
        frameGain,
        rimAlpha,
        rimEnabled,
        warpAmp,
        curvatureStrength,
        curvatureMode,
        surfaceBlend,
        surfaceRegion: surfaceRegionToEnum(surfaceRegion),
        surfEnabled,
        kurEnabled,
        volumeEnabled,
        useWallpaper,
        kernel: ke,
        alive,
        beta2,
        coupling: couplingEffective,
        couplingScale,
        composerExposure: composerUniforms.exposure,
        composerGamma: composerUniforms.gamma,
        composerWeight: composerUniforms.weight,
        composerBlendGain: composerUniforms.blendGain,
        tracer: {
          enabled: tracerEnabled,
          gain: tracerRuntime.gain,
          tau: tracerRuntime.tau,
          modulationDepth: tracerRuntime.modulationDepth,
          modulationFrequency: tracerRuntime.modulationFrequency,
          modulationPhase: tracerRuntime.modulationPhase,
          dt: tracerDt,
          reset: tracerReset,
        },
        su7: su7Uniforms,
        orientations: orientationCache,
        ops: gpuOps,
        center: [cx, cy],
      });

      gpuTracerRef.current.lastTime = tSeconds;
      gpuTracerRef.current.needsReset = !tracerEnabled;

      logFrameMetrics(tSeconds);

      if (telemetryActive) {
        recordTelemetry('renderGpu', performance.now() - renderStart);
      }
    },
    [
      rimFieldRef,
      surfaceFieldRef,
      basePixelsRef,
      dmt,
      arousal,
      blend,
      normPin,
      lambdas,
      lambdaRef,
      beta2,
      microsaccade,
      alive,
      orientations,
      getOrientationCache,
      wallGroup,
      surfEnabled,
      thetaMode,
      thetaGlobal,
      polBins,
      jitter,
      computeCouplingPair,
      sigma,
      contrast,
      rimAlpha,
      rimEnabled,
      warpAmp,
      curvatureStrength,
      curvatureMode,
      surfaceBlend,
      surfaceRegion,
      kurEnabled,
      su7Params,
      width,
      height,
      edgeThreshold,
      recordTelemetry,
      showRimDebug,
      showSurfaceDebug,
      ensureRimDebugBuffers,
      ensureSurfaceDebugBuffers,
      updateDebugSnapshots,
      updatePhaseDebug,
      showPhaseDebug,
      logFrameMetrics,
      tracerRuntime,
      deriveSu7ScheduleContext,
    ],
  );

  const advanceVolume = useCallback(
    (dt: number) => {
      if (!volumeEnabled) return;
      ensureVolumeState();
      const stub = volumeStubRef.current;
      if (!stub) return;
      stepVolumeStub(stub, dt);
      const field = snapshotVolumeStub(stub);
      assertVolumeField(field, 'volume:stub');
      volumeFieldRef.current = field;
      markFieldFresh('volume', field.resolution, 'volume:stub');
    },
    [volumeEnabled, ensureVolumeState, markFieldFresh],
  );

  const advanceKuramoto = useCallback(
    (dt: number, tSeconds: number) => {
      if (!kurEnabled) return;
      const teleStart = telemetryRef.current.enabled ? performance.now() : 0;
      if (kurSyncRef.current) {
        stepKuramotoCpu(dt);
        deriveKurFieldsCpu();
      } else {
        const worker = workerRef.current;
        if (worker && workerReadyRef.current) {
          const inflight = workerInflightRef.current;
          if (inflight < 2) {
            const frameId = workerNextFrameIdRef.current++;
            worker.postMessage({
              kind: 'tick',
              dt,
              timestamp: tSeconds,
              frameId,
            });
            workerInflightRef.current = inflight + 1;
          }
        }
        swapWorkerFrame();
      }
      if (teleStart) {
        recordTelemetry('kuramoto', performance.now() - teleStart);
      }
    },
    [kurEnabled, stepKuramotoCpu, deriveKurFieldsCpu, swapWorkerFrame, recordTelemetry],
  );

  const drawFrameCpu = useCallback(
    (ctx: CanvasRenderingContext2D, tSeconds: number) => {
      const telemetryActive = telemetryRef.current.enabled;
      const frameStart = telemetryActive ? performance.now() : 0;
      const dt = 0.016 * speed;
      advanceKuramoto(dt, tSeconds);
      advanceVolume(dt);
      const buffer = ensureFrameBuffer(ctx);
      const profiler = frameProfilerRef.current;
      let start = 0;
      if (profiler.enabled) {
        start = performance.now();
      }
      let renderStart = 0;
      if (telemetryActive) {
        renderStart = performance.now();
      }
      renderFrameCore(buffer.data, tSeconds, true);
      if (renderStart) {
        recordTelemetry('renderCpu', performance.now() - renderStart);
      }
      if (profiler.enabled) {
        const dt = performance.now() - start;
        profiler.samples.push(dt);
        if (profiler.samples.length >= profiler.maxSamples) {
          const count = profiler.samples.length;
          const sum = profiler.samples.reduce((acc, ms) => acc + ms, 0);
          const avg = sum / count;
          const sorted = [...profiler.samples].sort((a, b) => a - b);
          const idx = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95));
          const p95 = sorted[idx] ?? avg;
          console.log(
            `[${profiler.label}] avg ${avg.toFixed(3)}ms p95 ${p95.toFixed(
              3,
            )}ms over ${count} frames`,
          );
          profiler.samples = [];
          profiler.enabled = false;
        }
      }
      ctx.putImageData(buffer.image, 0, 0);
      logFrameMetrics(tSeconds);
      if (frameStart) {
        recordTelemetry('frame', performance.now() - frameStart);
      }
    },
    [
      advanceKuramoto,
      advanceVolume,
      speed,
      ensureFrameBuffer,
      renderFrameCore,
      recordTelemetry,
      logFrameMetrics,
    ],
  );

  const runRegressionHarness = useCallback(
    async (frameCount = 10) => {
      if (!rimFieldRef.current || !surfaceFieldRef.current) {
        console.warn('[regression] skipping: surface or rim field not ready.');
        return { maxDelta: 0, perFrameMax: [] as number[] };
      }
      if (!kurEnabled) {
        console.warn('[regression] Kuramoto disabled; nothing to compare.');
        return { maxDelta: 0, perFrameMax: [] as number[] };
      }

      const params = getKurParams();
      const total = width * height * 4;
      const dt = 0.016 * speed;
      const seed = 12345;
      const prevLastObs = lastObsRef.current;
      const kernelSnapshot = kernelEventRef.current;
      const operatorKernel = kernelSnapshot.spec;

      const cpuState = createKuramotoState(width, height);
      const cpuBuffer = new ArrayBuffer(derivedBufferSize(width, height));
      const cpuDerived = createDerivedViews(cpuBuffer, width, height);
      const cpuRand = createNormalGenerator(seed);
      initKuramotoState(cpuState, qInit, cpuDerived);

      const baselineFrames: Uint8ClampedArray[] = [];
      for (let i = 0; i < frameCount; i++) {
        stepKuramotoState(cpuState, params, dt, cpuRand, dt * (i + 1), {
          kernel: operatorKernel,
          controls: { dmt },
        });
        deriveKuramotoFieldsCore(cpuState, cpuDerived, {
          kernel: operatorKernel,
          controls: { dmt },
        });
        const buffer = new Uint8ClampedArray(total);
        const tSeconds = i * (1 / 60);
        renderFrameCore(buffer, tSeconds, false, cpuDerived);
        baselineFrames.push(buffer);
      }

      let simBuffers: ArrayBuffer[];
      try {
        simBuffers = await new Promise<ArrayBuffer[]>((resolve, reject) => {
          const worker = new Worker(new URL('./kuramotoWorker.ts', import.meta.url), {
            type: 'module',
          });
          worker.onmessage = (event: MessageEvent<WorkerIncomingMessage>) => {
            const msg = event.data;
            if (msg.kind === 'simulateResult') {
              worker.terminate();
              resolve(msg.buffers);
            } else if (msg.kind === 'log') {
              console.log(msg.message);
            }
          };
          worker.onerror = (error) => {
            worker.terminate();
            reject(error);
          };
          worker.postMessage({
            kind: 'simulate',
            frameCount,
            dt,
            params,
            width,
            height,
            qInit,
            seed,
          });
        });
      } catch (error) {
        console.error('[regression] worker simulation failed', error);
        lastObsRef.current = prevLastObs;
        return { maxDelta: Infinity, perFrameMax: [] as number[] };
      }

      const perFrameMax: number[] = [];
      let maxDelta = 0;
      for (let i = 0; i < frameCount; i++) {
        const derived = createDerivedViews(simBuffers[i], width, height);
        const buffer = new Uint8ClampedArray(total);
        const tSeconds = i * (1 / 60);
        renderFrameCore(buffer, tSeconds, false, derived);
        const baseline = baselineFrames[i];
        let frameMax = 0;
        for (let j = 0; j < total; j++) {
          const delta = Math.abs(buffer[j] - baseline[j]) / 255;
          if (delta > frameMax) {
            frameMax = delta;
          }
        }
        perFrameMax.push(frameMax);
        if (frameMax > maxDelta) {
          maxDelta = frameMax;
        }
      }

      lastObsRef.current = prevLastObs;
      console.log(`[regression] compared ${frameCount} frames, max normalized delta ${maxDelta}`);
      console.assert(maxDelta < 1e-6, `[regression] expected <=1e-6 delta, saw max ${maxDelta}`);
      return { maxDelta, perFrameMax };
    },
    [
      rimFieldRef,
      surfaceFieldRef,
      kurEnabled,
      width,
      height,
      renderFrameCore,
      getKurParams,
      speed,
      qInit,
      dmt,
    ],
  );

  const runGpuParityCheck = useCallback(async () => {
    if (!rimFieldRef.current || !basePixelsRef.current || !surfaceFieldRef.current) {
      console.warn('[gpu-regression] base pixels, surface, or rim data unavailable.');
      return null;
    }
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      premultipliedAlpha: false,
      antialias: false,
      preserveDrawingBuffer: true,
    });
    if (!gl) {
      console.warn('[gpu-regression] WebGL2 unavailable for parity check.');
      return null;
    }
    const renderer = createGpuRenderer(gl);
    renderer.resize(width, height);
    renderer.uploadBase(basePixelsRef.current);
    renderer.uploadRim(rimFieldRef.current);
    const phaseSource = cpuDerivedRef.current;
    const phaseField =
      phaseSource &&
      gradXRef.current &&
      gradYRef.current &&
      vortRef.current &&
      cohRef.current &&
      ampRef.current
        ? {
            kind: 'phase' as const,
            resolution: phaseSource.resolution,
            gradX: gradXRef.current,
            gradY: gradYRef.current,
            vort: vortRef.current,
            coh: cohRef.current,
            amp: ampRef.current,
          }
        : null;
    if (phaseField) {
      assertPhaseField(phaseField, 'gpu:parity-phase');
    }
    renderer.uploadPhase(phaseField);

    const state = { gl, renderer };
    const total = width * height * 4;
    const cpuBuffer = new Uint8ClampedArray(total);
    const gpuBuffer = new Uint8Array(total);
    const pixelCount = width * height;
    const scenes = [
      { label: 'scene-A', time: 0.0 },
      { label: 'scene-B', time: 0.37 },
      { label: 'scene-C', time: 0.73 },
    ];
    const results: ParitySceneSummary[] = [];
    const prevLastObs = lastObsRef.current;

    try {
      for (const scene of scenes) {
        const phaseUpload =
          phaseSource &&
          gradXRef.current &&
          gradYRef.current &&
          vortRef.current &&
          cohRef.current &&
          ampRef.current
            ? {
                kind: 'phase' as const,
                resolution: phaseSource.resolution,
                gradX: gradXRef.current,
                gradY: gradYRef.current,
                vort: vortRef.current,
                coh: cohRef.current,
                amp: ampRef.current,
              }
            : null;
        if (phaseUpload) {
          assertPhaseField(phaseUpload, 'gpu:parity-phase-loop');
        }
        renderer.uploadPhase(phaseUpload);
        renderFrameCore(cpuBuffer, scene.time, false);
        drawFrameGpu(state, scene.time, false);
        gl.finish();
        renderer.readPixels(gpuBuffer);
        let mismatched = 0;
        let maxDelta = 0;
        let worstIndex = -1;
        let worstCpu: [number, number, number] = [0, 0, 0];
        let worstGpu: [number, number, number] = [0, 0, 0];
        for (let i = 0; i < pixelCount; i++) {
          const idx = i * 4;
          const dr = Math.abs(cpuBuffer[idx] - gpuBuffer[idx]);
          const dg = Math.abs(cpuBuffer[idx + 1] - gpuBuffer[idx + 1]);
          const db = Math.abs(cpuBuffer[idx + 2] - gpuBuffer[idx + 2]);
          const delta = Math.max(dr, dg, db);
          if (delta > maxDelta) {
            maxDelta = delta;
            worstIndex = i;
            worstCpu = [cpuBuffer[idx], cpuBuffer[idx + 1], cpuBuffer[idx + 2]];
            worstGpu = [gpuBuffer[idx], gpuBuffer[idx + 1], gpuBuffer[idx + 2]];
          }
          if (delta > 1) mismatched++;
        }
        const coord =
          worstIndex >= 0
            ? ([worstIndex % width, Math.floor(worstIndex / width)] as [number, number])
            : ([0, 0] as [number, number]);
        if (maxDelta > 1) {
          console.warn(
            `[gpu-regression] ${scene.label} worst Δ${maxDelta.toFixed(
              2,
            )} at (${coord[0]},${coord[1]}) CPU ${printRgb(worstCpu)} GPU ${printRgb(worstGpu)}`,
          );
        }
        results.push({
          label: scene.label,
          mismatched,
          percent: (mismatched / pixelCount) * 100,
          maxDelta,
          maxCoord: coord,
          cpuColor: worstCpu,
          gpuColor: worstGpu,
        });
      }
    } finally {
      lastObsRef.current = prevLastObs;
      renderer.dispose();
    }

    return {
      scenes: results,
      tolerancePercent: 0.5,
    };
  }, [
    rimFieldRef,
    surfaceFieldRef,
    basePixelsRef,
    width,
    height,
    gradXRef,
    gradYRef,
    vortRef,
    cohRef,
    renderFrameCore,
    drawFrameGpu,
  ]);

  const measureRenderPerformance = useCallback(
    (frameCount = 60) => {
      if (!rimFieldRef.current || !basePixelsRef.current || !surfaceFieldRef.current) {
        console.warn('[perf] surface or rim field unavailable.');
        return null;
      }
      const prevLastObs = lastObsRef.current;
      const total = width * height * 4;
      const cpuBuffer = new Uint8ClampedArray(total);
      const cpuStart = performance.now();
      for (let i = 0; i < frameCount; i++) {
        const t = i * (1 / 60);
        renderFrameCore(cpuBuffer, t, false);
      }
      const cpuDuration = performance.now() - cpuStart;
      const cpuMs = cpuDuration / frameCount;
      const cpuFps = cpuMs > 0 ? 1000 / cpuMs : Infinity;

      const state = ensureGpuRenderer();
      if (!state) {
        console.warn('[perf] GPU renderer unavailable.');
        lastObsRef.current = prevLastObs;
        return null;
      }
      const gpuStart = performance.now();
      for (let i = 0; i < frameCount; i++) {
        const t = i * (1 / 60);
        drawFrameGpu(state, t, false);
      }
      state.gl.finish();
      const gpuDuration = performance.now() - gpuStart;
      const gpuMs = gpuDuration / frameCount;
      const gpuFps = gpuMs > 0 ? 1000 / gpuMs : Infinity;
      lastObsRef.current = prevLastObs;
      return {
        frameCount,
        cpuMs,
        gpuMs,
        cpuFps,
        gpuFps,
        throughputGain: cpuMs / gpuMs,
      };
    },
    [
      rimFieldRef,
      surfaceFieldRef,
      basePixelsRef,
      width,
      height,
      renderFrameCore,
      ensureGpuRenderer,
      drawFrameGpu,
    ],
  );

  const handleParityCheck = useCallback(async () => {
    const result = await runGpuParityCheck();
    if (!result) {
      setLastParityResult(null);
      return;
    }
    setLastParityResult({
      ...result,
      timestamp: Date.now(),
    });
  }, [runGpuParityCheck]);

  const handlePerfProbe = useCallback(
    (frameCount = 120) => {
      const snapshot = measureRenderPerformance(frameCount);
      if (!snapshot) {
        setLastPerfResult(null);
        return;
      }
      setLastPerfResult({
        ...snapshot,
        timestamp: Date.now(),
      });
    },
    [measureRenderPerformance],
  );

  const handleRendererToggle = useCallback(
    (useGpu: boolean) => {
      if (!useGpu && gpuStateRef.current) {
        gpuStateRef.current.renderer.dispose();
        gpuStateRef.current = null;
      }
      if (useGpu) {
        pendingStaticUploadRef.current = true;
      }
      setRenderBackend(useGpu ? 'gpu' : 'cpu');
    },
    [setRenderBackend],
  );

  const parityDisplay = useMemo(() => {
    if (!lastParityResult || lastParityResult.scenes.length === 0) return null;
    const worst = lastParityResult.scenes.reduce(
      (max, scene) => (scene.percent > max.percent ? scene : max),
      lastParityResult.scenes[0],
    );
    const within = worst.percent <= lastParityResult.tolerancePercent;
    return { worst, within, tolerance: lastParityResult.tolerancePercent };
  }, [lastParityResult]);

  const perfDisplay = useMemo(() => {
    if (!lastPerfResult) return null;
    return {
      cpuMs: lastPerfResult.cpuMs,
      gpuMs: lastPerfResult.gpuMs,
      cpuFps: lastPerfResult.cpuFps,
      gpuFps: lastPerfResult.gpuFps,
      throughputGain: lastPerfResult.throughputGain,
    };
  }, [lastPerfResult]);

  const availableCaptureFormats = captureSupport.supported;
  const recordingFormat = useMemo(() => {
    if (!captureSupport.checked || !recordingFormatId) return null;
    return availableCaptureFormats.find((candidate) => candidate.id === recordingFormatId) ?? null;
  }, [captureSupport.checked, recordingFormatId, availableCaptureFormats]);

  const isRecording = recordingStatus === 'recording';
  const isFinalizing = recordingStatus === 'finalizing';
  const mp4Unsupported =
    captureSupport.checked && !availableCaptureFormats.some((format) => format.container === 'mp4');
  const captureReady = captureSupport.checked && !!recordingFormat;
  const captureButtonDisabled = isFinalizing || (!isRecording && !captureReady);
  const recordingBitrateMbps = useMemo(
    () => (recordingBitrate / 1_000_000).toFixed(1),
    [recordingBitrate],
  );

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mediaRecorderCtor = (
      window as typeof window & {
        MediaRecorder?: typeof MediaRecorder;
      }
    ).MediaRecorder;
    if (!mediaRecorderCtor || typeof mediaRecorderCtor.isTypeSupported !== 'function') {
      setCaptureSupport({ checked: true, supported: [] });
      setRecordingFormatId(null);
      return;
    }
    const supported = CAPTURE_FORMAT_CANDIDATES.filter((candidate) => {
      try {
        return mediaRecorderCtor.isTypeSupported(candidate.mimeType);
      } catch {
        return false;
      }
    });
    setCaptureSupport({ checked: true, supported });
    setRecordingFormatId((prev) => {
      if (prev && supported.some((candidate) => candidate.id === prev)) {
        return prev;
      }
      return supported.length > 0 ? supported[0].id : null;
    });
  }, []);

  const startCapture = useCallback(() => {
    setRecordingError(null);
    if (recordingStatus !== 'idle') {
      return;
    }
    if (!captureSupport.checked) {
      setRecordingError('Detecting capture support, please try again in a moment.');
      return;
    }
    if (!recordingFormat) {
      if (availableCaptureFormats.length === 0) {
        setRecordingError('MediaRecorder is unavailable in this browser.');
      } else {
        const fallbackFormat = availableCaptureFormats[0];
        setRecordingFormatId(fallbackFormat.id);
        setRecordingError(
          `Selected recording format is no longer available. Switched to ${fallbackFormat.label}.`,
        );
      }
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) {
      setRecordingError('Canvas is not ready yet.');
      return;
    }
    if (typeof canvas.captureStream !== 'function') {
      setRecordingError('canvas.captureStream() is not supported in this environment.');
      return;
    }
    const mediaRecorderCtor = (
      window as typeof window & {
        MediaRecorder?: typeof MediaRecorder;
      }
    ).MediaRecorder;
    if (!mediaRecorderCtor) {
      setRecordingError('MediaRecorder constructor is missing on window.');
      return;
    }
    if (recordedUrlRef.current) {
      URL.revokeObjectURL(recordedUrlRef.current);
      recordedUrlRef.current = null;
    }
    setRecordingDownload(null);
    recordingChunksRef.current = [];
    recordingMimeTypeRef.current = recordingFormat.mimeType;
    recordingFormatRef.current = recordingFormat.id;
    try {
      const stream = canvas.captureStream(RECORDING_FPS);
      captureStreamRef.current = stream;
      const track = stream.getVideoTracks()[0];
      if (track && typeof track.applyConstraints === 'function') {
        const constraints: MediaTrackConstraints = {
          frameRate: { ideal: RECORDING_FPS, max: RECORDING_FPS },
          width: { ideal: canvas.width, min: canvas.width, max: canvas.width },
          height: { ideal: canvas.height, min: canvas.height, max: canvas.height },
        };
        track.applyConstraints(constraints).catch(() => {
          /* ignore */
        });
      }
      const recorder = new mediaRecorderCtor(stream, {
        mimeType: recordingFormat.mimeType,
        videoBitsPerSecond: recordingBitrate,
        bitsPerSecond: recordingBitrate,
      });
      recorderRef.current = recorder;
      setRecordingStatus('recording');
      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          recordingChunksRef.current.push(event.data);
        }
      };
      recorder.onerror = (event) => {
        const err = (event as { error?: DOMException }).error;
        const message = err?.message ?? 'Recorder error.';
        recordingChunksRef.current = [];
        recordingMimeTypeRef.current = null;
        stopCaptureStream();
        recorderRef.current = null;
        setRecordingStatus('idle');
        setRecordingError(message);
      };
      recorder.onstop = () => {
        const chunks = recordingChunksRef.current;
        recordingChunksRef.current = [];
        const lastFormatId = recordingFormatRef.current;
        const lastFormat = lastFormatId ? CAPTURE_FORMAT_BY_ID[lastFormatId] : null;
        const mimeType =
          recordingMimeTypeRef.current ??
          lastFormat?.mimeType ??
          (recordingFormat ? recordingFormat.mimeType : availableCaptureFormats[0]?.mimeType) ??
          'video/webm';
        recordingMimeTypeRef.current = null;
        stopCaptureStream();
        recorderRef.current = null;
        if (!chunks.length) {
          const activePreset = recordingPresetRef.current;
          const presetFallback = activePreset ? RECORDING_FALLBACK_PRESET[activePreset] : null;
          const activeFormat = recordingFormatRef.current;
          let formatFallback: CaptureFormatId | null = null;
          if (availableCaptureFormats.length > 0) {
            if (activeFormat) {
              const idx = availableCaptureFormats.findIndex((fmt) => fmt.id === activeFormat);
              const next = idx >= 0 ? availableCaptureFormats[idx + 1] : null;
              formatFallback = next ? next.id : null;
            } else {
              formatFallback = availableCaptureFormats[0]?.id ?? null;
            }
            if (formatFallback === activeFormat) {
              formatFallback = null;
            }
          }
          setRecordingStatus('idle');
          if (presetFallback) {
            setRecordingPreset(presetFallback);
          }
          if (formatFallback) {
            setRecordingFormatId(formatFallback);
          }
          const messages: string[] = [];
          if (presetFallback && RECORDING_PRESETS[presetFallback]) {
            messages.push(`Switched preset to ${RECORDING_PRESETS[presetFallback].label}.`);
          }
          if (formatFallback && CAPTURE_FORMAT_BY_ID[formatFallback]) {
            messages.push(`Switched format to ${CAPTURE_FORMAT_BY_ID[formatFallback].label}.`);
          }
          if (messages.length === 0) {
            messages.push('Try a lower recording preset or export WebM and re-encode.');
          }
          setRecordingError(`Recorder produced no data. ${messages.join(' ')}`);
          return;
        }
        const blob = new Blob(chunks, { type: mimeType });
        const safeDate = new Date().toISOString().replace(/[:.]/g, '-');
        const extension = mimeType.includes('mp4')
          ? 'mp4'
          : mimeType.includes('webm')
            ? 'webm'
            : 'mp4';
        const filename = `rainbow-perimeter-${safeDate}.${extension}`;
        if (recordedUrlRef.current) {
          URL.revokeObjectURL(recordedUrlRef.current);
        }
        const url = URL.createObjectURL(blob);
        recordedUrlRef.current = url;
        setRecordingDownload({
          url,
          size: blob.size,
          mimeType,
          filename,
        });
        setRecordingStatus('idle');
      };
      recorder.start();
    } catch (error) {
      recordingChunksRef.current = [];
      recordingMimeTypeRef.current = null;
      stopCaptureStream();
      recorderRef.current = null;
      setRecordingStatus('idle');
      setRecordingError(error instanceof Error ? error.message : String(error));
    }
  }, [
    availableCaptureFormats,
    canvasRef,
    captureSupport.checked,
    recordingBitrate,
    recordingFormat,
    recordingStatus,
    setRecordingDownload,
    setRecordingFormatId,
    setRecordingPreset,
    stopCaptureStream,
  ]);

  const stopCapture = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder) {
      return;
    }
    if (recorder.state === 'inactive') {
      stopCaptureStream();
      recorderRef.current = null;
      return;
    }
    setRecordingStatus('finalizing');
    try {
      recorder.stop();
    } catch (error) {
      stopCaptureStream();
      recorderRef.current = null;
      recordingChunksRef.current = [];
      recordingMimeTypeRef.current = null;
      setRecordingStatus('idle');
      setRecordingError(error instanceof Error ? error.message : String(error));
    }
  }, [stopCaptureStream]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const w = window as any;
    w.__setFrameProfiler = setFrameProfiler;
    w.__runFrameRegression = runRegressionHarness;
    w.__runGpuParityCheck = runGpuParityCheck;
    w.__measureRenderPerformance = measureRenderPerformance;
    const setBackend = (backend: 'gpu' | 'cpu') => {
      handleRendererToggle(backend === 'gpu');
    };
    w.__setRenderBackend = setBackend;
    w.__setTelemetryEnabled = (enabled: boolean) => {
      setTelemetryEnabled(Boolean(enabled));
    };
    w.__getTelemetryHistory = () => [...telemetryRef.current.history];
    return () => {
      if (w.__setFrameProfiler === setFrameProfiler) {
        delete w.__setFrameProfiler;
      }
      if (w.__runFrameRegression === runRegressionHarness) {
        delete w.__runFrameRegression;
      }
      if (w.__runGpuParityCheck === runGpuParityCheck) {
        delete w.__runGpuParityCheck;
      }
      if (w.__measureRenderPerformance === measureRenderPerformance) {
        delete w.__measureRenderPerformance;
      }
      if (w.__setRenderBackend === setBackend) {
        delete w.__setRenderBackend;
      }
      if (w.__setTelemetryEnabled === setTelemetryEnabled) {
        delete w.__setTelemetryEnabled;
      }
      if (w.__getTelemetryHistory) {
        delete w.__getTelemetryHistory;
      }
    };
  }, [
    setFrameProfiler,
    runRegressionHarness,
    runGpuParityCheck,
    measureRenderPerformance,
    handleRendererToggle,
    setTelemetryEnabled,
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let anim = true;
    let frameId = 0;
    const start = performance.now();

    if (renderBackend === 'gpu') {
      const render = () => {
        if (!anim) return;
        const state = ensureGpuRenderer();
        if (!state) {
          anim = false;
          return;
        }
        const frameStart = telemetryRef.current.enabled ? performance.now() : 0;
        const t = (performance.now() - start) * 0.001;
        const dt = 0.016 * speed;
        advanceKuramoto(dt, t);
        advanceVolume(dt);
        drawFrameGpu(state, t, true);
        if (frameStart) {
          recordTelemetry('frame', performance.now() - frameStart);
        }
        frameId = requestAnimationFrame(render);
      };
      render();
      return () => {
        anim = false;
        cancelAnimationFrame(frameId);
      };
    }

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;
    canvas.width = width;
    canvas.height = height;
    const renderCpu = () => {
      if (!anim) return;
      const t = (performance.now() - start) * 0.001;
      drawFrameCpu(ctx, t);
      frameId = requestAnimationFrame(renderCpu);
    };
    renderCpu();
    return () => {
      anim = false;
      cancelAnimationFrame(frameId);
    };
  }, [
    renderBackend,
    ensureGpuRenderer,
    drawFrameGpu,
    drawFrameCpu,
    advanceKuramoto,
    advanceVolume,
    speed,
    width,
    height,
    recordTelemetry,
  ]);

  useEffect(() => {
    return () => {
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== 'inactive') {
        try {
          recorder.stop();
        } catch {
          // ignore
        }
      }
      recorderRef.current = null;
      recordingChunksRef.current = [];
      recordingMimeTypeRef.current = null;
      stopCaptureStream();
      if (recordedUrlRef.current) {
        URL.revokeObjectURL(recordedUrlRef.current);
        recordedUrlRef.current = null;
      }
    };
  }, [stopCaptureStream]);

  const uploadImageBlob = useCallback(async (blob: Blob, originalName: string) => {
    if (typeof window === 'undefined') {
      throw new Error('Uploads require a browser environment.');
    }
    const arrayBuffer = await blob.arrayBuffer();
    const base64 = arrayBufferToBase64(arrayBuffer);
    const response = await fetch('/api/upload-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: originalName,
        type: blob.type,
        data: base64,
      }),
    });
    if (!response.ok) {
      throw new Error(`Upload failed (${response.status})`);
    }
    const payload = (await response.json()) as { path?: string };
    if (!payload?.path) {
      throw new Error('Upload response missing path');
    }
    return payload.path;
  }, []);

  const ingestImageData = useCallback(
    (image: ImageData, source: string) => {
      const newW = image.width;
      const newH = image.height;
      setWidth(newW);
      setHeight(newH);
      basePixelsRef.current = image;
      resetTracerState();
      const surfaceField = describeImageData(image);
      assertSurfaceField(surfaceField, 'io:surface');
      surfaceFieldRef.current = surfaceField;
      markFieldFresh('surface', surfaceField.resolution, source);

      const rimField = computeEdgeField({
        data: image.data,
        width: newW,
        height: newH,
      });
      assertRimField(rimField, 'io:rim');
      rimFieldRef.current = rimField;
      markFieldFresh('rim', rimField.resolution, source);
      pendingStaticUploadRef.current = true;
      const gpuState = gpuStateRef.current;
      if (gpuState) {
        gpuState.renderer.resize(newW, newH);
      }
      refreshGpuStaticTextures();
      normTargetRef.current = 0.6;
      lastObsRef.current = 0.6;
    },
    [markFieldFresh, refreshGpuStaticTextures, resetTracerState],
  );

  const loadImageAsset = useCallback(
    async (media: PresetMedia | null | undefined) => {
      if (!media?.imagePath) {
        setImageAsset(null);
        return;
      }
      try {
        const response = await fetch(media.imagePath);
        if (!response.ok) {
          throw new Error(`Fetch failed (${response.status})`);
        }
        const blob = await response.blob();
        const bitmap = await createImageBitmap(blob);
        const off = document.createElement('canvas');
        off.width = bitmap.width;
        off.height = bitmap.height;
        const octx = off.getContext('2d', { willReadFrequently: true });
        if (!octx) return;
        octx.drawImage(bitmap, 0, 0);
        const imageData = octx.getImageData(0, 0, bitmap.width, bitmap.height);
        ingestImageData(imageData, 'io:media');
        setImgBitmap((prev) => {
          if (prev && 'close' in prev) {
            try {
              (prev as ImageBitmap).close();
            } catch {
              // ignore close failure
            }
          }
          return bitmap;
        });
        setImageAsset({
          path: media.imagePath,
          name: media.imageName ?? null,
          width: bitmap.width,
          height: bitmap.height,
          mimeType: blob.type ?? null,
        });
        setIncludeImageInPreset(true);
      } catch (error) {
        console.error('[media] failed to load image asset', error);
        setImageAsset(null);
        setIncludeImageInPreset(false);
      }
    },
    [ingestImageData],
  );

  const loadSyntheticCase = useCallback(
    (caseId: SyntheticCaseId) => {
      const synthetic = SYNTHETIC_CASES.find((entry) => entry.id === caseId);
      if (!synthetic) {
        console.warn(`[synthetic] unknown case ${caseId}`);
        return;
      }
      setSelectedSyntheticCase(caseId);
      setImageAsset(null);
      setIncludeImageInPreset(false);
      const { width: defaultW, height: defaultH } = DEFAULT_SYNTHETIC_SIZE;
      const image = synthetic.generate(defaultW, defaultH);
      ingestImageData(image, `dev:${caseId}`);
      if (kurEnabled) {
        deriveKurFieldsCpu();
      }
      if (
        !metricsScratchRef.current ||
        metricsScratchRef.current.length !== defaultW * defaultH * 4
      ) {
        metricsScratchRef.current = new Uint8ClampedArray(defaultW * defaultH * 4);
      }
      const scratch = metricsScratchRef.current;
      const metrics = renderFrameCore(scratch, 0, false);
      if (metrics) {
        setSyntheticBaselines((prev) => ({
          ...prev,
          [caseId]: {
            metrics,
            timestamp: Date.now(),
          },
        }));
        const rimEnergy = metrics.composer.fields.rim.energy.toFixed(3);
        const surfaceEnergy = metrics.composer.fields.surface.energy.toFixed(3);
        const cohMean = metrics.gradient.cohMean.toFixed(3);
        console.log(
          `[synthetic] ${synthetic.label} rim=${rimEnergy} surface=${surfaceEnergy} |Z|=${cohMean}`,
        );
      }
    },
    [deriveKurFieldsCpu, ingestImageData, kurEnabled, renderFrameCore, setSelectedSyntheticCase],
  );

  const exportCouplingDiff = useCallback(
    (branch: 'rimToSurface' | 'surfaceToRim') => {
      if (typeof document === 'undefined') {
        console.warn('[coupling-diff] document unavailable for export');
        return;
      }
      if (!basePixelsRef.current || !rimFieldRef.current) {
        console.warn('[coupling-diff] base image or rim field not ready');
        return;
      }
      const total = width * height * 4;
      const currentBuffer = new Uint8ClampedArray(total);
      const toggledBuffer = new Uint8ClampedArray(total);
      const override: CouplingToggleState = {
        rimToSurface: branch === 'rimToSurface' ? false : couplingToggles.rimToSurface,
        surfaceToRim: branch === 'surfaceToRim' ? false : couplingToggles.surfaceToRim,
      };
      const baselineMetrics = renderFrameCore(currentBuffer, 0, false);
      const toggledMetrics = renderFrameCore(toggledBuffer, 0, false, undefined, {
        toggles: override,
      });
      if (!baselineMetrics || !toggledMetrics) {
        console.warn('[coupling-diff] unable to compute frame metrics');
        return;
      }
      const diff = new Uint8ClampedArray(total);
      let maxDelta = 0;
      for (let i = 0; i < total; i += 4) {
        const dr = Math.abs(currentBuffer[i] - toggledBuffer[i]);
        const dg = Math.abs(currentBuffer[i + 1] - toggledBuffer[i + 1]);
        const db = Math.abs(currentBuffer[i + 2] - toggledBuffer[i + 2]);
        const delta = Math.max(dr, dg, db);
        const amplified = Math.min(255, delta * 4);
        diff[i + 0] = amplified;
        diff[i + 1] = amplified;
        diff[i + 2] = amplified;
        diff[i + 3] = 255;
        if (delta > maxDelta) {
          maxDelta = delta;
        }
      }
      if (maxDelta <= 0) {
        console.warn(`[coupling-diff] delta is empty for ${branch}`);
        return;
      }
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        console.warn('[coupling-diff] failed to create canvas context');
        return;
      }
      const diffImage = new ImageData(diff, width, height);
      ctx.putImageData(diffImage, 0, 0);
      const url = canvas.toDataURL('image/png');
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `coupling-diff-${branch}-${timestamp}.png`;
      anchor.click();
      console.log(
        `[coupling-diff] ${branch} maxΔ=${maxDelta.toFixed(1)} rim=${baselineMetrics.composer.fields.rim.energy.toFixed(3)}→${toggledMetrics.composer.fields.rim.energy.toFixed(3)} surface=${baselineMetrics.composer.fields.surface.energy.toFixed(3)}→${toggledMetrics.composer.fields.surface.energy.toFixed(3)}`,
      );
    },
    [couplingToggles, renderFrameCore, width, height],
  );

  const runSyntheticDeck = useCallback(() => {
    SYNTHETIC_CASES.forEach((entry) => loadSyntheticCase(entry.id));
  }, [loadSyntheticCase]);

  const onFile = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      try {
        const originalBitmap = await createImageBitmap(file);
        const maxDim = 1000;
        const scale = Math.min(1, maxDim / Math.max(originalBitmap.width, originalBitmap.height));
        const newW = Math.max(1, Math.round(originalBitmap.width * scale));
        const newH = Math.max(1, Math.round(originalBitmap.height * scale));
        const off = document.createElement('canvas');
        off.width = newW;
        off.height = newH;
        const octx = off.getContext('2d', { willReadFrequently: true });
        if (!octx) return;
        octx.drawImage(originalBitmap, 0, 0, newW, newH);
        try {
          originalBitmap.close();
        } catch {
          // ignore close failure
        }
        const scaledBitmap = await createImageBitmap(off);
        setImgBitmap((prev) => {
          if (prev && 'close' in prev) {
            try {
              (prev as ImageBitmap).close();
            } catch {
              // ignore close failure
            }
          }
          return scaledBitmap;
        });
        const img = octx.getImageData(0, 0, newW, newH);
        ingestImageData(img, 'io:file');
        const blob = await new Promise<Blob | null>((resolve) =>
          off.toBlob((value) => resolve(value), 'image/png', 0.95),
        );
        if (blob) {
          try {
            const storedPath = await uploadImageBlob(blob, file.name);
            setImageAsset({
              path: storedPath,
              name: file.name,
              width: newW,
              height: newH,
              mimeType: blob.type ?? 'image/png',
            });
            setIncludeImageInPreset(true);
          } catch (error) {
            console.error('[upload] failed to persist image', error);
            setImageAsset(null);
            setIncludeImageInPreset(false);
          }
        } else {
          setImageAsset(null);
          setIncludeImageInPreset(false);
        }
      } catch (error) {
        console.error('[file] failed to process image', error);
        setImageAsset(null);
        setIncludeImageInPreset(false);
      }
    },
    [ingestImageData, uploadImageBlob],
  );

  const applyPreset = useCallback(
    (preset: Preset) => {
      pendingStaticUploadRef.current = true;

      const { params, surface, display, runtime, kuramoto, developer } = preset;

      handleRendererToggle(runtime.renderBackend === 'gpu');
      setVolumeEnabled(runtime.volumeEnabled);
      setRimEnabled(runtime.rimEnabled);
      setShowRimDebug(runtime.showRimDebug);
      setShowHyperbolicGrid(runtime.showHyperbolicGrid);
      setShowHyperbolicGuide(runtime.showHyperbolicGuide);
      setShowSurfaceDebug(runtime.showSurfaceDebug);
      setShowPhaseDebug(runtime.showPhaseDebug);
      setPhaseHeatmapEnabled(runtime.phaseHeatmapEnabled);
      setTelemetryEnabled(runtime.telemetryEnabled);
      setTelemetryOverlayEnabled(runtime.telemetryOverlayEnabled);
      setFrameLoggingEnabled(runtime.frameLoggingEnabled);

      setEdgeThreshold(params.edgeThreshold);
      setBlend(params.blend);
      setKernel(createKernelSpec(params.kernel));
      setDmt(params.dmt);
      setArousal(Math.min(Math.max(params.arousal ?? 0, 0), 1));
      setThetaMode(params.thetaMode);
      setThetaGlobal(params.thetaGlobal);
      setBeta2(params.beta2);
      setJitter(params.jitter);
      setSigma(params.sigma);
      setMicrosaccade(params.microsaccade);
      setSpeed(params.speed);
      setContrast(params.contrast);
      setPhasePin(params.phasePin);
      setAlive(params.alive);
      setRimAlpha(params.rimAlpha);
      const importedStrength = clampSu7ScheduleStrength(params.su7ScheduleStrength);
      const presetId = isSu7PresetId(params.su7Preset) ? params.su7Preset : null;
      const baseSchedule = presetId
        ? cloneSu7Schedule(SU7_PRESET_DEFINITIONS[presetId].schedule)
        : deriveSu7BaseSchedule(params.su7Schedule, importedStrength);
      su7BaseScheduleRef.current = baseSchedule;
      setSu7ScheduleStrength(importedStrength);
      const projectorWithDefault =
        presetId && SU7_PRESET_DEFINITIONS[presetId].projectorWeight != null
          ? {
              ...params.su7Projector,
              weight:
                params.su7Projector.weight ?? SU7_PRESET_DEFINITIONS[presetId].projectorWeight,
            }
          : params.su7Projector;
      setSu7Params(
        cloneSu7RuntimeParams({
          enabled: params.su7Enabled,
          gain: params.su7Gain,
          preset: params.su7Preset,
          seed: params.su7Seed,
          schedule: params.su7Schedule,
          projector: projectorWithDefault,
        }),
      );

      setSurfEnabled(surface.surfEnabled);
      setSurfaceBlend(surface.surfaceBlend);
      setWarpAmp(surface.warpAmp);
      setNOrient(surface.nOrient);
      setWallGroup(surface.wallGroup);
      setSurfaceRegion(surface.surfaceRegion);

      setDisplayMode(display.displayMode);
      setPolBins(display.polBins);
      setNormPin(display.normPin);
      setCurvatureStrength(clamp(display.curvatureStrength ?? 0, 0, MAX_CURVATURE_STRENGTH));
      setCurvatureMode(display.curvatureMode ?? 'poincare');
      setHyperbolicGuideSpacing(
        clamp(
          display.hyperbolicGuideSpacing ?? DEFAULT_HYPERBOLIC_GUIDE_SPACING,
          HYPERBOLIC_GUIDE_SPACING_MIN,
          HYPERBOLIC_GUIDE_SPACING_MAX,
        ),
      );
      setTracerConfig(cloneTracerConfig(preset.tracer ?? DEFAULT_TRACER_CONFIG));
      resetTracerState();

      setKurEnabled(kuramoto.kurEnabled);
      setKurSync(kuramoto.kurSync);
      setKurRegime(kuramoto.kurRegime);
      setK0(kuramoto.K0);
      setAlphaKur(kuramoto.alphaKur);
      setGammaKur(kuramoto.gammaKur);
      setOmega0(kuramoto.omega0);
      setEpsKur(kuramoto.epsKur);
      setFluxX(kuramoto.fluxX);
      setFluxY(kuramoto.fluxY);
      setQInit(kuramoto.qInit);
      setSmallWorldEnabled(kuramoto.smallWorldEnabled);
      setSmallWorldWeight(kuramoto.smallWorldWeight);
      setPSw(kuramoto.p_sw);
      setSmallWorldSeed(kuramoto.smallWorldSeed);
      setSmallWorldDegree(kuramoto.smallWorldDegree);

      setSelectedSyntheticCase(developer.selectedSyntheticCase);
      setIncludeImageInPreset(Boolean(preset.media?.imagePath));
      if (preset.media?.imagePath) {
        void loadImageAsset(preset.media);
      } else {
        setImageAsset(null);
      }

      setCoupling(cloneCouplingConfig(preset.coupling));
      setComposer(cloneComposerConfig(preset.composer));
      setCouplingToggles(cloneCouplingToggles(preset.couplingToggles));
    },
    [handleRendererToggle, loadImageAsset, resetTracerState, setSu7ScheduleStrength, setSu7Params],
  );

  const buildCurrentPreset = useCallback((): Preset => {
    const su7Snapshot = cloneSu7RuntimeParams(su7Params);
    return {
      name: PRESETS[presetIndex]?.name ?? 'Custom snapshot',
      params: {
        edgeThreshold,
        blend,
        kernel: kernelSpecToJSON(kernel),
        dmt,
        arousal,
        thetaMode,
        thetaGlobal,
        beta2,
        jitter,
        sigma,
        microsaccade,
        speed,
        contrast,
        phasePin,
        alive,
        rimAlpha,
        su7Enabled: su7Snapshot.enabled,
        su7Gain: su7Snapshot.gain,
        su7Preset: su7Snapshot.preset,
        su7Seed: su7Snapshot.seed,
        su7Schedule: su7Snapshot.schedule,
        su7Projector: su7Snapshot.projector,
        su7ScheduleStrength,
      },
      surface: {
        surfEnabled,
        surfaceBlend,
        warpAmp,
        nOrient,
        wallGroup,
        surfaceRegion,
      },
      display: {
        displayMode,
        polBins,
        normPin,
        curvatureStrength,
        curvatureMode,
        hyperbolicGuideSpacing,
      },
      tracer: cloneTracerConfig(tracerConfig),
      runtime: {
        renderBackend,
        rimEnabled,
        showRimDebug,
        showHyperbolicGrid,
        showHyperbolicGuide,
        showSurfaceDebug,
        showPhaseDebug,
        phaseHeatmapEnabled,
        volumeEnabled,
        telemetryEnabled,
        telemetryOverlayEnabled,
        frameLoggingEnabled,
      },
      kuramoto: {
        kurEnabled,
        kurSync,
        kurRegime,
        K0,
        alphaKur,
        gammaKur,
        omega0,
        epsKur,
        fluxX,
        fluxY,
        qInit,
        smallWorldEnabled,
        smallWorldWeight,
        p_sw: pSw,
        smallWorldSeed,
        smallWorldDegree,
      },
      developer: {
        selectedSyntheticCase,
      },
      media:
        includeImageInPreset && imageAsset?.path
          ? {
              imagePath: imageAsset.path,
              imageName: imageAsset.name ?? null,
            }
          : null,
      coupling: cloneCouplingConfig(coupling),
      composer: cloneComposerConfig(composer),
      couplingToggles: cloneCouplingToggles(couplingToggles),
    };
  }, [
    presetIndex,
    edgeThreshold,
    blend,
    kernel,
    dmt,
    thetaMode,
    thetaGlobal,
    beta2,
    jitter,
    sigma,
    microsaccade,
    speed,
    contrast,
    phasePin,
    alive,
    rimAlpha,
    su7Params,
    surfEnabled,
    surfaceBlend,
    warpAmp,
    nOrient,
    wallGroup,
    surfaceRegion,
    displayMode,
    polBins,
    normPin,
    curvatureStrength,
    curvatureMode,
    hyperbolicGuideSpacing,
    renderBackend,
    rimEnabled,
    showRimDebug,
    showHyperbolicGrid,
    showHyperbolicGuide,
    showSurfaceDebug,
    showPhaseDebug,
    phaseHeatmapEnabled,
    volumeEnabled,
    telemetryEnabled,
    telemetryOverlayEnabled,
    frameLoggingEnabled,
    kurEnabled,
    kurSync,
    kurRegime,
    K0,
    alphaKur,
    gammaKur,
    omega0,
    epsKur,
    fluxX,
    fluxY,
    qInit,
    smallWorldEnabled,
    smallWorldWeight,
    pSw,
    smallWorldSeed,
    smallWorldDegree,
    selectedSyntheticCase,
    includeImageInPreset,
    imageAsset,
    su7ScheduleStrength,
    coupling,
    composer,
    couplingToggles,
  ]);

  const handlePresetExport = useCallback(() => {
    const preset = buildCurrentPreset();
    const payload = { version: 3, preset };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const downloadName = `${preset.name.toLowerCase().replace(/[^a-z0-9]+/g, '-') || 'preset'}.json`;
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = downloadName;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [buildCurrentPreset]);

  const handleCreateShareLink = useCallback(async () => {
    if (typeof window === 'undefined') return;
    try {
      const preset = buildCurrentPreset();
      const payload = { version: 1, preset };
      const encoded = encodeURIComponent(btoa(JSON.stringify(payload)));
      const shareUrl = `${window.location.origin}${window.location.pathname}?share=${encoded}`;
      const nav = typeof navigator !== 'undefined' ? navigator : undefined;
      if (nav?.clipboard?.writeText) {
        await nav.clipboard.writeText(shareUrl);
        setShareStatus({ message: 'Share link copied to clipboard.', url: shareUrl });
      } else {
        setShareStatus({ message: 'Share link ready. Copy manually.', url: shareUrl });
        window.prompt('Share link', shareUrl);
      }
    } catch (error) {
      console.error('[share] failed to create link', error);
      setShareStatus({ message: 'Failed to create share link.' });
    }
  }, [buildCurrentPreset]);

  const handlePresetImport = useCallback(() => {
    const input = presetFileInputRef.current;
    if (!input) return;
    input.value = '';
    input.click();
  }, []);

  const handlePresetImportFile = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = '';
      if (!file) return;
      try {
        const text = await file.text();
        const parsed = JSON.parse(text);
        const payload = parsed?.preset ?? parsed;
        if (!payload?.params) {
          console.error('[preset] invalid payload');
          return;
        }
        const fallback = buildCurrentPreset();
        const importedPreset = sanitizePresetPayload(payload, fallback);
        if (!importedPreset) {
          console.error('[preset] invalid payload');
          return;
        }
        applyPreset(importedPreset);
      } catch (error) {
        console.error('[preset] failed to import', error);
      }
    },
    [applyPreset, buildCurrentPreset],
  );

  const exportFrameSequence = useCallback(async () => {
    if (frameExporting) return;
    const surfaceField = surfaceFieldRef.current;
    const rimField = rimFieldRef.current;
    if (!surfaceField || !rimField) {
      setFrameExportError('Load an image before exporting frames.');
      return;
    }
    if (width <= 0 || height <= 0) {
      setFrameExportError('Canvas size is zero; nothing to export.');
      return;
    }
    const duration = Math.max(0.5, frameExportDuration);
    const totalFrames = Math.max(1, Math.round(duration * RECORDING_FPS));
    const totalPixels = width * height * 4;
    const dt = 1 / RECORDING_FPS;
    frameExportAbortRef.current = { canceled: false };
    setFrameExportError(null);
    setFrameExporting(true);
    setFrameExportProgress({ current: 0, total: totalFrames, message: 'Preparing frame export…' });

    try {
      const kernelSnapshot = kernelEventRef.current;
      const kernelSpec = kernelSnapshot.spec;
      const couplingToggleSnapshot = cloneCouplingToggles(couplingToggles);
      const renderOptionsBase: RenderFrameOptions = {
        toggles: couplingToggleSnapshot,
      };

      const offscreen = document.createElement('canvas');
      offscreen.width = width;
      offscreen.height = height;
      const ctx = offscreen.getContext('2d', { willReadFrequently: true });
      if (!ctx) {
        throw new Error('Offscreen canvas context unavailable.');
      }

      const files: TarEntry[] = [];

      const localVolumeStub = volumeEnabled ? createVolumeStubState(width, height) : null;
      let localVolumeField =
        volumeEnabled && localVolumeStub ? snapshotVolumeStub(localVolumeStub) : null;

      let localKurState: KuramotoState | null = null;
      let localDerived: ReturnType<typeof createDerivedViews> | null = null;
      let localRand: ReturnType<typeof createNormalGenerator> | null = null;
      const params = getKurParams();
      if (kurEnabled) {
        localKurState = createKuramotoState(width, height);
        const derivedBuffer = new ArrayBuffer(derivedBufferSize(width, height));
        localDerived = createDerivedViews(derivedBuffer, width, height);
        initKuramotoState(localKurState, qInit, localDerived);
        deriveKuramotoFieldsCore(localKurState, localDerived, {
          kernel: kernelSpec,
          controls: { dmt },
        });
        localRand = createNormalGenerator(Math.floor(Math.random() * 1e9));
      }

      const metadataEntries: TarEntry[] = [];
      const frameName = (index: number) => `frames/frame_${String(index).padStart(5, '0')}.png`;

      for (let i = 0; i < totalFrames; i += 1) {
        if (frameExportAbortRef.current?.canceled) {
          throw new Error('Frame export cancelled');
        }
        const tSeconds = i * dt;

        if (volumeEnabled && localVolumeStub) {
          stepVolumeStub(localVolumeStub, dt);
          localVolumeField = snapshotVolumeStub(localVolumeStub);
        }

        if (kurEnabled && localKurState && localDerived && localRand) {
          stepKuramotoState(localKurState, params, dt, localRand, (i + 1) * dt, {
            kernel: kernelSpec,
            controls: { dmt },
          });
          deriveKuramotoFieldsCore(localKurState, localDerived, {
            kernel: kernelSpec,
            controls: { dmt },
          });
        }

        const buffer = new Uint8ClampedArray(totalPixels);
        const renderOptions: RenderFrameOptions = {
          ...renderOptionsBase,
          volumeFieldOverride: localVolumeField ?? null,
        };
        renderFrameCore(buffer, tSeconds, false, localDerived ?? undefined, renderOptions);
        const imageData = new ImageData(buffer, width, height);
        ctx.putImageData(imageData, 0, 0);
        const blob = await new Promise<Blob>((resolve, reject) =>
          offscreen.toBlob(
            (value) => (value ? resolve(value) : reject(new Error('Failed to encode frame'))),
            'image/png',
          ),
        );
        const frameBuffer = new Uint8Array(await blob.arrayBuffer());
        files.push({ name: frameName(i + 1), content: frameBuffer });

        if ((i + 1) % 5 === 0 || i + 1 === totalFrames) {
          setFrameExportProgress({
            current: i + 1,
            total: totalFrames,
            message: `Encoded frame ${i + 1} / ${totalFrames}`,
          });
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
      }

      const metadata = {
        width,
        height,
        fps: RECORDING_FPS,
        frameCount: totalFrames,
        durationSeconds: totalFrames / RECORDING_FPS,
        timestamp: new Date().toISOString(),
        includesImage: Boolean(includeImageInPreset && imageAsset?.path),
        image: imageAsset
          ? {
              path: imageAsset.path,
              name: imageAsset.name,
              width: imageAsset.width,
              height: imageAsset.height,
            }
          : null,
        preset: buildCurrentPreset().name,
      };
      metadataEntries.push({
        name: 'metadata.json',
        content: textEncoder.encode(JSON.stringify(metadata, null, 2)),
      });
      const readme = `Rainbow Perimeter frame export\n\nGenerated ${metadata.timestamp}\nFrames: ${metadata.frameCount}\nResolution: ${width}x${height}\nFrame rate: ${RECORDING_FPS} fps\n\nUse ffmpeg to assemble:\nffmpeg -framerate ${RECORDING_FPS} -i frames/frame_%05d.png -c:v libx264 -pix_fmt yuv420p rainbow-output.mp4\n`;
      metadataEntries.push({ name: 'README.txt', content: textEncoder.encode(readme) });

      files.push(...metadataEntries);

      setFrameExportProgress({
        current: totalFrames,
        total: totalFrames,
        message: 'Packaging frames…',
      });
      const tarBlob = createTarBlob(files);
      const compressed = await compressBlobGzip(tarBlob);
      setFrameExportProgress({
        current: totalFrames,
        total: totalFrames,
        message: 'Preparing download…',
      });
      const safeDate = new Date().toISOString().replace(/[:.]/g, '-');
      if (compressed) {
        downloadBlob(compressed, `rainbow-frames-${safeDate}.tar.gz`);
      } else {
        downloadBlob(tarBlob, `rainbow-frames-${safeDate}.tar`);
      }
      setFrameExportProgress(null);
    } catch (error) {
      if (error instanceof Error && error.message === 'Frame export cancelled') {
        setFrameExportError('Frame export cancelled.');
      } else {
        console.error('[ffmpeg-export] failed', error);
        setFrameExportError(error instanceof Error ? error.message : String(error));
      }
      setFrameExportProgress(null);
    } finally {
      frameExportAbortRef.current = null;
      setFrameExporting(false);
    }
  }, [
    frameExportDuration,
    frameExporting,
    surfaceFieldRef,
    rimFieldRef,
    width,
    height,
    couplingToggles,
    volumeEnabled,
    createVolumeStubState,
    snapshotVolumeStub,
    stepVolumeStub,
    getKurParams,
    kurEnabled,
    createKuramotoState,
    derivedBufferSize,
    createDerivedViews,
    initKuramotoState,
    deriveKuramotoFieldsCore,
    stepKuramotoState,
    dmt,
    renderFrameCore,
    includeImageInPreset,
    imageAsset,
    buildCurrentPreset,
  ]);

  const cancelFrameExport = useCallback(() => {
    if (frameExportAbortRef.current) {
      frameExportAbortRef.current.canceled = true;
    }
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const params = new URLSearchParams(window.location.search);
    const token = params.get('share');
    if (!token) return;
    try {
      const decoded = atob(decodeURIComponent(token));
      const parsed = JSON.parse(decoded);
      const payload = parsed?.preset ?? parsed;
      if (!payload?.params) {
        console.error('[share] invalid payload');
        return;
      }
      const fallback = buildCurrentPreset();
      const sharedPreset = sanitizePresetPayload(payload, fallback);
      if (!sharedPreset) {
        console.error('[share] failed to sanitize shared preset');
        return;
      }
      skipNextPresetApplyRef.current = true;
      applyPreset(sharedPreset);
      params.delete('share');
      const newSearch = params.toString();
      const newUrl = `${window.location.pathname}${newSearch ? `?${newSearch}` : ''}${window.location.hash}`;
      window.history.replaceState({}, '', newUrl);
    } catch (error) {
      console.error('[share] failed to apply shared preset', error);
    }
  }, [applyPreset, buildCurrentPreset]);

  useEffect(() => {
    if (skipNextPresetApplyRef.current) {
      skipNextPresetApplyRef.current = false;
      return;
    }
    applyPreset(PRESETS[presetIndex]);
  }, [presetIndex, applyPreset]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (shareStatusTimeoutRef.current !== null) {
      window.clearTimeout(shareStatusTimeoutRef.current);
      shareStatusTimeoutRef.current = null;
    }
    if (!shareStatus || shareStatus.url) return;
    shareStatusTimeoutRef.current = window.setTimeout(() => {
      setShareStatus(null);
      shareStatusTimeoutRef.current = null;
    }, 6000);
    return () => {
      if (shareStatusTimeoutRef.current !== null) {
        window.clearTimeout(shareStatusTimeoutRef.current);
        shareStatusTimeoutRef.current = null;
      }
    };
  }, [shareStatus]);

  useEffect(() => {
    return () => {
      if (gpuStateRef.current) {
        gpuStateRef.current.renderer.dispose();
        gpuStateRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    pendingStaticUploadRef.current = true;
    refreshGpuStaticTextures();
  }, [width, height, refreshGpuStaticTextures]);

  return (
    <main
      style={{
        height: '100vh',
        padding: '2rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.5rem',
        overflow: 'hidden',
      }}
    >
      <div>
        <h1 style={{ margin: 0, fontSize: '2.2rem' }}>Rainbow Perimeter Lab</h1>
        <p style={{ maxWidth: '52rem', color: '#94a3b8' }}>
          Upload a photo, apply the Rainbow Rims preset, then fine tune the kernel, DMT gain,
          surface wallpaper morph, and Kuramoto coupling to explore the hallucinatory perimeter
          lines described in the design brief.
        </p>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(22rem, 27rem) 1fr',
          gap: '1.5rem',
          alignItems: 'stretch',
          flex: 1,
          minHeight: 0,
        }}
      >
        <div
          className="panel"
          style={{
            gap: '1.25rem',
            overflowY: 'auto',
            height: '100%',
            minHeight: 0,
            paddingRight: '1.25rem',
          }}
        >
          <section className="panel">
            <h2>Image & Preset</h2>
            <div className="control">
              <label htmlFor="file-input">Upload image</label>
              <input id="file-input" type="file" accept="image/*" onChange={onFile} />
            </div>
            <SelectControl
              label="Preset"
              value={presetIndex.toString()}
              onChange={(v) => setPresetIndex(parseInt(v, 10))}
              options={PRESETS.map((p, i) => ({
                value: i.toString(),
                label: p.name,
              }))}
            />
            <button
              onClick={() => applyPreset(PRESETS[presetIndex])}
              style={{
                padding: '0.5rem 0.75rem',
                borderRadius: '0.6rem',
                border: '1px solid rgba(148,163,184,0.35)',
                background: 'rgba(14, 116, 144, 0.2)',
                color: '#f8fafc',
                cursor: 'pointer',
              }}
            >
              Apply preset
            </button>
            <div
              style={{
                display: 'flex',
                gap: '0.5rem',
                flexWrap: 'wrap',
              }}
            >
              <button
                onClick={handlePresetExport}
                style={{
                  padding: '0.4rem 0.65rem',
                  borderRadius: '0.55rem',
                  border: '1px solid rgba(148,163,184,0.35)',
                  background: 'rgba(15,118,110,0.18)',
                  color: '#e0f2fe',
                  cursor: 'pointer',
                }}
              >
                Export preset
              </button>
              <button
                onClick={handlePresetImport}
                style={{
                  padding: '0.4rem 0.65rem',
                  borderRadius: '0.55rem',
                  border: '1px solid rgba(148,163,184,0.35)',
                  background: 'rgba(30,64,175,0.18)',
                  color: '#e0e7ff',
                  cursor: 'pointer',
                }}
              >
                Import preset
              </button>
              <button
                onClick={handleCreateShareLink}
                style={{
                  padding: '0.4rem 0.65rem',
                  borderRadius: '0.55rem',
                  border: '1px solid rgba(148,163,184,0.35)',
                  background: 'rgba(59,130,246,0.18)',
                  color: '#bfdbfe',
                  cursor: 'pointer',
                }}
              >
                Create share link
              </button>
              <input
                ref={presetFileInputRef}
                type="file"
                accept="application/json"
                style={{ display: 'none' }}
                onChange={handlePresetImportFile}
              />
            </div>
            <label
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                marginTop: '0.6rem',
                fontSize: '0.85rem',
                color: imageAsset?.path ? '#cbd5f5' : '#64748b',
              }}
            >
              <input
                type="checkbox"
                checked={Boolean(includeImageInPreset && imageAsset?.path)}
                onChange={(event) => setIncludeImageInPreset(event.target.checked)}
                disabled={!imageAsset?.path}
              />
              <span>Include image reference when exporting or sharing</span>
            </label>
            {imageAsset?.path ? (
              <small style={{ display: 'block', color: '#94a3b8', marginTop: '0.35rem' }}>
                Using {imageAsset.name ?? imageAsset.path} ({imageAsset.width}×{imageAsset.height})
              </small>
            ) : (
              <small style={{ display: 'block', color: '#475569', marginTop: '0.35rem' }}>
                Upload an image to save or share it with presets.
              </small>
            )}
            {shareStatus ? (
              <div
                style={{
                  marginTop: '0.5rem',
                  fontSize: '0.8rem',
                  color: shareStatus.url ? '#38bdf8' : '#f87171',
                }}
              >
                <span>{shareStatus.message}</span>
                {shareStatus.url ? (
                  <>
                    {' '}
                    <a
                      href={shareStatus.url}
                      style={{ color: '#38bdf8', textDecoration: 'underline' }}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Open link
                    </a>
                  </>
                ) : null}
              </div>
            ) : null}
          </section>

          <section className="panel">
            <h2>Diagnostics</h2>
            <ToggleControl
              label="Use GPU renderer"
              value={renderBackend === 'gpu'}
              onChange={handleRendererToggle}
            />
            <ToggleControl
              label="Telemetry logging"
              value={telemetryEnabled}
              onChange={setTelemetryEnabled}
            />
            <ToggleControl
              label="Telemetry overlay"
              value={telemetryOverlayEnabled}
              onChange={setTelemetryOverlayEnabled}
              disabled={!telemetryEnabled}
            />
            <ToggleControl
              label="Frame metric logging"
              value={frameLoggingEnabled}
              onChange={setFrameLoggingEnabled}
            />
            <ToggleControl
              label="Enable rim generator"
              value={rimEnabled}
              onChange={setRimEnabled}
            />
            <ToggleControl
              label="Rim debug overlays"
              value={showRimDebug}
              onChange={setShowRimDebug}
            />
            <ToggleControl
              label="Hyperbolic grid overlay"
              value={showHyperbolicGrid}
              onChange={setShowHyperbolicGrid}
              disabled={!hyperbolicAtlas}
            />
            <ToggleControl
              label="Surface flow debug"
              value={showSurfaceDebug}
              onChange={setShowSurfaceDebug}
            />
            <ToggleControl
              label="Phase amplitude histogram"
              value={showPhaseDebug}
              onChange={setShowPhaseDebug}
            />
            <ToggleControl
              label="Phase alignment heatmap"
              value={phaseHeatmapEnabled}
              onChange={setPhaseHeatmapEnabled}
            />
            <div className="control">
              <button
                onClick={() => handleParityCheck()}
                style={{
                  padding: '0.5rem 0.75rem',
                  borderRadius: '0.6rem',
                  border: '1px solid rgba(148,163,184,0.35)',
                  background: 'rgba(147, 197, 253, 0.18)',
                  color: '#f8fafc',
                  cursor: 'pointer',
                }}
              >
                Run GPU parity check
              </button>
              {parityDisplay && (
                <small
                  style={{
                    display: 'block',
                    marginTop: '0.4rem',
                    color: parityDisplay.within ? '#38bdf8' : '#f87171',
                  }}
                >
                  worst {parityDisplay.worst.label}: Δ
                  <span style={{ fontFamily: 'monospace' }}>
                    {parityDisplay.worst.maxDelta.toFixed(2)}
                  </span>{' '}
                  | {parityDisplay.worst.percent.toFixed(2)}% &lt;={' '}
                  {parityDisplay.tolerance.toFixed(2)}%?{' '}
                  {parityDisplay.within ? 'OK' : 'Check reference'}
                  {!parityDisplay.within && (
                    <>
                      <br />
                      <span>
                        pixel ({parityDisplay.worst.maxCoord[0]}, {parityDisplay.worst.maxCoord[1]})
                        CPU {printRgb(parityDisplay.worst.cpuColor)} vs GPU{' '}
                        {printRgb(parityDisplay.worst.gpuColor)}
                      </span>
                    </>
                  )}
                </small>
              )}
            </div>
            <div className="control">
              <button
                onClick={() => handlePerfProbe()}
                style={{
                  padding: '0.5rem 0.75rem',
                  borderRadius: '0.6rem',
                  border: '1px solid rgba(148,163,184,0.35)',
                  background: 'rgba(94, 234, 212, 0.18)',
                  color: '#f8fafc',
                  cursor: 'pointer',
                }}
              >
                Measure render throughput (120 frames)
              </button>
              {perfDisplay && (
                <small
                  style={{
                    display: 'block',
                    marginTop: '0.4rem',
                    color: '#94a3b8',
                  }}
                >
                  GPU {perfDisplay.gpuFps.toFixed(0)} fps (~{perfDisplay.throughputGain.toFixed(1)}×
                  vs {perfDisplay.cpuMs.toFixed(1)} ms CPU)
                </small>
              )}
            </div>
            {(showRimDebug && rimDebugSnapshot) ||
            (showSurfaceDebug && surfaceDebugSnapshot) ||
            (showPhaseDebug && phaseDebugSnapshot) ? (
              <div
                className="control"
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.75rem',
                  background: 'rgba(15,23,42,0.45)',
                  borderRadius: '0.85rem',
                  padding: '0.75rem',
                }}
              >
                {showRimDebug && rimDebugSnapshot && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                    <HistogramPanel
                      title="Rim energy"
                      bins={rimDebugSnapshot.energyHist}
                      defaultColor="#f97316"
                      rangeLabel={`${rimDebugSnapshot.energyRange[0].toFixed(2)} – ${rimDebugSnapshot.energyRange[1].toFixed(2)}`}
                    />
                    <HistogramPanel
                      title="Rim hue"
                      bins={rimDebugSnapshot.hueHist}
                      colorForBin={(idx) =>
                        `hsl(${(idx / Math.max(rimDebugSnapshot.hueHist.length - 1, 1)) * 360}, 80%, 60%)`
                      }
                      rangeLabel={`${(rimDebugSnapshot.hueRange[0] * 360).toFixed(0)}° – ${(rimDebugSnapshot.hueRange[1] * 360).toFixed(0)}°`}
                    />
                  </div>
                )}
                {showSurfaceDebug && surfaceDebugSnapshot && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: '0.75rem',
                        color: '#cbd5f5',
                      }}
                    >
                      <span>Surface flow magnitudes</span>
                      <span>max {surfaceDebugSnapshot.magnitudeMax.toFixed(2)}</span>
                    </div>
                    {surfaceDebugSnapshot.magnitudeHist.map((bins, idx) => (
                      <HistogramPanel
                        key={`surface-hist-${idx}`}
                        title={`Orientation ${idx + 1}`}
                        bins={bins}
                        defaultColor="#38bdf8"
                      />
                    ))}
                  </div>
                )}
                {showPhaseDebug && phaseDebugSnapshot && (
                  <HistogramPanel
                    title="Phase amplitude |Z|"
                    bins={phaseDebugSnapshot.ampHist}
                    defaultColor="#a855f7"
                    rangeLabel={`${phaseDebugSnapshot.ampRange[0].toFixed(2)} – ${phaseDebugSnapshot.ampRange[1].toFixed(2)}`}
                  />
                )}
                {phaseHeatmapEnabled && phaseHeatmapSnapshot && (
                  <PhaseHeatmapPanel snapshot={phaseHeatmapSnapshot} />
                )}
              </div>
            ) : null}
            <div
              style={{
                marginTop: '0.75rem',
                display: 'grid',
                gap: '0.4rem',
              }}
            >
              {fieldStatusEntries.map((entry) => {
                const palette = FIELD_STATUS_STYLES[entry.state];
                const statusLabel = FIELD_STATUS_LABELS[entry.state];
                const resolutionText = entry.resolution
                  ? `${entry.resolution.width}×${entry.resolution.height}`
                  : '—';
                const stalenessText =
                  entry.state === 'missing'
                    ? 'offline'
                    : entry.stalenessMs === Number.POSITIVE_INFINITY
                      ? 'idle'
                      : `${entry.stalenessMs.toFixed(0)}ms`;
                return (
                  <div
                    key={entry.kind}
                    style={{
                      border: `1px solid ${palette.border}`,
                      background: palette.background,
                      color: palette.color,
                      borderRadius: '0.65rem',
                      padding: '0.45rem 0.6rem',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'baseline',
                    }}
                  >
                    <div style={{ fontWeight: 600 }}>{entry.label}</div>
                    <div style={{ fontSize: '0.75rem', fontFamily: 'monospace' }}>
                      {statusLabel} · {resolutionText} · {stalenessText}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="panel">
            <h2>SU7 Projection</h2>
            <ToggleControl
              label="Enable SU7 projector"
              value={su7Params.enabled}
              onChange={handleSu7EnabledChange}
            />
            <SelectControl
              label="Preset"
              value={su7PresetSelectValue}
              onChange={handleSu7PresetChange}
              options={su7PresetOptions}
            />
            {su7PresetMeta ? (
              <small
                style={{
                  display: 'block',
                  color: '#94a3b8',
                  marginTop: '-0.4rem',
                  marginBottom: '0.85rem',
                }}
              >
                {su7PresetMeta.description}
              </small>
            ) : null}
            <SliderControl
              label="Global gain"
              value={su7Params.gain}
              min={0}
              max={2}
              step={0.05}
              onChange={handleSu7GainChange}
              format={(v) => v.toFixed(2)}
              disabled={!su7Params.enabled}
            />
            <SliderControl
              label="Schedule strength"
              value={su7ScheduleStrength}
              min={SU7_SCHEDULE_STRENGTH_MIN}
              max={SU7_SCHEDULE_STRENGTH_MAX}
              step={0.05}
              onChange={handleSu7ScheduleStrengthChange}
              format={(v) => v.toFixed(2)}
              disabled={!su7Params.enabled}
            />
            <SelectControl
              label="Projector variant"
              value={su7ProjectorSelectValue}
              onChange={handleSu7ProjectorChange}
              options={su7ProjectorOptions}
            />
            <div className="control" style={!su7Params.enabled ? { opacity: 0.55 } : undefined}>
              <label>Seed</label>
              <input
                type="number"
                value={su7Params.seed}
                disabled={!su7Params.enabled}
                onChange={(event) => handleSu7SeedChange(Number.parseInt(event.target.value, 10))}
                style={{
                  padding: '0.45rem 0.6rem',
                  borderRadius: '0.6rem',
                  border: '1px solid rgba(148,163,184,0.35)',
                  background: 'rgba(15,23,42,0.7)',
                  color: '#e2e8f0',
                }}
              />
            </div>
          </section>

          <section className="panel">
            <h2>Dev Deck</h2>
            <SelectControl
              label="Synthetic case"
              value={selectedSyntheticCase}
              onChange={(value) => setSelectedSyntheticCase(value as SyntheticCaseId)}
              options={SYNTHETIC_CASES.map((entry) => ({ value: entry.id, label: entry.label }))}
            />
            <div
              style={{
                display: 'flex',
                gap: '0.5rem',
                flexWrap: 'wrap',
              }}
            >
              <button
                onClick={() => loadSyntheticCase(selectedSyntheticCase)}
                style={{
                  padding: '0.4rem 0.65rem',
                  borderRadius: '0.55rem',
                  border: '1px solid rgba(148,163,184,0.35)',
                  background: 'rgba(30, 64, 175, 0.18)',
                  color: '#e0e7ff',
                  cursor: 'pointer',
                }}
              >
                Load selected case
              </button>
              <button
                onClick={runSyntheticDeck}
                style={{
                  padding: '0.4rem 0.65rem',
                  borderRadius: '0.55rem',
                  border: '1px solid rgba(148,163,184,0.35)',
                  background: 'rgba(59, 130, 246, 0.2)',
                  color: '#e0f2fe',
                  cursor: 'pointer',
                }}
              >
                Run full deck
              </button>
            </div>
            <small style={{ display: 'block', color: '#94a3b8', marginTop: '0.35rem' }}>
              Records rim, surface, and |Z| baselines for analytics.
            </small>
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '0.5rem',
                marginTop: '0.75rem',
              }}
            >
              {SYNTHETIC_CASES.map((entry) => {
                const baseline = syntheticBaselines[entry.id];
                return (
                  <div
                    key={entry.id}
                    style={{
                      border: '1px solid rgba(148,163,184,0.3)',
                      borderRadius: '0.65rem',
                      background: 'rgba(15,23,42,0.4)',
                      padding: '0.55rem 0.65rem',
                    }}
                  >
                    <div style={{ fontWeight: 600, color: '#e2e8f0' }}>{entry.label}</div>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.25rem' }}>
                      {entry.description}
                    </div>
                    {baseline ? (
                      <div
                        style={{
                          marginTop: '0.4rem',
                          fontFamily: 'monospace',
                          fontSize: '0.8rem',
                          color: '#a5b4fc',
                        }}
                      >
                        rim {baseline.metrics.composer.fields.rim.energy.toFixed(3)} · surface{' '}
                        {baseline.metrics.composer.fields.surface.energy.toFixed(3)} · |Z|{' '}
                        {baseline.metrics.gradient.cohMean.toFixed(3)} ·{' '}
                        {new Date(baseline.timestamp).toLocaleTimeString()}
                      </div>
                    ) : (
                      <div
                        style={{
                          marginTop: '0.4rem',
                          fontFamily: 'monospace',
                          fontSize: '0.8rem',
                          color: '#64748b',
                        }}
                      >
                        No baseline recorded yet.
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>

          <section className="panel">
            <h2>Output</h2>
            <div className="control">
              <label>Canvas capture</label>
              <div className="control-row">
                <button
                  onClick={() => (isRecording ? stopCapture() : startCapture())}
                  disabled={captureButtonDisabled}
                  style={{
                    padding: '0.5rem 0.75rem',
                    borderRadius: '0.6rem',
                    border: '1px solid rgba(148,163,184,0.35)',
                    background: isRecording
                      ? 'rgba(248, 113, 113, 0.25)'
                      : 'rgba(59, 130, 246, 0.25)',
                    color: '#f8fafc',
                    cursor: captureButtonDisabled ? 'not-allowed' : 'pointer',
                    opacity: captureButtonDisabled ? 0.6 : 1,
                  }}
                >
                  {isRecording ? 'Stop capture' : isFinalizing ? 'Finalizing…' : 'Start capture'}
                </button>
                {(isRecording || isFinalizing) && (
                  <span style={{ color: '#fbbf24' }}>
                    {isRecording ? 'Recording…' : 'Finalizing…'}
                  </span>
                )}
              </div>
              {!captureSupport.checked && (
                <small style={{ color: '#94a3b8' }}>Checking browser capture support…</small>
              )}
              {captureSupport.checked && availableCaptureFormats.length === 0 && (
                <small style={{ color: '#f87171' }}>
                  MediaRecorder formats are unavailable in this browser.
                </small>
              )}
              <label
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.35rem',
                  color: '#94a3b8',
                  fontSize: '0.85rem',
                  marginTop: '0.5rem',
                }}
              >
                Recording format
                <select
                  value={recordingFormatId ?? ''}
                  onChange={handleRecordingFormatChange}
                  disabled={!captureSupport.checked || availableCaptureFormats.length === 0}
                  style={{
                    background: 'rgba(15,23,42,0.6)',
                    border: '1px solid rgba(148,163,184,0.35)',
                    borderRadius: '0.5rem',
                    color: '#e2e8f0',
                    padding: '0.35rem 0.5rem',
                    fontSize: '0.85rem',
                  }}
                >
                  {availableCaptureFormats.map((format) => (
                    <option key={format.id} value={format.id}>
                      {format.label}
                    </option>
                  ))}
                </select>
              </label>
              <label
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.35rem',
                  color: '#94a3b8',
                  fontSize: '0.85rem',
                }}
              >
                Recording quality
                <select
                  value={recordingPreset}
                  onChange={handleRecordingPresetChange}
                  style={{
                    background: 'rgba(15,23,42,0.6)',
                    border: '1px solid rgba(148,163,184,0.35)',
                    borderRadius: '0.5rem',
                    color: '#e2e8f0',
                    padding: '0.35rem 0.5rem',
                    fontSize: '0.85rem',
                  }}
                >
                  {RECORDING_PRESET_ORDER.map((presetId) => (
                    <option key={presetId} value={presetId}>
                      {RECORDING_PRESETS[presetId].label}
                    </option>
                  ))}
                </select>
              </label>
              <small style={{ color: '#94a3b8' }}>
                Canvas {width}×{height}px · {RECORDING_FPS} fps · target {recordingBitrateMbps} Mbps
                ({RECORDING_PRESETS[recordingPreset].label})
              </small>
              {recordingError && <small style={{ color: '#f87171' }}>{recordingError}</small>}
              {mp4Unsupported && (
                <small style={{ color: '#f97316' }}>
                  MP4 capture unsupported (available:{' '}
                  {availableCaptureFormats.length > 0
                    ? availableCaptureFormats
                        .filter((format) => format.container !== 'mp4')
                        .map((format) => format.label)
                        .join(', ') || 'none'
                    : 'none'}
                  ).
                </small>
              )}
              {recordingDownload && (
                <small style={{ color: '#38bdf8' }}>
                  Capture ready ({recordingDownload.mimeType}):{' '}
                  <a
                    href={recordingDownload.url}
                    download={recordingDownload.filename}
                    style={{ color: '#38bdf8' }}
                  >
                    Download ({formatBytes(recordingDownload.size)})
                  </a>
                </small>
              )}
              <div
                className="control"
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.35rem',
                  marginTop: '0.75rem',
                }}
              >
                <label style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
                  FFmpeg frame export duration (seconds)
                </label>
                <input
                  type="number"
                  min={1}
                  max={600}
                  step={1}
                  value={frameExportDuration}
                  disabled={frameExporting}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    if (Number.isFinite(value)) {
                      setFrameExportDuration(Math.min(600, Math.max(1, Math.round(value))));
                    }
                  }}
                  style={{
                    background: 'rgba(15,23,42,0.6)',
                    border: '1px solid rgba(148,163,184,0.35)',
                    borderRadius: '0.5rem',
                    color: '#e2e8f0',
                    padding: '0.35rem 0.5rem',
                    fontSize: '0.85rem',
                    width: '8rem',
                  }}
                />
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    flexWrap: 'wrap',
                  }}
                >
                  <button
                    onClick={exportFrameSequence}
                    disabled={frameExporting}
                    style={{
                      padding: '0.4rem 0.65rem',
                      borderRadius: '0.55rem',
                      border: '1px solid rgba(148,163,184,0.35)',
                      background: 'rgba(147,197,253,0.18)',
                      color: '#e0f2fe',
                      cursor: frameExporting ? 'not-allowed' : 'pointer',
                      opacity: frameExporting ? 0.6 : 1,
                    }}
                  >
                    {frameExporting ? 'Exporting frames…' : 'Export frames (FFmpeg)'}
                  </button>
                  {frameExporting ? (
                    <button
                      onClick={cancelFrameExport}
                      style={{
                        padding: '0.35rem 0.6rem',
                        borderRadius: '0.55rem',
                        border: '1px solid rgba(248,113,113,0.45)',
                        background: 'rgba(248,113,113,0.2)',
                        color: '#fee2e2',
                        cursor: 'pointer',
                      }}
                    >
                      Cancel export
                    </button>
                  ) : null}
                </div>
                {frameExportProgress ? (
                  <small style={{ color: '#38bdf8' }}>{frameExportProgress.message}</small>
                ) : null}
                {frameExportError ? (
                  <small style={{ color: '#f87171' }}>{frameExportError}</small>
                ) : null}
              </div>
            </div>
          </section>

          <section className="panel">
            <h2>Rainbow Rims</h2>
            <SliderControl
              label="Edge Threshold"
              value={edgeThreshold}
              min={0}
              max={1}
              step={0.01}
              onChange={setEdgeThreshold}
            />
            <SliderControl
              label="Kernel Blend"
              value={blend}
              min={0}
              max={1}
              step={0.01}
              onChange={setBlend}
            />
            <SliderControl
              label="Dispersion β₂"
              value={beta2}
              min={0}
              max={3}
              step={0.01}
              onChange={setBeta2}
            />
            <SliderControl
              label="Rim Thickness σ"
              value={sigma}
              min={0.3}
              max={6}
              step={0.05}
              onChange={setSigma}
            />
            <SliderControl
              label="Phase Jitter"
              value={jitter}
              min={0}
              max={2}
              step={0.02}
              onChange={setJitter}
            />
            <SliderControl
              label="Contrast"
              value={contrast}
              min={0.25}
              max={3}
              step={0.05}
              onChange={setContrast}
            />
            <SliderControl
              label="Animation Speed"
              value={speed}
              min={0}
              max={3}
              step={0.05}
              onChange={setSpeed}
            />
            <ToggleControl
              label="Microsaccade jitter"
              value={microsaccade}
              onChange={setMicrosaccade}
            />
            <ToggleControl label="Zero-mean pin" value={phasePin} onChange={setPhasePin} />
            <ToggleControl label="Alive microbreath" value={alive} onChange={setAlive} />
            <SliderControl
              label="Rim Alpha"
              value={rimAlpha}
              min={0}
              max={1}
              step={0.05}
              onChange={setRimAlpha}
            />
          </section>

          <section className="panel">
            <h2>Kernel + DMT</h2>
            <SliderControl
              label="Kernel Gain"
              value={kernel.gain}
              min={0}
              max={5}
              step={0.05}
              onChange={(v) => updateKernel({ gain: v })}
            />
            <SliderControl
              label="Spatial Frequency k₀"
              value={kernel.k0}
              min={0.01}
              max={0.4}
              step={0.01}
              onChange={(v) => updateKernel({ k0: v })}
            />
            <SliderControl
              label="Sharpness Q"
              value={kernel.Q}
              min={0.5}
              max={8}
              step={0.05}
              onChange={(v) => updateKernel({ Q: v })}
            />
            <SliderControl
              label="Anisotropy"
              value={kernel.anisotropy}
              min={0}
              max={1.5}
              step={0.05}
              onChange={(v) => updateKernel({ anisotropy: v })}
            />
            <SliderControl
              label="Chirality"
              value={kernel.chirality}
              min={0}
              max={2}
              step={0.05}
              onChange={(v) => updateKernel({ chirality: v })}
            />
            <SliderControl
              label="Transparency"
              value={kernel.transparency}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => updateKernel({ transparency: v })}
            />
            <SelectControl
              label="Coupling Kernel"
              value={kernel.couplingPreset}
              onChange={(v) => updateKernel({ couplingPreset: v as CouplingKernelPreset })}
              options={COUPLING_PRESET_OPTIONS}
            />
            <SliderControl
              label="DMT Gain"
              value={dmt}
              min={0}
              max={1}
              step={0.01}
              onChange={setDmt}
            />
            <SliderControl
              label="Arousal"
              value={arousal}
              min={0}
              max={1}
              step={0.01}
              onChange={setArousal}
            />
          </section>

          <section className="panel">
            <h2>Surface Morph</h2>
            <ToggleControl
              label="Enable surface morph"
              value={surfEnabled}
              onChange={setSurfEnabled}
            />
            <SliderControl
              label="Surface blend"
              value={surfaceBlend}
              min={0}
              max={1}
              step={0.02}
              onChange={setSurfaceBlend}
            />
            <SliderControl
              label="Warp amplitude"
              value={warpAmp}
              min={0}
              max={6}
              step={0.1}
              onChange={setWarpAmp}
            />
            <SliderControl
              label="Orientations"
              value={nOrient}
              min={2}
              max={8}
              step={1}
              onChange={setNOrient}
              format={(v) => v.toFixed(0)}
            />
            <SelectControl
              label="Wallpaper group"
              value={wallGroup}
              onChange={(v) => setWallGroup(v as WallpaperGroup)}
              options={[
                { value: 'off', label: 'Off' },
                { value: 'p1', label: 'p1' },
                { value: 'p2', label: 'p2' },
                { value: 'pm', label: 'pm' },
                { value: 'pg', label: 'pg' },
                { value: 'cm', label: 'cm' },
                { value: 'pmm', label: 'pmm' },
                { value: 'pmg', label: 'pmg' },
                { value: 'pgg', label: 'pgg' },
                { value: 'cmm', label: 'cmm' },
                { value: 'p4', label: 'p4' },
                { value: 'p4g', label: 'p4g' },
                { value: 'p4m', label: 'p4m' },
                { value: 'p3', label: 'p3' },
                { value: 'p31m', label: 'p31m' },
                { value: 'p3m1', label: 'p3m1' },
                { value: 'p6', label: 'p6' },
                { value: 'p6m', label: 'p6m' },
              ]}
            />
            <SelectControl
              label="Region"
              value={surfaceRegion}
              onChange={(v) => setSurfaceRegion(v as SurfaceRegion)}
              options={[
                { value: 'surfaces', label: 'Surfaces' },
                { value: 'edges', label: 'Edges' },
                { value: 'both', label: 'Both' },
              ]}
            />
          </section>

          <section className="panel">
            <h2>3D Volume</h2>
            <ToggleControl
              label="Enable 3D volume feed"
              value={volumeEnabled}
              onChange={setVolumeEnabled}
            />
            <div className="control-hint">
              Volume couplers wake once the feed is live. Press play and try the sliders below.
            </div>
          </section>

          <section className="panel">
            <h2>Coupling Fabric</h2>
            <ToggleControl
              label="Enable rims → surface coupling"
              value={couplingToggles.rimToSurface}
              onChange={setCouplingToggle('rimToSurface')}
            />
            <ToggleControl
              label="Enable surface → rims coupling"
              value={couplingToggles.surfaceToRim}
              onChange={setCouplingToggle('surfaceToRim')}
            />
            <h3>Rims → Surface</h3>
            <SliderControl
              label="Energy → surface blend"
              value={coupling.rimToSurfaceBlend}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue('rimToSurfaceBlend')}
              format={(v) => v.toFixed(2)}
            />
            <SliderControl
              label="Tangent alignment weight"
              value={coupling.rimToSurfaceAlign}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue('rimToSurfaceAlign')}
              format={(v) => v.toFixed(2)}
            />
            <h3>Surface → Rims</h3>
            <SliderControl
              label="Warp gradient → offset bias"
              value={coupling.surfaceToRimOffset}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue('surfaceToRimOffset')}
              format={(v) => v.toFixed(2)}
            />
            <SliderControl
              label="Warp gradient → sigma thinning"
              value={coupling.surfaceToRimSigma}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue('surfaceToRimSigma')}
              format={(v) => v.toFixed(2)}
            />
            <SliderControl
              label="Lattice phase → hue bias"
              value={coupling.surfaceToRimHue}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue('surfaceToRimHue')}
              format={(v) => v.toFixed(2)}
            />
            <div
              style={{
                display: 'flex',
                gap: '0.5rem',
                flexWrap: 'wrap',
                margin: '0.5rem 0 1rem',
              }}
            >
              <button
                onClick={() => exportCouplingDiff('rimToSurface')}
                style={{
                  padding: '0.4rem 0.65rem',
                  borderRadius: '0.55rem',
                  border: '1px solid rgba(148,163,184,0.35)',
                  background: 'rgba(59, 130, 246, 0.18)',
                  color: '#e0f2fe',
                  cursor: 'pointer',
                }}
              >
                Export rim→surface diff
              </button>
              <button
                onClick={() => exportCouplingDiff('surfaceToRim')}
                style={{
                  padding: '0.4rem 0.65rem',
                  borderRadius: '0.55rem',
                  border: '1px solid rgba(148,163,184,0.35)',
                  background: 'rgba(14, 165, 233, 0.18)',
                  color: '#cffafe',
                  cursor: 'pointer',
                }}
              >
                Export surface→rim diff
              </button>
            </div>
            <h3>Kuramoto Adapters</h3>
            <SliderControl
              label="|Z| → transparency"
              value={coupling.kurToTransparency}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue('kurToTransparency')}
              format={(v) => v.toFixed(2)}
            />
            <SliderControl
              label="∇θ → orientation blend"
              value={coupling.kurToOrientation}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue('kurToOrientation')}
              format={(v) => v.toFixed(2)}
            />
            <SliderControl
              label="Vorticity → chirality"
              value={coupling.kurToChirality}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue('kurToChirality')}
              format={(v) => v.toFixed(2)}
            />
            <h3>Volume → 2D</h3>
            <SliderControl
              label="Phase → rim hue"
              value={coupling.volumePhaseToHue}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue('volumePhaseToHue')}
              format={(v) => v.toFixed(2)}
            />
            <SliderControl
              label="Depth grad → warp amp"
              value={coupling.volumeDepthToWarp}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue('volumeDepthToWarp')}
              format={(v) => v.toFixed(2)}
            />
          </section>

          <section className="panel">
            <h2>Composer</h2>
            <SelectControl
              label="DMT routing"
              value={composer.dmtRouting}
              onChange={(value) => handleComposerRouting(value as DmtRoutingMode)}
              options={[
                { value: 'auto', label: 'Auto' },
                { value: 'rimBias', label: 'Rim bias' },
                { value: 'surfaceBias', label: 'Surface bias' },
              ]}
            />
            <SelectControl
              label="Solver regime"
              value={composer.solverRegime}
              onChange={(value) => handleComposerSolver(value as SolverRegime)}
              options={[
                { value: 'balanced', label: 'Balanced' },
                { value: 'rimLocked', label: 'Rim locked' },
                { value: 'surfaceLocked', label: 'Surface locked' },
              ]}
            />
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '1rem',
              }}
            >
              {COMPOSER_FIELD_LIST.map((field) => {
                const cfg = composer.fields[field];
                const label = COMPOSER_FIELD_LABELS[field];
                return (
                  <div
                    key={field}
                    style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}
                  >
                    <h3 style={{ margin: 0 }}>{label}</h3>
                    <SliderControl
                      label="Exposure"
                      value={cfg.exposure}
                      min={0}
                      max={4}
                      step={0.05}
                      onChange={setComposerFieldValue(field, 'exposure')}
                      format={(v) => v.toFixed(2)}
                    />
                    <SliderControl
                      label="Gamma"
                      value={cfg.gamma}
                      min={0.2}
                      max={3}
                      step={0.05}
                      onChange={setComposerFieldValue(field, 'gamma')}
                      format={(v) => v.toFixed(2)}
                    />
                    <SliderControl
                      label="Weight"
                      value={cfg.weight}
                      min={0}
                      max={2.5}
                      step={0.05}
                      onChange={setComposerFieldValue(field, 'weight')}
                      format={(v) => v.toFixed(2)}
                    />
                  </div>
                );
              })}
            </div>
          </section>

          <section className="panel">
            <h2>Kuramoto Field</h2>
            <ToggleControl label="Enable OA field" value={kurEnabled} onChange={setKurEnabled} />
            <ToggleControl label="Sync to main thread" value={kurSync} onChange={setKurSync} />
            <SelectControl
              label="Regime"
              value={kurRegime}
              onChange={(value) => {
                const next = value as KurRegime;
                if (next === 'custom') {
                  setKurRegime('custom');
                } else {
                  applyKurRegime(next);
                }
              }}
              options={[
                { value: 'locked', label: KUR_REGIME_PRESETS.locked.label },
                { value: 'highEnergy', label: KUR_REGIME_PRESETS.highEnergy.label },
                { value: 'chaotic', label: KUR_REGIME_PRESETS.chaotic.label },
                { value: 'custom', label: 'Custom (manual)' },
              ]}
            />
            {kurRegime !== 'custom' && (
              <small style={{ display: 'block', color: '#94a3b8', marginTop: '0.35rem' }}>
                {KUR_REGIME_PRESETS[kurRegime as Exclude<KurRegime, 'custom'>].description}
              </small>
            )}
            <SliderControl
              label="Coupling K₀"
              value={K0}
              min={0}
              max={2}
              step={0.05}
              onChange={(value) => {
                markKurCustom();
                setK0(value);
              }}
            />
            <SliderControl
              label="Phase lag α"
              value={alphaKur}
              min={-Math.PI}
              max={Math.PI}
              step={0.05}
              onChange={(value) => {
                markKurCustom();
                setAlphaKur(value);
              }}
            />
            <SliderControl
              label="Line width γ"
              value={gammaKur}
              min={0}
              max={1}
              step={0.02}
              onChange={(value) => {
                markKurCustom();
                setGammaKur(value);
              }}
            />
            <SliderControl
              label="Mean freq ω₀"
              value={omega0}
              min={-2}
              max={2}
              step={0.05}
              onChange={(value) => {
                markKurCustom();
                setOmega0(value);
              }}
            />
            <SliderControl
              label="Noise ε"
              value={epsKur}
              min={0}
              max={0.02}
              step={0.0005}
              onChange={(value) => {
                markKurCustom();
                setEpsKur(value);
              }}
              format={(v) => v.toFixed(4)}
            />
            <ToggleControl
              label="Small-world coupling"
              value={smallWorldEnabled}
              onChange={(value) => {
                markKurCustom();
                setSmallWorldEnabled(value);
              }}
            />
            <SliderControl
              label="Small-world weight"
              value={smallWorldWeight}
              min={0}
              max={2}
              step={0.05}
              onChange={(value) => {
                markKurCustom();
                setSmallWorldWeight(value);
              }}
              format={(v) => v.toFixed(2)}
              disabled={!smallWorldEnabled}
            />
            <SliderControl
              label="Signed gain p_sw"
              value={pSw}
              min={-0.15}
              max={0.15}
              step={0.005}
              onChange={(value) => {
                markKurCustom();
                setPSw(value);
              }}
              format={(v) => v.toFixed(3)}
              disabled={!smallWorldEnabled}
            />
            <div className="control" style={{ opacity: smallWorldEnabled ? 1 : 0.5 }}>
              <label>Small-world seed</label>
              <div className="control-row" style={{ gap: '0.5rem' }}>
                <input
                  type="number"
                  min={0}
                  max={Number.MAX_SAFE_INTEGER}
                  value={smallWorldSeed}
                  onChange={(event) => {
                    const next = Number.parseInt(event.target.value, 10);
                    const clamped = Number.isFinite(next) && next >= 0 ? next : 0;
                    markKurCustom();
                    setSmallWorldSeed(clamped);
                  }}
                  disabled={!smallWorldEnabled}
                  style={{ width: '6rem' }}
                />
                <button
                  type="button"
                  onClick={() => {
                    const nextSeed = Math.floor(Math.random() * 1e9);
                    markKurCustom();
                    setSmallWorldSeed(nextSeed);
                  }}
                  disabled={!smallWorldEnabled}
                >
                  Reseed
                </button>
              </div>
            </div>
            <SliderControl
              label="Flux φₓ"
              value={fluxX}
              min={0}
              max={Math.PI * 2}
              step={0.01}
              onChange={(value) => {
                markKurCustom();
                setFluxX(value);
              }}
              format={(v) => `${(v / Math.PI).toFixed(2)}π`}
            />
            <SliderControl
              label="Flux φ_y"
              value={fluxY}
              min={0}
              max={Math.PI * 2}
              step={0.01}
              onChange={(value) => {
                markKurCustom();
                setFluxY(value);
              }}
              format={(v) => `${(v / Math.PI).toFixed(2)}π`}
            />
            <SliderControl
              label="Init twist q"
              value={qInit}
              min={0}
              max={4}
              step={1}
              onChange={setQInit}
              format={(v) => v.toFixed(0)}
            />
            <button
              onClick={resetKuramotoField}
              style={{
                padding: '0.5rem 0.75rem',
                borderRadius: '0.6rem',
                border: '1px solid rgba(148,163,184,0.35)',
                background: 'rgba(30, 64, 175, 0.25)',
                color: '#f8fafc',
                cursor: 'pointer',
              }}
            >
              Reset Z-field
            </button>
          </section>

          <section className="panel">
            <h2>Display</h2>
            <SelectControl
              label="Display mode"
              value={displayMode}
              onChange={(v) => setDisplayMode(v as DisplayMode)}
              options={[
                { value: 'color', label: 'Color base + color rims' },
                {
                  value: 'grayBaseColorRims',
                  label: 'Gray base + color rims',
                },
                {
                  value: 'grayBaseGrayRims',
                  label: 'Gray base + gray rims',
                },
                {
                  value: 'colorBaseGrayRims',
                  label: 'Color base + gray rims',
                },
                {
                  value: 'colorBaseBlendedRims',
                  label: 'Color base + blended rims',
                },
              ]}
            />
            <SliderControl
              label="Polar bins"
              value={polBins}
              min={0}
              max={32}
              step={1}
              onChange={setPolBins}
              format={(v) => v.toFixed(0)}
            />
            <ToggleControl label="Normalization pin" value={normPin} onChange={setNormPin} />
            <SliderControl
              label="Curvature strength"
              value={curvatureStrength}
              min={0}
              max={MAX_CURVATURE_STRENGTH}
              step={0.01}
              onChange={setCurvatureStrength}
              format={(v) => v.toFixed(2)}
            />
            <ToggleControl
              label="Klein projection"
              value={curvatureMode === 'klein'}
              onChange={(enabled) => setCurvatureMode(enabled ? 'klein' : 'poincare')}
            />
            <ToggleControl
              label="Hyperbolic ruler overlay"
              value={showHyperbolicGuide}
              onChange={setShowHyperbolicGuide}
              disabled={!hyperbolicAtlas}
            />
            <SliderControl
              label="Ruler spacing (hyper units)"
              value={hyperbolicGuideSpacing}
              min={HYPERBOLIC_GUIDE_SPACING_MIN}
              max={HYPERBOLIC_GUIDE_SPACING_MAX}
              step={0.05}
              onChange={setHyperbolicGuideSpacing}
              format={(v) => `${v.toFixed(2)} σ`}
              disabled={!hyperbolicAtlas || !showHyperbolicGuide}
            />
            <h3>Tracer Loop</h3>
            <ToggleControl
              label="Tracer feedback"
              value={tracerConfig.enabled}
              onChange={setTracerValue('enabled')}
            />
            <SliderControl
              label="Tracer decay"
              value={tracerConfig.decay}
              min={0}
              max={1}
              step={0.01}
              onChange={setTracerValue('decay')}
              format={(v) => v.toFixed(2)}
              disabled={!tracerConfig.enabled}
            />
            <SliderControl
              label="Tracer gain"
              value={tracerConfig.gain}
              min={0}
              max={1}
              step={0.01}
              onChange={setTracerValue('gain')}
              format={(v) => v.toFixed(2)}
              disabled={!tracerConfig.enabled}
            />
            <ToggleControl
              label="Modulation"
              value={tracerConfig.modulationEnabled}
              onChange={setTracerValue('modulationEnabled')}
            />
            <SliderControl
              label="Mod freq (Hz)"
              value={tracerConfig.modulationFrequency}
              min={0}
              max={24}
              step={0.5}
              onChange={setTracerValue('modulationFrequency')}
              format={(v) => `${v.toFixed(1)} Hz`}
              disabled={!tracerConfig.enabled || !tracerConfig.modulationEnabled}
            />
            <SliderControl
              label="Mod depth"
              value={tracerConfig.modulationDepth}
              min={0}
              max={1}
              step={0.01}
              onChange={setTracerValue('modulationDepth')}
              format={(v) => `${Math.round(v * 100)}%`}
              disabled={!tracerConfig.enabled || !tracerConfig.modulationEnabled}
            />
          </section>
        </div>

        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '1rem',
            overflowY: 'auto',
            height: '100%',
            minHeight: 0,
            paddingRight: '0.75rem',
          }}
        >
          <div
            style={{
              position: 'relative',
              width: '100%',
              maxWidth: `${Math.min(width, 1000)}px`,
              maxHeight: `${Math.min(height, 1000)}px`,
              aspectRatio: `${width} / ${height}`,
              borderRadius: '1.25rem',
              border: '1px solid rgba(148,163,184,0.2)',
              boxShadow: '0 20px 60px rgba(15,23,42,0.65)',
              overflow: 'hidden',
            }}
          >
            <canvas
              ref={canvasRef}
              width={width}
              height={height}
              style={{
                width: '100%',
                height: '100%',
                display: 'block',
              }}
            />
            <canvas
              ref={hyperbolicGridCanvasRef}
              width={width}
              height={height}
              style={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                opacity: showHyperbolicGrid || showHyperbolicGuide ? 1 : 0,
                transition: 'opacity 120ms ease-out',
                mixBlendMode: 'screen',
              }}
            />
            <div
              style={{
                position: 'absolute',
                top: '0.75rem',
                left: '0.75rem',
                padding: '0.25rem 0.6rem',
                borderRadius: '9999px',
                background: 'rgba(15,23,42,0.65)',
                color: '#e2e8f0',
                fontSize: '0.75rem',
                letterSpacing: '0.02em',
                fontWeight: 500,
                pointerEvents: 'none',
              }}
            >
              <div>{`Flux (φₓ=${fluxX.toFixed(2)}, φ_y=${fluxY.toFixed(2)})`}</div>
              <div>
                {smallWorldEnabled && smallWorldWeight > 0
                  ? `Small-world: on (w=${smallWorldWeight.toFixed(2)}, p=${pSw.toFixed(3)})`
                  : 'Small-world: off'}
              </div>
            </div>
            {telemetryOverlayEnabled && telemetrySnapshot && telemetrySnapshot.frameSamples > 0 ? (
              <div
                style={{
                  position: 'absolute',
                  top: '0.75rem',
                  right: '0.75rem',
                  padding: '0.5rem 0.65rem',
                  borderRadius: '0.85rem',
                  background: 'rgba(15,23,42,0.7)',
                  border: '1px solid rgba(148,163,184,0.35)',
                  color: '#f8fafc',
                  fontSize: '0.7rem',
                  maxWidth: '15rem',
                  pointerEvents: 'none',
                  backdropFilter: 'blur(6px)',
                }}
              >
                <TelemetryOverlayContents snapshot={telemetrySnapshot} />
              </div>
            ) : null}
          </div>
          <p style={{ color: '#64748b', margin: 0 }}>
            Tip: enable the OA field once you have rims dialed in, then gradually increase K₀ and
            lower ε to let the oscillator field steer both rims and surface morph in a coherent way.
          </p>
          <section className="panel">
            <h2>Telemetry</h2>
            {telemetrySnapshot ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(11rem, 1fr))',
                    gap: '0.75rem',
                  }}
                >
                  {COMPOSER_FIELD_LIST.map((field) => {
                    const data = telemetrySnapshot.fields[field];
                    return (
                      <div
                        key={field}
                        style={{
                          background: 'rgba(15,23,42,0.45)',
                          borderRadius: '0.75rem',
                          padding: '0.75rem',
                          display: 'flex',
                          flexDirection: 'column',
                          gap: '0.35rem',
                        }}
                      >
                        <strong style={{ color: '#e2e8f0' }}>{COMPOSER_FIELD_LABELS[field]}</strong>
                        <span style={{ fontSize: '0.8rem', color: '#a5b4fc' }}>
                          Energy {data.energy.toFixed(3)}
                        </span>
                        <span style={{ fontSize: '0.8rem', color: '#cbd5f5' }}>
                          Blend {(data.blend * 100).toFixed(1)}%
                        </span>
                        <span style={{ fontSize: '0.8rem', color: '#cbd5f5' }}>
                          Share {(data.share * 100).toFixed(1)}%
                        </span>
                        <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                          Weight {data.weight.toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <div
                  style={{
                    background: 'rgba(15,23,42,0.45)',
                    borderRadius: '0.75rem',
                    padding: '0.75rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.5rem',
                  }}
                >
                  <div style={{ color: '#e2e8f0', fontWeight: 600 }}>
                    Coupling scale {telemetrySnapshot.coupling.scale.toFixed(2)}
                  </div>
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(13rem, 1fr))',
                      gap: '0.5rem',
                    }}
                  >
                    {Object.entries(telemetrySnapshot.coupling.effective).map(([key, value]) => (
                      <div key={key} style={{ fontSize: '0.75rem', color: '#cbd5f5' }}>
                        <strong style={{ color: '#facc15' }}>
                          {formatCouplingKey(key as keyof CouplingConfig)}
                        </strong>
                        : {value.toFixed(2)}
                      </div>
                    ))}
                  </div>
                </div>
                <div
                  style={{
                    background: 'rgba(15,23,42,0.45)',
                    borderRadius: '0.75rem',
                    padding: '0.75rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.35rem',
                  }}
                >
                  <div style={{ color: '#e2e8f0', fontWeight: 600 }}>SU7 Shell</div>
                  <span style={{ fontSize: '0.8rem', color: '#cbd5f5' }}>
                    Unitary μ {telemetrySnapshot.su7.unitaryError.toExponential(2)}
                  </span>
                  <span style={{ fontSize: '0.8rem', color: '#a5b4fc' }}>
                    Unitary frame {telemetrySnapshot.su7Unitary.latest.toExponential(2)} · ⟨frame⟩{' '}
                    {telemetrySnapshot.su7Unitary.mean.toExponential(2)} · max{' '}
                    {telemetrySnapshot.su7Unitary.max.toExponential(2)}
                  </span>
                  <span style={{ fontSize: '0.8rem', color: '#cbd5f5' }}>
                    Det drift {telemetrySnapshot.su7.determinantDrift.toExponential(2)}
                  </span>
                  <span style={{ fontSize: '0.8rem', color: '#cbd5f5' }}>
                    Norm Δ max {telemetrySnapshot.su7.normDeltaMax.toExponential(2)}
                  </span>
                  <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                    Norm Δ mean {telemetrySnapshot.su7.normDeltaMean.toExponential(2)}
                  </span>
                  <span style={{ fontSize: '0.8rem', color: '#eab308' }}>
                    Projector energy {telemetrySnapshot.su7.projectorEnergy.toFixed(3)}
                  </span>
                </div>
                <div
                  style={{
                    background: 'rgba(15,23,42,0.45)',
                    borderRadius: '0.75rem',
                    padding: '0.75rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.75rem',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'baseline',
                      gap: '0.5rem',
                    }}
                  >
                    <div style={{ color: '#e2e8f0', fontWeight: 600 }}>SU7 Distributions</div>
                    <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>
                      Samples {telemetrySnapshot.frameSamples}
                    </span>
                  </div>
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(12rem, 1fr))',
                      gap: '0.75rem',
                    }}
                  >
                    <HistogramPanel
                      title="Norm Δ mean"
                      bins={telemetrySnapshot.su7Histograms.normDeltaMean.bins}
                      defaultColor="#a855f7"
                      rangeLabel={formatTelemetryRange(
                        telemetrySnapshot.su7Histograms.normDeltaMean,
                        2,
                      )}
                    />
                    <HistogramPanel
                      title="Norm Δ max"
                      bins={telemetrySnapshot.su7Histograms.normDeltaMax.bins}
                      defaultColor="#f97316"
                      rangeLabel={formatTelemetryRange(
                        telemetrySnapshot.su7Histograms.normDeltaMax,
                        2,
                      )}
                    />
                    <HistogramPanel
                      title="Projector energy"
                      bins={telemetrySnapshot.su7Histograms.projectorEnergy.bins}
                      defaultColor="#38bdf8"
                      rangeLabel={formatTelemetryRange(
                        telemetrySnapshot.su7Histograms.projectorEnergy,
                        2,
                      )}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <p style={{ color: '#94a3b8', margin: 0 }}>
                {telemetryEnabled ? 'Collecting telemetry…' : 'Telemetry disabled.'}
              </p>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}

type SliderProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
  disabled?: boolean;
};

type RimDebugSnapshot = {
  energyRange: [number, number];
  hueRange: [number, number];
  energyHist: number[];
  hueHist: number[];
};

type SurfaceDebugSnapshot = {
  orientationCount: number;
  magnitudeMax: number;
  magnitudeHist: number[][];
};

type PhaseDebugSnapshot = {
  ampRange: [number, number];
  ampHist: number[];
};

type PhaseHeatmapSnapshot = {
  width: number;
  height: number;
  values: Float32Array;
  min: number;
  max: number;
};

type HistogramPanelProps = {
  title: string;
  bins: number[];
  defaultColor?: string;
  colorForBin?: (index: number, value: number, max: number) => string;
  rangeLabel?: string;
};

type TelemetryFieldSnapshot = {
  energy: number;
  blend: number;
  share: number;
  weight: number;
};

type TelemetryHistogramSnapshot = {
  bins: number[];
  min: number;
  max: number;
};

type TelemetrySnapshot = {
  fields: Record<ComposerFieldId, TelemetryFieldSnapshot>;
  coupling: {
    scale: number;
    base: CouplingConfig;
    effective: CouplingConfig;
  };
  su7: Su7Telemetry;
  su7Histograms: {
    normDeltaMean: TelemetryHistogramSnapshot;
    normDeltaMax: TelemetryHistogramSnapshot;
    projectorEnergy: TelemetryHistogramSnapshot;
  };
  su7Unitary: {
    latest: number;
    mean: number;
    max: number;
  };
  frameSamples: number;
  updatedAt: number;
};

const HistogramPanel = ({
  title,
  bins,
  defaultColor = '#38bdf8',
  colorForBin,
  rangeLabel,
}: HistogramPanelProps) => {
  const max = bins.reduce((acc, value) => (value > acc ? value : acc), 0);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '0.75rem',
          color: '#cbd5f5',
        }}
      >
        <span>{title}</span>
        {rangeLabel ? <span>{rangeLabel}</span> : null}
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${bins.length}, 1fr)`,
          alignItems: 'end',
          gap: '2px',
          height: '64px',
          background: 'rgba(15,23,42,0.45)',
          borderRadius: '0.6rem',
          padding: '6px',
        }}
      >
        {bins.map((value, idx) => {
          const normalized = max > 0 ? value / max : 0;
          const color = colorForBin ? colorForBin(idx, value, max) : defaultColor;
          return (
            <span
              key={idx}
              style={{
                display: 'block',
                width: '100%',
                height: `${Math.max(normalized * 100, 2)}%`,
                background: color,
                borderRadius: '0.35rem 0.35rem 0 0',
              }}
            />
          );
        })}
      </div>
    </div>
  );
};

function computeTelemetryHistogram(values: number[], binCount: number): TelemetryHistogramSnapshot {
  const finiteValues = values.filter((value) => Number.isFinite(value));
  if (finiteValues.length === 0) {
    return {
      bins: Array.from({ length: binCount }, () => 0),
      min: 0,
      max: 0,
    };
  }
  let min = finiteValues[0];
  let max = finiteValues[0];
  for (let i = 1; i < finiteValues.length; i += 1) {
    const value = finiteValues[i];
    if (value < min) min = value;
    if (value > max) max = value;
  }
  const bins = Array.from({ length: binCount }, () => 0);
  const range = max - min;
  if (range <= 1e-12) {
    const idx = Math.max(0, Math.min(binCount - 1, Math.floor(binCount / 2)));
    bins[idx] = finiteValues.length;
    return { bins, min, max };
  }
  const invRange = 1 / range;
  for (let i = 0; i < finiteValues.length; i += 1) {
    const value = finiteValues[i];
    let binIndex = Math.floor((value - min) * invRange * binCount);
    if (binIndex >= binCount) {
      binIndex = binCount - 1;
    } else if (binIndex < 0) {
      binIndex = 0;
    }
    bins[binIndex] += 1;
  }
  return { bins, min, max };
}

function hasTelemetryHistogramSamples(hist: TelemetryHistogramSnapshot): boolean {
  return hist.bins.some((value) => value > 0);
}

function formatTelemetryRange(hist: TelemetryHistogramSnapshot, digits = 2): string {
  if (!hasTelemetryHistogramSamples(hist)) {
    return 'no samples';
  }
  if (!Number.isFinite(hist.min) || !Number.isFinite(hist.max)) {
    return '—';
  }
  if (Math.abs(hist.max - hist.min) <= 1e-12) {
    return hist.min.toExponential(digits);
  }
  return `${hist.min.toExponential(digits)} – ${hist.max.toExponential(digits)}`;
}

type TinyHistogramProps = {
  bins: number[];
  color?: string;
  height?: number;
};

function TinyHistogram({ bins, color = '#38bdf8', height = 36 }: TinyHistogramProps) {
  const max = bins.reduce((acc, value) => (value > acc ? value : acc), 0);
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${bins.length}, 1fr)`,
        gap: '2px',
        alignItems: 'end',
        height: `${height}px`,
      }}
    >
      {bins.map((value, idx) => {
        const magnitude = max > 0 ? Math.max((value / max) * 100, 4) : 4;
        return (
          <span
            key={idx}
            style={{
              display: 'block',
              width: '100%',
              height: `${magnitude}%`,
              background: color,
              opacity: max > 0 ? 0.85 : 0.25,
              borderRadius: '0.25rem 0.25rem 0 0',
            }}
          />
        );
      })}
    </div>
  );
}

function TelemetryOverlayContents({ snapshot }: { snapshot: TelemetrySnapshot }) {
  const { su7, su7Histograms, su7Unitary, frameSamples } = snapshot;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'baseline',
          fontSize: '0.7rem',
          fontWeight: 600,
        }}
      >
        <span>SU7 overlay</span>
        <span style={{ fontSize: '0.65rem', color: '#cbd5f5' }}>n={frameSamples}</span>
      </div>
      <div style={{ fontSize: '0.65rem', color: '#cbd5f5' }}>
        Unitary frame {su7Unitary.latest.toExponential(2)} · μ {su7.unitaryError.toExponential(2)} ·
        max {su7Unitary.max.toExponential(2)}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '0.62rem',
            color: '#e9d5ff',
          }}
        >
          <span>Norm Δ mean</span>
          <span>{formatTelemetryRange(su7Histograms.normDeltaMean, 2)}</span>
        </div>
        <TinyHistogram bins={su7Histograms.normDeltaMean.bins} color="#a855f7" height={28} />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '0.62rem',
            color: '#fed7aa',
          }}
        >
          <span>Norm Δ max</span>
          <span>{formatTelemetryRange(su7Histograms.normDeltaMax, 2)}</span>
        </div>
        <TinyHistogram bins={su7Histograms.normDeltaMax.bins} color="#f97316" height={28} />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '0.62rem',
            color: '#bae6fd',
          }}
        >
          <span>Projector energy</span>
          <span>{formatTelemetryRange(su7Histograms.projectorEnergy, 2)}</span>
        </div>
        <TinyHistogram bins={su7Histograms.projectorEnergy.bins} color="#38bdf8" height={28} />
      </div>
    </div>
  );
}

const phaseHeatmapColor = (t: number): [number, number, number] => {
  const clamped = Math.min(1, Math.max(0, t));
  const r = Math.round(255 * clamped);
  const g = Math.round(255 * Math.sin(clamped * Math.PI));
  const b = Math.round(255 * (1 - clamped * 0.85));
  return [r, g, b];
};

const PhaseHeatmapPanel = ({ snapshot }: { snapshot: PhaseHeatmapSnapshot }) => {
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const { width, height, values, min, max } = snapshot;
    const range = Math.max(1e-6, max - min);
    const image = ctx.createImageData(width, height);
    for (let i = 0; i < values.length; i++) {
      const norm = (values[i] - min) / range;
      const [r, g, b] = phaseHeatmapColor(norm);
      const idx = i * 4;
      image.data[idx + 0] = r;
      image.data[idx + 1] = g;
      image.data[idx + 2] = b;
      image.data[idx + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }, [snapshot]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '0.75rem',
          color: '#cbd5f5',
        }}
      >
        <span>Phase alignment heatmap</span>
        <span>
          {snapshot.min.toFixed(3)} – {snapshot.max.toFixed(3)}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={snapshot.width}
        height={snapshot.height}
        style={{
          width: '160px',
          height: '160px',
          borderRadius: '0.6rem',
          border: '1px solid rgba(148,163,184,0.35)',
          background: 'rgba(15,23,42,0.35)',
        }}
      />
    </div>
  );
};

function SliderControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
  disabled = false,
}: SliderProps) {
  return (
    <div className="control" style={disabled ? { opacity: 0.55 } : undefined}>
      <label>{label}</label>
      <div className="control-row">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(parseFloat(event.target.value))}
        />
        <span>{format ? format(value) : value.toFixed(2)}</span>
      </div>
    </div>
  );
}

type ToggleProps = {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
};

function ToggleControl({ label, value, onChange, disabled = false }: ToggleProps) {
  return (
    <div className="control" style={disabled ? { opacity: 0.55 } : undefined}>
      <label>{label}</label>
      <div className="control-row">
        <input
          type="checkbox"
          checked={value}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span>{value ? 'On' : 'Off'}</span>
      </div>
    </div>
  );
}

type Option = { value: string; label: string };

type SelectProps = {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Option[];
};

function SelectControl({ label, value, onChange, options }: SelectProps) {
  return (
    <div className="control">
      <label>{label}</label>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        style={{
          padding: '0.45rem 0.6rem',
          borderRadius: '0.6rem',
          border: '1px solid rgba(148,163,184,0.35)',
          background: 'rgba(15,23,42,0.7)',
          color: '#e2e8f0',
        }}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}
