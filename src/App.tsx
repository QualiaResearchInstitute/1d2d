import React, { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  createDerivedViews,
  createKuramotoState,
  createNormalGenerator,
  derivedBufferSize,
  deriveKuramotoFields as deriveKuramotoFieldsCore,
  initKuramotoState,
  stepKuramotoState,
  createWavePlateStep,
  createPolarizerStep,
  createPolarizationMatrixStep,
  type PhaseField,
  type KuramotoParams,
  type KuramotoState,
  type KuramotoTelemetrySnapshot,
  type IrradianceFrameBuffer,
  type KuramotoInstrumentationSnapshot,
  type ThinElementSchedule,
  type PolarizationMatrix,
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
  resolveSu7Runtime,
  buildSu7UnitaryFromGains,
  toGpuOps,
  COMPOSER_FIELD_LIST,
  type RainbowFrameMetrics,
  type QualiaMetrics,
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
  sanitizeGateList,
  type Gate,
  type GateList,
  type HopfLensControlTarget,
  type HopfLensDescriptor,
  type Su7ProjectorDescriptor,
  type Su7RuntimeParams,
  type Su7Schedule,
  type Su7Telemetry,
  type Su7GuardrailEvent,
  type Su7GuardrailStatus,
} from './pipeline/su7/types.js';
import { DEFAULT_HOPF_LENSES, resolveHopfLenses } from './pipeline/su7/projector.js';
import { embedToC7 } from './pipeline/su7/embed.js';
import { detectFlickerGuardrail } from './pipeline/su7/math.js';
import { su7UnitaryColumnToPolarizationMatrix } from './pipeline/polarizationBridge.js';
import {
  computeFluxOverlayState,
  type FluxOverlayFrameData,
  type FluxSource,
} from './qcd/overlays';
import {
  initializeQcdRuntime,
  restoreQcdRuntime,
  runGpuSubstep,
  runCpuSweep,
  runTemperatureScan,
  buildQcdOverlay,
  buildQcdProbeFrame,
  buildQcdSnapshot,
  hashQcdSnapshot,
  deriveLatticeResolution,
  type QcdRuntimeState,
  type QcdSnapshot,
  type QcdAnnealConfig,
  type QcdObservables,
} from './qcd/runtime';
import type { ProbeTransportFrameData } from './qcd/probeTransport.js';
import { mulberry32 } from './qcd/updateCpu';
import {
  ensureSu7TileBuffer,
  ensureSu7VectorBuffers,
  fillSu7TileBuffer,
  fillSu7VectorBuffers,
  fillSu7VectorBuffersFromPacked,
  SU7_TILE_SIZE,
  SU7_TILE_TEXTURE_ROWS_PER_TILE,
  SU7_TILE_TEXTURE_WIDTH,
  type Su7VectorBuffers,
} from './pipeline/su7/gpuPacking.js';
import {
  Su7GpuKernel,
  packSu7Unitary,
  packSu7Vectors,
  SU7_GPU_KERNEL_VECTOR_STRIDE,
  Su7GpuKernelStats,
  Su7GpuKernelProfile,
  Su7GpuKernelWarningEvent,
} from './pipeline/su7/gpuKernel.js';
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
import {
  Timeline,
  TimelinePlayer,
  type TimelineFrameEvaluation,
  type TimelineInterpolation,
  type TimelineValue,
  serializeTimeline,
  deriveSeedFromHash,
} from './timeline/index.js';

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
  su7GateAppends: Gate[];
};

type TimelineParameterKind = 'number' | 'boolean' | 'enum';

type TimelineParameterOption = {
  value: string;
  label: string;
};

type TimelineParameterConfig = {
  id: string;
  label: string;
  kind: TimelineParameterKind;
  min?: number;
  max?: number;
  step?: number;
  options?: TimelineParameterOption[];
  getValue: () => TimelineValue;
};

type TimelineEditorKeyframe = {
  frame: number;
  value: TimelineValue;
};

type TimelineEditorLane = {
  id: string;
  label: string;
  kind: TimelineParameterKind;
  interpolation: TimelineInterpolation;
  keyframes: TimelineEditorKeyframe[];
};

const sortTimelineKeyframes = (
  entries: readonly TimelineEditorKeyframe[],
): TimelineEditorKeyframe[] => [...entries].sort((a, b) => a.frame - b.frame);

const formatTimelineValue = (value: TimelineValue | undefined): string => {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return 'NaN';
    }
    const abs = Math.abs(value);
    const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : abs >= 1 ? 2 : 3;
    return value.toFixed(digits);
  }
  if (typeof value === 'boolean') {
    return value ? 'True' : 'False';
  }
  if (typeof value === 'string') {
    return value;
  }
  return '—';
};

const coerceTimelineNumber = (value: TimelineValue, fallback = 0): number => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
};

type MacroEvent =
  | { kind: 'set'; target: string; value: TimelineValue; at: number }
  | { kind: 'action'; action: 'applyPreset'; presetName: string; at: number };

type MacroScript = {
  id: string;
  label: string;
  createdAt: string;
  events: MacroEvent[];
};

type HopfLensControlsProps = {
  lenses: HopfLensDescriptor[];
  metrics?: RainbowFrameMetrics['hopf']['lenses'];
  onAxisChange: (index: number, which: 'a' | 'b', value: number) => void;
  onBaseMixChange: (index: number, value: number) => void;
  onFiberMixChange: (index: number, value: number) => void;
  onControlTargetChange: (index: number, target: HopfLensControlTarget) => void;
};

const HopfLensControls: React.FC<HopfLensControlsProps> = ({
  lenses,
  metrics,
  onAxisChange,
  onBaseMixChange,
  onFiberMixChange,
  onControlTargetChange,
}) => {
  const axisOptions = useMemo(() => Array.from({ length: 7 }, (_, idx) => idx), []);
  return (
    <div
      className="control"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.75rem',
        background: 'rgba(15,23,42,0.35)',
        borderRadius: '0.85rem',
        padding: '0.75rem',
      }}
    >
      <h3 style={{ margin: 0, fontSize: '1rem', color: '#e2e8f0' }}>Hopf lenses</h3>
      {lenses.map((lens, index) => {
        const metric = metrics?.find((entry) => entry.index === index) ?? null;
        const baseText = metric
          ? metric.base.map((component) => component.toFixed(2)).join(', ')
          : '–';
        const fiberDeg = metric ? (metric.fiber * 180) / Math.PI : 0;
        const shareText = metric ? metric.share.toFixed(3) : '—';
        return (
          <div
            key={`hopf-lens-${index}`}
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.55rem',
              background: 'rgba(15,23,42,0.45)',
              borderRadius: '0.65rem',
              padding: '0.6rem 0.6rem 0.7rem',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'baseline',
                color: '#cbd5f5',
                fontSize: '0.85rem',
                fontWeight: 600,
              }}
            >
              <span>Lens {index + 1}</span>
              <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
                {(lens.label ?? '').trim()}
              </span>
            </div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(2, minmax(5rem, 1fr))',
                gap: '0.45rem',
                fontSize: '0.72rem',
                color: '#cbd5f5',
              }}
            >
              <label htmlFor={`hopf-axis-a-${index}`}>Axis A</label>
              <select
                id={`hopf-axis-a-${index}`}
                value={lens.axes[0]}
                onChange={(event) => onAxisChange(index, 'a', Number(event.target.value))}
                style={{
                  background: 'rgba(15,23,42,0.7)',
                  border: '1px solid rgba(148,163,184,0.35)',
                  color: '#e2e8f0',
                  borderRadius: '0.45rem',
                  padding: '0.3rem 0.45rem',
                }}
              >
                {axisOptions.map((axis) => (
                  <option key={`hopf-axis-a-${index}-${axis}`} value={axis}>
                    {axis + 1}
                  </option>
                ))}
              </select>
              <label htmlFor={`hopf-axis-b-${index}`}>Axis B</label>
              <select
                id={`hopf-axis-b-${index}`}
                value={lens.axes[1]}
                onChange={(event) => onAxisChange(index, 'b', Number(event.target.value))}
                style={{
                  background: 'rgba(15,23,42,0.7)',
                  border: '1px solid rgba(148,163,184,0.35)',
                  color: '#e2e8f0',
                  borderRadius: '0.45rem',
                  padding: '0.3rem 0.45rem',
                }}
              >
                {axisOptions.map((axis) => (
                  <option key={`hopf-axis-b-${index}-${axis}`} value={axis}>
                    {axis + 1}
                  </option>
                ))}
              </select>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <label style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Base mix</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem' }}>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={lens.baseMix ?? 1}
                  onChange={(event) => onBaseMixChange(index, Number(event.target.value))}
                />
                <span style={{ fontSize: '0.7rem', color: '#cbd5f5' }}>
                  {(lens.baseMix ?? 1).toFixed(2)}
                </span>
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <label style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Fiber mix</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem' }}>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={lens.fiberMix ?? 1}
                  onChange={(event) => onFiberMixChange(index, Number(event.target.value))}
                />
                <span style={{ fontSize: '0.7rem', color: '#cbd5f5' }}>
                  {(lens.fiberMix ?? 1).toFixed(2)}
                </span>
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              <label style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Control target</label>
              <select
                value={lens.controlTarget ?? 'none'}
                onChange={(event) =>
                  onControlTargetChange(index, event.target.value as HopfLensControlTarget)
                }
                style={{
                  background: 'rgba(15,23,42,0.7)',
                  border: '1px solid rgba(148,163,184,0.35)',
                  color: '#e2e8f0',
                  borderRadius: '0.45rem',
                  padding: '0.3rem 0.45rem',
                }}
              >
                <option value="none">None</option>
                <option value="base">Base (S²)</option>
                <option value="fiber">Fiber (S¹)</option>
              </select>
            </div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
                gap: '0.5rem',
                fontSize: '0.7rem',
                color: '#94a3b8',
              }}
            >
              <div>
                <strong style={{ color: '#cbd5f5' }}>Base</strong>
                <div>{baseText}</div>
              </div>
              <div>
                <strong style={{ color: '#cbd5f5' }}>Fiber</strong>
                <div>{metric ? `${fiberDeg.toFixed(1)}°` : '–'}</div>
              </div>
              <div>
                <strong style={{ color: '#cbd5f5' }}>Share</strong>
                <div>{shareText}</div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
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

type Su7GpuFrameResult = {
  width: number;
  height: number;
  vectors: Float32Array;
  norms: Float32Array;
  profile: Su7GpuKernelProfile | null;
  timestamp: number;
};

type GuardrailAuditKind = Su7GuardrailEvent['kind'] | 'manualReorthon' | 'manualAutoGain';

type GuardrailAuditEntry = {
  id: number;
  kind: GuardrailAuditKind;
  message: string;
  severity: 'info' | 'warn';
  timestamp: number;
};

type GuardrailConsoleState = {
  unitaryError: number;
  determinantDrift: number;
  frameTimeMs: number;
  energyEma: number;
  lastEnergy: number | null;
  auditLog: GuardrailAuditEntry[];
};

type GuardrailSummary = {
  message: string;
  severity: 'info' | 'warn';
};

const summarizeGuardrailEvent = (event: Su7GuardrailEvent): GuardrailSummary => {
  switch (event.kind) {
    case 'autoReorthon': {
      const before = event.before.toExponential(2);
      const after = event.after.toExponential(2);
      const message = event.forced
        ? `Manual re-orthon completed (${before} → ${after})`
        : `Auto re-orthon corrected unitary (${before} → ${after})`;
      return { message, severity: event.forced ? 'info' : 'warn' };
    }
    case 'autoGain': {
      const message = `Auto gain correction ${event.before.toFixed(3)} → ${event.after.toFixed(
        3,
      )} (target ${event.target.toFixed(3)})`;
      return { message, severity: 'info' };
    }
    case 'flicker': {
      const message = `Flicker guardrail ${event.frequencyHz.toFixed(1)} Hz Δ ${(event.deltaRatio * 100).toFixed(1)}%`;
      return { message, severity: 'warn' };
    }
    default:
      return { message: 'Guardrail event', severity: 'info' };
  }
};

const MAX_CURVATURE_STRENGTH = 0.95;
const HYPERBOLIC_GUIDE_SPACING_MIN = 0.25;
const HYPERBOLIC_GUIDE_SPACING_MAX = 2.5;
const DEFAULT_HYPERBOLIC_GUIDE_SPACING = 0.75;

type PresetEarlyVision = {
  dogEnabled: boolean;
  orientationEnabled: boolean;
  motionEnabled: boolean;
  opacity: number;
  dogSigma: number;
  dogRatio: number;
  dogGain: number;
  downsample: number;
  orientationGain: number;
  orientationSharpness: number;
  orientationCount: number;
  motionGain: number;
  frameModulo: number;
  viewMode: 'blend' | 'overlay';
};

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
  macroBinding: MacroBinding | null;
  macroKnobValue: number;
  earlyVision: PresetEarlyVision;
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
  polarizationEnabled: boolean;
  wavePlateEnabled: boolean;
  wavePlatePhaseDeg: number;
  wavePlateOrientationDeg: number;
  su7PolarizationEnabled: boolean;
  su7PolarizationColumn: number;
  su7PolarizationGain: number;
  su7PolarizationBlend: number;
  polarizerEnabled: boolean;
  polarizerOrientationDeg: number;
  polarizerExtinction: number;
};

type PresetQcd = {
  beta: number;
  stepsPerSecond: number;
  smearingAlpha: number;
  smearingIterations: number;
  overRelaxationSteps: number;
  baseSeed: number;
  depth: number;
  temporalExtent: number;
  batchLayers: number;
  temperatureSchedule: number[];
  snapshot: {
    data: QcdSnapshot;
    hash: string;
  } | null;
};

type PresetDeveloper = {
  selectedSyntheticCase: SyntheticCaseId;
};

type MacroBinding = {
  gateLabel: string;
  axis: number;
  thetaScale: number;
  phiScale: number;
  phiBase: number;
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
  qcd: PresetQcd;
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

const TELEMETRY_STREAM_QUEUE_LIMIT = 256;
const TELEMETRY_RECORDING_LIMIT = 7200;

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
  { value: 'hopflens', label: 'Hopf lens overlays' },
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
      su7GateAppends: [],
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
      macroBinding: null,
      macroKnobValue: 0,
      earlyVision: {
        dogEnabled: false,
        orientationEnabled: false,
        motionEnabled: false,
        opacity: 0.65,
        dogSigma: 1.2,
        dogRatio: 1.6,
        dogGain: 2.4,
        downsample: 1,
        orientationGain: 0.9,
        orientationSharpness: 2,
        orientationCount: 4,
        motionGain: 6,
        frameModulo: 1,
        viewMode: 'blend',
      },
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
      polarizationEnabled: false,
      wavePlateEnabled: true,
      wavePlatePhaseDeg: 90,
      wavePlateOrientationDeg: 0,
      su7PolarizationEnabled: false,
      su7PolarizationColumn: 0,
      su7PolarizationGain: 1,
      su7PolarizationBlend: 1,
      polarizerEnabled: false,
      polarizerOrientationDeg: 0,
      polarizerExtinction: 0,
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
    qcd: {
      beta: 5.25,
      stepsPerSecond: 3,
      smearingAlpha: 0.5,
      smearingIterations: 1,
      overRelaxationSteps: 1,
      baseSeed: 2024,
      depth: 1,
      temporalExtent: 1,
      batchLayers: 1,
      temperatureSchedule: [],
      snapshot: null,
    },
  },
  {
    name: 'Polarization SU7 Demo',
    params: {
      edgeThreshold: 0.1,
      blend: 0.42,
      kernel: createKernelSpec({
        gain: 2.6,
        k0: 0.24,
        Q: 4.2,
        anisotropy: 0.9,
        chirality: 1.25,
        transparency: 0.35,
      }),
      dmt: 0.3,
      arousal: 0.45,
      thetaMode: 'gradient',
      thetaGlobal: 0,
      beta2: 1.5,
      jitter: 0.9,
      sigma: 3.6,
      microsaccade: true,
      speed: 1.05,
      contrast: 1.55,
      phasePin: true,
      alive: true,
      rimAlpha: 0.92,
      su7Enabled: true,
      su7Gain: 1.2,
      su7Preset: 'identity',
      su7Seed: 11,
      su7Schedule: [],
      su7Projector: { id: 'hopfLens', weight: 0.65 },
      su7ScheduleStrength: 1,
      su7GateAppends: [],
    },
    surface: {
      surfEnabled: true,
      surfaceBlend: 0.4,
      warpAmp: 0.95,
      nOrient: 5,
      wallGroup: 'p4m',
      surfaceRegion: 'both',
    },
    display: {
      displayMode: 'colorBaseBlendedRims',
      polBins: 24,
      normPin: true,
      curvatureStrength: 0.2,
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
      frameLoggingEnabled: false,
      macroBinding: null,
      macroKnobValue: 0,
      earlyVision: {
        dogEnabled: false,
        orientationEnabled: false,
        motionEnabled: false,
        opacity: 0.65,
        dogSigma: 1.2,
        dogRatio: 1.6,
        dogGain: 2.4,
        downsample: 1,
        orientationGain: 0.9,
        orientationSharpness: 2,
        orientationCount: 4,
        motionGain: 6,
        frameModulo: 1,
        viewMode: 'blend',
      },
    },
    kuramoto: {
      kurEnabled: true,
      kurSync: true,
      kurRegime: 'custom',
      K0: 0.72,
      alphaKur: 0.24,
      gammaKur: 0.18,
      omega0: 0.08,
      epsKur: 0.0018,
      fluxX: Math.PI / 3,
      fluxY: Math.PI / 6,
      qInit: 1,
      smallWorldEnabled: false,
      smallWorldWeight: 0.45,
      p_sw: 0,
      smallWorldSeed: 2048,
      smallWorldDegree: 10,
      polarizationEnabled: true,
      wavePlateEnabled: true,
      wavePlatePhaseDeg: 90,
      wavePlateOrientationDeg: 45,
      su7PolarizationEnabled: true,
      su7PolarizationColumn: 2,
      su7PolarizationGain: 1.1,
      su7PolarizationBlend: 0.85,
      polarizerEnabled: true,
      polarizerOrientationDeg: 20,
      polarizerExtinction: 0.08,
    },
    developer: {
      selectedSyntheticCase: 'waves',
    },
    media: null,
    coupling: {
      rimToSurfaceBlend: 0.42,
      rimToSurfaceAlign: 0.5,
      surfaceToRimOffset: 0.35,
      surfaceToRimSigma: 0.55,
      surfaceToRimHue: 0.48,
      kurToTransparency: 0.4,
      kurToOrientation: 0.45,
      kurToChirality: 0.55,
      volumePhaseToHue: 0.4,
      volumeDepthToWarp: 0.33,
    },
    composer: createDefaultComposerConfig(),
    couplingToggles: DEFAULT_COUPLING_TOGGLES,
    qcd: {
      beta: 5.25,
      stepsPerSecond: 3,
      smearingAlpha: 0.5,
      smearingIterations: 1,
      overRelaxationSteps: 1,
      baseSeed: 2024,
      depth: 1,
      temporalExtent: 1,
      batchLayers: 1,
      temperatureSchedule: [],
      snapshot: null,
    },
  },
  {
    name: 'Early Vision Circles Diagnostic',
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
      su7GateAppends: [],
    },
    surface: {
      surfEnabled: false,
      surfaceBlend: 0.35,
      warpAmp: 1.0,
      nOrient: 4,
      wallGroup: 'p4',
      surfaceRegion: 'both',
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
      macroBinding: null,
      macroKnobValue: 0,
      earlyVision: {
        dogEnabled: true,
        orientationEnabled: true,
        motionEnabled: true,
        opacity: 0.7,
        dogSigma: 1.6,
        dogRatio: 1.6,
        dogGain: 3.0,
        downsample: 2,
        orientationGain: 1.2,
        orientationSharpness: 2.5,
        orientationCount: 6,
        motionGain: 7.5,
        frameModulo: 2,
        viewMode: 'blend',
      },
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
      polarizationEnabled: false,
      wavePlateEnabled: true,
      wavePlatePhaseDeg: 90,
      wavePlateOrientationDeg: 0,
      su7PolarizationEnabled: false,
      su7PolarizationColumn: 0,
      su7PolarizationGain: 1,
      su7PolarizationBlend: 1,
      polarizerEnabled: false,
      polarizerOrientationDeg: 0,
      polarizerExtinction: 0,
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
    qcd: {
      beta: 5.25,
      stepsPerSecond: 3,
      smearingAlpha: 0.5,
      smearingIterations: 1,
      overRelaxationSteps: 1,
      baseSeed: 2024,
      depth: 1,
      temporalExtent: 1,
      batchLayers: 1,
      temperatureSchedule: [],
      snapshot: null,
    },
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
const TAU = Math.PI * 2;

const wrapAngle = (value: number): number => {
  if (!Number.isFinite(value)) {
    return 0;
  }
  let angle = value % TAU;
  if (angle <= -Math.PI) {
    angle += TAU;
  } else if (angle > Math.PI) {
    angle -= TAU;
  }
  return angle;
};

const radiansToDegrees = (value: number): number => (value / Math.PI) * 180;

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

const cloneMacroBinding = (binding: MacroBinding | null): MacroBinding | null => {
  if (!binding) return null;
  return {
    gateLabel: binding.gateLabel,
    axis: binding.axis,
    thetaScale: binding.thetaScale,
    phiScale: binding.phiScale,
    phiBase: binding.phiBase,
  };
};

const sanitizeMacroBinding = (
  value: unknown,
  fallback: MacroBinding | null,
): MacroBinding | null => {
  if (!value || typeof value !== 'object') {
    return fallback ? cloneMacroBinding(fallback) : null;
  }
  const source = value as Partial<MacroBinding> & { [key: string]: unknown };
  const labelSource = source.gateLabel;
  const gateLabel =
    typeof labelSource === 'string' && labelSource.length > 0
      ? labelSource
      : (fallback?.gateLabel ?? '');
  if (!gateLabel) {
    return fallback ? cloneMacroBinding(fallback) : null;
  }
  const axisSource = source.axis;
  const axis =
    typeof axisSource === 'number' && Number.isFinite(axisSource)
      ? Math.max(0, Math.min(6, Math.trunc(axisSource)))
      : (fallback?.axis ?? 0);
  const thetaScaleSource = source.thetaScale;
  const phiScaleSource = source.phiScale;
  const phiBaseSource = source.phiBase;
  const thetaScale =
    typeof thetaScaleSource === 'number' && Number.isFinite(thetaScaleSource)
      ? thetaScaleSource
      : (fallback?.thetaScale ?? 0);
  const phiScale =
    typeof phiScaleSource === 'number' && Number.isFinite(phiScaleSource)
      ? phiScaleSource
      : (fallback?.phiScale ?? 0);
  const phiBaseValue =
    typeof phiBaseSource === 'number' && Number.isFinite(phiBaseSource)
      ? phiBaseSource
      : (fallback?.phiBase ?? 0);
  const phiBase = wrapAngle(phiBaseValue);
  return {
    gateLabel,
    axis,
    thetaScale,
    phiScale,
    phiBase,
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

const sanitizeQcdSnapshot = (
  raw: unknown,
  fallback: PresetQcd['snapshot'],
): PresetQcd['snapshot'] => {
  if (!raw || typeof raw !== 'object') {
    return fallback ?? null;
  }
  const source = raw as { data?: unknown; hash?: unknown };
  if (!source.data || typeof source.data !== 'object') {
    return fallback ?? null;
  }
  try {
    const snapshot = source.data as QcdSnapshot;
    restoreQcdRuntime(snapshot);
    const hash = typeof source.hash === 'string' ? (source.hash as string) : (fallback?.hash ?? '');
    return {
      data: snapshot,
      hash,
    };
  } catch {
    return fallback ?? null;
  }
};

const sanitizePresetQcd = (raw: unknown, fallback: PresetQcd): PresetQcd => {
  if (!raw || typeof raw !== 'object') {
    return fallback;
  }
  const source = raw as Record<string, unknown>;
  const beta = sanitizeNumber(source.beta, fallback.beta);
  const steps = sanitizeNumber(source.stepsPerSecond, fallback.stepsPerSecond);
  const smearingAlpha = sanitizeNumber(source.smearingAlpha, fallback.smearingAlpha);
  const smearingIterations = clampInt(
    sanitizeNumber(source.smearingIterations, fallback.smearingIterations),
    0,
    32,
  );
  const overRelaxationSteps = clampInt(
    sanitizeNumber(source.overRelaxationSteps, fallback.overRelaxationSteps),
    0,
    16,
  );
  const baseSeed = clampInt(
    sanitizeNumber(source.baseSeed, fallback.baseSeed),
    0,
    Number.MAX_SAFE_INTEGER,
  );
  const depth = clampInt(sanitizeNumber(source.depth, fallback.depth), 1, 64);
  const temporalExtent = clampInt(
    sanitizeNumber(source.temporalExtent, fallback.temporalExtent),
    1,
    256,
  );
  const planeCap = Math.max(1, depth * temporalExtent);
  const batchLayers = clampInt(
    sanitizeNumber(source.batchLayers, fallback.batchLayers),
    1,
    planeCap,
  );
  const scheduleSource = Array.isArray(source.temperatureSchedule)
    ? (source.temperatureSchedule as unknown[])
    : fallback.temperatureSchedule;
  const temperatureSchedule = Array.isArray(scheduleSource)
    ? scheduleSource
        .map((value) => {
          if (typeof value === 'number') return value;
          if (typeof value === 'string') {
            const parsed = Number.parseFloat(value);
            return Number.isFinite(parsed) ? parsed : null;
          }
          return null;
        })
        .filter((value): value is number => value != null)
        .slice(0, 64)
    : [...fallback.temperatureSchedule];
  const snapshot = sanitizeQcdSnapshot(source.snapshot, fallback.snapshot);
  return {
    beta,
    stepsPerSecond: clamp(steps, 0.1, 60),
    smearingAlpha: clamp(smearingAlpha, 0, 1),
    smearingIterations,
    overRelaxationSteps,
    baseSeed,
    depth,
    temporalExtent,
    batchLayers,
    temperatureSchedule,
    snapshot,
  };
};

const sanitizePresetPayload = (raw: unknown, fallback: Preset): Preset | null => {
  if (!raw || typeof raw !== 'object') return null;
  const source = raw as Record<string, unknown>;
  const paramsSource = (source.params as Record<string, unknown>) ?? {};
  const kernelSource = (paramsSource.kernel as Record<string, unknown>) ?? {};
  const surfaceSource = (source.surface as Record<string, unknown>) ?? {};
  const displaySource = (source.display as Record<string, unknown>) ?? {};
  const tracerSource = source.tracer as TracerConfig | undefined;
  const runtimeSource = (source.runtime as Record<string, unknown>) ?? {};
  const kuramotoSource = (source.kuramoto as Record<string, unknown>) ?? {};
  const developerSource = (source.developer as Record<string, unknown>) ?? {};
  const togglesSource = (source.couplingToggles as Record<string, unknown>) ?? {};

  const fallbackParams = fallback.params;
  const fallbackSurface = fallback.surface;
  const fallbackDisplay = fallback.display;
  const fallbackRuntime = fallback.runtime;
  const earlyVisionSource = (runtimeSource.earlyVision as Record<string, unknown>) ?? {};
  const fallbackEarlyVision = fallbackRuntime.earlyVision;
  const fallbackKuramoto = fallback.kuramoto;
  const fallbackDeveloper = fallback.developer;
  const fallbackQcd = fallback.qcd;
  const fallbackToggles = fallback.couplingToggles;
  const fallbackSu7: Su7RuntimeParams = {
    enabled: fallbackParams.su7Enabled,
    gain: fallbackParams.su7Gain,
    preset: fallbackParams.su7Preset,
    seed: fallbackParams.su7Seed,
    schedule: fallbackParams.su7Schedule,
    projector: fallbackParams.su7Projector,
    gateAppends: fallbackParams.su7GateAppends,
  };
  const sanitizedSu7 = sanitizeSu7RuntimeParams(
    {
      enabled: paramsSource.su7Enabled,
      gain: paramsSource.su7Gain,
      preset: paramsSource.su7Preset,
      seed: paramsSource.su7Seed,
      schedule: paramsSource.su7Schedule as unknown,
      projector: paramsSource.su7Projector,
      gateAppends: paramsSource.su7GateAppends,
    } as Partial<Su7RuntimeParams>,
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
      su7GateAppends: sanitizedSu7.gateAppends,
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
    tracer: sanitizeTracerConfig(tracerSource ?? fallback.tracer),
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
      macroBinding: sanitizeMacroBinding(
        runtimeSource.macroBinding,
        fallbackRuntime.macroBinding ?? null,
      ),
      macroKnobValue: sanitizeNumber(
        runtimeSource.macroKnobValue,
        typeof fallbackRuntime.macroKnobValue === 'number' ? fallbackRuntime.macroKnobValue : 0,
      ),
      earlyVision: {
        dogEnabled: sanitizeBoolean(earlyVisionSource.dogEnabled, fallbackEarlyVision.dogEnabled),
        orientationEnabled: sanitizeBoolean(
          earlyVisionSource.orientationEnabled,
          fallbackEarlyVision.orientationEnabled,
        ),
        motionEnabled: sanitizeBoolean(
          earlyVisionSource.motionEnabled,
          fallbackEarlyVision.motionEnabled,
        ),
        opacity: sanitizeNumber(earlyVisionSource.opacity, fallbackEarlyVision.opacity),
        dogSigma: sanitizeNumber(earlyVisionSource.dogSigma, fallbackEarlyVision.dogSigma),
        dogRatio: sanitizeNumber(earlyVisionSource.dogRatio, fallbackEarlyVision.dogRatio),
        dogGain: sanitizeNumber(earlyVisionSource.dogGain, fallbackEarlyVision.dogGain),
        downsample: sanitizeNumber(earlyVisionSource.downsample, fallbackEarlyVision.downsample),
        orientationGain: sanitizeNumber(
          earlyVisionSource.orientationGain,
          fallbackEarlyVision.orientationGain,
        ),
        orientationSharpness: sanitizeNumber(
          earlyVisionSource.orientationSharpness,
          fallbackEarlyVision.orientationSharpness,
        ),
        orientationCount: clampInt(
          sanitizeNumber(earlyVisionSource.orientationCount, fallbackEarlyVision.orientationCount),
          1,
          8,
        ),
        motionGain: sanitizeNumber(earlyVisionSource.motionGain, fallbackEarlyVision.motionGain),
        frameModulo: clampInt(
          sanitizeNumber(earlyVisionSource.frameModulo, fallbackEarlyVision.frameModulo),
          1,
          16,
        ),
        viewMode: sanitizeEnum(
          earlyVisionSource.viewMode,
          ['blend', 'overlay'],
          fallbackEarlyVision.viewMode,
        ),
      },
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
      polarizationEnabled: sanitizeBoolean(
        kuramotoSource.polarizationEnabled,
        fallbackKuramoto.polarizationEnabled ?? false,
      ),
      wavePlateEnabled: sanitizeBoolean(
        kuramotoSource.wavePlateEnabled,
        fallbackKuramoto.wavePlateEnabled ?? true,
      ),
      wavePlatePhaseDeg: sanitizeNumber(
        kuramotoSource.wavePlatePhaseDeg,
        fallbackKuramoto.wavePlatePhaseDeg ?? 90,
      ),
      wavePlateOrientationDeg: sanitizeNumber(
        kuramotoSource.wavePlateOrientationDeg,
        fallbackKuramoto.wavePlateOrientationDeg ?? 0,
      ),
      su7PolarizationEnabled: sanitizeBoolean(
        kuramotoSource.su7PolarizationEnabled,
        fallbackKuramoto.su7PolarizationEnabled ?? false,
      ),
      su7PolarizationColumn: clampInt(
        sanitizeNumber(
          kuramotoSource.su7PolarizationColumn,
          fallbackKuramoto.su7PolarizationColumn ?? 0,
        ),
        0,
        6,
      ),
      su7PolarizationGain: sanitizeNumber(
        kuramotoSource.su7PolarizationGain,
        fallbackKuramoto.su7PolarizationGain ?? 1,
      ),
      su7PolarizationBlend: clamp(
        sanitizeNumber(
          kuramotoSource.su7PolarizationBlend,
          fallbackKuramoto.su7PolarizationBlend ?? 1,
        ),
        0,
        1,
      ),
      polarizerEnabled: sanitizeBoolean(
        kuramotoSource.polarizerEnabled,
        fallbackKuramoto.polarizerEnabled ?? false,
      ),
      polarizerOrientationDeg: sanitizeNumber(
        kuramotoSource.polarizerOrientationDeg,
        fallbackKuramoto.polarizerOrientationDeg ?? 0,
      ),
      polarizerExtinction: clamp(
        sanitizeNumber(
          kuramotoSource.polarizerExtinction,
          fallbackKuramoto.polarizerExtinction ?? 0,
        ),
        0,
        1,
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
    qcd: sanitizePresetQcd(source.qcd, fallbackQcd),
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
  const parts: BlobPart[] = [];
  const nowSeconds = Math.floor(Date.now() / 1000);
  const pushBytes = (bytes: Uint8Array) => {
    const needsCopy = bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength;
    if (needsCopy) {
      parts.push(bytes.slice().buffer as ArrayBuffer);
    } else {
      parts.push(bytes.buffer as ArrayBuffer);
    }
  };

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
    pushBytes(header);
    pushBytes(content);
    const remainder = content.length % TAR_BLOCK_SIZE;
    if (remainder !== 0) {
      pushBytes(new Uint8Array(TAR_BLOCK_SIZE - remainder));
    }
  });

  pushBytes(new Uint8Array(TAR_BLOCK_SIZE));
  pushBytes(new Uint8Array(TAR_BLOCK_SIZE));

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

const formatCouplingKey = (key: keyof CouplingConfig) => {
  const label = String(key);
  return label.replace(/([A-Z])/g, ' $1').replace(/^./, (ch) => ch.toUpperCase());
};

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

type TelemetryExportEntry = {
  timestamp: number;
  backend: FrameMetricsEntry['backend'];
  kernelVersion: number;
  qualia: QualiaMetrics;
  motion: RainbowFrameMetrics['motionEnergy'];
  parallax: RainbowFrameMetrics['parallax'];
  texture: {
    wallpapericity: number;
    beatEnergy: number;
    sampleCount: number;
  };
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

const FALLBACK_TIMELINE_HASH = '0000000000000000000000000000000000000000000000000000000000000000';

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const hyperbolicGridCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const presetFileInputRef = useRef<HTMLInputElement | null>(null);
  const gpuStateRef = useRef<{ gl: WebGL2RenderingContext; renderer: GpuRenderer } | null>(null);
  const pendingStaticUploadRef = useRef(true);
  const su7VectorBuffersRef = useRef<Su7VectorBuffers | null>(null);
  const su7TileBufferRef = useRef<Float32Array | null>(null);
  const su7GpuKernelRef = useRef<Su7GpuKernel | null>(null);
  const su7GpuPackedMatrixRef = useRef<Float32Array | null>(null);
  const su7GpuPackedInputRef = useRef<Float32Array | null>(null);
  const su7GpuTransformedRef = useRef<Float32Array | null>(null);
  const su7GpuPendingRef = useRef<Promise<void> | null>(null);
  const su7GpuReadyRef = useRef<Su7GpuFrameResult | null>(null);
  const su7GpuStatsRef = useRef<Su7GpuKernelStats | null>(null);
  const su7GpuWarningRef = useRef<Su7GpuKernelWarningEvent | null>(null);
  const su7GpuLastProfileRef = useRef<Su7GpuKernelProfile | null>(null);

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
  const earlyVisionFrameRef = useRef(0);
  const earlyVisionForceUpdateRef = useRef(false);
  const timelinePlayerRef = useRef<TimelinePlayer | null>(null);
  const timelineEvaluationRef = useRef<TimelineFrameEvaluation | null>(null);
  const timelineHashRef = useRef<string>(FALLBACK_TIMELINE_HASH);
  const timelineActiveRef = useRef(false);
  const timelineLastFrameRef = useRef<number | null>(null);
  const timelineClockRef = useRef(0);
  const [timelineCurrentTime, setTimelineCurrentTime] = useState(0);
  const [timelinePlaying, setTimelinePlaying] = useState(false);
  const [timelineLoop, setTimelineLoop] = useState(true);
  const [timelineFps, setTimelineFps] = useState(60);
  const [timelineDurationSeconds, setTimelineDurationSeconds] = useState(10);
  const [timelineAutoKeyframe, setTimelineAutoKeyframe] = useState(true);
  const [timelineLanes, setTimelineLanes] = useState<TimelineEditorLane[]>([]);
  const [timelineEvaluationState, setTimelineEvaluationState] =
    useState<TimelineFrameEvaluation | null>(null);
  const [timelineActive, setTimelineActive] = useState(false);
  const durationFrames = useMemo(
    () => Math.max(1, Math.round(timelineDurationSeconds * timelineFps)),
    [timelineDurationSeconds, timelineFps],
  );

  const [macroRecording, setMacroRecording] = useState(false);
  const macroRecordingRef = useRef(false);
  const macroStartTimeRef = useRef(0);
  const [macroEvents, setMacroEvents] = useState<MacroEvent[]>([]);
  const macroEventsRef = useRef<MacroEvent[]>([]);
  const [macroLibrary, setMacroLibrary] = useState<MacroScript[]>([]);
  const macroPlaybackRef = useRef<{ timers: number[] } | null>(null);
  const [macroPlaybackId, setMacroPlaybackId] = useState<string | null>(null);
  const macroIdCounterRef = useRef(1);

  const recordMacroEvent = useCallback(
    (entry: Omit<MacroEvent, 'at'>) => {
      if (!macroRecordingRef.current) {
        return;
      }
      const now =
        typeof performance !== 'undefined' && typeof performance.now === 'function'
          ? performance.now()
          : Date.now();
      const at = Math.max(0, now - macroStartTimeRef.current);
      const event: MacroEvent = { ...entry, at };
      macroEventsRef.current = [...macroEventsRef.current, event];
      setMacroEvents((prev) => [...prev, event]);
    },
    [setMacroEvents],
  );

  const recordMacroParameterChange = useCallback(
    (target: string, value: TimelineValue) => {
      recordMacroEvent({ kind: 'set', target, value });
    },
    [recordMacroEvent],
  );

  const recordMacroPresetAction = useCallback(
    (presetName: string) => {
      recordMacroEvent({ kind: 'action', action: 'applyPreset', presetName });
    },
    [recordMacroEvent],
  );

  const cancelMacroPlayback = useCallback(() => {
    const playback = macroPlaybackRef.current;
    if (playback) {
      playback.timers.forEach((timerId) => globalThis.clearTimeout(timerId));
      macroPlaybackRef.current = null;
    }
    setMacroPlaybackId(null);
  }, []);

  const startMacroRecording = useCallback(() => {
    if (macroRecordingRef.current) {
      return;
    }
    cancelMacroPlayback();
    macroRecordingRef.current = true;
    setMacroRecording(true);
    macroEventsRef.current = [];
    setMacroEvents([]);
    macroStartTimeRef.current =
      typeof performance !== 'undefined' && typeof performance.now === 'function'
        ? performance.now()
        : Date.now();
  }, [cancelMacroPlayback]);

  const stopMacroRecording = useCallback(
    (label?: string) => {
      if (!macroRecordingRef.current) {
        return;
      }
      macroRecordingRef.current = false;
      setMacroRecording(false);
      const recorded = macroEventsRef.current;
      if (recorded.length > 0) {
        const macroIndex = macroIdCounterRef.current++;
        const script: MacroScript = {
          id: `macro-${macroIndex}`,
          label: label ?? `Macro ${macroIndex}`,
          createdAt: new Date().toISOString(),
          events: recorded.map((event) => ({ ...event })),
        };
        setMacroLibrary((prev) => [...prev, script]);
      }
      macroEventsRef.current = [];
      setMacroEvents([]);
    },
    [setMacroLibrary],
  );

  const cancelMacroRecording = useCallback(() => {
    if (!macroRecordingRef.current) {
      return;
    }
    macroRecordingRef.current = false;
    setMacroRecording(false);
    macroEventsRef.current = [];
    setMacroEvents([]);
  }, []);

  useEffect(() => {
    if (macroLibrary.length === 0 && PRESETS.length > 0) {
      const demoEvents: MacroEvent[] = PRESETS.slice(0, Math.min(3, PRESETS.length)).map(
        (preset, index) => ({
          kind: 'action',
          action: 'applyPreset',
          presetName: preset.name,
          at: index * 2000,
        }),
      );
      const demoMacro: MacroScript = {
        id: 'macro-demo-cycle-presets',
        label: 'Cycle Presets Demo',
        createdAt: new Date().toISOString(),
        events: demoEvents,
      };
      setMacroLibrary([demoMacro]);
    }
  }, [macroLibrary.length]);

  const getTimelineNumber = (id: string, fallback: number): number => {
    const evaluation = timelineEvaluationRef.current;
    const value = evaluation?.values[id];
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
    return fallback;
  };

  const getTimelineBoolean = (id: string, fallback: boolean): boolean => {
    const evaluation = timelineEvaluationRef.current;
    const value = evaluation?.values[id];
    if (typeof value === 'boolean') {
      return value;
    }
    return fallback;
  };

  const getTimelineString = <T extends string>(id: string, fallback: T): T => {
    const evaluation = timelineEvaluationRef.current;
    const value = evaluation?.values[id];
    if (typeof value === 'string') {
      return value as T;
    }
    return fallback;
  };

  const updateTimelineForTime = useCallback(
    (timeSeconds: number): TimelineFrameEvaluation | null => {
      const player = timelinePlayerRef.current;
      if (!player || !timelineActiveRef.current) {
        if (timelineEvaluationRef.current || timelineLastFrameRef.current !== null) {
          timelineEvaluationRef.current = null;
          timelineLastFrameRef.current = null;
          setTimelineEvaluationState(null);
        }
        return null;
      }
      const evaluation = player.evaluateAtTimeSeconds(timeSeconds);
      const previous = timelineEvaluationRef.current;
      let changed = false;
      if (!previous) {
        changed = true;
      } else if (previous.frameIndex !== evaluation.frameIndex) {
        changed = true;
      } else {
        const prevValues = previous.values;
        const nextValues = evaluation.values;
        const prevKeys = Object.keys(prevValues);
        const nextKeys = Object.keys(nextValues);
        if (prevKeys.length !== nextKeys.length) {
          changed = true;
        } else {
          for (const key of nextKeys) {
            if (nextValues[key] !== prevValues[key]) {
              changed = true;
              break;
            }
          }
        }
      }
      timelineEvaluationRef.current = evaluation;
      timelineLastFrameRef.current = evaluation.frameIndex;
      if (changed) {
        setTimelineEvaluationState(evaluation);
      }
      return timelineEvaluationRef.current;
    },
    [setTimelineEvaluationState],
  );

  const getTimelineSeed = useCallback((scope: string, timeSeconds: number): number => {
    const player = timelinePlayerRef.current;
    if (player && timelineActiveRef.current) {
      return player.getSeedAtTime(scope, timeSeconds);
    }
    const frame =
      player && timelineActiveRef.current
        ? player.getFrameForTime(timeSeconds)
        : Math.max(0, Math.round(timeSeconds * RECORDING_FPS));
    return deriveSeedFromHash(timelineHashRef.current, scope, frame);
  }, []);

  const loadTimeline = useCallback(
    (timeline: Timeline) => {
      const player = new TimelinePlayer(timeline);
      timelinePlayerRef.current = player;
      timelineHashRef.current = player.hash;
      timelineActiveRef.current = true;
      const currentFrame = player.getFrameForTime(timelineClockRef.current);
      const evaluation = player.evaluateAtFrame(currentFrame);
      timelineEvaluationRef.current = evaluation;
      timelineLastFrameRef.current = evaluation.frameIndex;
      setTimelineEvaluationState(evaluation);
      setTimelineActive(true);
      console.info(
        `[timeline] loaded hash=${player.hash} fps=${player.fps} frames=${player.durationFrames}`,
      );
      return { hash: player.hash, fps: player.fps, durationFrames: player.durationFrames };
    },
    [setTimelineActive, setTimelineEvaluationState],
  );

  const clearTimeline = useCallback(() => {
    timelinePlayerRef.current = null;
    timelineActiveRef.current = false;
    timelineEvaluationRef.current = null;
    timelineLastFrameRef.current = null;
    timelineHashRef.current = FALLBACK_TIMELINE_HASH;
    setTimelineActive(false);
    setTimelineEvaluationState(null);
    setTimelinePlaying(false);
    console.info('[timeline] cleared');
  }, [setTimelineActive, setTimelineEvaluationState, setTimelinePlaying]);

  const exportTimeline = useCallback(() => {
    const player = timelinePlayerRef.current;
    if (!player) {
      return null;
    }
    const { json, hash } = serializeTimeline(player.timeline, { indent: 2 });
    return { json, hash, fps: player.fps, durationFrames: player.durationFrames };
  }, []);

  const loadTimelineFromJson = useCallback(
    (json: string) => {
      const parsed = JSON.parse(json) as Timeline;
      return loadTimeline(parsed);
    },
    [loadTimeline],
  );

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
  const [earlyVisionDogEnabled, setEarlyVisionDogEnabled] = useState(false);
  const [earlyVisionOrientationEnabled, setEarlyVisionOrientationEnabled] = useState(false);
  const [earlyVisionMotionEnabled, setEarlyVisionMotionEnabled] = useState(false);
  const [earlyVisionOpacity, setEarlyVisionOpacity] = useState(0.65);
  const [earlyVisionDoGSigma, setEarlyVisionDoGSigma] = useState(1.2);
  const [earlyVisionDoGRatio, setEarlyVisionDoGRatio] = useState(1.6);
  const [earlyVisionDoGGain, setEarlyVisionDoGGain] = useState(2.4);
  const [earlyVisionDownsample, setEarlyVisionDownsample] = useState(1);
  const [earlyVisionOrientationGain, setEarlyVisionOrientationGain] = useState(0.9);
  const [earlyVisionOrientationSharpness, setEarlyVisionOrientationSharpness] = useState(2);
  const [earlyVisionOrientationCount, setEarlyVisionOrientationCount] = useState(4);
  const [earlyVisionMotionGain, setEarlyVisionMotionGain] = useState(6);
  const [earlyVisionFrameModulo, setEarlyVisionFrameModulo] = useState(1);
  const [earlyVisionViewMode, setEarlyVisionViewMode] = useState<'blend' | 'overlay'>('blend');
  const earlyVisionOrientationAngles = useMemo(() => {
    const count = Math.min(8, Math.max(1, earlyVisionOrientationCount));
    const result = new Float32Array(count);
    for (let index = 0; index < count; index += 1) {
      result[index] = (Math.PI * index) / count;
    }
    return result;
  }, [earlyVisionOrientationCount]);
  const earlyVisionOrientationCos = useMemo(() => {
    const result = new Float32Array(earlyVisionOrientationAngles.length);
    earlyVisionOrientationAngles.forEach((angle, index) => {
      result[index] = Math.cos(angle);
    });
    return result;
  }, [earlyVisionOrientationAngles]);
  const earlyVisionOrientationSin = useMemo(() => {
    const result = new Float32Array(earlyVisionOrientationAngles.length);
    earlyVisionOrientationAngles.forEach((angle, index) => {
      result[index] = Math.sin(angle);
    });
    return result;
  }, [earlyVisionOrientationAngles]);
  useEffect(() => {
    earlyVisionForceUpdateRef.current = true;
  }, [
    earlyVisionDogEnabled,
    earlyVisionOrientationEnabled,
    earlyVisionMotionEnabled,
    earlyVisionOpacity,
    earlyVisionDoGSigma,
    earlyVisionDoGRatio,
    earlyVisionDoGGain,
    earlyVisionDownsample,
    earlyVisionOrientationGain,
    earlyVisionOrientationSharpness,
    earlyVisionOrientationCount,
    earlyVisionMotionGain,
    earlyVisionFrameModulo,
    earlyVisionViewMode,
  ]);
  useEffect(() => {
    earlyVisionForceUpdateRef.current = true;
    earlyVisionFrameRef.current = 0;
  }, [width, height]);

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
  const gateLabelCounterRef = useRef(0);
  const [macroBinding, setMacroBinding] = useState<MacroBinding | null>(null);
  const [macroKnobValue, setMacroKnobValue] = useState(0);
  const [macroLearnMode, setMacroLearnMode] = useState(false);
  const guardrailAuditIdRef = useRef(0);
  const guardrailFrameTimeRef = useRef(0);
  const guardrailCommandsRef = useRef<{ forceReorthon: boolean }>({ forceReorthon: false });
  const [guardrailConsoleInternal, setGuardrailConsoleInternal] = useState<GuardrailConsoleState>({
    unitaryError: 0,
    determinantDrift: 0,
    frameTimeMs: 0,
    energyEma: 0,
    lastEnergy: null,
    auditLog: [],
  });
  const guardrailConsoleRef = useRef(guardrailConsoleInternal);
  useEffect(() => {
    guardrailConsoleRef.current = guardrailConsoleInternal;
  }, [guardrailConsoleInternal]);
  const setGuardrailConsole = useCallback(
    (updater: (prev: GuardrailConsoleState) => GuardrailConsoleState) => {
      setGuardrailConsoleInternal((prev) => {
        const next = updater(prev);
        guardrailConsoleRef.current = next;
        return next;
      });
    },
    [setGuardrailConsoleInternal],
  );
  const guardrailConsole = guardrailConsoleInternal;
  const nextGateLabel = useCallback((prefix: string) => {
    const id = gateLabelCounterRef.current++;
    return `${prefix}-${id}`;
  }, []);
  const updateSu7GateAppends = useCallback(
    (mutator: (gates: Gate[]) => Gate[]) => {
      setSu7Params((prev) => {
        const current = prev.gateAppends ?? [];
        const next = mutator(current);
        if (next === current) {
          return prev;
        }
        return {
          ...prev,
          gateAppends: next,
        };
      });
    },
    [setSu7Params],
  );
  const clearSu7GateAppends = useCallback(() => {
    setSu7Params((prev) => ({ ...prev, gateAppends: [] }));
    setMacroBinding(null);
    setMacroKnobValue(0);
  }, [setSu7Params, setMacroBinding, setMacroKnobValue]);
  const removeSu7Gate = useCallback(
    (label: string | null, index: number) => {
      updateSu7GateAppends((current) => {
        if (label) {
          return current.filter((gate) => gate.label !== label);
        }
        return current.filter((_, idx) => idx !== index);
      });
      if (label && macroBinding?.gateLabel === label) {
        setMacroBinding(null);
        setMacroKnobValue(0);
      }
    },
    [macroBinding, setMacroBinding, setMacroKnobValue, updateSu7GateAppends],
  );
  const applyMacroGateValue = useCallback(
    (value: number, binding: MacroBinding | null) => {
      if (!binding) {
        return;
      }
      const theta = binding.thetaScale * value;
      const phase = wrapAngle(binding.phiBase + binding.phiScale * value);
      updateSu7GateAppends((current) => {
        let found = false;
        let changed = false;
        const next = current.map((gate) => {
          if (gate.label === binding.gateLabel && gate.kind === 'pulse') {
            found = true;
            const thetaChanged = Math.abs(gate.theta - theta) > 1e-6;
            const axisChanged = gate.axis !== binding.axis;
            const phaseChanged = Math.abs(wrapAngle(gate.phase - phase)) > 1e-6;
            if (!thetaChanged && !axisChanged && !phaseChanged) {
              return gate;
            }
            changed = true;
            return {
              kind: 'pulse' as const,
              axis: binding.axis,
              theta,
              phase,
              label: binding.gateLabel,
            };
          }
          return gate;
        });
        if (!found) {
          changed = true;
          next.push({
            kind: 'pulse',
            axis: binding.axis,
            theta,
            phase,
            label: binding.gateLabel,
          });
        }
        return changed ? next : current;
      });
    },
    [updateSu7GateAppends],
  );
  useEffect(() => {
    applyMacroGateValue(macroKnobValue, macroBinding);
  }, [applyMacroGateValue, macroBinding, macroKnobValue]);
  const su7RuntimeEval = useMemo(
    () => resolveSu7Runtime(su7Params, undefined, { emitGuardrailEvents: false }),
    [su7Params],
  );
  const su7DisplayGateList = su7RuntimeEval.gateList;
  const su7Unitary = su7RuntimeEval.unitary;
  const su7SquashedGateCount = su7DisplayGateList?.squashedAppends ?? 0;
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
  const handleGuardrailReorthon = useCallback(() => {
    guardrailCommandsRef.current.forceReorthon = true;
    setGuardrailConsole((prev) => {
      const now = Date.now();
      const message = 'Manual re-orthonormalization requested';
      const lastEntry = prev.auditLog[prev.auditLog.length - 1];
      if (lastEntry && lastEntry.message === message && now - lastEntry.timestamp < 250) {
        return prev;
      }
      const nextLog = [
        ...prev.auditLog,
        {
          id: guardrailAuditIdRef.current++,
          kind: 'manualReorthon' as GuardrailAuditKind,
          message,
          severity: 'info' as const,
          timestamp: now,
        },
      ];
      if (nextLog.length > 12) {
        nextLog.shift();
      }
      return { ...prev, auditLog: nextLog };
    });
  }, [setGuardrailConsole]);
  const handleGuardrailAutoGain = useCallback(() => {
    const ema = guardrailConsoleRef.current.energyEma;
    if (!Number.isFinite(ema) || ema <= 1e-6) {
      setGuardrailConsole((prev) => {
        const now = Date.now();
        const message = 'Auto-gain skipped: insufficient energy samples';
        const lastEntry = prev.auditLog[prev.auditLog.length - 1];
        if (lastEntry && lastEntry.message === message && now - lastEntry.timestamp < 250) {
          return prev;
        }
        const nextLog = [
          ...prev.auditLog,
          {
            id: guardrailAuditIdRef.current++,
            kind: 'manualAutoGain' as GuardrailAuditKind,
            message,
            severity: 'info' as const,
            timestamp: now,
          },
        ];
        if (nextLog.length > 12) {
          nextLog.shift();
        }
        return { ...prev, auditLog: nextLog };
      });
      return;
    }
    const correction = ema > 0 ? clamp(1 / ema, 0.2, 5) : 1;
    setSu7Params((prev) => {
      const nextGain = clamp(prev.gain * correction, 0.05, 5);
      if (!Number.isFinite(nextGain)) {
        return prev;
      }
      return { ...prev, gain: nextGain };
    });
    setGuardrailConsole((prev) => {
      const now = Date.now();
      const message = `Manual auto-gain applied (×${correction.toFixed(2)})`;
      const nextLog = [
        ...prev.auditLog,
        {
          id: guardrailAuditIdRef.current++,
          kind: 'manualAutoGain' as GuardrailAuditKind,
          message,
          severity: 'info' as const,
          timestamp: now,
        },
      ];
      if (nextLog.length > 12) {
        nextLog.shift();
      }
      return { ...prev, auditLog: nextLog };
    });
  }, [setGuardrailConsole, setSu7Params]);
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
      const id = value.toLowerCase();
      setSu7Params((prev) => {
        const previousProjector = prev.projector ?? { id };
        let nextHopf = previousProjector.hopf;
        if (id === 'hopflens') {
          const sanitized = resolveHopfLenses(previousProjector);
          nextHopf =
            sanitized.length > 0
              ? { lenses: sanitized.map((lens) => ({ ...lens })) }
              : {
                  lenses: DEFAULT_HOPF_LENSES.map((lens) => ({ ...lens })),
                };
        }
        return {
          ...prev,
          projector: {
            ...previousProjector,
            id,
            hopf: nextHopf,
          },
        };
      });
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
        gateAppends: [],
      }));
      setMacroBinding(null);
      setMacroKnobValue(0);
      setMacroLearnMode(false);
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
  const handleMacroKnobChange = useCallback(
    (value: number) => {
      const clamped = Math.max(-2, Math.min(2, value));
      setMacroKnobValue(clamped);
      applyMacroGateValue(clamped, macroBinding);
    },
    [applyMacroGateValue, macroBinding, setMacroKnobValue],
  );
  const handleEdgeGesture = useCallback(
    ({ axis, deltaTheta, deltaPhi }: { axis: number; deltaTheta: number; deltaPhi: number }) => {
      const theta = Math.max(-Math.PI, Math.min(Math.PI, deltaTheta));
      const phase = wrapAngle(deltaPhi);
      const label = nextGateLabel('edge');
      updateSu7GateAppends((current) => [
        ...current,
        {
          kind: 'pulse' as const,
          axis,
          theta,
          phase,
          label,
        },
      ]);
      if (macroLearnMode) {
        const binding: MacroBinding = {
          gateLabel: label,
          axis,
          thetaScale: theta,
          phiScale: deltaPhi,
          phiBase: 0,
        };
        const initialValue = Math.abs(theta) > 1e-6 || Math.abs(deltaPhi) > 1e-6 ? 1 : 0;
        setMacroBinding(binding);
        setMacroKnobValue(initialValue);
        applyMacroGateValue(initialValue, binding);
        setMacroLearnMode(false);
      }
    },
    [
      applyMacroGateValue,
      macroLearnMode,
      nextGateLabel,
      setMacroBinding,
      setMacroKnobValue,
      setMacroLearnMode,
      updateSu7GateAppends,
    ],
  );
  const su7PresetOptions = useMemo(() => {
    if (su7Params.preset && !isSu7PresetId(su7Params.preset)) {
      const label = su7Params.preset.length > 0 ? su7Params.preset : 'Custom preset';
      return [...SU7_PRESET_OPTIONS, { value: su7Params.preset, label }];
    }
    return SU7_PRESET_OPTIONS;
  }, [su7Params.preset]);
  const su7GateAppends = su7Params.gateAppends ?? [];
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
  const hopfLenses = useMemo(() => resolveHopfLenses(su7Params.projector), [su7Params.projector]);
  const updateHopfLenses = useCallback(
    (mutator: (lenses: HopfLensDescriptor[]) => boolean | void) => {
      setSu7Params((prev) => {
        const current = resolveHopfLenses(prev.projector);
        const next = current.map((lens) => ({ ...lens }));
        const result = mutator(next);
        if (result === false) {
          return prev;
        }
        return {
          ...prev,
          projector: {
            ...prev.projector,
            hopf: { lenses: next },
          },
        };
      });
    },
    [setSu7Params],
  );
  const handleHopfAxisChange = useCallback(
    (index: number, which: 'a' | 'b', value: number) => {
      const axis = Math.max(0, Math.min(6, Math.trunc(value)));
      updateHopfLenses((lenses) => {
        const lens = lenses[index];
        if (!lens) return false;
        const axes = [...lens.axes] as [number, number];
        const position = which === 'a' ? 0 : 1;
        if (axes[position] === axis) {
          return false;
        }
        axes[position] = axis;
        lenses[index] = { ...lens, axes };
        return true;
      });
    },
    [updateHopfLenses],
  );
  const handleHopfBaseMixChange = useCallback(
    (index: number, value: number) => {
      const mix = Math.max(0, Math.min(1, value));
      updateHopfLenses((lenses) => {
        const lens = lenses[index];
        if (!lens) return false;
        const previous = lens.baseMix ?? 1;
        if (Math.abs(previous - mix) <= 1e-6) {
          return false;
        }
        lenses[index] = { ...lens, baseMix: mix };
        return true;
      });
    },
    [updateHopfLenses],
  );
  const handleHopfFiberMixChange = useCallback(
    (index: number, value: number) => {
      const mix = Math.max(0, Math.min(1, value));
      updateHopfLenses((lenses) => {
        const lens = lenses[index];
        if (!lens) return false;
        const previous = lens.fiberMix ?? 1;
        if (Math.abs(previous - mix) <= 1e-6) {
          return false;
        }
        lenses[index] = { ...lens, fiberMix: mix };
        return true;
      });
    },
    [updateHopfLenses],
  );
  const handleHopfControlTargetChange = useCallback(
    (index: number, target: HopfLensControlTarget) => {
      updateHopfLenses((lenses) => {
        let changed = false;
        for (let i = 0; i < lenses.length; i++) {
          const lens = lenses[i];
          if (!lens) continue;
          if (i === index) {
            if (lens.controlTarget !== target) {
              lenses[i] = { ...lens, controlTarget: target };
              changed = true;
            }
          } else if (target !== 'none' && lens.controlTarget !== 'none') {
            lenses[i] = { ...lens, controlTarget: 'none' };
            changed = true;
          }
        }
        return changed;
      });
    },
    [updateHopfLenses],
  );
  useEffect(() => {
    if (su7ProjectorId !== 'hopflens') {
      if (macroBinding && macroBinding.gateLabel.startsWith('hopf-')) {
        setMacroBinding(null);
        setMacroKnobValue(0);
      }
      return;
    }
    const activeIndex = hopfLenses.findIndex(
      (lens) => lens.controlTarget && lens.controlTarget !== 'none',
    );
    if (activeIndex === -1) {
      if (macroBinding && macroBinding.gateLabel.startsWith('hopf-')) {
        setMacroBinding(null);
        setMacroKnobValue(0);
      }
      return;
    }
    const lens = hopfLenses[activeIndex];
    const target = lens.controlTarget ?? 'none';
    if (target === 'none') {
      return;
    }
    if (macroBinding && !macroBinding.gateLabel.startsWith('hopf-')) {
      return;
    }
    const label = `hopf-${target}-${activeIndex}`;
    const axis = lens.axes[0];
    const thetaScale = target === 'base' ? 1 : 0;
    const phiScale = target === 'fiber' ? 1 : 0;
    if (
      macroBinding &&
      macroBinding.gateLabel === label &&
      macroBinding.axis === axis &&
      macroBinding.thetaScale === thetaScale &&
      macroBinding.phiScale === phiScale
    ) {
      return;
    }
    setMacroBinding({
      gateLabel: label,
      axis,
      thetaScale,
      phiScale,
      phiBase: 0,
    });
    setMacroKnobValue(0);
  }, [hopfLenses, macroBinding, setMacroBinding, setMacroKnobValue, su7ProjectorId]);

  useEffect(() => {
    let canceled = false;
    void Su7GpuKernel.create({
      backend: 'auto',
      label: 'su7-transform-kernel',
      onWarning: (event) => {
        if (canceled) return;
        su7GpuWarningRef.current = { ...event };
        console.warn(
          `[su7-gpu] median frame time drift ${(event.drift * 100).toFixed(1)}% (median ${event.medianMs.toFixed(3)}ms)`,
        );
      },
    })
      .then((kernelInstance) => {
        if (canceled) {
          kernelInstance.dispose();
          return;
        }
        su7GpuKernelRef.current = kernelInstance;
      })
      .catch((error) => {
        console.warn('[su7-gpu] kernel unavailable', error);
      });
    return () => {
      canceled = true;
      if (su7GpuKernelRef.current) {
        su7GpuKernelRef.current.dispose();
        su7GpuKernelRef.current = null;
      }
      su7GpuReadyRef.current = null;
      su7GpuPendingRef.current = null;
      su7GpuStatsRef.current = null;
      su7GpuWarningRef.current = null;
      su7GpuLastProfileRef.current = null;
    };
  }, []);

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

  const timelineParameterConfigs = useMemo<TimelineParameterConfig[]>(
    () => [
      {
        id: 'edgeThreshold',
        label: 'Edge threshold',
        kind: 'number',
        min: 0,
        max: 1,
        step: 0.01,
        getValue: () => edgeThreshold,
      },
      {
        id: 'blend',
        label: 'Kernel blend',
        kind: 'number',
        min: 0,
        max: 1,
        step: 0.01,
        getValue: () => blend,
      },
      {
        id: 'beta2',
        label: 'Dispersion β₂',
        kind: 'number',
        min: 0,
        max: 3,
        step: 0.01,
        getValue: () => beta2,
      },
      {
        id: 'sigma',
        label: 'Rim thickness σ',
        kind: 'number',
        min: 0.3,
        max: 6,
        step: 0.05,
        getValue: () => sigma,
      },
      {
        id: 'jitter',
        label: 'Phase jitter',
        kind: 'number',
        min: 0,
        max: 2,
        step: 0.02,
        getValue: () => jitter,
      },
      {
        id: 'contrast',
        label: 'Contrast',
        kind: 'number',
        min: 0.25,
        max: 3,
        step: 0.05,
        getValue: () => contrast,
      },
      {
        id: 'rimAlpha',
        label: 'Rim alpha',
        kind: 'number',
        min: 0,
        max: 1,
        step: 0.01,
        getValue: () => rimAlpha,
      },
      {
        id: 'rimEnabled',
        label: 'Rim enabled',
        kind: 'boolean',
        getValue: () => rimEnabled,
      },
      {
        id: 'microsaccade',
        label: 'Microsaccade',
        kind: 'boolean',
        getValue: () => microsaccade,
      },
      {
        id: 'normPin',
        label: 'Normalization pin',
        kind: 'boolean',
        getValue: () => normPin,
      },
      {
        id: 'phasePin',
        label: 'Phase pin',
        kind: 'boolean',
        getValue: () => phasePin,
      },
      {
        id: 'alive',
        label: 'Alive',
        kind: 'boolean',
        getValue: () => alive,
      },
      {
        id: 'dmt',
        label: 'DMT gain',
        kind: 'number',
        min: 0,
        max: 1,
        step: 0.01,
        getValue: () => dmt,
      },
      {
        id: 'arousal',
        label: 'Arousal',
        kind: 'number',
        min: 0,
        max: 1,
        step: 0.01,
        getValue: () => arousal,
      },
      {
        id: 'surfaceBlend',
        label: 'Surface blend',
        kind: 'number',
        min: 0,
        max: 1,
        step: 0.02,
        getValue: () => surfaceBlend,
      },
      {
        id: 'warpAmp',
        label: 'Warp amplitude',
        kind: 'number',
        min: 0,
        max: 6,
        step: 0.1,
        getValue: () => warpAmp,
      },
      {
        id: 'thetaMode',
        label: 'Theta mode',
        kind: 'enum',
        options: [
          { value: 'gradient', label: 'Gradient' },
          { value: 'global', label: 'Global' },
        ],
        getValue: () => thetaMode,
      },
      {
        id: 'thetaGlobal',
        label: 'Theta (global)',
        kind: 'number',
        min: -Math.PI,
        max: Math.PI,
        step: 0.05,
        getValue: () => thetaGlobal,
      },
    ],
    [
      edgeThreshold,
      blend,
      beta2,
      sigma,
      jitter,
      contrast,
      rimAlpha,
      rimEnabled,
      microsaccade,
      normPin,
      phasePin,
      alive,
      dmt,
      arousal,
      surfaceBlend,
      warpAmp,
      thetaMode,
      thetaGlobal,
    ],
  );

  const availableTimelineParameters = useMemo(
    () =>
      timelineParameterConfigs.filter(
        (config) => !timelineLanes.some((lane) => lane.id === config.id),
      ),
    [timelineParameterConfigs, timelineLanes],
  );

  const timelineMaxSeconds = durationFrames / timelineFps;

  const clampFrameIndex = useCallback(
    (frame: number) => Math.max(0, Math.min(durationFrames, frame)),
    [durationFrames],
  );

  const upsertTimelineKeyframe = useCallback(
    (
      parameterId: string,
      config: TimelineParameterConfig,
      frame: number,
      nextValue: TimelineValue,
      previousValue: TimelineValue,
    ) => {
      setTimelineLanes((lanes) => {
        const laneIndex = lanes.findIndex((lane) => lane.id === parameterId);
        const clampedFrame = clampFrameIndex(frame);
        if (laneIndex === -1) {
          const baselineValue = clampedFrame > 0 ? (previousValue ?? nextValue) : nextValue;
          const keyframes =
            clampedFrame > 0
              ? sortTimelineKeyframes([
                  { frame: 0, value: baselineValue },
                  { frame: clampedFrame, value: nextValue },
                ])
              : sortTimelineKeyframes([{ frame: 0, value: nextValue }]);
          const interpolation =
            config.kind === 'boolean' || config.kind === 'enum' ? 'step' : 'linear';
          return [
            ...lanes,
            {
              id: parameterId,
              label: config.label,
              kind: config.kind,
              interpolation,
              keyframes,
            },
          ];
        }
        const lane = lanes[laneIndex]!;
        const keyframes = lane.keyframes.slice();
        const existingIndex = keyframes.findIndex((entry) => entry.frame === clampedFrame);
        if (existingIndex >= 0) {
          if (keyframes[existingIndex].value === nextValue) {
            return lanes;
          }
          keyframes[existingIndex] = { frame: clampedFrame, value: nextValue };
        } else {
          keyframes.push({ frame: clampedFrame, value: nextValue });
        }
        if (clampedFrame !== 0 && keyframes.every((entry) => entry.frame !== 0)) {
          keyframes.push({ frame: 0, value: previousValue ?? nextValue });
        }
        const sorted = sortTimelineKeyframes(keyframes);
        const unchanged =
          sorted.length === lane.keyframes.length &&
          sorted.every(
            (entry, idx) =>
              entry.frame === lane.keyframes[idx]?.frame &&
              entry.value === lane.keyframes[idx]?.value,
          );
        if (unchanged) {
          return lanes;
        }
        const nextLane: TimelineEditorLane = {
          ...lane,
          keyframes: sorted,
        };
        const nextLanes = lanes.slice();
        nextLanes[laneIndex] = nextLane;
        return nextLanes;
      });
    },
    [clampFrameIndex, setTimelineLanes],
  );

  const maybeRecordTimelineKeyframe = useCallback(
    (parameterId: string, nextValue: TimelineValue, previousValue: TimelineValue) => {
      if (!timelineAutoKeyframe || timelinePlaying) {
        return;
      }
      const config = timelineParameterConfigs.find((entry) => entry.id === parameterId);
      if (!config) {
        return;
      }
      const clampedTime = Math.max(0, Math.min(timelineMaxSeconds, timelineClockRef.current));
      const frame = clampFrameIndex(Math.round(clampedTime * timelineFps));
      upsertTimelineKeyframe(parameterId, config, frame, nextValue, previousValue);
    },
    [
      timelineAutoKeyframe,
      timelinePlaying,
      timelineParameterConfigs,
      timelineMaxSeconds,
      clampFrameIndex,
      timelineFps,
      upsertTimelineKeyframe,
    ],
  );

  const handleAddTimelineLane = useCallback(
    (parameterId: string) => {
      const config = timelineParameterConfigs.find((entry) => entry.id === parameterId);
      if (!config) {
        return;
      }
      setTimelineLanes((lanes) => {
        if (lanes.some((lane) => lane.id === parameterId)) {
          return lanes;
        }
        const currentTime = Math.max(0, Math.min(timelineMaxSeconds, timelineClockRef.current));
        const frame = clampFrameIndex(Math.round(currentTime * timelineFps));
        const baseline = config.getValue();
        const currentValue = timelineEvaluationRef.current?.values[parameterId] ?? baseline;
        const keyframes =
          frame > 0
            ? sortTimelineKeyframes([
                { frame: 0, value: baseline },
                { frame, value: currentValue },
              ])
            : sortTimelineKeyframes([{ frame, value: currentValue }]);
        const interpolation =
          config.kind === 'boolean' || config.kind === 'enum' ? 'step' : 'linear';
        return [
          ...lanes,
          {
            id: parameterId,
            label: config.label,
            kind: config.kind,
            interpolation,
            keyframes,
          },
        ];
      });
    },
    [timelineParameterConfigs, timelineMaxSeconds, clampFrameIndex, timelineFps, setTimelineLanes],
  );

  const handleRemoveTimelineLane = useCallback(
    (laneId: string) => {
      setTimelineLanes((lanes) => lanes.filter((lane) => lane.id !== laneId));
    },
    [setTimelineLanes],
  );

  const handleAddTimelineKeyframe = useCallback(
    (laneId: string) => {
      const config = timelineParameterConfigs.find((entry) => entry.id === laneId);
      if (!config) {
        return;
      }
      const lane = timelineLanes.find((entry) => entry.id === laneId);
      const baseline =
        lane?.keyframes.find((entry) => entry.frame === 0)?.value ?? config.getValue();
      const currentTime = Math.max(0, Math.min(timelineMaxSeconds, timelineClockRef.current));
      const frame = clampFrameIndex(Math.round(currentTime * timelineFps));
      const currentValue = timelineEvaluationRef.current?.values[laneId] ?? config.getValue();
      upsertTimelineKeyframe(laneId, config, frame, currentValue, baseline);
    },
    [
      timelineParameterConfigs,
      timelineLanes,
      timelineMaxSeconds,
      clampFrameIndex,
      timelineFps,
      upsertTimelineKeyframe,
    ],
  );

  const handleTimelineKeyframeTimeChange = useCallback(
    (laneId: string, index: number, nextTimeSeconds: number) => {
      setTimelineLanes((lanes) => {
        const laneIndex = lanes.findIndex((lane) => lane.id === laneId);
        if (laneIndex === -1) {
          return lanes;
        }
        const lane = lanes[laneIndex]!;
        if (!lane.keyframes[index]) {
          return lanes;
        }
        const clampedTime = Math.max(0, Math.min(timelineMaxSeconds, nextTimeSeconds));
        const frame = clampFrameIndex(Math.round(clampedTime * timelineFps));
        const keyframes = lane.keyframes.slice();
        const value = keyframes[index]!.value;
        keyframes.splice(index, 1);
        const existingIndex = keyframes.findIndex((entry) => entry.frame === frame);
        if (existingIndex >= 0) {
          keyframes[existingIndex] = { frame, value };
        } else {
          keyframes.push({ frame, value });
        }
        const sorted = sortTimelineKeyframes(keyframes);
        const nextLane: TimelineEditorLane = {
          ...lane,
          keyframes: sorted,
        };
        const nextLanes = lanes.slice();
        nextLanes[laneIndex] = nextLane;
        return nextLanes;
      });
    },
    [clampFrameIndex, timelineFps, timelineMaxSeconds, setTimelineLanes],
  );

  const handleTimelineKeyframeValueChange = useCallback(
    (laneId: string, index: number, value: TimelineValue) => {
      setTimelineLanes((lanes) => {
        const laneIndex = lanes.findIndex((lane) => lane.id === laneId);
        if (laneIndex === -1) {
          return lanes;
        }
        const lane = lanes[laneIndex]!;
        if (!lane.keyframes[index]) {
          return lanes;
        }
        const keyframes = lane.keyframes.slice();
        if (keyframes[index]!.value === value) {
          return lanes;
        }
        keyframes[index] = { ...keyframes[index]!, value };
        const nextLane: TimelineEditorLane = {
          ...lane,
          keyframes,
        };
        const nextLanes = lanes.slice();
        nextLanes[laneIndex] = nextLane;
        return nextLanes;
      });
    },
    [setTimelineLanes],
  );

  const handleTimelineRemoveKeyframe = useCallback(
    (laneId: string, index: number) => {
      setTimelineLanes((lanes) => {
        const laneIndex = lanes.findIndex((lane) => lane.id === laneId);
        if (laneIndex === -1) {
          return lanes;
        }
        const lane = lanes[laneIndex]!;
        if (!lane.keyframes[index]) {
          return lanes;
        }
        if (lane.keyframes.length <= 1) {
          return lanes.filter((entry) => entry.id !== laneId);
        }
        const keyframes = lane.keyframes.slice();
        keyframes.splice(index, 1);
        const nextLane: TimelineEditorLane = {
          ...lane,
          keyframes,
        };
        const nextLanes = lanes.slice();
        nextLanes[laneIndex] = nextLane;
        return nextLanes;
      });
    },
    [setTimelineLanes],
  );

  const handleTimelineInterpolationChange = useCallback(
    (laneId: string, interpolation: TimelineInterpolation) => {
      setTimelineLanes((lanes) =>
        lanes.map((lane) => (lane.id === laneId ? { ...lane, interpolation } : lane)),
      );
    },
    [setTimelineLanes],
  );

  const setTimelineTime = useCallback(
    (timeSeconds: number) => {
      const clamped = Math.max(0, Math.min(timelineMaxSeconds, timeSeconds));
      timelineClockRef.current = clamped;
      setTimelineCurrentTime(clamped);
      updateTimelineForTime(clamped);
    },
    [timelineMaxSeconds, updateTimelineForTime],
  );

  const handleTimelineScrub = useCallback(
    (timeSeconds: number) => {
      setTimelinePlaying(false);
      setTimelineTime(timeSeconds);
    },
    [setTimelineTime],
  );

  const handleTimelineTogglePlay = useCallback(() => {
    setTimelinePlaying((prev) => !prev);
  }, []);

  const handleTimelineStop = useCallback(() => {
    setTimelinePlaying(false);
    setTimelineTime(0);
  }, [setTimelineTime]);

  const handleTimelineLoopToggle = useCallback((value: boolean) => {
    setTimelineLoop(value);
  }, []);

  const handleTimelineFpsChange = useCallback((value: number) => {
    if (!Number.isFinite(value)) {
      return;
    }
    const clamped = Math.max(1, Math.min(240, Math.round(value)));
    setTimelineFps(clamped);
  }, []);

  const handleTimelineDurationChange = useCallback((value: number) => {
    if (!Number.isFinite(value)) {
      return;
    }
    const clamped = Math.max(0.1, Math.min(600, value));
    setTimelineDurationSeconds(clamped);
  }, []);

  const handleTimelineAutoKeyframeToggle = useCallback((value: boolean) => {
    setTimelineAutoKeyframe(value);
  }, []);

  const handleTimelineClear = useCallback(() => {
    setTimelinePlaying(false);
    timelineClockRef.current = 0;
    setTimelineCurrentTime(0);
    setTimelineLanes([]);
    clearTimeline();
    updateTimelineForTime(0);
  }, [clearTimeline, setTimelineLanes, updateTimelineForTime]);

  const handleTimelineStepPrev = useCallback(() => {
    if (timelineLanes.length === 0) {
      setTimelinePlaying(false);
      setTimelineTime(0);
      return;
    }
    const frames = new Set<number>();
    timelineLanes.forEach((lane) => lane.keyframes.forEach((kf) => frames.add(kf.frame)));
    if (frames.size === 0) {
      setTimelinePlaying(false);
      setTimelineTime(0);
      return;
    }
    const sorted = Array.from(frames)
      .map((frame) => clampFrameIndex(frame))
      .sort((a, b) => a - b);
    const currentFrame = clampFrameIndex(Math.round(timelineClockRef.current * timelineFps));
    let targetFrame = sorted[0]!;
    for (let i = sorted.length - 1; i >= 0; i -= 1) {
      const frame = sorted[i]!;
      if (frame < currentFrame) {
        targetFrame = frame;
        break;
      }
    }
    setTimelinePlaying(false);
    setTimelineTime(targetFrame / timelineFps);
  }, [timelineLanes, clampFrameIndex, timelineFps, setTimelineTime]);

  const handleTimelineStepNext = useCallback(() => {
    if (timelineLanes.length === 0) {
      setTimelinePlaying(false);
      setTimelineTime(0);
      return;
    }
    const frames = new Set<number>();
    timelineLanes.forEach((lane) => lane.keyframes.forEach((kf) => frames.add(kf.frame)));
    if (frames.size === 0) {
      setTimelinePlaying(false);
      setTimelineTime(0);
      return;
    }
    const sorted = Array.from(frames)
      .map((frame) => clampFrameIndex(frame))
      .sort((a, b) => a - b);
    const currentFrame = clampFrameIndex(Math.round(timelineClockRef.current * timelineFps));
    let targetFrame = sorted[sorted.length - 1]!;
    for (let i = 0; i < sorted.length; i += 1) {
      const frame = sorted[i]!;
      if (frame > currentFrame) {
        targetFrame = frame;
        break;
      }
    }
    setTimelinePlaying(false);
    setTimelineTime(targetFrame / timelineFps);
  }, [timelineLanes, clampFrameIndex, timelineFps, setTimelineTime]);

  const timelineRuntime = useMemo((): Timeline | null => {
    if (timelineLanes.length === 0) {
      return null;
    }
    const lanes = timelineLanes
      .map((lane) => ({
        id: lane.id,
        label: lane.label,
        interpolation: lane.interpolation,
        keyframes: sortTimelineKeyframes(
          lane.keyframes.map((entry) => ({
            frame: clampFrameIndex(entry.frame),
            value: entry.value,
          })),
        ),
      }))
      .filter((lane) => lane.keyframes.length > 0);
    if (lanes.length === 0) {
      return null;
    }
    return {
      version: 1,
      fps: timelineFps,
      durationFrames,
      lanes,
      seeds: [],
    };
  }, [timelineLanes, clampFrameIndex, timelineFps, durationFrames]);

  useEffect(() => {
    if (!timelineRuntime) {
      if (timelineActiveRef.current) {
        clearTimeline();
      }
      return;
    }
    loadTimeline(timelineRuntime);
    const currentTime = Math.max(0, Math.min(timelineMaxSeconds, timelineClockRef.current));
    updateTimelineForTime(currentTime);
  }, [timelineRuntime, loadTimeline, updateTimelineForTime, timelineMaxSeconds, clearTimeline]);

  useEffect(() => {
    if (!timelinePlaying) {
      return;
    }
    let animationFrame = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      let next = timelineClockRef.current + dt;
      let continuePlayback = true;
      if (next > timelineMaxSeconds) {
        if (timelineLoop && timelineMaxSeconds > 0) {
          next = next % timelineMaxSeconds;
        } else {
          next = timelineMaxSeconds;
          setTimelinePlaying(false);
          continuePlayback = false;
        }
      }
      timelineClockRef.current = next;
      setTimelineCurrentTime(next);
      updateTimelineForTime(next);
      if (continuePlayback) {
        animationFrame = requestAnimationFrame(tick);
      }
    };
    animationFrame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animationFrame);
  }, [
    timelinePlaying,
    timelineLoop,
    timelineMaxSeconds,
    updateTimelineForTime,
    setTimelinePlaying,
    setTimelineCurrentTime,
  ]);

  useEffect(() => {
    const clamped = Math.max(0, Math.min(timelineMaxSeconds, timelineClockRef.current));
    if (!Number.isFinite(clamped)) {
      return;
    }
    if (Math.abs(clamped - timelineClockRef.current) > 1e-6) {
      timelineClockRef.current = clamped;
      setTimelineCurrentTime(clamped);
      updateTimelineForTime(clamped);
    }
  }, [timelineMaxSeconds, updateTimelineForTime, setTimelineCurrentTime]);

  const handleEdgeThresholdChange = useCallback(
    (value: number) => {
      const previous = edgeThreshold;
      setEdgeThreshold(value);
      maybeRecordTimelineKeyframe('edgeThreshold', value, previous);
      recordMacroParameterChange('edgeThreshold', value);
    },
    [edgeThreshold, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handleBlendChange = useCallback(
    (value: number) => {
      const previous = blend;
      setBlend(value);
      maybeRecordTimelineKeyframe('blend', value, previous);
      recordMacroParameterChange('blend', value);
    },
    [blend, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handleBeta2Change = useCallback(
    (value: number) => {
      const previous = beta2;
      setBeta2(value);
      maybeRecordTimelineKeyframe('beta2', value, previous);
      recordMacroParameterChange('beta2', value);
    },
    [beta2, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handleSigmaChange = useCallback(
    (value: number) => {
      const previous = sigma;
      setSigma(value);
      maybeRecordTimelineKeyframe('sigma', value, previous);
      recordMacroParameterChange('sigma', value);
    },
    [sigma, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handleJitterChange = useCallback(
    (value: number) => {
      const previous = jitter;
      setJitter(value);
      maybeRecordTimelineKeyframe('jitter', value, previous);
      recordMacroParameterChange('jitter', value);
    },
    [jitter, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handleContrastChange = useCallback(
    (value: number) => {
      const previous = contrast;
      setContrast(value);
      maybeRecordTimelineKeyframe('contrast', value, previous);
      recordMacroParameterChange('contrast', value);
    },
    [contrast, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handleRimAlphaChange = useCallback(
    (value: number) => {
      const previous = rimAlpha;
      setRimAlpha(value);
      maybeRecordTimelineKeyframe('rimAlpha', value, previous);
      recordMacroParameterChange('rimAlpha', value);
    },
    [rimAlpha, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handleRimEnabledChange = useCallback(
    (value: boolean) => {
      const previous = rimEnabled;
      setRimEnabled(value);
      maybeRecordTimelineKeyframe('rimEnabled', value, previous);
      recordMacroParameterChange('rimEnabled', value);
    },
    [rimEnabled, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handleMicrosaccadeChange = useCallback(
    (value: boolean) => {
      const previous = microsaccade;
      setMicrosaccade(value);
      maybeRecordTimelineKeyframe('microsaccade', value, previous);
      recordMacroParameterChange('microsaccade', value);
    },
    [microsaccade, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handleNormPinChange = useCallback(
    (value: boolean) => {
      const previous = normPin;
      setNormPin(value);
      maybeRecordTimelineKeyframe('normPin', value, previous);
      recordMacroParameterChange('normPin', value);
    },
    [normPin, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handlePhasePinChange = useCallback(
    (value: boolean) => {
      const previous = phasePin;
      setPhasePin(value);
      maybeRecordTimelineKeyframe('phasePin', value, previous);
      recordMacroParameterChange('phasePin', value);
    },
    [phasePin, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handleAliveChange = useCallback(
    (value: boolean) => {
      const previous = alive;
      setAlive(value);
      maybeRecordTimelineKeyframe('alive', value, previous);
      recordMacroParameterChange('alive', value);
    },
    [alive, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handleDmtChange = useCallback(
    (value: number) => {
      const previous = dmt;
      setDmt(value);
      maybeRecordTimelineKeyframe('dmt', value, previous);
      recordMacroParameterChange('dmt', value);
    },
    [dmt, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handleArousalChange = useCallback(
    (value: number) => {
      const previous = arousal;
      setArousal(value);
      maybeRecordTimelineKeyframe('arousal', value, previous);
      recordMacroParameterChange('arousal', value);
    },
    [arousal, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handleSurfaceBlendChange = useCallback(
    (value: number) => {
      const previous = surfaceBlend;
      setSurfaceBlend(value);
      maybeRecordTimelineKeyframe('surfaceBlend', value, previous);
      recordMacroParameterChange('surfaceBlend', value);
    },
    [surfaceBlend, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const handleWarpAmpChange = useCallback(
    (value: number) => {
      const previous = warpAmp;
      setWarpAmp(value);
      maybeRecordTimelineKeyframe('warpAmp', value, previous);
      recordMacroParameterChange('warpAmp', value);
    },
    [warpAmp, maybeRecordTimelineKeyframe, recordMacroParameterChange],
  );

  const macroActionHandlers = useMemo<Record<string, (value: TimelineValue) => void>>(
    () => ({
      edgeThreshold: (value) =>
        handleEdgeThresholdChange(coerceTimelineNumber(value, edgeThreshold)),
      blend: (value) => handleBlendChange(coerceTimelineNumber(value, blend)),
      beta2: (value) => handleBeta2Change(coerceTimelineNumber(value, beta2)),
      sigma: (value) => handleSigmaChange(coerceTimelineNumber(value, sigma)),
      jitter: (value) => handleJitterChange(coerceTimelineNumber(value, jitter)),
      contrast: (value) => handleContrastChange(coerceTimelineNumber(value, contrast)),
      rimAlpha: (value) => handleRimAlphaChange(coerceTimelineNumber(value, rimAlpha)),
      rimEnabled: (value) => handleRimEnabledChange(Boolean(value)),
      microsaccade: (value) => handleMicrosaccadeChange(Boolean(value)),
      normPin: (value) => handleNormPinChange(Boolean(value)),
      phasePin: (value) => handlePhasePinChange(Boolean(value)),
      alive: (value) => handleAliveChange(Boolean(value)),
      dmt: (value) => handleDmtChange(coerceTimelineNumber(value, dmt)),
      arousal: (value) => handleArousalChange(coerceTimelineNumber(value, arousal)),
      surfaceBlend: (value) => handleSurfaceBlendChange(coerceTimelineNumber(value, surfaceBlend)),
      warpAmp: (value) => handleWarpAmpChange(coerceTimelineNumber(value, warpAmp)),
    }),
    [
      edgeThreshold,
      blend,
      beta2,
      sigma,
      jitter,
      contrast,
      rimAlpha,
      dmt,
      arousal,
      surfaceBlend,
      warpAmp,
      handleEdgeThresholdChange,
      handleBlendChange,
      handleBeta2Change,
      handleSigmaChange,
      handleJitterChange,
      handleContrastChange,
      handleRimAlphaChange,
      handleRimEnabledChange,
      handleMicrosaccadeChange,
      handleNormPinChange,
      handlePhasePinChange,
      handleAliveChange,
      handleDmtChange,
      handleArousalChange,
      handleSurfaceBlendChange,
      handleWarpAmpChange,
    ],
  );

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
  const [polarizationEnabled, setPolarizationEnabled] = useState(false);
  const [wavePlateEnabled, setWavePlateEnabled] = useState(true);
  const [wavePlatePhaseDeg, setWavePlatePhaseDeg] = useState(90);
  const [wavePlateOrientationDeg, setWavePlateOrientationDeg] = useState(0);
  const [polarizerEnabled, setPolarizerEnabled] = useState(false);
  const [polarizerOrientationDeg, setPolarizerOrientationDeg] = useState(0);
  const [polarizerExtinction, setPolarizerExtinction] = useState(0);
  const [su7PolarizationEnabled, setSu7PolarizationEnabled] = useState(false);
  const [su7PolarizationColumn, setSu7PolarizationColumn] = useState(0);
  const [su7PolarizationGain, setSu7PolarizationGain] = useState(1);
  const [su7PolarizationBlend, setSu7PolarizationBlend] = useState(1);
  const su7PolarizationMatrix = useMemo<PolarizationMatrix | null>(() => {
    if (!su7PolarizationEnabled) {
      return null;
    }
    const column = Math.min(6, Math.max(0, Math.floor(su7PolarizationColumn)));
    const gain = Number.isFinite(su7PolarizationGain) ? su7PolarizationGain : 1;
    return su7UnitaryColumnToPolarizationMatrix(su7Unitary, {
      column,
      gain,
    });
  }, [su7PolarizationEnabled, su7Unitary, su7PolarizationColumn, su7PolarizationGain]);
  const polarizationSchedule = useMemo<ThinElementSchedule | undefined>(() => {
    if (!polarizationEnabled) {
      return undefined;
    }
    const steps: any[] = [];
    const toRad = (deg: number) => (Number.isFinite(deg) ? (deg * Math.PI) / 180 : 0);
    if (wavePlateEnabled) {
      const phaseRad = toRad(wavePlatePhaseDeg);
      const orientationRad = toRad(wavePlateOrientationDeg);
      steps.push(createWavePlateStep(phaseRad, orientationRad, 'Wave plate'));
    }
    if (su7PolarizationEnabled && su7PolarizationMatrix) {
      const blend = clamp(su7PolarizationBlend, 0, 1);
      const mixMatrix = (matrix: PolarizationMatrix, weight: number): PolarizationMatrix => {
        if (weight >= 0.999) {
          return matrix;
        }
        const w = Math.max(0, weight);
        const inv = 1 - w;
        const mixEntry = (
          entry: { re: number; im: number },
          identityRe: number,
        ): { re: number; im: number } => ({
          re: inv * identityRe + w * entry.re,
          im: w * entry.im,
        });
        return {
          m00: mixEntry(matrix.m00, 1),
          m01: mixEntry(matrix.m01, 0),
          m10: mixEntry(matrix.m10, 0),
          m11: mixEntry(matrix.m11, 1),
        };
      };
      const matrix = mixMatrix(su7PolarizationMatrix, blend);
      steps.push(createPolarizationMatrixStep(matrix, 'SU7 gate'));
    }
    if (polarizerEnabled) {
      const orientationRad = toRad(polarizerOrientationDeg);
      const extinction = clamp(polarizerExtinction, 0, 1);
      steps.push(createPolarizerStep(orientationRad, extinction, 'Polarizer'));
    }
    return [
      ...steps,
      { kind: 'operator', operator: 'amplitude' as const },
      { kind: 'operator', operator: 'phase' as const },
    ] as ThinElementSchedule;
  }, [
    polarizationEnabled,
    wavePlateEnabled,
    wavePlatePhaseDeg,
    wavePlateOrientationDeg,
    su7PolarizationEnabled,
    su7PolarizationMatrix,
    su7PolarizationBlend,
    polarizerEnabled,
    polarizerOrientationDeg,
    polarizerExtinction,
  ]);
  const [fluxSources, setFluxSources] = useState<FluxSource[]>([]);
  const [qcdBeta, setQcdBeta] = useState(5.25);
  const [qcdStepsPerSecond, setQcdStepsPerSecond] = useState(3);
  const [qcdSmearingAlpha, setQcdSmearingAlpha] = useState(0.5);
  const [qcdSmearingIterations, setQcdSmearingIterations] = useState(1);
  const [qcdBaseSeed, setQcdBaseSeed] = useState(2024);
  const [qcdDepth, setQcdDepth] = useState(1);
  const [qcdTemporalExtent, setQcdTemporalExtent] = useState(1);
  const [qcdBatchLayers, setQcdBatchLayers] = useState(1);
  const [qcdTemperatureScheduleText, setQcdTemperatureScheduleText] = useState('');
  const [qcdPolyakovSchedule, setQcdPolyakovSchedule] = useState<number[]>([]);
  const [qcdPerfLog, setQcdPerfLog] = useState<string[]>([]);
  const [qcdRunning, setQcdRunning] = useState(false);
  const [qcdObservables, setQcdObservables] = useState<QcdObservables | null>(null);
  const [qcdSnapshotHash, setQcdSnapshotHash] = useState<string | null>(null);
  const [qcdOverlayState, setQcdOverlayState] = useState<FluxOverlayFrameData | null>(null);
  const [qcdProbeFrame, setQcdProbeFrame] = useState<ProbeTransportFrameData | null>(null);
  const qcdTemperatureSchedule = useMemo(() => {
    if (!qcdTemperatureScheduleText.trim()) {
      return [] as number[];
    }
    const tokens = qcdTemperatureScheduleText
      .split(/[\s,]+/)
      .map((token) => Number.parseFloat(token))
      .filter((value) => Number.isFinite(value));
    return tokens.slice(0, 64);
  }, [qcdTemperatureScheduleText]);
  const qcdDepthInt = Math.max(1, Math.floor(qcdDepth));
  const qcdTemporalExtentInt = Math.max(1, Math.floor(qcdTemporalExtent));
  useEffect(() => {
    setQcdBatchLayers((prev) => {
      const normalized = Math.max(1, Math.floor(prev));
      const maxLayers = Math.max(1, qcdDepthInt * qcdTemporalExtentInt);
      const clamped = Math.min(normalized, maxLayers);
      return clamped === prev ? prev : clamped;
    });
  }, [qcdDepthInt, qcdTemporalExtentInt]);
  const qcdBatchLayersInt = Math.max(1, Math.floor(qcdBatchLayers));
  const appendQcdPerfLog = useCallback((entry: string) => {
    setQcdPerfLog((prev) => {
      const next = [...prev, entry];
      return next.length > 12 ? next.slice(next.length - 12) : next;
    });
  }, []);
  const schedulesEqual = (left: readonly number[], right: readonly number[]): boolean => {
    if (left.length !== right.length) {
      return false;
    }
    for (let i = 0; i < left.length; i++) {
      if (Math.abs(left[i]! - right[i]!) > 1e-6) {
        return false;
      }
    }
    return true;
  };
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
  const [telemetrySeriesSelection, setTelemetrySeriesSelection] =
    useState<TelemetrySeriesSelection>({
      indraIndex: true,
      symmetry: true,
      colorfulness: false,
      edgeDensity: false,
    });
  const [telemetryStreamEnabled, setTelemetryStreamEnabled] = useState(false);
  const [telemetryStreamUrl, setTelemetryStreamUrl] = useState('ws://localhost:8090/telemetry');
  const [telemetryStreamStatus, setTelemetryStreamStatus] = useState<
    'idle' | 'connecting' | 'connected' | 'error'
  >('idle');
  const [telemetryStreamError, setTelemetryStreamError] = useState<string | null>(null);
  const telemetryStreamSocketRef = useRef<WebSocket | null>(null);
  const telemetryStreamQueueRef = useRef<string[]>([]);
  const telemetryStreamFrameIdRef = useRef(0);
  const telemetryStreamErrorRef = useRef<string | null>(null);
  const telemetryRecordingRef = useRef<{
    active: boolean;
    startedAt: number | null;
    samples: TelemetryExportEntry[];
  }>({ active: false, startedAt: null, samples: [] });
  const [telemetryRecordingActive, setTelemetryRecordingActive] = useState(false);
  const [telemetryRecordingCount, setTelemetryRecordingCount] = useState(0);
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
  const toggleTelemetrySeries = useCallback((key: QualiaSeriesKey) => {
    setTelemetrySeriesSelection((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  }, []);

  const flushTelemetryStreamQueue = (socket: WebSocket) => {
    const queue = telemetryStreamQueueRef.current;
    while (queue.length > 0 && socket.readyState === WebSocket.OPEN) {
      const message = queue.shift();
      if (message) {
        socket.send(message);
      }
    }
  };

  const closeTelemetryStream = useCallback(() => {
    const socket = telemetryStreamSocketRef.current;
    if (socket) {
      try {
        if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
          socket.close();
        }
      } catch (error) {
        console.warn('[telemetry-stream] failed to close socket', error);
      }
    }
    telemetryStreamSocketRef.current = null;
    telemetryStreamQueueRef.current = [];
    telemetryStreamErrorRef.current = null;
    setTelemetryStreamStatus('idle');
  }, []);

  const openTelemetryStream = useCallback(() => {
    if (typeof window === 'undefined' || !telemetryStreamEnabled) {
      return;
    }
    const existing = telemetryStreamSocketRef.current;
    if (existing) {
      if (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING) {
        return;
      }
    }
    try {
      const socket = new WebSocket(telemetryStreamUrl);
      telemetryStreamSocketRef.current = socket;
      setTelemetryStreamStatus('connecting');
      setTelemetryStreamError(null);
      socket.addEventListener('open', () => {
        setTelemetryStreamStatus('connected');
        flushTelemetryStreamQueue(socket);
      });
      socket.addEventListener('error', (event) => {
        console.warn('[telemetry-stream] socket error', event);
        telemetryStreamErrorRef.current = 'Unable to connect';
        setTelemetryStreamStatus('error');
        setTelemetryStreamError('Unable to connect');
      });
      socket.addEventListener('close', () => {
        telemetryStreamSocketRef.current = null;
        if (telemetryStreamEnabled) {
          setTelemetryStreamStatus('error');
          setTelemetryStreamError(telemetryStreamErrorRef.current ?? 'Stream closed');
        } else {
          setTelemetryStreamStatus('idle');
        }
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      telemetryStreamErrorRef.current = message;
      setTelemetryStreamStatus('error');
      setTelemetryStreamError(message);
    }
  }, [telemetryStreamEnabled, telemetryStreamUrl]);

  const streamFrame = useCallback(
    (payload: unknown) => {
      if (!telemetryStreamEnabled) {
        return;
      }
      const data = JSON.stringify(payload);
      const socket = telemetryStreamSocketRef.current;
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(data);
        return;
      }
      if (!socket) {
        openTelemetryStream();
      }
      const queue = telemetryStreamQueueRef.current;
      if (queue.length >= TELEMETRY_STREAM_QUEUE_LIMIT) {
        queue.shift();
      }
      queue.push(data);
    },
    [openTelemetryStream, telemetryStreamEnabled],
  );

  const startTelemetryRecording = useCallback(() => {
    telemetryRecordingRef.current = {
      active: true,
      startedAt: Date.now(),
      samples: [],
    };
    setTelemetryRecordingActive(true);
    setTelemetryRecordingCount(0);
  }, []);

  const stopTelemetryRecording = useCallback(() => {
    const record = telemetryRecordingRef.current;
    if (!record.active) {
      return;
    }
    record.active = false;
    setTelemetryRecordingActive(false);
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      record.samples = [];
      setTelemetryRecordingCount(0);
      return;
    }
    if (record.samples.length === 0) {
      setTelemetryRecordingCount(0);
      return;
    }
    const samples = record.samples.slice();
    const lines = samples.map((sample) => JSON.stringify(sample));
    const blob = new Blob([lines.join('\n')], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    const timestamp = record.startedAt ? new Date(record.startedAt) : new Date();
    const safeName = timestamp.toISOString().replace(/[:]/g, '-');
    anchor.href = url;
    anchor.download = `indra-telemetry-${safeName}.jsonl`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
    record.samples = [];
    record.startedAt = null;
    setTelemetryRecordingCount(0);
  }, []);

  const downloadTelemetrySnapshot = useCallback(() => {
    if (typeof document === 'undefined') {
      return;
    }
    const latest = metricsRef.current[metricsRef.current.length - 1];
    if (!latest) {
      return;
    }
    const blob = new Blob([JSON.stringify(latest, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `indra-telemetry-snapshot-${new Date().toISOString().replace(/[:]/g, '-')}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }, []);

  const captureFrameMetrics = useCallback(
    (
      backend: FrameMetricsEntry['backend'],
      metrics: RainbowFrameMetrics,
      kernelVersion: number,
    ) => {
      const entry: FrameMetricsEntry = {
        backend,
        ts: performance.now(),
        metrics,
        kernelVersion,
      };
      metricsRef.current.push(entry);
      if (metricsRef.current.length > 240) {
        metricsRef.current.shift();
      }
      if (telemetryStreamEnabled) {
        const payload = {
          type: 'qualia-frame',
          frame: telemetryStreamFrameIdRef.current++,
          timestamp: Date.now(),
          backend: entry.backend,
          kernelVersion: entry.kernelVersion,
          qualia: entry.metrics.qualia,
          motion: entry.metrics.motionEnergy,
          parallax: entry.metrics.parallax,
          texture: {
            wallpapericity: entry.metrics.texture.wallpapericity,
            beatEnergy: entry.metrics.texture.beatEnergy,
            sampleCount: entry.metrics.texture.sampleCount,
          },
          composer: entry.metrics.composer.fields,
        };
        streamFrame(payload);
      }
      if (telemetryRecordingRef.current.active) {
        const record = telemetryRecordingRef.current;
        const exportEntry: TelemetryExportEntry = {
          timestamp: Date.now(),
          backend: entry.backend,
          kernelVersion: entry.kernelVersion,
          qualia: entry.metrics.qualia,
          motion: entry.metrics.motionEnergy,
          parallax: entry.metrics.parallax,
          texture: {
            wallpapericity: entry.metrics.texture.wallpapericity,
            beatEnergy: entry.metrics.texture.beatEnergy,
            sampleCount: entry.metrics.texture.sampleCount,
          },
        };
        if (record.samples.length >= TELEMETRY_RECORDING_LIMIT) {
          record.samples.shift();
        }
        record.samples.push(exportEntry);
        setTelemetryRecordingCount(record.samples.length);
      }
    },
    [streamFrame, telemetryStreamEnabled],
  );

  const toggleTelemetryRecording = useCallback(() => {
    if (telemetryRecordingRef.current.active) {
      stopTelemetryRecording();
    } else {
      startTelemetryRecording();
    }
  }, [startTelemetryRecording, stopTelemetryRecording]);

  const handleTelemetryReconnect = useCallback(() => {
    closeTelemetryStream();
    openTelemetryStream();
  }, [closeTelemetryStream, openTelemetryStream]);
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
  const fluxOverlayFallback = useMemo<FluxOverlayFrameData | null>(() => {
    if (width <= 0 || height <= 0) {
      return null;
    }
    if (fluxSources.length < 2) {
      return null;
    }
    return computeFluxOverlayState({ width, height, sources: fluxSources }) ?? null;
  }, [width, height, fluxSources]);
  const activeFluxOverlay = qcdOverlayState ?? fluxOverlayFallback;
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
    qcdRunningRef.current = qcdRunning;
  }, [qcdRunning]);

  useEffect(() => {
    if (!tracerRuntime.enabled) {
      resetTracerState();
    }
  }, [tracerRuntime.enabled, resetTracerState]);

  useEffect(() => {
    resetTracerState();
  }, [width, height, resetTracerState]);

  const nextFluxChargeRef = useRef<1 | -1>(1);
  const normTargetRef = useRef(0.6);
  const lastObsRef = useRef(0.6);
  const qcdRuntimeRef = useRef<QcdRuntimeState | null>(null);
  const qcdSnapshotRef = useRef<QcdSnapshot | null>(null);
  const qcdTimerRef = useRef<number | null>(null);
  const qcdRunningRef = useRef(false);
  const qcdStepInFlightRef = useRef(false);
  const qcdCpuRngRef = useRef<(() => number) | null>(null);
  const qcdBaseSeedRef = useRef(qcdBaseSeed);
  const qcdSweeps = qcdRuntimeRef.current?.sweepIndex ?? 0;
  const cloneQcdObservables = (obs: QcdObservables): QcdObservables => ({
    averagePlaquette: obs.averagePlaquette,
    plaquetteHistory: [...obs.plaquetteHistory],
    plaquetteEstimate: { ...obs.plaquetteEstimate },
    wilsonLoops: obs.wilsonLoops.map((entry) => ({ ...entry })),
    creutzRatio: obs.creutzRatio ? { ...obs.creutzRatio } : undefined,
    polyakovSamples: obs.polyakovSamples
      ? obs.polyakovSamples.map((sample) => ({
          axis: sample.axis,
          extent: sample.extent,
          magnitude: sample.magnitude,
          sampleCount: sample.sampleCount,
          average: { ...sample.average },
        }))
      : undefined,
  });

  const removeFluxSource = useCallback((index: number) => {
    setFluxSources((prev) => prev.filter((_, idx) => idx !== index));
  }, []);

  const flipFluxSourceCharge = useCallback((index: number) => {
    setFluxSources((prev) =>
      prev.map((entry, idx) =>
        idx === index ? { ...entry, charge: entry.charge >= 0 ? -1 : 1 } : entry,
      ),
    );
  }, []);

  const clearFluxSources = useCallback(() => {
    setFluxSources([]);
    nextFluxChargeRef.current = 1;
  }, []);

  const handleCanvasPointerDown = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || width <= 0 || height <= 0) return;
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      const pointerX = ((event.clientX - rect.left) / rect.width) * width;
      const pointerY = ((event.clientY - rect.top) / rect.height) * height;
      if (!Number.isFinite(pointerX) || !Number.isFinite(pointerY)) return;
      const clampedX = clamp(Math.round(pointerX), 0, width - 1);
      const clampedY = clamp(Math.round(pointerY), 0, height - 1);
      const searchRadius = Math.max(6, Math.min(width, height) * 0.04);
      let nearestIndex = -1;
      let nearestDistance = Number.POSITIVE_INFINITY;
      for (let idx = 0; idx < fluxSources.length; idx++) {
        const source = fluxSources[idx]!;
        const dx = source.x - clampedX;
        const dy = source.y - clampedY;
        const dist = Math.hypot(dx, dy);
        if (dist < nearestDistance) {
          nearestDistance = dist;
          nearestIndex = idx;
        }
      }
      const removing = event.button === 2 || event.shiftKey;
      if (removing) {
        if (nearestIndex >= 0 && nearestDistance <= searchRadius) {
          removeFluxSource(nearestIndex);
        }
        event.preventDefault();
        return;
      }
      if (nearestIndex >= 0 && nearestDistance <= Math.max(3, searchRadius * 0.35)) {
        flipFluxSourceCharge(nearestIndex);
        event.preventDefault();
        return;
      }
      const explicitCharge = event.altKey || event.metaKey ? -1 : event.ctrlKey ? 1 : null;
      const charge =
        explicitCharge != null ? (explicitCharge > 0 ? 1 : -1) : nextFluxChargeRef.current;
      setFluxSources((prev) => [...prev, { x: clampedX, y: clampedY, charge, strength: 1 }]);
      if (explicitCharge == null) {
        nextFluxChargeRef.current = charge === 1 ? -1 : 1;
      }
      event.preventDefault();
    },
    [fluxSources, flipFluxSourceCharge, height, removeFluxSource, width],
  );

  const handleCanvasContextMenu = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    event.preventDefault();
  }, []);

  const getQcdConfig = useCallback((): QcdAnnealConfig => {
    const maxBatchLayers = Math.max(1, qcdDepthInt * qcdTemporalExtentInt);
    const batchLayers = Math.min(qcdBatchLayersInt, maxBatchLayers);
    return {
      beta: qcdBeta,
      overRelaxationSteps: 1,
      smearing: {
        alpha: clamp(qcdSmearingAlpha, 0, 1),
        iterations: Math.max(0, Math.floor(qcdSmearingIterations)),
      },
      depth: qcdDepthInt,
      temporalExtent: qcdTemporalExtentInt,
      batchLayers,
      temperatureSchedule: qcdTemperatureSchedule.length > 0 ? [...qcdTemperatureSchedule] : [],
    };
  }, [
    qcdBeta,
    qcdSmearingAlpha,
    qcdSmearingIterations,
    qcdDepthInt,
    qcdTemporalExtentInt,
    qcdBatchLayersInt,
    qcdTemperatureSchedule,
  ]);

  const ensureQcdRuntime = useCallback(
    (options?: {
      reinitialize?: boolean;
      snapshot?: QcdSnapshot;
      sourcesOverride?: FluxSource[];
    }) => {
      const sourceList = options?.sourcesOverride ?? fluxSources;

      const applyPolyakovSchedule = (
        runtime: QcdRuntimeState,
        schedule: readonly number[],
        reason: string,
      ): boolean => {
        if (!schedule || schedule.length === 0) {
          const hadSamples =
            runtime.polyakovScan.length > 0 ||
            (runtime.observables.polyakovSamples?.length ?? 0) > 0;
          runtime.polyakovScan = [];
          runtime.observables.polyakovSamples = undefined;
          setQcdPolyakovSchedule([]);
          if (hadSamples) {
            appendQcdPerfLog(`[polyakov] ${reason}: cleared`);
            return true;
          }
          return false;
        }
        const axis = runtime.lattice.axes.includes('t') ? ('t' as const) : runtime.lattice.axes[0];
        if (!axis) {
          runtime.polyakovScan = [];
          runtime.observables.polyakovSamples = undefined;
          setQcdPolyakovSchedule([]);
          appendQcdPerfLog(`[polyakov] ${reason}: skipped (no active axis)`);
          return true;
        }
        runTemperatureScan(runtime, schedule, axis);
        setQcdPolyakovSchedule([...schedule]);
        appendQcdPerfLog(`[polyakov] ${reason}: ${schedule.length} β along ${axis}`);
        return true;
      };

      if (options?.snapshot) {
        try {
          const runtime = restoreQcdRuntime(options.snapshot);
          const restoredSchedule = Array.isArray(runtime.config.temperatureSchedule)
            ? runtime.config.temperatureSchedule.filter((value) => Number.isFinite(value))
            : [];
          const latticeAxisCount = Math.max(
            1,
            runtime.lattice.depth * runtime.lattice.temporalExtent,
          );
          const restoredBatchLayers = Math.max(
            1,
            Math.min(Math.floor(runtime.config.batchLayers ?? qcdBatchLayersInt), latticeAxisCount),
          );
          runtime.config = {
            ...runtime.config,
            beta: qcdBeta,
            overRelaxationSteps: runtime.config.overRelaxationSteps ?? 1,
            smearing: {
              alpha: clamp(qcdSmearingAlpha, 0, 1),
              iterations: Math.max(0, Math.floor(qcdSmearingIterations)),
            },
            depth: runtime.lattice.depth,
            temporalExtent: runtime.lattice.temporalExtent,
            batchLayers: restoredBatchLayers,
            temperatureSchedule: restoredSchedule,
          };
          qcdRuntimeRef.current = runtime;
          qcdCpuRngRef.current = mulberry32(runtime.baseSeed >>> 0);
          setQcdDepth(runtime.lattice.depth);
          setQcdTemporalExtent(runtime.lattice.temporalExtent);
          setQcdBatchLayers(restoredBatchLayers);
          setQcdTemperatureScheduleText(
            restoredSchedule.length > 0
              ? restoredSchedule.map((value) => value.toFixed(3)).join(', ')
              : '',
          );
          setQcdPerfLog([]);
          const mutated = applyPolyakovSchedule(runtime, restoredSchedule, 'restored');
          const overlay = buildQcdOverlay(runtime, sourceList, width, height);
          setQcdOverlayState(overlay);
          const probeFrame = buildQcdProbeFrame(runtime, sourceList);
          setQcdProbeFrame(probeFrame);
          setQcdObservables(cloneQcdObservables(runtime.observables));
          if (!mutated && restoredSchedule.length === 0) {
            appendQcdPerfLog('[polyakov] restored: no schedule');
          }
          qcdSnapshotRef.current = options.snapshot;
          const { hash } = hashQcdSnapshot(options.snapshot);
          setQcdSnapshotHash(hash);
          return runtime;
        } catch (error) {
          console.error('[qcd] failed to restore snapshot', error);
          return null;
        }
      }

      if (width <= 0 || height <= 0) {
        return null;
      }

      const config = getQcdConfig();
      const baseSeed = qcdBaseSeed >>> 0;
      const latticeSize = deriveLatticeResolution(width, height);
      const existing = qcdRuntimeRef.current;
      if (
        existing &&
        existing.baseSeed === baseSeed &&
        existing.lattice.width > 0 &&
        !options?.reinitialize
      ) {
        const dimensionChanged =
          existing.lattice.width !== latticeSize.width ||
          existing.lattice.height !== latticeSize.height ||
          existing.lattice.depth !== config.depth ||
          existing.lattice.temporalExtent !== config.temporalExtent;
        if (!dimensionChanged) {
          const currentSchedule = existing.config.temperatureSchedule ?? [];
          const scheduleChanged = !schedulesEqual(currentSchedule, config.temperatureSchedule);
          existing.config = config;
          if (scheduleChanged) {
            const mutated = applyPolyakovSchedule(existing, config.temperatureSchedule, 'updated');
            if (mutated || currentSchedule.length > 0 || config.temperatureSchedule.length > 0) {
              setQcdObservables(cloneQcdObservables(existing.observables));
            }
          }
          return existing;
        }
      }

      setQcdPerfLog([]);
      const runtime = initializeQcdRuntime({
        latticeSize,
        config,
        baseSeed,
        startMode: 'cold',
      });
      qcdRuntimeRef.current = runtime;
      qcdCpuRngRef.current = mulberry32(baseSeed);
      const overlay = buildQcdOverlay(runtime, sourceList, width, height);
      setQcdOverlayState(overlay);
      const probeFrame = buildQcdProbeFrame(runtime, sourceList);
      setQcdProbeFrame(probeFrame);
      applyPolyakovSchedule(runtime, config.temperatureSchedule, 'initialized');
      setQcdObservables(cloneQcdObservables(runtime.observables));
      const snapshot = buildQcdSnapshot(runtime, sourceList);
      const { hash } = hashQcdSnapshot(snapshot);
      qcdSnapshotRef.current = snapshot;
      setQcdSnapshotHash(hash);
      if (config.temperatureSchedule.length === 0) {
        appendQcdPerfLog('[polyakov] initialized: no schedule');
      }
      return runtime;
    },
    [
      appendQcdPerfLog,
      cloneQcdObservables,
      fluxSources,
      getQcdConfig,
      height,
      qcdBaseSeed,
      qcdBatchLayersInt,
      qcdSmearingAlpha,
      qcdSmearingIterations,
      qcdBeta,
      schedulesEqual,
      width,
    ],
  );

  const runQcdStep = useCallback(async () => {
    if (qcdStepInFlightRef.current) {
      return;
    }
    const runtime = ensureQcdRuntime();
    if (!runtime) {
      return;
    }
    qcdStepInFlightRef.current = true;
    try {
      let success = false;
      const gpuState = gpuStateRef.current;
      const axes = runtime.lattice.axes;
      const axisIndex = axes.length > 0 ? Math.floor(runtime.phaseIndex / 2) % axes.length : -1;
      const activeAxis = axisIndex >= 0 ? axes[axisIndex]! : null;
      const parity = (runtime.phaseIndex % 2) as 0 | 1;
      if (gpuState?.renderer) {
        success = await runGpuSubstep(runtime, gpuState.renderer, 'interactive-qcd');
      }
      if (success) {
        appendQcdPerfLog(
          `[gpu] axis ${activeAxis ?? '?'} parity ${parity} batch ${runtime.config.batchLayers}`,
        );
      }
      if (!success) {
        appendQcdPerfLog('[gpu] sweep unavailable, fallback to CPU');
        if (!qcdCpuRngRef.current) {
          qcdCpuRngRef.current = mulberry32(runtime.baseSeed >>> 0);
        }
        runCpuSweep(runtime, qcdCpuRngRef.current!);
        appendQcdPerfLog('[cpu] heatbath sweep completed');
      }
      if (runtime.phaseIndex === 0) {
        setQcdObservables(cloneQcdObservables(runtime.observables));
        const overlay = buildQcdOverlay(runtime, fluxSources, width, height);
        setQcdOverlayState(overlay);
        const probeFrame = buildQcdProbeFrame(runtime, fluxSources);
        setQcdProbeFrame(probeFrame);
        const snapshot = buildQcdSnapshot(runtime, fluxSources);
        const { hash } = hashQcdSnapshot(snapshot);
        qcdSnapshotRef.current = snapshot;
        setQcdSnapshotHash(hash);
      }
    } catch (error) {
      console.warn('[qcd] anneal step failed', error);
    } finally {
      qcdStepInFlightRef.current = false;
    }
  }, [appendQcdPerfLog, cloneQcdObservables, ensureQcdRuntime, fluxSources, height, width]);

  const stopQcdAnneal = useCallback(() => {
    if (qcdTimerRef.current != null) {
      window.clearTimeout(qcdTimerRef.current);
      qcdTimerRef.current = null;
    }
    qcdRunningRef.current = false;
    setQcdRunning(false);
  }, []);

  const startQcdAnneal = useCallback(() => {
    if (qcdRunningRef.current) {
      return;
    }
    const runtime = ensureQcdRuntime();
    if (!runtime) {
      return;
    }
    qcdRunningRef.current = true;
    setQcdRunning(true);
    const loop = async () => {
      if (!qcdRunningRef.current) {
        return;
      }
      await runQcdStep();
      if (!qcdRunningRef.current) {
        return;
      }
      const intervalMs = Math.max(16, Math.round(1000 / clamp(qcdStepsPerSecond, 0.1, 60)));
      qcdTimerRef.current = window.setTimeout(loop, intervalMs);
    };
    void loop();
  }, [ensureQcdRuntime, qcdStepsPerSecond, runQcdStep]);

  const handlePolyakovScan = useCallback(() => {
    const schedule = qcdTemperatureSchedule;
    if (schedule.length === 0) {
      appendQcdPerfLog('[polyakov] manual scan skipped: empty schedule');
      return;
    }
    const runtime = ensureQcdRuntime();
    if (!runtime) {
      appendQcdPerfLog('[polyakov] manual scan skipped: runtime unavailable');
      return;
    }
    const axis = runtime.lattice.axes.includes('t') ? ('t' as const) : runtime.lattice.axes[0];
    if (!axis) {
      appendQcdPerfLog('[polyakov] manual scan skipped: no active axis');
      return;
    }
    runTemperatureScan(runtime, schedule, axis);
    setQcdPolyakovSchedule([...schedule]);
    setQcdObservables(cloneQcdObservables(runtime.observables));
    appendQcdPerfLog(`[polyakov] manual scan: ${schedule.length} β along ${axis}`);
  }, [appendQcdPerfLog, cloneQcdObservables, ensureQcdRuntime, qcdTemperatureSchedule]);

  useEffect(() => {
    if (!qcdRunning) {
      ensureQcdRuntime();
    }
  }, [ensureQcdRuntime, qcdRunning, fluxSources, width, height]);

  useEffect(() => stopQcdAnneal, [stopQcdAnneal]);

  useEffect(() => {
    if (qcdRuntimeRef.current && !qcdRunning) {
      const overlay = buildQcdOverlay(qcdRuntimeRef.current, fluxSources, width, height);
      setQcdOverlayState(overlay);
      const probeFrame = buildQcdProbeFrame(qcdRuntimeRef.current, fluxSources);
      setQcdProbeFrame(probeFrame);
      const snapshot = buildQcdSnapshot(qcdRuntimeRef.current, fluxSources);
      const { hash } = hashQcdSnapshot(snapshot);
      qcdSnapshotRef.current = snapshot;
      setQcdSnapshotHash(hash);
    }
  }, [fluxSources, width, height, qcdRunning]);

  useEffect(() => {
    if (qcdBaseSeedRef.current === qcdBaseSeed) {
      return;
    }
    qcdBaseSeedRef.current = qcdBaseSeed;
    if (!qcdRuntimeRef.current) {
      return;
    }
    if (qcdRunningRef.current) {
      stopQcdAnneal();
    }
    ensureQcdRuntime({ reinitialize: true });
  }, [ensureQcdRuntime, qcdBaseSeed, stopQcdAnneal]);

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

  const recordTelemetry = useCallback(
    (phase: TelemetryPhase, ms: number) => {
      if (phase === 'frame') {
        guardrailFrameTimeRef.current = ms;
        setGuardrailConsole((prev) => {
          if (Math.abs(prev.frameTimeMs - ms) <= 1e-3) {
            return prev;
          }
          return { ...prev, frameTimeMs: ms };
        });
      }
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
    },
    [setGuardrailConsole],
  );

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
    if (!telemetryStreamEnabled) {
      setTelemetryStreamError(null);
      telemetryStreamQueueRef.current = [];
      return;
    }
    openTelemetryStream();
    return () => {
      closeTelemetryStream();
    };
  }, [telemetryStreamEnabled, openTelemetryStream, closeTelemetryStream]);

  useEffect(() => {
    if (!telemetryStreamEnabled) {
      return;
    }
    closeTelemetryStream();
    openTelemetryStream();
  }, [telemetryStreamUrl, telemetryStreamEnabled, closeTelemetryStream, openTelemetryStream]);

  useEffect(() => {
    return () => {
      closeTelemetryStream();
    };
  }, [closeTelemetryStream]);

  useEffect(() => {
    return () => {
      telemetryRecordingRef.current.active = false;
      telemetryRecordingRef.current.samples = [];
    };
  }, []);

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
        geodesicFallbacks: 0,
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
        su7Totals.geodesicFallbacks += su7.geodesicFallbacks;
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
      const su7GpuStats = su7GpuStatsRef.current;
      const su7GpuWarning = su7GpuWarningRef.current;
      const su7GpuProfile = su7GpuLastProfileRef.current;
      const qualiaSeriesLength = Math.min(history.length, 90);
      const qualiaHistory = history.slice(-qualiaSeriesLength);
      const qualiaLatestMetrics =
        qualiaHistory.length > 0
          ? qualiaHistory[qualiaHistory.length - 1]!.metrics.qualia
          : history[history.length - 1]!.metrics.qualia;
      const projectQualiaSeries = (selector: (metrics: QualiaMetrics) => number) =>
        qualiaHistory.map((entry) => {
          const value = selector(entry.metrics.qualia);
          return Number.isFinite(value) ? clamp01(value) : 0;
        });
      let fps = 0;
      if (history.length > 1) {
        const span = history[history.length - 1]!.ts - history[0]!.ts;
        if (span > 1) {
          fps = ((history.length - 1) * 1000) / span;
        }
      }
      const frameMs = fps > 0 ? 1000 / fps : 0;
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
          geodesicFallbacks: su7Totals.geodesicFallbacks / count,
        },
        hopf:
          history.length > 0
            ? {
                lenses: history[history.length - 1].metrics.hopf.lenses.map((lens) => ({
                  ...lens,
                  axes: [lens.axes[0], lens.axes[1]] as [number, number],
                  base: [lens.base[0], lens.base[1], lens.base[2]] as [number, number, number],
                })),
              }
            : { lenses: [] },
        qualia: {
          latest: qualiaLatestMetrics,
          series: {
            indraIndex: projectQualiaSeries((q) => q.indraIndex),
            symmetry: projectQualiaSeries((q) => q.symmetry),
            colorfulness: projectQualiaSeries((q) => q.colorfulness),
            edgeDensity: projectQualiaSeries((q) => q.edgeDensity),
          },
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
        performance: {
          fps,
          frameMs,
        },
        frameSamples: history.length,
        updatedAt: Date.now(),
        su7Gpu: su7GpuStats
          ? {
              backend: su7GpuStats.backend,
              medianMs: su7GpuStats.medianMs,
              meanMs: su7GpuStats.meanMs,
              sampleCount: su7GpuStats.sampleCount,
              baselineMs: su7GpuStats.baselineMs,
              drift: su7GpuStats.drift,
              warning: su7GpuStats.warning,
              lastProfile: su7GpuProfile,
              warningEvent: su7GpuWarning ?? null,
            }
          : null,
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
    }, 200);
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
    const desiredComponents = polarizationEnabled ? 2 : 1;
    if (
      !kurStateRef.current ||
      kurStateRef.current.width !== width ||
      kurStateRef.current.height !== height ||
      kurStateRef.current.componentCount !== desiredComponents
    ) {
      kurStateRef.current = createKuramotoState(width, height, undefined, {
        componentCount: desiredComponents,
      });
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
  }, [width, height, polarizationEnabled]);

  const ensureVolumeState = useCallback(() => {
    if (width <= 0 || height <= 0) {
      volumeStubRef.current = null;
      volumeFieldRef.current = null;
      return;
    }
    const stub = volumeStubRef.current;
    if (!stub || stub.width !== width || stub.height !== height) {
      volumeStubRef.current = createVolumeStubState(
        width,
        height,
        getTimelineSeed('volumeNoise', 0),
      );
    }
  }, [width, height, getTimelineSeed]);

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
        buffers = {
          phases,
          magnitudes,
          magnitudeHist: new Float32Array(orientationCount * SURFACE_HIST_BINS),
          orientationCount,
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
    (dt: number, tSeconds: number) => {
      if (!kurEnabled) return;
      ensureKurCpuState();
      if (!kurStateRef.current) return;
      const kernelSnapshot = kernelEventRef.current;
      kurAppliedKernelVersionRef.current = kernelSnapshot.version;
      const timestamp = typeof performance !== 'undefined' ? performance.now() : Date.now();
      const timelineTime = timelineClockRef.current;
      const seed = getTimelineSeed('kuramotoNoise', timelineTime);
      const frameRand = createNormalGenerator(seed);
      const result = stepKuramotoState(
        kurStateRef.current,
        getKurParams(),
        dt,
        frameRand,
        timestamp,
        {
          kernel: kernelSnapshot.spec,
          controls: { dmt },
          telemetry: { kernelVersion: kernelSnapshot.version },
          schedule: polarizationSchedule,
        },
      );
      kurTelemetryRef.current = result.telemetry;
      kurIrradianceRef.current = result.irradiance;
      logKurTelemetry(result.telemetry);
    },
    [
      kurEnabled,
      ensureKurCpuState,
      getKurParams,
      getTimelineSeed,
      dmt,
      logKurTelemetry,
      polarizationSchedule,
    ],
  );

  const deriveKurFieldsCpu = useCallback(() => {
    if (!kurEnabled) return;
    ensureKurCpuState();
    if (!kurStateRef.current || !cpuDerivedRef.current) return;
    const kernelSnapshot = kernelEventRef.current;
    const activeDmt = getTimelineNumber('dmt', dmt);
    deriveKuramotoFieldsCore(kurStateRef.current, cpuDerivedRef.current, {
      kernel: kernelSnapshot.spec,
      controls: { dmt: activeDmt },
      schedule: polarizationSchedule,
    });
    markFieldFresh('phase', cpuDerivedRef.current.resolution, 'cpu');
  }, [kurEnabled, ensureKurCpuState, dmt, markFieldFresh, getTimelineNumber, polarizationSchedule]);

  const resetKuramotoField = useCallback(() => {
    initKuramotoCpu(qInit);
    if (!kurSyncRef.current && workerRef.current && workerReadyRef.current) {
      const currentEval = timelineEvaluationRef.current;
      const fps = timelinePlayerRef.current?.fps ?? RECORDING_FPS;
      const seedTime = currentEval && fps > 0 ? currentEval.frameIndex / fps : 0;
      workerRef.current.postMessage({
        kind: 'reset',
        qInit,
        seed: getTimelineSeed('kuramotoNoise', seedTime),
      });
    }
  }, [initKuramotoCpu, qInit, getTimelineSeed]);

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
    const currentEval = timelineEvaluationRef.current;
    const currentFps = timelinePlayerRef.current?.fps ?? RECORDING_FPS;
    const seedTime = currentEval && currentFps > 0 ? currentEval.frameIndex / currentFps : 0;
    worker.postMessage(
      {
        kind: 'init',
        width,
        height,
        params: getKurParams(),
        qInit,
        buffers,
        seed: getTimelineSeed('kuramotoNoise', seedTime),
        componentCount: polarizationEnabled ? 2 : 1,
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
    getTimelineSeed,
    polarizationEnabled,
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

  const processGuardrailFrame = useCallback(
    (metrics: RainbowFrameMetrics | null, guardrails?: Su7GuardrailStatus) => {
      if (!metrics) {
        return;
      }
      const events = guardrails?.events ?? [];
      const frameTimeMs = guardrailFrameTimeRef.current;
      setGuardrailConsole((prev) => {
        const energy = Number.isFinite(metrics.su7.projectorEnergy)
          ? metrics.su7.projectorEnergy
          : 0;
        const alpha = 0.2;
        const energyEma =
          prev.lastEnergy == null ? energy : prev.energyEma + (energy - prev.energyEma) * alpha;
        const entries: { kind: GuardrailAuditKind; summary: GuardrailSummary }[] = [];
        for (const event of events) {
          entries.push({ kind: event.kind, summary: summarizeGuardrailEvent(event) });
        }
        const flickerEvent = detectFlickerGuardrail(prev.lastEnergy, energy, frameTimeMs, {
          ratioThreshold: 0.12,
          frequencyThreshold: 30,
          minEnergy: 0.02,
        });
        if (flickerEvent) {
          entries.push({ kind: flickerEvent.kind, summary: summarizeGuardrailEvent(flickerEvent) });
        }
        let auditLog = prev.auditLog.slice();
        if (entries.length > 0) {
          const now = Date.now();
          for (const { kind, summary } of entries) {
            const message = summary.message;
            const lastEntry = auditLog[auditLog.length - 1];
            if (!lastEntry || lastEntry.message !== message || now - lastEntry.timestamp > 250) {
              auditLog = [
                ...auditLog,
                {
                  id: guardrailAuditIdRef.current++,
                  kind,
                  message,
                  severity: summary.severity,
                  timestamp: now,
                },
              ];
              if (auditLog.length > 12) {
                auditLog.shift();
              }
            }
          }
        }
        return {
          ...prev,
          unitaryError: metrics.su7.unitaryError,
          determinantDrift: metrics.su7.determinantDrift,
          frameTimeMs: frameTimeMs > 0 ? frameTimeMs : prev.frameTimeMs,
          energyEma,
          lastEnergy: energy,
          auditLog,
        };
      });
    },
    [setGuardrailConsole],
  );

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

      updateTimelineForTime(timelineClockRef.current);
      const activeDmt = getTimelineNumber('dmt', dmt);
      const activeArousal = getTimelineNumber('arousal', arousal);
      const activeBlend = getTimelineNumber('blend', blend);
      const activeNormPin = getTimelineBoolean('normPin', normPin);
      const activeMicrosaccade = getTimelineBoolean('microsaccade', microsaccade);
      const activePhasePin = getTimelineBoolean('phasePin', phasePin);
      const activeEdgeThreshold = getTimelineNumber('edgeThreshold', edgeThreshold);
      const activeJitter = getTimelineNumber('jitter', jitter);
      const activeSigma = getTimelineNumber('sigma', sigma);
      const activeContrast = getTimelineNumber('contrast', contrast);
      const activeRimAlpha = getTimelineNumber('rimAlpha', rimAlpha);
      const activeRimEnabled = getTimelineBoolean('rimEnabled', rimEnabled);
      const activeBeta2 = getTimelineNumber('beta2', beta2);
      const activeAlive = getTimelineBoolean('alive', alive);
      const activeSurfaceBlend = getTimelineNumber('surfaceBlend', surfaceBlend);
      const activeWarpAmp = getTimelineNumber('warpAmp', warpAmp);
      const activeThetaMode = getTimelineString<'gradient' | 'global'>('thetaMode', thetaMode);
      const activeThetaGlobal = getTimelineNumber('thetaGlobal', thetaGlobal);
      const rimDebugRequest = showRimDebug ? ensureRimDebugBuffers() : null;
      const surfaceDebugRequest =
        showSurfaceDebug && orientations.length > 0
          ? ensureSurfaceDebugBuffers(orientations.length)
          : null;

      const { base: couplingBase, effective: couplingEffective } = computeCouplingPair(
        options?.toggles,
      );

      const guardrailForce = guardrailCommandsRef.current.forceReorthon;
      const guardrailOpts = commitObs
        ? { forceReorthon: guardrailForce, emitGuardrailEvents: true }
        : { emitGuardrailEvents: false };

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
        dmt: activeDmt,
        arousal: activeArousal,
        blend: activeBlend,
        normPin: activeNormPin,
        normTarget: normTargetRef.current,
        lastObs: lastObsRef.current,
        lambdaRef,
        lambdas,
        beta2: activeBeta2,
        microsaccade: activeMicrosaccade,
        alive: activeAlive,
        phasePin: activePhasePin,
        edgeThreshold: activeEdgeThreshold,
        wallpaperGroup: wallGroup,
        surfEnabled,
        orientationAngles: orientations,
        thetaMode: activeThetaMode,
        thetaGlobal: activeThetaGlobal,
        polBins,
        jitter: activeJitter,
        coupling: couplingEffective,
        couplingBase,
        sigma: activeSigma,
        contrast: activeContrast,
        rimAlpha: activeRimAlpha,
        rimEnabled: activeRimEnabled,
        displayMode,
        surfaceBlend: activeSurfaceBlend,
        surfaceRegion,
        warpAmp: activeWarpAmp,
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
        guardrailOptions: guardrailOpts,
        composer,
        kurTelemetry: kurTelemetryRef.current ?? undefined,
        fluxOverlay: activeFluxOverlay ?? undefined,
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
        captureFrameMetrics('cpu', result.metrics, kernelSnapshot.version);
      }
      updateDebugSnapshots(commitObs, rimDebugRequest, surfaceDebugRequest, result.debug);
      updatePhaseDebug(commitObs, phaseField);
      if (commitObs) {
        processGuardrailFrame(result.metrics, result.guardrails);
        guardrailCommandsRef.current.forceReorthon = false;
      }
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
      activeFluxOverlay,
      basePixelsRef,
      getTimelineNumber,
      getTimelineBoolean,
      getTimelineString,
      updateTimelineForTime,
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

      updateTimelineForTime(timelineClockRef.current);
      const activeDmt = getTimelineNumber('dmt', dmt);
      const activeArousal = getTimelineNumber('arousal', arousal);
      const activeBlend = getTimelineNumber('blend', blend);
      const activeNormPin = getTimelineBoolean('normPin', normPin);
      const activeBeta2 = getTimelineNumber('beta2', beta2);
      const activeMicrosaccade = getTimelineBoolean('microsaccade', microsaccade);
      const activeAlive = getTimelineBoolean('alive', alive);
      const activePhasePin = getTimelineBoolean('phasePin', phasePin);
      const activeEdgeThreshold = getTimelineNumber('edgeThreshold', edgeThreshold);
      const activeJitter = getTimelineNumber('jitter', jitter);
      const activeSigma = getTimelineNumber('sigma', sigma);
      const activeContrast = getTimelineNumber('contrast', contrast);
      const activeRimAlpha = getTimelineNumber('rimAlpha', rimAlpha);
      const activeRimEnabled = getTimelineBoolean('rimEnabled', rimEnabled);
      const activeThetaMode = getTimelineString<'gradient' | 'global'>('thetaMode', thetaMode);
      const activeThetaGlobal = getTimelineNumber('thetaGlobal', thetaGlobal);
      const activeSurfaceBlend = getTimelineNumber('surfaceBlend', surfaceBlend);
      const activeWarpAmp = getTimelineNumber('warpAmp', warpAmp);

      const guardrailForce = guardrailCommandsRef.current.forceReorthon;
      const guardrailNeedsMetrics = su7Params.enabled && su7Params.gain > 1e-4;

      const rimDebugRequest = showRimDebug ? ensureRimDebugBuffers() : null;
      const surfaceDebugRequest =
        showSurfaceDebug && orientations.length > 0
          ? ensureSurfaceDebugBuffers(orientations.length)
          : null;

      const { base: couplingBase, effective: couplingEffective } = computeCouplingPair();

      const telemetryActive = telemetryRef.current.enabled && commitObs;
      const needsCpuCompositor =
        commitObs &&
        (telemetryActive ||
          rimDebugRequest != null ||
          surfaceDebugRequest != null ||
          guardrailNeedsMetrics);
      const renderStart = telemetryActive ? performance.now() : 0;

      const ke = kEff(kernelSpec, activeDmt);
      const effectiveBlend = clamp01(activeBlend + ke.transparency * 0.5);
      const eps = 1e-6;
      const frameGain = activeNormPin
        ? Math.pow((normTargetRef.current + eps) / (lastObsRef.current + eps), 0.5)
        : 1.0;

      const baseOffsets = {
        L: activeBeta2 * (lambdaRef / lambdas.L - 1),
        M: activeBeta2 * (lambdaRef / lambdas.M - 1),
        S: activeBeta2 * (lambdaRef / lambdas.S - 1),
      } as const;

      const jitterPhase = activeMicrosaccade ? tSeconds * 6.0 : 0.0;
      const breath = activeAlive ? 0.15 * Math.sin(2 * Math.PI * 0.55 * tSeconds) : 0.0;

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
      if (activePhasePin && activeMicrosaccade) {
        let muSum = 0;
        let muCount = 0;
        for (let yy = 0; yy < height; yy += 8) {
          for (let xx = 0; xx < width; xx += 8) {
            const idx = yy * width + xx;
            if (mag[idx] >= activeEdgeThreshold) {
              muSum += Math.sin(jitterPhase + hash2(xx, yy) * Math.PI * 2);
              muCount++;
            }
          }
        }
        muJ = muCount ? muSum / muCount : 0;
      }

      let metricDebug: ReturnType<typeof renderRainbowFrame>['debug'] | null = null;
      let guardrailMetrics: ReturnType<typeof renderRainbowFrame> | null = null;
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
          dmt: activeDmt,
          arousal: activeArousal,
          blend: activeBlend,
          normPin: activeNormPin,
          normTarget: normTargetRef.current,
          lastObs: lastObsRef.current,
          lambdaRef,
          lambdas,
          beta2: activeBeta2,
          microsaccade: activeMicrosaccade,
          alive: activeAlive,
          phasePin: activePhasePin,
          edgeThreshold: activeEdgeThreshold,
          wallpaperGroup: wallGroup,
          surfEnabled,
          orientationAngles: orientations,
          thetaMode: activeThetaMode,
          thetaGlobal: activeThetaGlobal,
          polBins,
          jitter: activeJitter,
          coupling: couplingEffective,
          couplingBase,
          sigma: activeSigma,
          contrast: activeContrast,
          rimAlpha: activeRimAlpha,
          rimEnabled: activeRimEnabled,
          displayMode,
          surfaceBlend: activeSurfaceBlend,
          surfaceRegion,
          warpAmp: activeWarpAmp,
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
          guardrailOptions: {
            forceReorthon: guardrailForce,
            emitGuardrailEvents: true,
          },
          composer,
          kurTelemetry: kurTelemetryRef.current ?? undefined,
          fluxOverlay: activeFluxOverlay ?? undefined,
        });
        metricDebug = metricsResult.debug;
        guardrailMetrics = metricsResult;
        if (telemetryActive) {
          captureFrameMetrics('gpu', metricsResult.metrics, kernelSnapshot.version);
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

      if (commitObs && guardrailMetrics) {
        processGuardrailFrame(guardrailMetrics.metrics, guardrailMetrics.guardrails);
      }

      guardrailCommandsRef.current.forceReorthon = false;

      renderer.uploadPhase(phaseField);
      renderer.uploadVolume(volumeField ?? null);

      if (!su7Params.enabled) {
        su7GpuReadyRef.current = null;
      }

      let su7Uniforms: Su7Uniforms = {
        enabled: false,
        gain: su7Params.gain,
        decimationStride: 1,
        decimationMode: 'hybrid',
        projectorMode: 'identity',
        projectorWeight: 0,
        projectorMatrix: null,
        pretransformed: false,
        hopfLenses: null,
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
          const kernel = su7GpuKernelRef.current;
          const ready = su7GpuReadyRef.current;
          if (
            kernel &&
            ready &&
            ready.width === width &&
            ready.height === height &&
            ready.vectors.length >= texelCount * SU7_GPU_KERNEL_VECTOR_STRIDE
          ) {
            try {
              fillSu7VectorBuffersFromPacked(ready.vectors, ready.norms, texelCount, vectorBuffers);
              su7Uniforms.pretransformed = true;
              su7GpuStatsRef.current = kernel.getStats();
              su7GpuLastProfileRef.current = ready.profile ?? null;
            } catch (error) {
              console.warn('[su7-gpu] failed to apply pretransformed vectors', error);
              su7Uniforms.pretransformed = false;
              fillSu7VectorBuffers(
                embedResult.vectors,
                embedResult.norms,
                texelCount,
                vectorBuffers,
              );
            }
          } else {
            su7Uniforms.pretransformed = false;
            fillSu7VectorBuffers(embedResult.vectors, embedResult.norms, texelCount, vectorBuffers);
            if (
              ready &&
              (ready.width !== width || ready.height !== height || ready.vectors.length === 0)
            ) {
              su7GpuReadyRef.current = null;
            }
          }
          su7VectorBuffersRef.current = vectorBuffers;
          const su7ContextGpu =
            su7Params.enabled && su7Params.gain > 1e-4
              ? deriveSu7ScheduleContext({
                  width,
                  height,
                  phase: phaseField,
                  rim: rimField,
                  volume: volumeField,
                  dmt: activeDmt,
                  arousal: activeArousal,
                  curvatureStrength,
                })
              : null;
          const su7RuntimeGpu = resolveSu7Runtime(su7Params, su7ContextGpu ?? undefined, {
            forceReorthon: guardrailForce,
            emitGuardrailEvents: false,
          });
          const unitary = su7RuntimeGpu.unitary;
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
          } else if (projectorId === 'hopflens') {
            projectorMode = 'hopfLens';
          }
          const projectorWeight = Math.min(
            1,
            Math.max(0, Math.abs(su7Params.projector?.weight ?? 1)),
          );
          const pretransformedFlag = su7Uniforms.pretransformed ?? false;
          const hopfLensUniforms =
            projectorMode === 'hopfLens'
              ? resolveHopfLenses(su7Params.projector).map((lens) => ({
                  axes: lens.axes,
                  baseMix: lens.baseMix ?? 1,
                  fiberMix: lens.fiberMix ?? 1,
                }))
              : null;
          su7Uniforms = {
            enabled: true,
            gain: su7Params.gain,
            decimationStride,
            decimationMode,
            projectorMode,
            projectorWeight,
            projectorMatrix: null,
            pretransformed: pretransformedFlag,
            hopfLenses: hopfLensUniforms,
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
            pretransformed: pretransformedFlag,
          };
          if (kernel && !su7GpuPendingRef.current) {
            const packedUnitary = packSu7Unitary(unitary, su7GpuPackedMatrixRef.current);
            su7GpuPackedMatrixRef.current = packedUnitary;
            const packedVectors = packSu7Vectors(embedResult.vectors, su7GpuPackedInputRef.current);
            su7GpuPackedInputRef.current = packedVectors;
            const expectedFloats = packedVectors.length;
            const outputBuffer =
              su7GpuTransformedRef.current && su7GpuTransformedRef.current.length === expectedFloats
                ? su7GpuTransformedRef.current
                : new Float32Array(expectedFloats);
            su7GpuTransformedRef.current = outputBuffer;
            const normsCopy = embedResult.norms.slice();
            su7GpuPendingRef.current = kernel
              .dispatch({
                unitary: packedUnitary,
                input: packedVectors,
                vectorCount: texelCount,
                output: outputBuffer,
              })
              .then((result) => {
                su7GpuReadyRef.current = {
                  width,
                  height,
                  vectors: result,
                  norms: normsCopy,
                  profile: kernel.getLastProfile(),
                  timestamp: typeof performance !== 'undefined' ? performance.now() : Date.now(),
                };
                su7GpuStatsRef.current = kernel.getStats();
              })
              .catch((error) => {
                console.warn('[su7-gpu] dispatch failed', error);
              })
              .finally(() => {
                su7GpuPendingRef.current = null;
              });
          } else if (!kernel) {
            su7GpuReadyRef.current = null;
          }
        }
      }

      renderer.uploadSu7(su7Payload);

      const couplingScale = 1 + 0.65 * activeDmt;

      const lastGpuTracerTime = gpuTracerRef.current.lastTime;
      const tracerDelta = lastGpuTracerTime != null ? Math.max(0, tSeconds - lastGpuTracerTime) : 0;
      const tracerDt =
        lastGpuTracerTime != null ? Math.max(1 / 480, Math.min(tracerDelta, 0.25)) : 1 / 60;
      const tracerEnabled = tracerRuntime.enabled;
      const tracerReset = gpuTracerRef.current.needsReset || !tracerEnabled;
      const earlyVisionFrame = earlyVisionFrameRef.current;
      earlyVisionFrameRef.current = (earlyVisionFrame + 1) >>> 0;
      const earlyVisionForceUpdate = earlyVisionForceUpdateRef.current;

      renderer.render({
        time: tSeconds,
        edgeThreshold: activeEdgeThreshold,
        effectiveBlend,
        displayMode: displayModeToEnum(displayMode),
        baseOffsets: [baseOffsets.L, baseOffsets.M, baseOffsets.S],
        sigma: activeSigma,
        jitter: activeJitter,
        jitterPhase,
        breath,
        muJ,
        phasePin: activePhasePin,
        microsaccade: activeMicrosaccade,
        polBins,
        thetaMode: activeThetaMode === 'gradient' ? 0 : 1,
        thetaGlobal: activeThetaGlobal,
        contrast: activeContrast,
        frameGain,
        rimAlpha: activeRimAlpha,
        rimEnabled: activeRimEnabled,
        warpAmp: activeWarpAmp,
        curvatureStrength,
        curvatureMode,
        surfaceBlend: activeSurfaceBlend,
        surfaceRegion: surfaceRegionToEnum(surfaceRegion),
        surfEnabled,
        kurEnabled,
        volumeEnabled,
        useWallpaper,
        kernel: ke,
        alive: activeAlive,
        beta2: activeBeta2,
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
        earlyVision: {
          dogEnabled: earlyVisionDogEnabled,
          orientationEnabled: earlyVisionOrientationEnabled,
          motionEnabled: earlyVisionMotionEnabled,
          opacity: earlyVisionOpacity,
          dogSigma: earlyVisionDoGSigma,
          dogRatio: earlyVisionDoGRatio,
          dogGain: earlyVisionDoGGain,
          downsample: Math.max(1, earlyVisionDownsample),
          orientationGain: earlyVisionOrientationGain,
          orientationSharpness: earlyVisionOrientationSharpness,
          orientationCount: earlyVisionOrientationAngles.length,
          orientationCos: earlyVisionOrientationCos,
          orientationSin: earlyVisionOrientationSin,
          motionGain: earlyVisionMotionGain,
          frameModulo: Math.max(1, earlyVisionFrameModulo),
          frameIndex: earlyVisionFrame,
          forceUpdate: earlyVisionForceUpdate,
          viewMode: earlyVisionViewMode,
        },
        orientations: orientationCache,
        ops: gpuOps,
        center: [cx, cy],
      });

      if (earlyVisionForceUpdate) {
        earlyVisionForceUpdateRef.current = false;
      }
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
      earlyVisionDogEnabled,
      earlyVisionOrientationEnabled,
      earlyVisionMotionEnabled,
      earlyVisionOpacity,
      earlyVisionDoGSigma,
      earlyVisionDoGRatio,
      earlyVisionDoGGain,
      earlyVisionDownsample,
      earlyVisionOrientationGain,
      earlyVisionOrientationSharpness,
      earlyVisionOrientationCount,
      earlyVisionMotionGain,
      earlyVisionFrameModulo,
      earlyVisionViewMode,
      earlyVisionOrientationCos,
      earlyVisionOrientationSin,
      earlyVisionOrientationAngles,
      tracerRuntime,
      deriveSu7ScheduleContext,
      getTimelineNumber,
      getTimelineBoolean,
      getTimelineString,
      updateTimelineForTime,
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
      const timelineTime = timelineClockRef.current;
      updateTimelineForTime(timelineTime);
      if (kurSyncRef.current) {
        stepKuramotoCpu(dt, tSeconds);
        deriveKurFieldsCpu();
      } else {
        const worker = workerRef.current;
        if (worker && workerReadyRef.current) {
          const inflight = workerInflightRef.current;
          if (inflight < 2) {
            const frameId = workerNextFrameIdRef.current++;
            const seed = getTimelineSeed('kuramotoNoise', timelineTime);
            worker.postMessage({
              kind: 'tick',
              dt,
              timestamp: tSeconds,
              frameId,
              seed,
              schedule: polarizationSchedule ?? null,
              componentCount: polarizationEnabled ? 2 : 1,
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
    [
      kurEnabled,
      stepKuramotoCpu,
      deriveKurFieldsCpu,
      swapWorkerFrame,
      recordTelemetry,
      updateTimelineForTime,
      getTimelineSeed,
      polarizationSchedule,
      polarizationEnabled,
    ],
  );

  const drawFrameCpu = useCallback(
    (ctx: CanvasRenderingContext2D, tSeconds: number) => {
      const telemetryActive = telemetryRef.current.enabled;
      const frameBegin = performance.now();
      const frameStart = telemetryActive ? frameBegin : 0;
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
      recordTelemetry('frame', performance.now() - frameBegin);
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

      const cpuState = createKuramotoState(width, height, undefined, {
        componentCount: polarizationEnabled ? 2 : 1,
      });
      const cpuBuffer = new ArrayBuffer(derivedBufferSize(width, height));
      const cpuDerived = createDerivedViews(cpuBuffer, width, height);
      const cpuRand = createNormalGenerator(seed);
      initKuramotoState(cpuState, qInit, cpuDerived);

      const baselineFrames: Uint8ClampedArray[] = [];
      for (let i = 0; i < frameCount; i++) {
        stepKuramotoState(cpuState, params, dt, cpuRand, dt * (i + 1), {
          kernel: operatorKernel,
          controls: { dmt },
          schedule: polarizationSchedule,
        });
        deriveKuramotoFieldsCore(cpuState, cpuDerived, {
          kernel: operatorKernel,
          controls: { dmt },
          schedule: polarizationSchedule,
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
            schedule: polarizationSchedule ?? null,
            componentCount: polarizationEnabled ? 2 : 1,
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
      polarizationSchedule,
      polarizationEnabled,
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
  const telemetryStreamStatusLabel =
    telemetryStreamStatus === 'connected'
      ? 'Connected'
      : telemetryStreamStatus === 'connecting'
        ? 'Connecting…'
        : telemetryStreamStatus === 'error'
          ? 'Error'
          : 'Idle';
  const telemetryStreamStatusColor =
    telemetryStreamStatus === 'connected'
      ? '#34d399'
      : telemetryStreamStatus === 'error'
        ? '#f87171'
        : telemetryStreamStatus === 'connecting'
          ? '#fbbf24'
          : '#94a3b8';

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
    const loadTimelineHandler = (payload: Timeline | string) =>
      typeof payload === 'string' ? loadTimelineFromJson(payload) : loadTimeline(payload);
    w.__loadTimeline = loadTimelineHandler;
    w.__clearTimeline = clearTimeline;
    w.__exportTimeline = exportTimeline;
    w.__getTimelineHash = () => timelineHashRef.current;
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
      if (w.__loadTimeline === loadTimelineHandler) {
        delete w.__loadTimeline;
      }
      if (w.__clearTimeline === clearTimeline) {
        delete w.__clearTimeline;
      }
      if (w.__exportTimeline === exportTimeline) {
        delete w.__exportTimeline;
      }
      if (w.__getTimelineHash) {
        delete w.__getTimelineHash;
      }
    };
  }, [
    setFrameProfiler,
    runRegressionHarness,
    runGpuParityCheck,
    measureRenderPerformance,
    handleRendererToggle,
    setTelemetryEnabled,
    loadTimeline,
    loadTimelineFromJson,
    clearTimeline,
    exportTimeline,
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let anim = true;
    let frameId = 0;
    const start = performance.now();

    if (renderBackend === 'gpu') {
      const render = (timestamp: number) => {
        if (!anim) return;
        const state = ensureGpuRenderer();
        if (!state) {
          anim = false;
          return;
        }
        const frameBegin = performance.now();
        const frameStart = telemetryRef.current.enabled ? frameBegin : 0;
        const t = (timestamp - start) * 0.001;
        const dt = 0.016 * speed;
        advanceKuramoto(dt, t);
        advanceVolume(dt);
        drawFrameGpu(state, t, true);
        recordTelemetry('frame', performance.now() - frameBegin);
        frameId = requestAnimationFrame(render);
      };
      frameId = requestAnimationFrame(render);
      return () => {
        anim = false;
        cancelAnimationFrame(frameId);
      };
    }

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;
    canvas.width = width;
    canvas.height = height;
    const renderCpu = (timestamp: number) => {
      if (!anim) return;
      const t = (timestamp - start) * 0.001;
      drawFrameCpu(ctx, t);
      frameId = requestAnimationFrame(renderCpu);
    };
    frameId = requestAnimationFrame(renderCpu);
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

      recordMacroPresetAction(preset.name);

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
      setMacroBinding(cloneMacroBinding(runtime.macroBinding));
      setMacroKnobValue(
        typeof runtime.macroKnobValue === 'number' && Number.isFinite(runtime.macroKnobValue)
          ? runtime.macroKnobValue
          : 0,
      );
      setMacroLearnMode(false);
      const runtimeEarlyVision = runtime.earlyVision ?? {
        dogEnabled: false,
        orientationEnabled: false,
        motionEnabled: false,
        opacity: 0.65,
        dogSigma: 1.2,
        dogRatio: 1.6,
        dogGain: 2.4,
        downsample: 1,
        orientationGain: 0.9,
        orientationSharpness: 2,
        orientationCount: 4,
        motionGain: 6,
        frameModulo: 1,
        viewMode: 'blend' as const,
      };
      setEarlyVisionDogEnabled(Boolean(runtimeEarlyVision.dogEnabled));
      setEarlyVisionOrientationEnabled(Boolean(runtimeEarlyVision.orientationEnabled));
      setEarlyVisionMotionEnabled(Boolean(runtimeEarlyVision.motionEnabled));
      setEarlyVisionOpacity(
        Number.isFinite(runtimeEarlyVision.opacity) ? runtimeEarlyVision.opacity : 0.65,
      );
      setEarlyVisionDoGSigma(
        Number.isFinite(runtimeEarlyVision.dogSigma) ? runtimeEarlyVision.dogSigma : 1.2,
      );
      setEarlyVisionDoGRatio(
        Number.isFinite(runtimeEarlyVision.dogRatio) ? runtimeEarlyVision.dogRatio : 1.6,
      );
      setEarlyVisionDoGGain(
        Number.isFinite(runtimeEarlyVision.dogGain) ? runtimeEarlyVision.dogGain : 2.4,
      );
      setEarlyVisionDownsample(
        Number.isFinite(runtimeEarlyVision.downsample)
          ? Math.max(1, Math.round(runtimeEarlyVision.downsample))
          : 1,
      );
      setEarlyVisionOrientationGain(
        Number.isFinite(runtimeEarlyVision.orientationGain)
          ? runtimeEarlyVision.orientationGain
          : 0.9,
      );
      setEarlyVisionOrientationSharpness(
        Number.isFinite(runtimeEarlyVision.orientationSharpness)
          ? runtimeEarlyVision.orientationSharpness
          : 2,
      );
      setEarlyVisionOrientationCount(
        Number.isFinite(runtimeEarlyVision.orientationCount)
          ? Math.max(1, Math.min(8, Math.round(runtimeEarlyVision.orientationCount)))
          : 4,
      );
      setEarlyVisionMotionGain(
        Number.isFinite(runtimeEarlyVision.motionGain) ? runtimeEarlyVision.motionGain : 6,
      );
      setEarlyVisionFrameModulo(
        Number.isFinite(runtimeEarlyVision.frameModulo)
          ? Math.max(1, Math.round(runtimeEarlyVision.frameModulo))
          : 1,
      );
      setEarlyVisionViewMode(runtimeEarlyVision.viewMode === 'overlay' ? 'overlay' : 'blend');
      earlyVisionForceUpdateRef.current = true;
      earlyVisionFrameRef.current = 0;

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
          gateAppends: params.su7GateAppends,
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
      setPolarizationEnabled(kuramoto.polarizationEnabled ?? false);
      setWavePlateEnabled(kuramoto.wavePlateEnabled ?? true);
      setWavePlatePhaseDeg(kuramoto.wavePlatePhaseDeg ?? 90);
      setWavePlateOrientationDeg(kuramoto.wavePlateOrientationDeg ?? 0);
      setSu7PolarizationEnabled(kuramoto.su7PolarizationEnabled ?? false);
      setSu7PolarizationColumn(kuramoto.su7PolarizationColumn ?? 0);
      setSu7PolarizationGain(kuramoto.su7PolarizationGain ?? 1);
      setSu7PolarizationBlend(kuramoto.su7PolarizationBlend ?? 1);
      setPolarizerEnabled(kuramoto.polarizerEnabled ?? false);
      setPolarizerOrientationDeg(kuramoto.polarizerOrientationDeg ?? 0);
      setPolarizerExtinction(kuramoto.polarizerExtinction ?? 0);

      stopQcdAnneal();
      setQcdBeta(preset.qcd.beta);
      setQcdStepsPerSecond(preset.qcd.stepsPerSecond);
      setQcdSmearingAlpha(preset.qcd.smearingAlpha);
      setQcdSmearingIterations(preset.qcd.smearingIterations);
      setQcdBaseSeed(preset.qcd.baseSeed);
      const presetDepth = Math.max(1, Math.floor(preset.qcd.depth ?? 1));
      const presetTemporal = Math.max(1, Math.floor(preset.qcd.temporalExtent ?? 1));
      const presetBatch = Math.max(
        1,
        Math.min(Math.floor(preset.qcd.batchLayers ?? 1), presetDepth * presetTemporal),
      );
      const presetSchedule = Array.isArray(preset.qcd.temperatureSchedule)
        ? preset.qcd.temperatureSchedule.filter((value) => Number.isFinite(value)).slice(0, 64)
        : [];
      setQcdDepth(presetDepth);
      setQcdTemporalExtent(presetTemporal);
      setQcdBatchLayers(presetBatch);
      setQcdTemperatureScheduleText(
        presetSchedule.length > 0 ? presetSchedule.map((value) => value.toFixed(3)).join(', ') : '',
      );
      setQcdPolyakovSchedule([]);
      setQcdPerfLog([]);
      if (preset.qcd.snapshot) {
        const snapshotSources = preset.qcd.snapshot.data.sources.map((source) => ({
          x: source.x,
          y: source.y,
          charge: source.charge,
          strength: source.strength,
        }));
        setFluxSources(snapshotSources);
        setQcdSnapshotHash(preset.qcd.snapshot.hash);
        setTimeout(() => {
          ensureQcdRuntime({
            snapshot: preset.qcd.snapshot!.data,
            sourcesOverride: snapshotSources,
          });
        }, 0);
        const snapshotSchedule = Array.isArray(preset.qcd.snapshot.data.config.temperatureSchedule)
          ? preset.qcd.snapshot.data.config.temperatureSchedule.filter((value) =>
              Number.isFinite(value),
            )
          : [];
        if (snapshotSchedule.length > 0) {
          setQcdPolyakovSchedule([...snapshotSchedule]);
        }
      } else {
        qcdRuntimeRef.current = null;
        qcdSnapshotRef.current = null;
        setQcdObservables(null);
        setQcdOverlayState(null);
        setQcdProbeFrame(null);
        setQcdSnapshotHash(null);
        setTimeout(() => {
          ensureQcdRuntime({ reinitialize: true });
        }, 0);
      }

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
    [
      handleRendererToggle,
      loadImageAsset,
      resetTracerState,
      setSu7ScheduleStrength,
      setSu7Params,
      stopQcdAnneal,
      ensureQcdRuntime,
      recordMacroPresetAction,
    ],
  );

  const applyPresetRef = useRef(applyPreset);

  useEffect(() => {
    applyPresetRef.current = applyPreset;
  }, [applyPreset]);

  const applyMacroEvent = useCallback(
    (event: MacroEvent) => {
      if (event.kind === 'set') {
        const handler = macroActionHandlers[event.target];
        if (handler) {
          handler(event.value);
        }
        return;
      }
      switch (event.action) {
        case 'applyPreset': {
          const preset = PRESETS.find((entry) => entry.name === event.presetName);
          if (preset) {
            applyPresetRef.current?.(preset);
          }
          break;
        }
        default:
          break;
      }
    },
    [macroActionHandlers],
  );

  const playMacro = useCallback(
    (macroId: string) => {
      cancelMacroPlayback();
      const script = macroLibrary.find((entry) => entry.id === macroId);
      if (!script || script.events.length === 0) {
        return;
      }
      setMacroPlaybackId(macroId);
      let executed = 0;
      const timers: number[] = [];
      script.events.forEach((event) => {
        const timerId = globalThis.setTimeout(
          () => {
            applyMacroEvent(event);
            executed += 1;
            if (executed >= script.events.length) {
              macroPlaybackRef.current = null;
              setMacroPlaybackId(null);
            }
          },
          Math.max(0, event.at),
        );
        timers.push(timerId);
      });
      macroPlaybackRef.current = { timers };
    },
    [applyMacroEvent, cancelMacroPlayback, macroLibrary],
  );

  const deleteMacro = useCallback(
    (macroId: string) => {
      setMacroLibrary((prev) => prev.filter((entry) => entry.id !== macroId));
      if (macroPlaybackId === macroId) {
        cancelMacroPlayback();
      }
    },
    [cancelMacroPlayback, macroPlaybackId],
  );

  const renameMacro = useCallback((macroId: string, label: string) => {
    setMacroLibrary((prev) =>
      prev.map((entry) =>
        entry.id === macroId ? { ...entry, label: label.trim() || entry.label } : entry,
      ),
    );
  }, []);

  const buildCurrentPreset = useCallback((): Preset => {
    const su7Snapshot = cloneSu7RuntimeParams(su7Params);
    let qcdSnapshotEntry: PresetQcd['snapshot'] = null;
    const runtime = qcdRuntimeRef.current;
    if (runtime) {
      const snapshot = qcdSnapshotRef.current ?? buildQcdSnapshot(runtime, fluxSources);
      const { hash } = hashQcdSnapshot(snapshot);
      qcdSnapshotEntry = { data: snapshot, hash };
    }
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
        su7GateAppends: su7Snapshot.gateAppends,
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
        macroBinding: cloneMacroBinding(macroBinding),
        macroKnobValue: Number.isFinite(macroKnobValue) ? macroKnobValue : 0,
        earlyVision: {
          dogEnabled: earlyVisionDogEnabled,
          orientationEnabled: earlyVisionOrientationEnabled,
          motionEnabled: earlyVisionMotionEnabled,
          opacity: earlyVisionOpacity,
          dogSigma: earlyVisionDoGSigma,
          dogRatio: earlyVisionDoGRatio,
          dogGain: earlyVisionDoGGain,
          downsample: Math.max(1, earlyVisionDownsample),
          orientationGain: earlyVisionOrientationGain,
          orientationSharpness: earlyVisionOrientationSharpness,
          orientationCount: earlyVisionOrientationCount,
          motionGain: earlyVisionMotionGain,
          frameModulo: Math.max(1, earlyVisionFrameModulo),
          viewMode: earlyVisionViewMode,
        },
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
        polarizationEnabled,
        wavePlateEnabled,
        wavePlatePhaseDeg,
        wavePlateOrientationDeg,
        su7PolarizationEnabled,
        su7PolarizationColumn,
        su7PolarizationGain,
        su7PolarizationBlend,
        polarizerEnabled,
        polarizerOrientationDeg,
        polarizerExtinction,
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
      qcd: {
        beta: qcdBeta,
        stepsPerSecond: qcdStepsPerSecond,
        smearingAlpha: qcdSmearingAlpha,
        smearingIterations: qcdSmearingIterations,
        overRelaxationSteps: 1,
        baseSeed: qcdBaseSeed,
        depth: qcdDepthInt,
        temporalExtent: qcdTemporalExtentInt,
        batchLayers: qcdBatchLayersInt,
        temperatureSchedule: qcdTemperatureSchedule.length > 0 ? [...qcdTemperatureSchedule] : [],
        snapshot: qcdSnapshotEntry,
      },
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
    polarizationEnabled,
    wavePlateEnabled,
    wavePlatePhaseDeg,
    wavePlateOrientationDeg,
    su7PolarizationEnabled,
    su7PolarizationColumn,
    su7PolarizationGain,
    su7PolarizationBlend,
    polarizerEnabled,
    polarizerOrientationDeg,
    polarizerExtinction,
    selectedSyntheticCase,
    includeImageInPreset,
    imageAsset,
    qcdBeta,
    qcdDepthInt,
    qcdTemporalExtentInt,
    qcdBatchLayersInt,
    qcdTemperatureSchedule,
    qcdStepsPerSecond,
    qcdSmearingAlpha,
    qcdSmearingIterations,
    qcdBaseSeed,
    fluxSources,
    su7ScheduleStrength,
    coupling,
    composer,
    couplingToggles,
    earlyVisionDogEnabled,
    earlyVisionOrientationEnabled,
    earlyVisionMotionEnabled,
    earlyVisionOpacity,
    earlyVisionDoGSigma,
    earlyVisionDoGRatio,
    earlyVisionDoGGain,
    earlyVisionDownsample,
    earlyVisionOrientationGain,
    earlyVisionOrientationSharpness,
    earlyVisionOrientationCount,
    earlyVisionMotionGain,
    earlyVisionFrameModulo,
    earlyVisionViewMode,
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

  const handleExportEarlyVisionOverlay = useCallback(() => {
    if (!earlyVisionDogEnabled && !earlyVisionOrientationEnabled && !earlyVisionMotionEnabled) {
      console.warn('[early-vision] enable at least one analyzer before exporting an overlay.');
      return;
    }
    const gpuState = gpuStateRef.current;
    if (!gpuState) {
      console.warn('[early-vision] GPU renderer unavailable.');
      return;
    }
    const total = width * height * 4;
    const buffer = new Uint8Array(total);
    const ok = gpuState.renderer.readEarlyVisionOverlay(buffer);
    if (!ok) {
      console.warn('[early-vision] overlay texture not available yet.');
      return;
    }
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      console.warn('[early-vision] failed to allocate canvas context.');
      return;
    }
    const imageData = new ImageData(new Uint8ClampedArray(buffer), width, height);
    ctx.putImageData(imageData, 0, 0);
    const finalizeDownload = (blob: Blob | null) => {
      if (!blob) {
        console.warn('[early-vision] failed to encode overlay image.');
        return;
      }
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `early-vision-overlay-${stamp}.png`;
      anchor.click();
      URL.revokeObjectURL(url);
    };
    if (canvas.toBlob) {
      canvas.toBlob(finalizeDownload, 'image/png');
    } else {
      const url = canvas.toDataURL('image/png');
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `early-vision-overlay-${stamp}.png`;
      anchor.click();
    }
  }, [
    earlyVisionDogEnabled,
    earlyVisionMotionEnabled,
    earlyVisionOrientationEnabled,
    width,
    height,
  ]);

  const handleExportEarlyVisionMetrics = useCallback(() => {
    const latest = metricsRef.current[metricsRef.current.length - 1];
    if (!latest) {
      console.warn('[early-vision] metrics history is empty.');
      return;
    }
    const evMetrics = latest.metrics.texture.earlyVision;
    const payload = {
      timestamp: new Date().toISOString(),
      backend: latest.backend,
      frameStamp: latest.ts,
      analyzer: {
        dogMean: evMetrics.dogMean,
        dogStd: evMetrics.dogStd,
        orientationMean: evMetrics.orientationMean,
        orientationStd: evMetrics.orientationStd,
        divisiveMean: evMetrics.divisiveMean,
        divisiveStd: evMetrics.divisiveStd,
        sampleCount: evMetrics.sampleCount,
      },
      motionEnergy: latest.metrics.motionEnergy,
      configuration: {
        dogEnabled: earlyVisionDogEnabled,
        orientationEnabled: earlyVisionOrientationEnabled,
        motionEnabled: earlyVisionMotionEnabled,
        opacity: earlyVisionOpacity,
        dogSigma: earlyVisionDoGSigma,
        dogRatio: earlyVisionDoGRatio,
        dogGain: earlyVisionDoGGain,
        downsample: Math.max(1, earlyVisionDownsample),
        orientationGain: earlyVisionOrientationGain,
        orientationSharpness: earlyVisionOrientationSharpness,
        orientationCount: earlyVisionOrientationCount,
        motionGain: earlyVisionMotionGain,
        frameModulo: Math.max(1, earlyVisionFrameModulo),
        viewMode: earlyVisionViewMode,
      },
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `early-vision-metrics-${stamp}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [
    earlyVisionDogEnabled,
    earlyVisionDoGGain,
    earlyVisionDoGRatio,
    earlyVisionDoGSigma,
    earlyVisionDownsample,
    earlyVisionFrameModulo,
    earlyVisionMotionEnabled,
    earlyVisionMotionGain,
    earlyVisionOpacity,
    earlyVisionOrientationCount,
    earlyVisionOrientationEnabled,
    earlyVisionOrientationGain,
    earlyVisionOrientationSharpness,
    earlyVisionViewMode,
  ]);
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

      updateTimelineForTime(0);
      const initialDmt = getTimelineNumber('dmt', dmt);

      const localVolumeStub = volumeEnabled
        ? createVolumeStubState(width, height, getTimelineSeed('volumeNoise', 0))
        : null;
      let localVolumeField =
        volumeEnabled && localVolumeStub ? snapshotVolumeStub(localVolumeStub) : null;

      let localKurState: KuramotoState | null = null;
      let localDerived: ReturnType<typeof createDerivedViews> | null = null;
      const params = getKurParams();
      if (kurEnabled) {
        localKurState = createKuramotoState(width, height, undefined, {
          componentCount: polarizationEnabled ? 2 : 1,
        });
        const derivedBuffer = new ArrayBuffer(derivedBufferSize(width, height));
        localDerived = createDerivedViews(derivedBuffer, width, height);
        initKuramotoState(localKurState, qInit, localDerived);
        deriveKuramotoFieldsCore(localKurState, localDerived, {
          kernel: kernelSpec,
          controls: { dmt: initialDmt },
          schedule: polarizationSchedule,
        });
      }

      const metadataEntries: TarEntry[] = [];
      const frameName = (index: number) => `frames/frame_${String(index).padStart(5, '0')}.png`;

      for (let i = 0; i < totalFrames; i += 1) {
        if (frameExportAbortRef.current?.canceled) {
          throw new Error('Frame export cancelled');
        }
        const tSeconds = i * dt;
        updateTimelineForTime(tSeconds);
        const frameDmt = getTimelineNumber('dmt', dmt);

        if (volumeEnabled && localVolumeStub) {
          stepVolumeStub(localVolumeStub, dt);
          localVolumeField = snapshotVolumeStub(localVolumeStub);
        }

        if (kurEnabled && localKurState && localDerived) {
          const frameSeed = getTimelineSeed('kuramotoNoise', tSeconds);
          const localRand = createNormalGenerator(frameSeed);
          stepKuramotoState(localKurState, params, dt, localRand, (i + 1) * dt, {
            kernel: kernelSpec,
            controls: { dmt: frameDmt },
            schedule: polarizationSchedule,
          });
          deriveKuramotoFieldsCore(localKurState, localDerived, {
            kernel: kernelSpec,
            controls: { dmt: frameDmt },
            schedule: polarizationSchedule,
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

      const activeTimeline = timelinePlayerRef.current;
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
        timeline: activeTimeline
          ? {
              hash: activeTimeline.hash,
              fps: activeTimeline.fps,
              durationFrames: activeTimeline.durationFrames,
            }
          : null,
      };
      metadataEntries.push({
        name: 'metadata.json',
        content: textEncoder.encode(JSON.stringify(metadata, null, 2)),
      });
      if (activeTimeline) {
        metadataEntries.push({
          name: 'timeline.json',
          content: textEncoder.encode(activeTimeline.json),
        });
        metadataEntries.push({
          name: 'timeline.hash',
          content: textEncoder.encode(`${activeTimeline.hash}\n`),
        });
      }
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
    updateTimelineForTime,
    getTimelineNumber,
    getTimelineSeed,
    polarizationEnabled,
    polarizationSchedule,
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
      applyPresetRef.current(sharedPreset);
      params.delete('share');
      const newSearch = params.toString();
      const newUrl = `${window.location.pathname}${newSearch ? `?${newSearch}` : ''}${window.location.hash}`;
      window.history.replaceState({}, '', newUrl);
    } catch (error) {
      console.error('[share] failed to apply shared preset', error);
    }
  }, [buildCurrentPreset]);

  useEffect(() => {
    if (skipNextPresetApplyRef.current) {
      skipNextPresetApplyRef.current = false;
      return;
    }
    applyPresetRef.current(PRESETS[presetIndex]);
  }, [presetIndex]);

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
    if (!su7Params.enabled) {
      setGuardrailConsole((prev) => ({
        ...prev,
        unitaryError: 0,
        determinantDrift: 0,
        energyEma: 0,
        lastEnergy: null,
      }));
    }
  }, [su7Params.enabled, setGuardrailConsole]);

  useEffect(() => {
    pendingStaticUploadRef.current = true;
    if (renderBackend === 'gpu') {
      const state = ensureGpuRenderer();
      if (!state && pendingStaticUploadRef.current) {
        refreshGpuStaticTextures();
      }
    } else {
      refreshGpuStaticTextures();
    }
  }, [width, height, renderBackend, ensureGpuRenderer, refreshGpuStaticTextures]);

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
              onChange={handleRimEnabledChange}
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
            <div style={{ marginTop: '0.75rem' }}>
              <strong style={{ color: '#cbd5f5', fontSize: '0.9rem' }}>
                Early vision analyzer
              </strong>
            </div>
            <ToggleControl
              label="Retina edge map"
              value={earlyVisionDogEnabled}
              onChange={setEarlyVisionDogEnabled}
            />
            <ToggleControl
              label="Orientation map"
              value={earlyVisionOrientationEnabled}
              onChange={setEarlyVisionOrientationEnabled}
            />
            <ToggleControl
              label="Motion highlight"
              value={earlyVisionMotionEnabled}
              onChange={setEarlyVisionMotionEnabled}
            />
            <SliderControl
              label="Overlay opacity"
              min={0}
              max={1}
              step={0.05}
              value={earlyVisionOpacity}
              onChange={setEarlyVisionOpacity}
              format={(value) => value.toFixed(2)}
            />
            <SliderControl
              label="DoG sigma"
              min={0.3}
              max={4}
              step={0.1}
              value={earlyVisionDoGSigma}
              onChange={setEarlyVisionDoGSigma}
              format={(value) => value.toFixed(2)}
            />
            <SliderControl
              label="DoG ratio"
              min={1.1}
              max={3}
              step={0.05}
              value={earlyVisionDoGRatio}
              onChange={setEarlyVisionDoGRatio}
              format={(value) => value.toFixed(2)}
            />
            <SliderControl
              label="DoG gain"
              min={0.5}
              max={4}
              step={0.1}
              value={earlyVisionDoGGain}
              onChange={setEarlyVisionDoGGain}
              format={(value) => value.toFixed(2)}
            />
            <SliderControl
              label="DoG downsample"
              min={1}
              max={4}
              step={1}
              value={earlyVisionDownsample}
              onChange={(value) => setEarlyVisionDownsample(Math.max(1, Math.round(value)))}
              format={(value) => value.toFixed(0)}
            />
            <SliderControl
              label="Orientation count"
              min={1}
              max={8}
              step={1}
              value={earlyVisionOrientationCount}
              onChange={(value) => setEarlyVisionOrientationCount(Math.max(1, Math.round(value)))}
              format={(value) => value.toFixed(0)}
            />
            <SliderControl
              label="Orientation gain"
              min={0}
              max={3}
              step={0.05}
              value={earlyVisionOrientationGain}
              onChange={setEarlyVisionOrientationGain}
              format={(value) => value.toFixed(2)}
            />
            <SliderControl
              label="Orientation sharpness"
              min={0.5}
              max={4}
              step={0.05}
              value={earlyVisionOrientationSharpness}
              onChange={setEarlyVisionOrientationSharpness}
              format={(value) => value.toFixed(2)}
            />
            <SliderControl
              label="Motion gain"
              min={0}
              max={12}
              step={0.25}
              value={earlyVisionMotionGain}
              onChange={setEarlyVisionMotionGain}
              format={(value) => value.toFixed(2)}
            />
            <SliderControl
              label="Analysis frame skip"
              min={1}
              max={8}
              step={1}
              value={earlyVisionFrameModulo}
              onChange={(value) => setEarlyVisionFrameModulo(Math.max(1, Math.round(value)))}
              format={(value) => `x${value.toFixed(0)}`}
            />
            <SelectControl
              label="Overlay view"
              value={earlyVisionViewMode}
              onChange={(value) => setEarlyVisionViewMode(value as 'blend' | 'overlay')}
              options={[
                { value: 'blend', label: 'Blend with base' },
                { value: 'overlay', label: 'Overlay only' },
              ]}
            />
            <div className="control">
              <button
                onClick={handleExportEarlyVisionOverlay}
                style={{
                  padding: '0.5rem 0.75rem',
                  borderRadius: '0.6rem',
                  border: '1px solid rgba(185, 248, 255, 0.35)',
                  background: 'rgba(14, 165, 233, 0.2)',
                  color: '#e0f2fe',
                  cursor: 'pointer',
                }}
              >
                Download overlay PNG
              </button>
            </div>
            <div className="control">
              <button
                onClick={handleExportEarlyVisionMetrics}
                style={{
                  padding: '0.5rem 0.75rem',
                  borderRadius: '0.6rem',
                  border: '1px solid rgba(190, 242, 100, 0.35)',
                  background: 'rgba(132, 204, 22, 0.18)',
                  color: '#ecfccb',
                  cursor: 'pointer',
                }}
              >
                Export analyzer metrics
              </button>
            </div>
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
            <div
              className="control"
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '0.6rem',
                background: 'rgba(15,23,42,0.45)',
                borderRadius: '0.85rem',
                padding: '0.8rem',
                marginTop: '0.5rem',
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
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#e2e8f0' }}>
                  SU7 Debug Console
                </span>
                <span style={{ fontSize: '0.65rem', color: '#94a3b8' }}>
                  {su7Params.enabled ? 'Active' : 'SU7 disabled'}
                </span>
              </div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                  gap: '0.35rem 0.75rem',
                  fontSize: '0.72rem',
                  color: '#cbd5f5',
                }}
              >
                <span>Unitary ε {guardrailConsole.unitaryError.toExponential(2)}</span>
                <span>Det drift {guardrailConsole.determinantDrift.toExponential(2)}</span>
                <span>Frame {guardrailConsole.frameTimeMs.toFixed(2)} ms</span>
                <span>Energy EMA {guardrailConsole.energyEma.toFixed(3)}</span>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                <button
                  onClick={handleGuardrailReorthon}
                  style={{
                    padding: '0.45rem 0.7rem',
                    borderRadius: '0.6rem',
                    border: '1px solid rgba(148,163,184,0.35)',
                    background: 'rgba(59,130,246,0.18)',
                    color: '#e0f2fe',
                    cursor: 'pointer',
                  }}
                >
                  Re-orthonormalize
                </button>
                <button
                  onClick={handleGuardrailAutoGain}
                  style={{
                    padding: '0.45rem 0.7rem',
                    borderRadius: '0.6rem',
                    border: '1px solid rgba(148,163,184,0.35)',
                    background: 'rgba(16,185,129,0.18)',
                    color: '#bbf7d0',
                    cursor: 'pointer',
                  }}
                >
                  Auto-gain
                </button>
              </div>
              {guardrailConsole.auditLog.length > 0 ? (
                <ul
                  style={{
                    listStyle: 'none',
                    margin: 0,
                    padding: 0,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.35rem',
                    fontSize: '0.7rem',
                  }}
                >
                  {guardrailConsole.auditLog.slice(-6).map((entry) => (
                    <li
                      key={entry.id}
                      style={{
                        color: entry.severity === 'warn' ? '#f97316' : '#cbd5f5',
                        display: 'flex',
                        justifyContent: 'space-between',
                        gap: '0.5rem',
                      }}
                    >
                      <span>{entry.message}</span>
                      <span style={{ opacity: 0.7 }}>
                        {new Date(entry.timestamp).toLocaleTimeString([], {
                          hour: '2-digit',
                          minute: '2-digit',
                          second: '2-digit',
                        })}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <span style={{ fontSize: '0.7rem', color: '#64748b' }}>
                  No guardrail events yet.
                </span>
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
            {su7ProjectorId === 'hopflens' ? (
              <HopfLensControls
                lenses={hopfLenses}
                metrics={telemetrySnapshot?.hopf?.lenses}
                onAxisChange={handleHopfAxisChange}
                onBaseMixChange={handleHopfBaseMixChange}
                onFiberMixChange={handleHopfFiberMixChange}
                onControlTargetChange={handleHopfControlTargetChange}
              />
            ) : null}
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
              <label htmlFor="su7-seed-input">Seed</label>
              <input
                id="su7-seed-input"
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
            <div
              className="control"
              style={{
                border: '1px solid rgba(148,163,184,0.2)',
                borderRadius: '0.85rem',
                padding: '0.75rem',
                background: 'rgba(15,23,42,0.35)',
              }}
            >
              <div
                style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}
              >
                <h3 style={{ margin: 0, fontSize: '1rem', color: '#e2e8f0' }}>Glyph controls</h3>
                {macroLearnMode ? (
                  <span style={{ color: '#facc15', fontSize: '0.75rem' }}>Learn mode</span>
                ) : null}
              </div>
              <Su7GlyphControl
                gateList={su7DisplayGateList}
                macroBinding={macroBinding}
                macroLearnMode={macroLearnMode}
                onEdgeGesture={handleEdgeGesture}
              />
              <div
                style={{
                  marginTop: '0.9rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.7rem',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    gap: '0.5rem',
                    flexWrap: 'wrap',
                  }}
                >
                  <button
                    onClick={() => setMacroLearnMode((prev) => !prev)}
                    aria-pressed={macroLearnMode}
                    style={{
                      padding: '0.4rem 0.65rem',
                      borderRadius: '0.55rem',
                      border: '1px solid rgba(148,163,184,0.35)',
                      background: macroLearnMode
                        ? 'rgba(250,204,21,0.18)'
                        : 'rgba(59,130,246,0.18)',
                      color: macroLearnMode ? '#facc15' : '#bfdbfe',
                      cursor: 'pointer',
                    }}
                  >
                    {macroLearnMode ? 'Cancel macro learn' : 'Macro learn mode'}
                  </button>
                  <button
                    onClick={clearSu7GateAppends}
                    disabled={su7GateAppends.length === 0}
                    style={{
                      padding: '0.4rem 0.65rem',
                      borderRadius: '0.55rem',
                      border: '1px solid rgba(148,163,184,0.35)',
                      background:
                        su7GateAppends.length === 0
                          ? 'rgba(15,23,42,0.25)'
                          : 'rgba(239,68,68,0.18)',
                      color: su7GateAppends.length === 0 ? '#64748b' : '#fecaca',
                      cursor: su7GateAppends.length === 0 ? 'not-allowed' : 'pointer',
                    }}
                  >
                    Clear appended gates
                  </button>
                </div>
                <SliderControl
                  label="Macro knob"
                  value={macroKnobValue}
                  min={-1.5}
                  max={1.5}
                  step={0.05}
                  onChange={handleMacroKnobChange}
                  format={(v) => v.toFixed(2)}
                  disabled={!macroBinding}
                />
                <small style={{ color: '#94a3b8' }}>
                  {macroBinding
                    ? `Knob bound to edge ${macroBinding.axis + 1} (${macroBinding.gateLabel}).`
                    : 'Enable learn mode then drag an edge to bind the knob.'}
                </small>
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.45rem',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'baseline',
                      justifyContent: 'space-between',
                      fontWeight: 600,
                      color: '#e2e8f0',
                      fontSize: '0.85rem',
                    }}
                  >
                    <span>Gate appends</span>
                    {su7SquashedGateCount > 0 ? (
                      <span style={{ color: '#fbbf24', fontSize: '0.7rem', fontWeight: 500 }}>
                        {su7SquashedGateCount} squashed
                      </span>
                    ) : null}
                  </div>
                  {su7SquashedGateCount > 0 ? (
                    <small style={{ color: '#fbbf24' }}>
                      Oldest gates rebased into the preset for this session.
                    </small>
                  ) : null}
                  {su7GateAppends.length === 0 ? (
                    <small style={{ color: '#94a3b8' }}>No appended gates.</small>
                  ) : (
                    su7GateAppends.map((gate, index) => {
                      if (gate.kind === 'phase') {
                        return (
                          <div
                            key={`gate-${gate.label ?? index}`}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              background: 'rgba(15,23,42,0.45)',
                              borderRadius: '0.6rem',
                              padding: '0.4rem 0.6rem',
                              fontSize: '0.8rem',
                            }}
                          >
                            <span>Phase gate ({gate.label ?? `#${index + 1}`})</span>
                            <button
                              onClick={() => removeSu7Gate(gate.label ?? null, index)}
                              style={{
                                background: 'transparent',
                                border: 'none',
                                color: '#fca5a5',
                                cursor: 'pointer',
                                fontSize: '0.75rem',
                              }}
                              aria-label={`Remove phase gate ${gate.label ?? index + 1}`}
                            >
                              Remove
                            </button>
                          </div>
                        );
                      }
                      const thetaDeg = radiansToDegrees(gate.theta);
                      const phiDeg = radiansToDegrees(gate.phase);
                      const label = gate.label ?? `gate-${index + 1}`;
                      return (
                        <div
                          key={`gate-${label}`}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            background:
                              macroBinding?.gateLabel === gate.label
                                ? 'rgba(250,204,21,0.12)'
                                : 'rgba(15,23,42,0.45)',
                            borderRadius: '0.6rem',
                            padding: '0.4rem 0.6rem',
                            fontSize: '0.8rem',
                          }}
                        >
                          <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <span style={{ color: '#e2e8f0', fontWeight: 600 }}>
                              Edge {gate.axis + 1}
                            </span>
                            <span style={{ color: '#94a3b8' }}>
                              Δθ {thetaDeg.toFixed(2)}° · Δφ {phiDeg.toFixed(2)}°
                            </span>
                          </div>
                          <button
                            onClick={() => removeSu7Gate(gate.label ?? null, index)}
                            style={{
                              background: 'transparent',
                              border: 'none',
                              color: '#fca5a5',
                              cursor: 'pointer',
                              fontSize: '0.75rem',
                            }}
                            aria-label={`Remove gate ${label}`}
                          >
                            Remove
                          </button>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
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
              onChange={handleEdgeThresholdChange}
            />
            <SliderControl
              label="Kernel Blend"
              value={blend}
              min={0}
              max={1}
              step={0.01}
              onChange={handleBlendChange}
            />
            <SliderControl
              label="Dispersion β₂"
              value={beta2}
              min={0}
              max={3}
              step={0.01}
              onChange={handleBeta2Change}
            />
            <SliderControl
              label="Rim Thickness σ"
              value={sigma}
              min={0.3}
              max={6}
              step={0.05}
              onChange={handleSigmaChange}
            />
            <SliderControl
              label="Phase Jitter"
              value={jitter}
              min={0}
              max={2}
              step={0.02}
              onChange={handleJitterChange}
            />
            <SliderControl
              label="Contrast"
              value={contrast}
              min={0.25}
              max={3}
              step={0.05}
              onChange={handleContrastChange}
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
              onChange={handleMicrosaccadeChange}
            />
            <ToggleControl label="Zero-mean pin" value={phasePin} onChange={handlePhasePinChange} />
            <ToggleControl label="Alive microbreath" value={alive} onChange={handleAliveChange} />
            <SliderControl
              label="Rim Alpha"
              value={rimAlpha}
              min={0}
              max={1}
              step={0.05}
              onChange={handleRimAlphaChange}
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
              onChange={handleDmtChange}
            />
            <SliderControl
              label="Arousal"
              value={arousal}
              min={0}
              max={1}
              step={0.01}
              onChange={handleArousalChange}
            />
          </section>

          <section className="panel">
            <h2>Timeline</h2>
            <TimelineEditorPanel
              active={timelineActive}
              playing={timelinePlaying}
              loop={timelineLoop}
              fps={timelineFps}
              durationSeconds={timelineDurationSeconds}
              durationFrames={durationFrames}
              currentTime={timelineCurrentTime}
              lanes={timelineLanes}
              parameterConfigs={timelineParameterConfigs}
              availableParameters={availableTimelineParameters}
              evaluation={timelineEvaluationState}
              autoKeyframe={timelineAutoKeyframe}
              onTogglePlay={handleTimelineTogglePlay}
              onStop={handleTimelineStop}
              onToggleLoop={handleTimelineLoopToggle}
              onTimeChange={handleTimelineScrub}
              onFpsChange={handleTimelineFpsChange}
              onDurationChange={handleTimelineDurationChange}
              onAddLane={handleAddTimelineLane}
              onRemoveLane={handleRemoveTimelineLane}
              onClear={handleTimelineClear}
              onAddKeyframe={handleAddTimelineKeyframe}
              onKeyframeTimeChange={handleTimelineKeyframeTimeChange}
              onKeyframeValueChange={handleTimelineKeyframeValueChange}
              onKeyframeRemove={handleTimelineRemoveKeyframe}
              onInterpolationChange={handleTimelineInterpolationChange}
              onPrevKeyframe={handleTimelineStepPrev}
              onNextKeyframe={handleTimelineStepNext}
              onAutoKeyframeToggle={handleTimelineAutoKeyframeToggle}
            />
          </section>

          <section className="panel">
            <h2>Macro Recorder</h2>
            <div className="macro-recorder">
              <div className="macro-recorder__controls">
                <button
                  type="button"
                  className="macro-button"
                  onClick={
                    macroRecording ? () => stopMacroRecording() : () => startMacroRecording()
                  }
                >
                  {macroRecording ? 'Stop & save macro' : 'Record macro'}
                </button>
                {macroRecording ? (
                  <button type="button" className="macro-button" onClick={cancelMacroRecording}>
                    Cancel
                  </button>
                ) : macroPlaybackId ? (
                  <button type="button" className="macro-button" onClick={cancelMacroPlayback}>
                    Stop playback
                  </button>
                ) : null}
                <span className="macro-recorder__status">
                  {macroRecording
                    ? `Recording… ${macroEvents.length} event${macroEvents.length === 1 ? '' : 's'}`
                    : macroLibrary.length === 0
                      ? 'No macros recorded yet'
                      : `${macroLibrary.length} saved ${macroLibrary.length === 1 ? 'macro' : 'macros'}`}
                </span>
              </div>
              {macroLibrary.length === 0 ? (
                <p className="macro-recorder__empty">
                  Start recording to capture parameter changes, preset loads, and other actions.
                </p>
              ) : (
                <ul className="macro-recorder__list">
                  {macroLibrary.map((macro) => {
                    const createdLabel = new Date(macro.createdAt).toLocaleString();
                    const isActive = macroPlaybackId === macro.id;
                    return (
                      <li
                        key={macro.id}
                        className={isActive ? 'macro-item macro-item--active' : 'macro-item'}
                      >
                        <div className="macro-item__header">
                          <input
                            value={macro.label}
                            onChange={(event) => renameMacro(macro.id, event.target.value)}
                            className="macro-item__title"
                          />
                          <span className="macro-item__meta">
                            {macro.events.length} event{macro.events.length === 1 ? '' : 's'} •{' '}
                            {createdLabel}
                          </span>
                        </div>
                        <div className="macro-item__actions">
                          <button
                            type="button"
                            className="macro-button"
                            onClick={() => (isActive ? cancelMacroPlayback() : playMacro(macro.id))}
                          >
                            {isActive ? 'Stop' : 'Play'}
                          </button>
                          <button
                            type="button"
                            className="macro-button macro-button--danger"
                            onClick={() => deleteMacro(macro.id)}
                          >
                            Delete
                          </button>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
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
              onChange={handleSurfaceBlendChange}
            />
            <SliderControl
              label="Warp amplitude"
              value={warpAmp}
              min={0}
              max={6}
              step={0.1}
              onChange={handleWarpAmpChange}
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
            <h3 style={{ marginTop: '0.75rem' }}>Polarization</h3>
            <ToggleControl
              label="Enable polarization"
              value={polarizationEnabled}
              onChange={(value) => {
                markKurCustom();
                setPolarizationEnabled(value);
              }}
            />
            {polarizationEnabled ? (
              <>
                <ToggleControl
                  label="Wave plate"
                  value={wavePlateEnabled}
                  onChange={(value) => {
                    markKurCustom();
                    setWavePlateEnabled(value);
                  }}
                />
                <SliderControl
                  label="Phase delay"
                  value={wavePlatePhaseDeg}
                  min={-360}
                  max={360}
                  step={1}
                  onChange={(value) => {
                    markKurCustom();
                    setWavePlatePhaseDeg(value);
                  }}
                  format={(v) => `${v.toFixed(0)}°`}
                  disabled={!wavePlateEnabled}
                />
                <SliderControl
                  label="Wave plate orientation"
                  value={wavePlateOrientationDeg}
                  min={-90}
                  max={90}
                  step={1}
                  onChange={(value) => {
                    markKurCustom();
                    setWavePlateOrientationDeg(value);
                  }}
                  format={(v) => `${v.toFixed(0)}°`}
                  disabled={!wavePlateEnabled}
                />
                <ToggleControl
                  label="SU(7) gate coupling"
                  value={su7PolarizationEnabled}
                  onChange={(value) => {
                    markKurCustom();
                    setSu7PolarizationEnabled(value);
                  }}
                />
                <SliderControl
                  label="SU(7) blend"
                  value={su7PolarizationBlend}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={(value) => {
                    markKurCustom();
                    setSu7PolarizationBlend(value);
                  }}
                  format={(v) => `${(v * 100).toFixed(0)}%`}
                  disabled={!su7PolarizationEnabled}
                />
                <SliderControl
                  label="SU(7) column"
                  value={su7PolarizationColumn}
                  min={0}
                  max={6}
                  step={1}
                  onChange={(value) => {
                    markKurCustom();
                    setSu7PolarizationColumn(Math.round(value));
                  }}
                  format={(v) => v.toFixed(0)}
                  disabled={!su7PolarizationEnabled}
                />
                <SliderControl
                  label="SU(7) gain"
                  value={su7PolarizationGain}
                  min={0}
                  max={2.5}
                  step={0.05}
                  onChange={(value) => {
                    markKurCustom();
                    setSu7PolarizationGain(value);
                  }}
                  format={(v) => v.toFixed(2)}
                  disabled={!su7PolarizationEnabled}
                />
                <ToggleControl
                  label="Polarizer"
                  value={polarizerEnabled}
                  onChange={(value) => {
                    markKurCustom();
                    setPolarizerEnabled(value);
                  }}
                />
                <SliderControl
                  label="Polarizer orientation"
                  value={polarizerOrientationDeg}
                  min={-90}
                  max={90}
                  step={1}
                  onChange={(value) => {
                    markKurCustom();
                    setPolarizerOrientationDeg(value);
                  }}
                  format={(v) => `${v.toFixed(0)}°`}
                  disabled={!polarizerEnabled}
                />
                <SliderControl
                  label="Extinction ratio"
                  value={polarizerExtinction}
                  min={0}
                  max={1}
                  step={0.02}
                  onChange={(value) => {
                    markKurCustom();
                    setPolarizerExtinction(value);
                  }}
                  format={(v) => v.toFixed(2)}
                  disabled={!polarizerEnabled}
                />
              </>
            ) : null}
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
            <h2>Quark Sources</h2>
            <p style={{ color: '#94a3b8', marginTop: 0 }}>
              Click the canvas to drop alternating quark/antiquark sources. Shift-click removes, Alt
              (or ⌥) places a negative source.
            </p>
            <div
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}
            >
              <button
                type="button"
                onClick={clearFluxSources}
                disabled={fluxSources.length === 0}
                style={{
                  padding: '0.4rem 0.7rem',
                  borderRadius: '0.55rem',
                  border: '1px solid rgba(148,163,184,0.35)',
                  background:
                    fluxSources.length === 0 ? 'rgba(15,23,42,0.3)' : 'rgba(59,130,246,0.2)',
                  color: fluxSources.length === 0 ? '#64748b' : '#bfdbfe',
                  cursor: fluxSources.length === 0 ? 'not-allowed' : 'pointer',
                }}
              >
                Clear sources
              </button>
              <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>
                {fluxSources.length} active
              </span>
            </div>
            {fluxSources.length > 0 ? (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.35rem',
                  marginTop: '0.6rem',
                }}
              >
                {fluxSources.map((source, idx) => (
                  <div
                    key={`flux-source-row-${idx}`}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '0.5rem',
                      padding: '0.35rem 0.5rem',
                      borderRadius: '0.55rem',
                      background: 'rgba(15,23,42,0.4)',
                      border: '1px solid rgba(71,85,105,0.4)',
                    }}
                  >
                    <span style={{ color: '#e2e8f0', fontSize: '0.85rem' }}>
                      {idx + 1}.{source.charge >= 0 ? ' +' : ' -'} ({Math.round(source.x)},{' '}
                      {Math.round(source.y)})
                    </span>
                    <div style={{ display: 'flex', gap: '0.35rem' }}>
                      <button
                        type="button"
                        onClick={() => flipFluxSourceCharge(idx)}
                        style={{
                          padding: '0.25rem 0.45rem',
                          borderRadius: '0.45rem',
                          border: '1px solid rgba(96,165,250,0.55)',
                          background: 'rgba(59,130,246,0.15)',
                          color: '#bfdbfe',
                          cursor: 'pointer',
                          fontSize: '0.75rem',
                        }}
                      >
                        Flip
                      </button>
                      <button
                        type="button"
                        onClick={() => removeFluxSource(idx)}
                        style={{
                          padding: '0.25rem 0.45rem',
                          borderRadius: '0.45rem',
                          border: '1px solid rgba(248,113,113,0.5)',
                          background: 'rgba(248,113,113,0.18)',
                          color: '#fecaca',
                          cursor: 'pointer',
                          fontSize: '0.75rem',
                        }}
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: '#64748b', marginTop: '0.6rem' }}>
                Add at least two sources to form a flux tube overlay.
              </p>
            )}
          </section>

          <section className="panel">
            <h2>QCD Anneal</h2>
            <SliderControl
              label="Heatbath β"
              value={qcdBeta}
              min={3}
              max={7}
              step={0.05}
              onChange={setQcdBeta}
              format={(v) => v.toFixed(2)}
            />
            <SliderControl
              label="Steps per second"
              value={qcdStepsPerSecond}
              min={0.5}
              max={20}
              step={0.5}
              onChange={setQcdStepsPerSecond}
              format={(v) => v.toFixed(1)}
            />
            <SliderControl
              label="Smearing α"
              value={qcdSmearingAlpha}
              min={0}
              max={1}
              step={0.02}
              onChange={setQcdSmearingAlpha}
              format={(v) => v.toFixed(2)}
            />
            <SliderControl
              label="Smear iterations"
              value={qcdSmearingIterations}
              min={0}
              max={8}
              step={1}
              onChange={setQcdSmearingIterations}
              format={(v) => v.toFixed(0)}
            />
            <SliderControl
              label="Lattice depth"
              value={qcdDepthInt}
              min={1}
              max={16}
              step={1}
              onChange={(value) => setQcdDepth(Math.max(1, Math.round(value)))}
              format={(v) => v.toFixed(0)}
            />
            <SliderControl
              label="Temporal extent"
              value={qcdTemporalExtentInt}
              min={1}
              max={24}
              step={1}
              onChange={(value) => setQcdTemporalExtent(Math.max(1, Math.round(value)))}
              format={(v) => v.toFixed(0)}
            />
            <SliderControl
              label="GPU batch planes"
              value={qcdBatchLayersInt}
              min={1}
              max={Math.max(1, qcdDepthInt * qcdTemporalExtentInt)}
              step={1}
              onChange={(value) => setQcdBatchLayers(Math.max(1, Math.round(value)))}
              format={(v) => v.toFixed(0)}
            />
            <small style={{ color: '#64748b', fontSize: '0.72rem' }}>
              Plane budget: {qcdDepthInt} × {qcdTemporalExtentInt} ={' '}
              {Math.max(1, qcdDepthInt * qcdTemporalExtentInt)}
            </small>
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '0.35rem',
                marginTop: '0.5rem',
              }}
            >
              <label style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Seed</label>
              <input
                type="number"
                value={qcdBaseSeed}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  if (Number.isFinite(value)) {
                    setQcdBaseSeed(Math.max(0, Math.trunc(value)));
                  }
                }}
                style={{
                  background: 'rgba(15,23,42,0.7)',
                  border: '1px solid rgba(148,163,184,0.35)',
                  borderRadius: '0.5rem',
                  color: '#e2e8f0',
                  padding: '0.35rem 0.55rem',
                  width: '100%',
                }}
              />
            </div>
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '0.35rem',
                marginTop: '0.6rem',
              }}
            >
              <label style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                Temperature schedule (β list)
              </label>
              <textarea
                value={qcdTemperatureScheduleText}
                onChange={(event) => setQcdTemperatureScheduleText(event.target.value)}
                rows={2}
                placeholder="5.20, 5.25, 5.30"
                style={{
                  background: 'rgba(15,23,42,0.7)',
                  border: '1px solid rgba(148,163,184,0.35)',
                  borderRadius: '0.5rem',
                  color: '#e2e8f0',
                  padding: '0.45rem 0.55rem',
                  width: '100%',
                  fontFamily: 'monospace',
                  fontSize: '0.75rem',
                  resize: 'vertical',
                }}
              />
              <small style={{ color: '#64748b', fontSize: '0.72rem' }}>
                Parsed {qcdTemperatureSchedule.length}{' '}
                {qcdTemperatureSchedule.length === 1 ? 'β' : 'β values'} · Axis{' '}
                {qcdTemporalExtentInt > 1 ? 't' : 'spatial'}
              </small>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  type="button"
                  onClick={handlePolyakovScan}
                  disabled={qcdTemperatureSchedule.length === 0 || qcdRunning}
                  style={{
                    padding: '0.4rem 0.65rem',
                    borderRadius: '0.5rem',
                    border: '1px solid rgba(147,197,253,0.45)',
                    background:
                      qcdTemperatureSchedule.length === 0 || qcdRunning
                        ? 'rgba(71,85,105,0.35)'
                        : 'rgba(59,130,246,0.2)',
                    color: '#bfdbfe',
                    cursor:
                      qcdTemperatureSchedule.length === 0 || qcdRunning ? 'not-allowed' : 'pointer',
                    fontWeight: 600,
                  }}
                >
                  Run temperature scan
                </button>
              </div>
            </div>
            <div
              style={{
                display: 'flex',
                gap: '0.6rem',
                alignItems: 'center',
                marginTop: '0.75rem',
              }}
            >
              <button
                type="button"
                onClick={startQcdAnneal}
                disabled={qcdRunning}
                style={{
                  padding: '0.45rem 0.7rem',
                  borderRadius: '0.55rem',
                  border: '1px solid rgba(74,222,128,0.35)',
                  background: qcdRunning ? 'rgba(21,94,32,0.25)' : 'rgba(34,197,94,0.2)',
                  color: qcdRunning ? '#4ade80' : '#bbf7d0',
                  cursor: qcdRunning ? 'not-allowed' : 'pointer',
                  fontWeight: 600,
                }}
              >
                Start anneal
              </button>
              <button
                type="button"
                onClick={stopQcdAnneal}
                disabled={!qcdRunning}
                style={{
                  padding: '0.45rem 0.7rem',
                  borderRadius: '0.55rem',
                  border: '1px solid rgba(248,113,113,0.45)',
                  background: qcdRunning ? 'rgba(248,113,113,0.22)' : 'rgba(71,85,105,0.4)',
                  color: qcdRunning ? '#fecaca' : '#cbd5f5',
                  cursor: qcdRunning ? 'pointer' : 'not-allowed',
                  fontWeight: 600,
                }}
              >
                Stop
              </button>
              <span style={{ color: '#94a3b8', fontSize: '0.78rem' }}>
                Status: {qcdRunning ? 'Annealing' : 'Idle'} · Sweeps {qcdSweeps.toLocaleString()}
              </span>
            </div>
            {qcdObservables ? (
              <div
                style={{
                  display: 'grid',
                  gap: '0.35rem',
                  marginTop: '0.75rem',
                  fontSize: '0.8rem',
                  color: '#cbd5f5',
                }}
              >
                <div>⟨P⟩ {qcdObservables.averagePlaquette.toFixed(4)}</div>
                <div>
                  σ<sub>P</sub> {qcdObservables.plaquetteEstimate.standardError.toExponential(2)}
                </div>
                {qcdObservables.creutzRatio ? (
                  <div>
                    χ
                    <sub>{`${qcdObservables.creutzRatio.extentX}×${qcdObservables.creutzRatio.extentY}`}</sub>{' '}
                    {qcdObservables.creutzRatio.value.toFixed(3)}
                  </div>
                ) : null}
                {qcdObservables.polyakovSamples && qcdObservables.polyakovSamples.length > 0 ? (
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.15rem',
                    }}
                  >
                    <span>Polyakov |P|</span>
                    {qcdObservables.polyakovSamples.map((sample, idx) => {
                      const beta = qcdPolyakovSchedule[idx];
                      return (
                        <span key={`polyakov-${sample.axis}-${idx}`}>
                          β {beta != null ? beta.toFixed(3) : `#${idx + 1}`} · |P|{' '}
                          {sample.magnitude.toFixed(3)} (axis {sample.axis})
                        </span>
                      );
                    })}
                  </div>
                ) : null}
                {qcdObservables.wilsonLoops.slice(0, 2).map((loop) => (
                  <div key={`wilson-${loop.extentX}x${loop.extentY}`}>
                    W<sub>{`${loop.extentX}×${loop.extentY}`}</sub> {loop.value.toFixed(3)}
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: '#64748b', marginTop: '0.75rem' }}>
                Run the annealer to accumulate Wilson loop statistics.
              </p>
            )}
            <div
              style={{
                marginTop: '0.6rem',
                color: '#94a3b8',
                fontSize: '0.75rem',
                wordBreak: 'break-all',
              }}
            >
              Snapshot hash: {qcdSnapshotHash ?? '–'}
            </div>
            {qcdPerfLog.length > 0 ? (
              <div
                style={{
                  marginTop: '0.6rem',
                  color: '#94a3b8',
                  fontSize: '0.72rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.2rem',
                }}
              >
                <div style={{ color: '#cbd5f5', fontWeight: 600 }}>QCD perf log</div>
                {qcdPerfLog
                  .slice(-10)
                  .reverse()
                  .map((entry, idx) => (
                    <span key={`qcd-perf-${idx}`} style={{ fontFamily: 'monospace' }}>
                      {entry}
                    </span>
                  ))}
              </div>
            ) : null}
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
            <ToggleControl
              label="Normalization pin"
              value={normPin}
              onChange={handleNormPinChange}
            />
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
              onPointerDown={handleCanvasPointerDown}
              onContextMenu={handleCanvasContextMenu}
              style={{
                width: '100%',
                height: '100%',
                display: 'block',
                cursor: 'crosshair',
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
                inset: 0,
                pointerEvents: 'none',
              }}
            >
              {qcdProbeFrame ? (
                <svg
                  width={width}
                  height={height}
                  viewBox={`0 0 ${Math.max(1, qcdProbeFrame.latticeWidth)} ${Math.max(1, qcdProbeFrame.latticeHeight)}`}
                  preserveAspectRatio="none"
                  style={{
                    position: 'absolute',
                    inset: 0,
                    width: '100%',
                    height: '100%',
                    mixBlendMode: 'screen',
                    filter: 'drop-shadow(0 0 4px rgba(15,23,42,0.45))',
                  }}
                >
                  {qcdProbeFrame.segments.map((segment) => {
                    const from = qcdProbeFrame.nodes[segment.fromIndex];
                    const to = qcdProbeFrame.nodes[segment.toIndex];
                    if (!from || !to) {
                      return null;
                    }
                    const color = `rgba(${Math.round(segment.rgb[0] * 255)}, ${Math.round(segment.rgb[1] * 255)}, ${Math.round(segment.rgb[2] * 255)}, 0.88)`;
                    const strokeWidth = Math.max(
                      0.18,
                      Math.min(
                        0.45,
                        Math.min(qcdProbeFrame.latticeWidth, qcdProbeFrame.latticeHeight) * 0.01,
                      ),
                    );
                    return (
                      <line
                        key={`probe-segment-${segment.stepIndex}`}
                        x1={from.coord.x + 0.5}
                        y1={from.coord.y + 0.5}
                        x2={to.coord.x + 0.5}
                        y2={to.coord.y + 0.5}
                        stroke={color}
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                        vectorEffect="non-scaling-stroke"
                      />
                    );
                  })}
                  {qcdProbeFrame.nodes.map((node, idx) => {
                    const fill = `rgba(${Math.round(node.rgb[0] * 255)}, ${Math.round(node.rgb[1] * 255)}, ${Math.round(node.rgb[2] * 255)}, 0.95)`;
                    const radius = Math.max(
                      0.22,
                      Math.min(
                        0.55,
                        Math.min(qcdProbeFrame.latticeWidth, qcdProbeFrame.latticeHeight) * 0.012,
                      ),
                    );
                    return (
                      <circle
                        key={`probe-node-${idx}`}
                        cx={node.coord.x + 0.5}
                        cy={node.coord.y + 0.5}
                        r={radius}
                        fill={fill}
                        stroke="rgba(15,23,42,0.65)"
                        strokeWidth={radius * 0.35}
                        vectorEffect="non-scaling-stroke"
                      />
                    );
                  })}
                </svg>
              ) : null}
              {fluxSources.map((source, idx) => {
                const denomX = Math.max(width - 1, 1);
                const denomY = Math.max(height - 1, 1);
                const leftPercent = (source.x / denomX) * 100;
                const topPercent = (source.y / denomY) * 100;
                const tint = source.charge >= 0 ? 'rgba(96,165,250,0.9)' : 'rgba(248,113,113,0.9)';
                return (
                  <div
                    key={`flux-source-${idx}`}
                    style={{
                      position: 'absolute',
                      left: `${leftPercent}%`,
                      top: `${topPercent}%`,
                      transform: 'translate(-50%, -50%)',
                      width: '0.85rem',
                      height: '0.85rem',
                      borderRadius: '9999px',
                      border: '1px solid rgba(15,23,42,0.55)',
                      background: tint,
                      boxShadow: '0 0 6px rgba(15,23,42,0.45)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#0f172a',
                      fontSize: '0.55rem',
                      fontWeight: 600,
                    }}
                  >
                    {source.charge >= 0 ? '+' : '-'}
                  </div>
                );
              })}
            </div>
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
                <TelemetryOverlayContents
                  snapshot={telemetrySnapshot}
                  seriesSelection={telemetrySeriesSelection}
                  onToggleSeries={toggleTelemetrySeries}
                />
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
                    gap: '0.65rem',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'baseline',
                    }}
                  >
                    <span style={{ color: '#5eead4', fontWeight: 600 }}>Qualia dashboard</span>
                    <span style={{ color: '#94a3b8', fontSize: '0.78rem' }}>
                      {telemetrySnapshot.performance.fps > 0
                        ? `${telemetrySnapshot.performance.fps.toFixed(1)} fps · ${telemetrySnapshot.performance.frameMs.toFixed(2)} ms`
                        : 'fps —'}
                    </span>
                  </div>
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(14rem, 1fr))',
                      gap: '0.75rem',
                    }}
                  >
                    {QUALIA_SERIES_META.map((entry) => {
                      const values = telemetrySnapshot.qualia.series[entry.key];
                      const latest = telemetrySnapshot.qualia.latest[entry.key];
                      const seriesValues = values.length > 0 ? values : [latest];
                      return (
                        <div
                          key={`panel-qualia-${entry.key}`}
                          style={{
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '0.35rem',
                            padding: '0.5rem 0.6rem',
                            borderRadius: '0.65rem',
                            background: 'rgba(14,36,52,0.55)',
                          }}
                        >
                          <div
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              fontSize: '0.8rem',
                              color: '#e2e8f0',
                              fontWeight: 600,
                            }}
                          >
                            <span>{entry.label}</span>
                            <span style={{ color: entry.color }}>{entry.format(latest)}</span>
                          </div>
                          <div style={{ height: '48px' }}>
                            <Sparkline values={seriesValues} color={entry.color} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
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
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'baseline',
                    }}
                  >
                    <span style={{ color: '#38bdf8', fontWeight: 600 }}>Live telemetry stream</span>
                    <span style={{ color: telemetryStreamStatusColor, fontSize: '0.78rem' }}>
                      {telemetryStreamStatusLabel}
                    </span>
                  </div>
                  <label
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.3rem',
                      fontSize: '0.75rem',
                      color: '#cbd5f5',
                    }}
                  >
                    <span>WebSocket endpoint</span>
                    <input
                      type="text"
                      value={telemetryStreamUrl}
                      onChange={(event) => setTelemetryStreamUrl(event.target.value)}
                      style={{
                        padding: '0.5rem 0.6rem',
                        borderRadius: '0.55rem',
                        border: '1px solid rgba(148,163,184,0.35)',
                        background: 'rgba(15,23,42,0.6)',
                        color: '#e2e8f0',
                      }}
                    />
                  </label>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <button
                      type="button"
                      onClick={() => setTelemetryStreamEnabled((prev) => !prev)}
                      style={{
                        padding: '0.45rem 0.8rem',
                        borderRadius: '0.6rem',
                        border: '1px solid rgba(56,189,248,0.35)',
                        background: telemetryStreamEnabled
                          ? 'rgba(22,78,99,0.6)'
                          : 'rgba(30,64,175,0.4)',
                        color: '#e2e8f0',
                        fontSize: '0.75rem',
                      }}
                    >
                      {telemetryStreamEnabled ? 'Stop streaming' : 'Start streaming'}
                    </button>
                    <button
                      type="button"
                      onClick={handleTelemetryReconnect}
                      disabled={!telemetryStreamEnabled}
                      style={{
                        padding: '0.45rem 0.8rem',
                        borderRadius: '0.6rem',
                        border: '1px solid rgba(148,163,184,0.35)',
                        background: telemetryStreamEnabled
                          ? 'rgba(15,23,42,0.6)'
                          : 'rgba(15,23,42,0.3)',
                        color: telemetryStreamEnabled ? '#e2e8f0' : '#94a3b8',
                        fontSize: '0.75rem',
                      }}
                    >
                      Reconnect
                    </button>
                  </div>
                  {telemetryStreamError ? (
                    <div style={{ fontSize: '0.72rem', color: '#f87171' }}>
                      {telemetryStreamError}
                    </div>
                  ) : (
                    <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
                      Frames send as JSON lines when the socket is connected.
                    </div>
                  )}
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
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'baseline',
                    }}
                  >
                    <span style={{ color: '#fcd34d', fontWeight: 600 }}>Metrics recording</span>
                    <span style={{ color: '#94a3b8', fontSize: '0.78rem' }}>
                      {telemetryRecordingActive ? `${telemetryRecordingCount} frames` : 'Idle'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <button
                      type="button"
                      onClick={toggleTelemetryRecording}
                      style={{
                        padding: '0.45rem 0.8rem',
                        borderRadius: '0.6rem',
                        border: '1px solid rgba(251,191,36,0.45)',
                        background: telemetryRecordingActive
                          ? 'rgba(146,64,14,0.6)'
                          : 'rgba(124,45,18,0.45)',
                        color: '#fef9c3',
                        fontSize: '0.75rem',
                      }}
                    >
                      {telemetryRecordingActive ? 'Stop & download' : 'Start recording'}
                    </button>
                    <button
                      type="button"
                      onClick={downloadTelemetrySnapshot}
                      style={{
                        padding: '0.45rem 0.8rem',
                        borderRadius: '0.6rem',
                        border: '1px solid rgba(148,163,184,0.35)',
                        background: 'rgba(15,23,42,0.5)',
                        color: '#e2e8f0',
                        fontSize: '0.75rem',
                      }}
                    >
                      Export snapshot
                    </button>
                  </div>
                  <div
                    style={{
                      fontSize: '0.72rem',
                      color: telemetryRecordingActive ? '#facc15' : '#94a3b8',
                    }}
                  >
                    {telemetryRecordingActive
                      ? 'Recording to memory; stop to download a JSONL log.'
                      : 'Recording outputs newline-delimited JSON with qualia metrics.'}
                  </div>
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
                  <span style={{ fontSize: '0.8rem', color: '#f87171' }}>
                    Geodesic fallbacks {telemetrySnapshot.su7.geodesicFallbacks}
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

type QualiaSeriesKey = 'indraIndex' | 'symmetry' | 'colorfulness' | 'edgeDensity';

type TelemetrySeriesSelection = Record<QualiaSeriesKey, boolean>;

type TelemetrySnapshot = {
  fields: Record<ComposerFieldId, TelemetryFieldSnapshot>;
  coupling: {
    scale: number;
    base: CouplingConfig;
    effective: CouplingConfig;
  };
  su7: Su7Telemetry;
  hopf: RainbowFrameMetrics['hopf'];
  qualia: {
    latest: QualiaMetrics;
    series: Record<QualiaSeriesKey, number[]>;
  };
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
  performance: {
    fps: number;
    frameMs: number;
  };
  frameSamples: number;
  updatedAt: number;
  su7Gpu: {
    backend: 'gpu' | 'cpu';
    medianMs: number;
    meanMs: number;
    sampleCount: number;
    baselineMs: number | null;
    drift: number | null;
    warning: boolean;
    lastProfile: Su7GpuKernelProfile | null;
    warningEvent: Su7GpuKernelWarningEvent | null;
  } | null;
};

const formatPercentValue = (value: number) => `${(clamp01(value) * 100).toFixed(1)}%`;

const QUALIA_SERIES_META: readonly {
  key: QualiaSeriesKey;
  label: string;
  color: string;
  format: (value: number) => string;
}[] = [
  { key: 'indraIndex', label: 'Indra Index', color: '#facc15', format: formatPercentValue },
  { key: 'symmetry', label: 'Symmetry', color: '#34d399', format: formatPercentValue },
  { key: 'colorfulness', label: 'Colorfulness', color: '#60a5fa', format: formatPercentValue },
  { key: 'edgeDensity', label: 'Edge Density', color: '#f97316', format: formatPercentValue },
];

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

type SparklineProps = {
  values: readonly number[];
  color: string;
  width?: number;
  height?: number;
  strokeWidth?: number;
};

function Sparkline({ values, color, width = 160, height = 56, strokeWidth = 2 }: SparklineProps) {
  if (!values.length) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '0.65rem',
          color: '#94a3b8',
          background: 'rgba(15,23,42,0.35)',
          borderRadius: '0.5rem',
        }}
      >
        No data
      </div>
    );
  }
  const min = values.reduce((acc, value) => (value < acc ? value : acc), values[0] ?? 0);
  const max = values.reduce((acc, value) => (value > acc ? value : acc), values[0] ?? 0);
  const range = Math.max(1e-6, max - min);
  const step = values.length > 1 ? width / (values.length - 1) : width;
  let path = '';
  for (let i = 0; i < values.length; i++) {
    const value = values[i]!;
    const norm = clamp01(range <= 1e-6 ? 0.5 : (value - min) / range);
    const x = i === values.length - 1 ? width : i * step;
    const y = height - norm * height;
    path += i === 0 ? `M${x},${y}` : ` L${x},${y}`;
  }
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      style={{
        width: '100%',
        height: '100%',
        display: 'block',
        background: 'rgba(15,23,42,0.35)',
        borderRadius: '0.5rem',
      }}
    >
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

function TelemetryOverlayContents({
  snapshot,
  seriesSelection,
  onToggleSeries,
}: {
  snapshot: TelemetrySnapshot;
  seriesSelection: TelemetrySeriesSelection;
  onToggleSeries: (key: QualiaSeriesKey) => void;
}) {
  const { su7, su7Histograms, su7Unitary, frameSamples, su7Gpu, qualia, performance } = snapshot;
  const selectedQualiaSeries = QUALIA_SERIES_META.filter((entry) => seriesSelection[entry.key]);
  const performanceLabel =
    performance.fps > 0
      ? `${performance.fps.toFixed(1)} fps · ${performance.frameMs.toFixed(2)} ms`
      : 'fps —';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.45rem',
          padding: '0.5rem',
          borderRadius: '0.75rem',
          background: 'rgba(13,36,52,0.65)',
          border: '1px solid rgba(45,212,191,0.18)',
        }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'baseline',
          }}
        >
          <span
            style={{
              fontWeight: 700,
              fontSize: '0.72rem',
              letterSpacing: '0.02em',
              color: '#5eead4',
            }}
          >
            Qualia metrics
          </span>
          <div
            style={{
              display: 'flex',
              alignItems: 'baseline',
              gap: '0.5rem',
              fontSize: '0.64rem',
              color: '#bae6fd',
            }}
          >
            <span style={{ color: '#94a3b8' }}>{performanceLabel}</span>
            <span>Indra {formatPercentValue(qualia.latest.indraIndex)}</span>
          </div>
        </div>
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '0.4rem',
          }}
        >
          {QUALIA_SERIES_META.map((entry) => {
            const active = seriesSelection[entry.key];
            return (
              <label
                key={`qualia-toggle-${entry.key}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                  padding: '0.25rem 0.55rem',
                  borderRadius: '9999px',
                  fontSize: '0.64rem',
                  cursor: 'pointer',
                  background: active ? 'rgba(45,212,191,0.2)' : 'rgba(15,23,42,0.45)',
                  color: active ? entry.color : '#94a3b8',
                  border: '1px solid rgba(94,234,212,0.25)',
                }}
              >
                <input
                  type="checkbox"
                  checked={active}
                  onChange={() => onToggleSeries(entry.key)}
                />
                <span>{entry.label}</span>
              </label>
            );
          })}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.55rem' }}>
          {selectedQualiaSeries.length === 0 ? (
            <div style={{ fontSize: '0.65rem', color: '#94a3b8' }}>
              Enable a metric above to visualize its trend over the last frames.
            </div>
          ) : (
            selectedQualiaSeries.map((entry) => {
              const values = qualia.series[entry.key];
              const seriesValues =
                values && values.length > 0 ? values : [qualia.latest[entry.key]];
              return (
                <div
                  key={`qualia-series-${entry.key}`}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.3rem',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      fontSize: '0.66rem',
                      color: '#e2e8f0',
                    }}
                  >
                    <span>{entry.label}</span>
                    <span>{entry.format(qualia.latest[entry.key])}</span>
                  </div>
                  <div style={{ height: '56px' }}>
                    <Sparkline values={seriesValues} color={entry.color} />
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
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
      {su7Gpu ? (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '0.25rem',
            fontSize: '0.62rem',
            color: '#cbd5f5',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'baseline',
            }}
          >
            <span>GPU transform ({su7Gpu.backend})</span>
            <span>samples {su7Gpu.sampleCount}</span>
          </div>
          <div>
            Median {su7Gpu.medianMs.toFixed(3)} ms · Mean {su7Gpu.meanMs.toFixed(3)} ms
          </div>
          {su7Gpu.baselineMs != null && <div>Baseline {su7Gpu.baselineMs.toFixed(3)} ms</div>}
          {su7Gpu.drift != null && <div>Drift {(su7Gpu.drift * 100).toFixed(1)}%</div>}
          {su7Gpu.lastProfile && (
            <div>
              Last dispatch {su7Gpu.lastProfile.timeMs.toFixed(3)} ms ·{' '}
              {su7Gpu.lastProfile.vectorCount.toLocaleString()} vectors
            </div>
          )}
          {su7Gpu.warning && (
            <div style={{ color: '#f97316', fontWeight: 600 }}>
              Warning: performance drift
              {su7Gpu.warningEvent
                ? ` ${(su7Gpu.warningEvent.drift * 100).toFixed(1)}%`
                : su7Gpu.drift != null
                  ? ` ${(su7Gpu.drift * 100).toFixed(1)}%`
                  : ''}
              {su7Gpu.warningEvent
                ? ` · baseline ${su7Gpu.warningEvent.baselineMs.toFixed(3)} ms`
                : ''}
            </div>
          )}
        </div>
      ) : null}
      <div style={{ fontSize: '0.65rem', color: '#cbd5f5' }}>
        Unitary frame {su7Unitary.latest.toExponential(2)} · μ {su7.unitaryError.toExponential(2)} ·
        max {su7Unitary.max.toExponential(2)}
      </div>
      {snapshot.hopf.lenses.length > 0 ? (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '0.25rem',
            fontSize: '0.62rem',
            color: '#a5f3fc',
          }}
        >
          <span style={{ fontWeight: 600, fontSize: '0.65rem', color: '#67e8f9' }}>
            Hopf lenses
          </span>
          {snapshot.hopf.lenses.map((lens) => (
            <div
              key={`overlay-hopf-${lens.index}`}
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '0.15rem',
                background: 'rgba(14,24,42,0.55)',
                borderRadius: '0.45rem',
                padding: '0.35rem 0.45rem',
              }}
            >
              <span style={{ color: '#e2e8f0', fontWeight: 600 }}>
                Lens {lens.index + 1} · share {lens.share.toFixed(3)}
              </span>
              <span>Base ({lens.base.map((value) => value.toFixed(2)).join(', ')})</span>
              <span>Fiber {((lens.fiber * 180) / Math.PI).toFixed(1)}°</span>
            </div>
          ))}
        </div>
      ) : null}
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

type Su7GlyphControlProps = {
  gateList: GateList | null;
  macroBinding: MacroBinding | null;
  macroLearnMode: boolean;
  onEdgeGesture: (args: { axis: number; deltaTheta: number; deltaPhi: number }) => void;
};

type DragPreview = {
  axis: number;
  deltaTheta: number;
  deltaPhi: number;
  x: number;
  y: number;
};

const EDGE_THETA_SCALE = Math.PI;
const KEY_THETA_STEP = 0.18;
const KEY_PHI_STEP = 0.14;
const NODE_RADIUS = 82;

const phaseStrokeFor = (phase: number): { color: string; dash: string } => {
  const magnitude = Math.min(1, Math.abs(phase) / Math.PI);
  if (Math.abs(phase) < 1e-3) {
    return { color: '#a3a3a3', dash: '1 4' };
  }
  if (phase >= 0) {
    return {
      color: `rgba(59,130,246,${0.45 + 0.45 * magnitude})`,
      dash: '0',
    };
  }
  return {
    color: `rgba(248,113,113,${0.45 + 0.45 * magnitude})`,
    dash: '4 2',
  };
};

const formatDegrees = (value: number): string => `${value >= 0 ? '+' : ''}${value.toFixed(1)}°`;

const Su7GlyphControl: React.FC<Su7GlyphControlProps> = ({
  gateList,
  macroBinding,
  macroLearnMode,
  onEdgeGesture,
}) => {
  const svgRef = React.useRef<SVGSVGElement | null>(null);
  const pointerStateRef = React.useRef<{
    axis: number;
    startX: number;
    startY: number;
    tangent: { x: number; y: number };
    normal: { x: number; y: number };
    length: number;
    pointerId: number;
  } | null>(null);
  const [dragPreview, setDragPreview] = React.useState<DragPreview | null>(null);

  const phases = gateList ? Array.from(gateList.gains.phaseAngles) : new Array(7).fill(0);
  const pulses = gateList ? Array.from(gateList.gains.pulseAngles) : new Array(7).fill(0);

  const nodes = React.useMemo(() => {
    return Array.from({ length: 7 }, (_, axis) => {
      const angle = (axis / 7) * TAU - Math.PI / 2;
      return {
        axis,
        angle,
        x: NODE_RADIUS * Math.cos(angle),
        y: NODE_RADIUS * Math.sin(angle),
      };
    });
  }, []);

  const getLocalPoint = (event: React.PointerEvent<Element>) => {
    const svg = svgRef.current;
    if (!svg) {
      return { x: 0, y: 0 };
    }
    const rect = svg.getBoundingClientRect();
    const nx = (event.clientX - rect.left) / rect.width;
    const ny = (event.clientY - rect.top) / rect.height;
    return {
      x: nx * 240 - 120,
      y: ny * 240 - 120,
    };
  };

  const computePreview = (local: { x: number; y: number }) => {
    const state = pointerStateRef.current;
    if (!state) return null;
    const dx = local.x - state.startX;
    const dy = local.y - state.startY;
    const projT = dx * state.tangent.x + dy * state.tangent.y;
    const projN = dx * state.normal.x + dy * state.normal.y;
    const deltaTheta = (projT / state.length) * EDGE_THETA_SCALE;
    const deltaPhi = (projN / state.length) * EDGE_THETA_SCALE;
    return {
      axis: state.axis,
      deltaTheta,
      deltaPhi,
      x: local.x,
      y: local.y,
    } satisfies DragPreview;
  };

  const handlePointerMove = (event: React.PointerEvent<Element>) => {
    if (!pointerStateRef.current) return;
    event.preventDefault();
    event.stopPropagation();
    const preview = computePreview(getLocalPoint(event));
    if (preview) {
      setDragPreview(preview);
    }
  };

  const finishPointerGesture = (event: React.PointerEvent<Element>, cancel = false) => {
    const state = pointerStateRef.current;
    if (!state) return;
    const svg = svgRef.current;
    if (svg) {
      try {
        svg.releasePointerCapture(state.pointerId);
      } catch {
        // ignore
      }
    }
    if (!cancel) {
      const preview = computePreview(getLocalPoint(event));
      if (preview) {
        onEdgeGesture({
          axis: preview.axis,
          deltaTheta: preview.deltaTheta,
          deltaPhi: preview.deltaPhi,
        });
      }
    }
    pointerStateRef.current = null;
    setDragPreview(null);
  };

  const beginPointerGesture = (axis: number, event: React.PointerEvent<SVGLineElement>) => {
    if (!svgRef.current) return;
    event.preventDefault();
    event.stopPropagation();
    const local = getLocalPoint(event);
    const current = nodes[axis];
    const next = nodes[(axis + 1) % nodes.length];
    const edgeX = next.x - current.x;
    const edgeY = next.y - current.y;
    const length = Math.hypot(edgeX, edgeY);
    if (length <= 1e-3) return;
    const tangent = { x: edgeX / length, y: edgeY / length };
    const normal = { x: -tangent.y, y: tangent.x };
    pointerStateRef.current = {
      axis,
      startX: local.x,
      startY: local.y,
      tangent,
      normal,
      length,
      pointerId: event.pointerId,
    };
    svgRef.current!.setPointerCapture(event.pointerId);
  };

  const handlePointerDown = (axis: number) => (event: React.PointerEvent<SVGLineElement>) => {
    beginPointerGesture(axis, event);
  };

  const handlePointerUp = (event: React.PointerEvent<Element>) => {
    if (pointerStateRef.current) {
      event.preventDefault();
      event.stopPropagation();
    }
    finishPointerGesture(event, false);
  };

  const handlePointerCancel = (event: React.PointerEvent<Element>) => {
    finishPointerGesture(event, true);
  };

  const handleEdgeKeyDown = (axis: number) => (event: React.KeyboardEvent<SVGLineElement>) => {
    if (event.defaultPrevented) return;
    const key = event.key;
    let handled = false;
    if (key === 'ArrowRight') {
      onEdgeGesture({ axis, deltaTheta: KEY_THETA_STEP, deltaPhi: 0 });
      handled = true;
    } else if (key === 'ArrowLeft') {
      onEdgeGesture({ axis, deltaTheta: -KEY_THETA_STEP, deltaPhi: 0 });
      handled = true;
    } else if (key === 'ArrowUp') {
      onEdgeGesture({ axis, deltaTheta: 0, deltaPhi: KEY_PHI_STEP });
      handled = true;
    } else if (key === 'ArrowDown') {
      onEdgeGesture({ axis, deltaTheta: 0, deltaPhi: -KEY_PHI_STEP });
      handled = true;
    }
    if (handled) {
      event.preventDefault();
      event.stopPropagation();
    }
  };

  const macroAxis = macroBinding?.axis ?? null;

  const edgeSummary = (axis: number) => {
    const thetaDeg = radiansToDegrees(pulses[axis]);
    const phaseDeg = radiansToDegrees(phases[axis]);
    return `Δθ ${thetaDeg.toFixed(1)}°, φ ${phaseDeg.toFixed(1)}°`;
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.75rem',
      }}
    >
      <svg
        ref={svgRef}
        viewBox="-120 -120 240 240"
        role="img"
        aria-label="SU7 phase and pulse glyph"
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerCancel}
        onPointerCancel={handlePointerCancel}
        style={{ width: '100%', maxWidth: '260px', alignSelf: 'center' }}
      >
        <circle
          cx={0}
          cy={0}
          r={110}
          fill="rgba(15,23,42,0.4)"
          stroke="rgba(148,163,184,0.35)"
          strokeWidth={2}
        />
        {nodes.map((node) => {
          const phase = phases[node.axis] ?? 0;
          const pulse = pulses[node.axis] ?? 0;
          const stroke = phaseStrokeFor(phase);
          const isMacro = macroAxis === node.axis;
          return (
            <g key={`node-${node.axis}`}>
              <circle
                cx={node.x}
                cy={node.y}
                r={14}
                fill={isMacro ? 'rgba(250,204,21,0.18)' : 'rgba(15,23,42,0.75)'}
                stroke={stroke.color}
                strokeWidth={3}
                strokeDasharray={stroke.dash}
              />
              <text x={node.x} y={node.y - 24} textAnchor="middle" fontSize="9" fill="#cbd5f5">
                {formatDegrees(radiansToDegrees(phase))}
              </text>
              <text x={node.x} y={node.y + 28} textAnchor="middle" fontSize="9" fill="#94a3b8">
                {formatDegrees(radiansToDegrees(pulse))}
              </text>
            </g>
          );
        })}
        {nodes.map((node) => {
          const axis = node.axis;
          const next = nodes[(axis + 1) % nodes.length];
          const isMacro = macroAxis === axis;
          const baseStroke = isMacro ? '#facc15' : '#94a3b8';
          const strokeWidth = isMacro ? 4 : 3;
          return (
            <line
              key={`edge-${axis}`}
              x1={node.x}
              y1={node.y}
              x2={next.x}
              y2={next.y}
              stroke={baseStroke}
              strokeWidth={strokeWidth}
              strokeDasharray={isMacro ? 'none' : '6 4'}
              opacity={macroLearnMode && !isMacro ? 0.55 : 0.8}
              cursor="grab"
              role="slider"
              aria-label={`Edge ${axis + 1} control`}
              aria-valuemin={-Math.PI}
              aria-valuemax={Math.PI}
              aria-valuenow={pulses[axis] ?? 0}
              aria-valuetext={edgeSummary(axis)}
              tabIndex={0}
              onPointerDown={handlePointerDown(axis)}
              onKeyDown={handleEdgeKeyDown(axis)}
            />
          );
        })}
        {dragPreview && (
          <g>
            <line
              x1={nodes[dragPreview.axis].x}
              y1={nodes[dragPreview.axis].y}
              x2={dragPreview.x}
              y2={dragPreview.y}
              stroke="#38bdf8"
              strokeWidth={2}
              strokeDasharray="3 3"
            />
            <text
              x={dragPreview.x}
              y={dragPreview.y - 10}
              textAnchor="middle"
              fontSize="9"
              fill="#38bdf8"
            >
              {`${formatDegrees(radiansToDegrees(dragPreview.deltaTheta))} | ${formatDegrees(radiansToDegrees(dragPreview.deltaPhi))}`}
            </text>
          </g>
        )}
      </svg>
      <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
        <div>
          Drag an edge to append a pulse gate using the captured tangential (Δθ) and normal (Δφ)
          deltas.{' '}
          {macroLearnMode
            ? 'Macro learn is active — the next gesture will bind the knob.'
            : 'Use arrow keys on an edge for keyboard adjustments.'}
        </div>
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(9rem, 1fr))',
          gap: '0.5rem',
          fontSize: '0.75rem',
          color: '#cbd5f5',
        }}
      >
        {nodes.map((node) => {
          const thetaDeg = radiansToDegrees(pulses[node.axis] ?? 0);
          const phaseDeg = radiansToDegrees(phases[node.axis] ?? 0);
          return (
            <div
              key={`summary-${node.axis}`}
              style={{
                background: 'rgba(15,23,42,0.45)',
                borderRadius: '0.5rem',
                padding: '0.4rem 0.6rem',
              }}
            >
              <div style={{ fontWeight: 600, color: '#e2e8f0' }}>Axis {node.axis + 1}</div>
              <div>Δθ {thetaDeg.toFixed(1)}°</div>
              <div>φ {phaseDeg.toFixed(1)}°</div>
            </div>
          );
        })}
      </div>
    </div>
  );
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
  const inputId = React.useId();
  return (
    <div className="control" style={disabled ? { opacity: 0.55 } : undefined}>
      <label htmlFor={inputId}>{label}</label>
      <div className="control-row">
        <input
          id={inputId}
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
  const inputId = React.useId();
  return (
    <div className="control" style={disabled ? { opacity: 0.55 } : undefined}>
      <label htmlFor={inputId}>{label}</label>
      <div className="control-row">
        <input
          id={inputId}
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
  const selectId = React.useId();
  return (
    <div className="control">
      <label htmlFor={selectId}>{label}</label>
      <select
        id={selectId}
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

type TimelineEditorPanelProps = {
  active: boolean;
  playing: boolean;
  loop: boolean;
  fps: number;
  durationSeconds: number;
  durationFrames: number;
  currentTime: number;
  lanes: TimelineEditorLane[];
  parameterConfigs: TimelineParameterConfig[];
  availableParameters: TimelineParameterConfig[];
  evaluation: TimelineFrameEvaluation | null;
  autoKeyframe: boolean;
  onTogglePlay: () => void;
  onStop: () => void;
  onToggleLoop: (value: boolean) => void;
  onTimeChange: (timeSeconds: number) => void;
  onFpsChange: (value: number) => void;
  onDurationChange: (value: number) => void;
  onAddLane: (parameterId: string) => void;
  onRemoveLane: (laneId: string) => void;
  onClear: () => void;
  onAddKeyframe: (laneId: string) => void;
  onKeyframeTimeChange: (laneId: string, index: number, timeSeconds: number) => void;
  onKeyframeValueChange: (laneId: string, index: number, value: TimelineValue) => void;
  onKeyframeRemove: (laneId: string, index: number) => void;
  onInterpolationChange: (laneId: string, interpolation: TimelineInterpolation) => void;
  onPrevKeyframe: () => void;
  onNextKeyframe: () => void;
  onAutoKeyframeToggle: (value: boolean) => void;
};

function TimelineEditorPanel({
  active,
  playing,
  loop,
  fps,
  durationSeconds,
  durationFrames,
  currentTime,
  lanes,
  parameterConfigs,
  availableParameters,
  evaluation,
  autoKeyframe,
  onTogglePlay,
  onStop,
  onToggleLoop,
  onTimeChange,
  onFpsChange,
  onDurationChange,
  onAddLane,
  onRemoveLane,
  onClear,
  onAddKeyframe,
  onKeyframeTimeChange,
  onKeyframeValueChange,
  onKeyframeRemove,
  onInterpolationChange,
  onPrevKeyframe,
  onNextKeyframe,
  onAutoKeyframeToggle,
}: TimelineEditorPanelProps) {
  const [selectedParameter, setSelectedParameter] = React.useState<string>(
    availableParameters[0]?.id ?? '',
  );

  React.useEffect(() => {
    if (availableParameters.length === 0) {
      setSelectedParameter('');
      return;
    }
    if (!availableParameters.some((entry) => entry.id === selectedParameter)) {
      setSelectedParameter(availableParameters[0]!.id);
    }
  }, [availableParameters, selectedParameter]);

  const configMap = React.useMemo(() => {
    const map = new Map<string, TimelineParameterConfig>();
    parameterConfigs.forEach((config) => {
      map.set(config.id, config);
    });
    return map;
  }, [parameterConfigs]);

  const timeStep = fps > 0 ? 1 / fps : 0.0166667;
  const safeDuration = durationSeconds > 0 ? durationSeconds : timeStep;
  const sliderMax = Math.max(safeDuration, timeStep);
  const playheadFraction = safeDuration > 0 ? Math.min(currentTime / safeDuration, 1) : 0;
  const hasKeyframes = lanes.some((lane) => lane.keyframes.length > 0);

  const handleAddLaneClick = () => {
    if (selectedParameter) {
      onAddLane(selectedParameter);
    }
  };

  const handleTimeInput = (value: number) => {
    if (!Number.isFinite(value)) {
      return;
    }
    onTimeChange(Math.max(0, value));
  };

  return (
    <div className="timeline-editor">
      <div className="timeline-editor__status">
        <span>
          {playing ? 'Playing' : 'Paused'} • {currentTime.toFixed(2)} s / {safeDuration.toFixed(2)}
           s • {Math.round(fps)} fps
        </span>
        {!active ? <span className="timeline-editor__badge">Timeline inactive</span> : null}
      </div>
      <div className="timeline-editor__controls">
        <button type="button" onClick={onTogglePlay} className="timeline-editor__button">
          {playing ? 'Pause' : 'Play'}
        </button>
        <button
          type="button"
          onClick={onStop}
          className="timeline-editor__button"
          disabled={currentTime <= 0 && !playing}
        >
          Stop
        </button>
        <button
          type="button"
          onClick={onPrevKeyframe}
          className="timeline-editor__button"
          disabled={!hasKeyframes}
        >
          Prev KF
        </button>
        <button
          type="button"
          onClick={onNextKeyframe}
          className="timeline-editor__button"
          disabled={!hasKeyframes}
        >
          Next KF
        </button>
        <label className="timeline-editor__toggle">
          <input
            type="checkbox"
            checked={loop}
            onChange={(event) => onToggleLoop(event.target.checked)}
          />{' '}
          Loop
        </label>
        <label className="timeline-editor__toggle">
          <input
            type="checkbox"
            checked={autoKeyframe}
            onChange={(event) => onAutoKeyframeToggle(event.target.checked)}
          />{' '}
          Auto keyframe on tweak
        </label>
        <button
          type="button"
          onClick={onClear}
          className="timeline-editor__button timeline-editor__button--danger"
          disabled={lanes.length === 0}
        >
          Clear timeline
        </button>
      </div>
      <div className="timeline-editor__timeline">
        <label className="timeline-editor__slider">
          <span>Time</span>
          <input
            type="range"
            min={0}
            max={sliderMax}
            step={timeStep}
            value={Math.max(0, Math.min(currentTime, sliderMax))}
            onChange={(event) => handleTimeInput(Number(event.target.value))}
          />
          <span>{currentTime.toFixed(3)} s</span>
        </label>
        <div className="timeline-editor__settings">
          <label>
            Duration (s)
            <input
              type="number"
              min={0.1}
              max={600}
              step={0.1}
              value={Number(durationSeconds.toFixed(2))}
              onChange={(event) => onDurationChange(Number(event.target.value))}
            />
          </label>
          <label>
            FPS
            <input
              type="number"
              min={1}
              max={240}
              step={1}
              value={Math.round(fps)}
              onChange={(event) => onFpsChange(Number(event.target.value))}
            />
          </label>
          <label>
            Current time (s)
            <input
              type="number"
              min={0}
              max={sliderMax}
              step={timeStep}
              value={Number(currentTime.toFixed(3))}
              onChange={(event) => handleTimeInput(Number(event.target.value))}
            />
          </label>
        </div>
        <div className="timeline-editor__track">
          <div className="timeline-track">
            <div
              className="timeline-track__playhead"
              style={{ left: `${Math.max(0, Math.min(playheadFraction, 1)) * 100}%` }}
            />
          </div>
        </div>
      </div>
      <div className="timeline-editor__add">
        <label htmlFor="timeline-parameter-select">Add parameter</label>
        <div className="timeline-editor__add-controls">
          <select
            id="timeline-parameter-select"
            value={selectedParameter}
            onChange={(event) => setSelectedParameter(event.target.value)}
          >
            {availableParameters.length === 0 ? (
              <option value="" disabled>
                All parameters added
              </option>
            ) : (
              availableParameters.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label}
                </option>
              ))
            )}
          </select>
          <button
            type="button"
            className="timeline-editor__button"
            onClick={handleAddLaneClick}
            disabled={!selectedParameter}
          >
            Add lane
          </button>
        </div>
      </div>
      {lanes.length === 0 ? (
        <p className="timeline-editor__empty">
          No lanes yet. Add a parameter or enable auto keyframe and tweak a control.
        </p>
      ) : (
        lanes.map((lane) => {
          const config = configMap.get(lane.id);
          const interpolationOptions: TimelineInterpolation[] =
            config?.kind === 'number' ? ['linear', 'step'] : ['step'];
          const activeValue = evaluation?.values[lane.id];
          return (
            <div key={lane.id} className="timeline-lane">
              <header className="timeline-lane__header">
                <div>
                  <div className="timeline-lane__title">{config?.label ?? lane.id}</div>
                  <div className="timeline-lane__meta">
                    Active value: {formatTimelineValue(activeValue)}
                  </div>
                </div>
                <div className="timeline-lane__actions">
                  <label>
                    Interpolation
                    <select
                      value={lane.interpolation}
                      onChange={(event) =>
                        onInterpolationChange(lane.id, event.target.value as TimelineInterpolation)
                      }
                    >
                      {interpolationOptions.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    type="button"
                    className="timeline-editor__button"
                    onClick={() => onAddKeyframe(lane.id)}
                  >
                    Keyframe @ {currentTime.toFixed(2)} s
                  </button>
                  <button
                    type="button"
                    className="timeline-editor__button timeline-editor__button--danger"
                    onClick={() => onRemoveLane(lane.id)}
                  >
                    Remove lane
                  </button>
                </div>
              </header>
              <div className="timeline-track timeline-track--lane">
                <div
                  className="timeline-track__playhead"
                  style={{ left: `${Math.max(0, Math.min(playheadFraction, 1)) * 100}%` }}
                />
                {lane.keyframes.map((keyframe, index) => {
                  const timeSeconds = keyframe.frame / fps;
                  const fraction = safeDuration > 0 ? Math.min(timeSeconds / safeDuration, 1) : 0;
                  return (
                    <button
                      type="button"
                      key={`${lane.id}-kf-${index}`}
                      className="timeline-keyframe"
                      style={{ left: `${fraction * 100}%` }}
                      onClick={() => onTimeChange(timeSeconds)}
                      title={`${timeSeconds.toFixed(3)}s`}
                    />
                  );
                })}
              </div>
              <div className="timeline-keyframes">
                {lane.keyframes.map((keyframe, index) => {
                  const timeSeconds = keyframe.frame / fps;
                  const configForLane = config ?? null;
                  const renderValueInput = () => {
                    if (configForLane?.kind === 'boolean') {
                      const current = Boolean(keyframe.value);
                      return (
                        <select
                          value={current ? 'true' : 'false'}
                          onChange={(event) =>
                            onKeyframeValueChange(lane.id, index, event.target.value === 'true')
                          }
                        >
                          <option value="true">True</option>
                          <option value="false">False</option>
                        </select>
                      );
                    }
                    if (configForLane?.kind === 'enum') {
                      return (
                        <select
                          value={String(keyframe.value)}
                          onChange={(event) =>
                            onKeyframeValueChange(lane.id, index, event.target.value)
                          }
                        >
                          {(configForLane.options ?? []).map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label ?? option.value}
                            </option>
                          ))}
                        </select>
                      );
                    }
                    const min = configForLane?.min;
                    const max = configForLane?.max;
                    const step = configForLane?.step ?? 0.01;
                    const numericValue =
                      typeof keyframe.value === 'number' && Number.isFinite(keyframe.value)
                        ? keyframe.value
                        : 0;
                    return (
                      <input
                        type="number"
                        value={numericValue}
                        step={step}
                        min={min}
                        max={max}
                        onChange={(event) => {
                          const next = Number(event.target.value);
                          if (Number.isFinite(next)) {
                            onKeyframeValueChange(lane.id, index, next);
                          }
                        }}
                      />
                    );
                  };
                  return (
                    <div key={`${lane.id}-row-${index}`} className="timeline-keyframe-row">
                      <div className="timeline-keyframe-field">
                        <label>Time (s)</label>
                        <input
                          type="number"
                          min={0}
                          max={sliderMax}
                          step={timeStep}
                          value={Number(timeSeconds.toFixed(3))}
                          onChange={(event) => {
                            const nextTime = Number(event.target.value);
                            if (Number.isFinite(nextTime)) {
                              onKeyframeTimeChange(lane.id, index, nextTime);
                            }
                          }}
                        />
                      </div>
                      <div className="timeline-keyframe-field">
                        <label>Value</label>
                        {renderValueInput()}
                      </div>
                      <button
                        type="button"
                        className="timeline-editor__button timeline-editor__button--danger"
                        onClick={() => onKeyframeRemove(lane.id, index)}
                      >
                        Remove
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
