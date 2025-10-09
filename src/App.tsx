import React, {
  ChangeEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
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
  type KuramotoInstrumentationSnapshot
} from "./kuramotoCore";
import {
  createGpuRenderer,
  type GpuRenderer
} from "./gpuRenderer";
import {
  applyOp,
  clamp,
  computeComposerBlendGain,
  createDefaultComposerConfig,
  groupOps,
  hash2,
  kEff,
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
  type SolverRegime
} from "./pipeline/rainbowFrame";
import {
  clampKernelSpec,
  createKernelSpec,
  getDefaultKernelSpec,
  kernelSpecToJSON,
  type KernelSpec
} from "./kernel/kernelSpec";
import { getKernelSpecHub } from "./kernel/kernelHub";
import { computeEdgeField } from "./pipeline/edgeDetection";
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
  type VolumeField
} from "./fields/contracts";
import type { OpticalFieldMetadata } from "./fields/opticalField.js";
import {
  createInitialStatuses,
  markFieldUnavailable as setFieldUnavailable,
  markFieldUpdate as setFieldAvailable,
  refreshFieldStaleness,
  type FieldStatusMap
} from "./fields/state";
import {
  createVolumeStubState,
  snapshotVolumeStub,
  stepVolumeStub,
  type VolumeStubState
} from "./volumeStub";
import {
  DEFAULT_SYNTHETIC_SIZE,
  SYNTHETIC_CASES,
  type SyntheticCaseId
} from "./dev/syntheticDeck";

type Preset = {
  name: string;
  params: {
    edgeThreshold: number;
    blend: number;
    kernel: KernelSpec;
    dmt: number;
    thetaMode: "gradient" | "global";
    thetaGlobal: number;
    beta2: number;
    jitter: number;
    sigma: number;
    microsaccade: boolean;
    speed: number;
    contrast: number;
  };
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
};

const DEFAULT_COUPLING_TOGGLES: CouplingToggleState = {
  rimToSurface: true,
  surfaceToRim: true
};

const cloneCouplingToggles = (value: CouplingToggleState): CouplingToggleState => ({
  rimToSurface: value.rimToSurface,
  surfaceToRim: value.surfaceToRim
});

const applyCouplingToggles = (
  config: CouplingConfig,
  toggles: CouplingToggleState
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
    name: "Rainbow Rims + DMT Kernel Effects",
    params: {
      edgeThreshold: 0.08,
      blend: 0.39,
      kernel: createKernelSpec({
        gain: 3.0,
        k0: 0.2,
        Q: 4.6,
        anisotropy: 0.95,
        chirality: 1.46,
        transparency: 0.28
      }),
      dmt: 0.2,
      thetaMode: "gradient",
      thetaGlobal: 0,
      beta2: 1.9,
      jitter: 1.16,
      sigma: 4.0,
      microsaccade: true,
      speed: 1.32,
      contrast: 1.62
    },
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
      volumeDepthToWarp: 0.3
    },
    composer: createDefaultComposerConfig(),
    couplingToggles: DEFAULT_COUPLING_TOGGLES
  }
];

type KurRegime = "locked" | "highEnergy" | "chaotic" | "custom";

const KUR_REGIME_PRESETS: Record<Exclude<KurRegime, "custom">, {
  label: string;
  description: string;
  params: {
    K0: number;
    alphaKur: number;
    gammaKur: number;
    omega0: number;
    epsKur: number;
  };
}> = {
  locked: {
    label: "Locked coherence",
    description: "Low-noise lattice with stable phase alignment.",
    params: {
      K0: 0.85,
      alphaKur: 0.18,
      gammaKur: 0.16,
      omega0: 0.0,
      epsKur: 0.0025
    }
  },
  highEnergy: {
    label: "High-energy flux",
    description: "Strong coupling and drive yielding intense wavefronts.",
    params: {
      K0: 1.35,
      alphaKur: 0.12,
      gammaKur: 0.1,
      omega0: 0.55,
      epsKur: 0.006
    }
  },
  chaotic: {
    label: "Chaotic drift",
    description: "Loose locking with broadband oscillations and noise.",
    params: {
      K0: 1.1,
      alphaKur: 0.28,
      gammaKur: 0.12,
      omega0: 0.35,
      epsKur: 0.012
    }
  }
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
  surface: "Surface",
  rim: "Rim",
  kur: "Kuramoto",
  volume: "Volume"
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
  volumeDepthToWarp: value.volumeDepthToWarp
});

const sanitizeCouplingConfig = (
  value: Partial<CouplingConfig> | null | undefined,
  fallback: CouplingConfig
): CouplingConfig => {
  const source = value ?? {};
  return {
    rimToSurfaceBlend: typeof source.rimToSurfaceBlend === "number" ? source.rimToSurfaceBlend : fallback.rimToSurfaceBlend,
    rimToSurfaceAlign: typeof source.rimToSurfaceAlign === "number" ? source.rimToSurfaceAlign : fallback.rimToSurfaceAlign,
    surfaceToRimOffset: typeof source.surfaceToRimOffset === "number" ? source.surfaceToRimOffset : fallback.surfaceToRimOffset,
    surfaceToRimSigma: typeof source.surfaceToRimSigma === "number" ? source.surfaceToRimSigma : fallback.surfaceToRimSigma,
    surfaceToRimHue: typeof source.surfaceToRimHue === "number" ? source.surfaceToRimHue : fallback.surfaceToRimHue,
    kurToTransparency: typeof source.kurToTransparency === "number" ? source.kurToTransparency : fallback.kurToTransparency,
    kurToOrientation: typeof source.kurToOrientation === "number" ? source.kurToOrientation : fallback.kurToOrientation,
    kurToChirality: typeof source.kurToChirality === "number" ? source.kurToChirality : fallback.kurToChirality,
    volumePhaseToHue: typeof source.volumePhaseToHue === "number" ? source.volumePhaseToHue : fallback.volumePhaseToHue,
    volumeDepthToWarp: typeof source.volumeDepthToWarp === "number" ? source.volumeDepthToWarp : fallback.volumeDepthToWarp
  };
};

const cloneComposerConfig = (config: ComposerConfig): ComposerConfig => ({
  fields: {
    surface: { ...config.fields.surface },
    rim: { ...config.fields.rim },
    kur: { ...config.fields.kur },
    volume: { ...config.fields.volume }
  },
  dmtRouting: config.dmtRouting,
  solverRegime: config.solverRegime
});

const sanitizeComposerImport = (
  value: Partial<ComposerConfig> | null | undefined
): ComposerConfig => {
  const defaults = createDefaultComposerConfig();
  if (!value) return defaults;
  const result = cloneComposerConfig(defaults);
  result.dmtRouting = (value.dmtRouting as DmtRoutingMode) ?? defaults.dmtRouting;
  result.solverRegime = (value.solverRegime as SolverRegime) ?? defaults.solverRegime;
  COMPOSER_FIELD_LIST.forEach((field) => {
    const incoming = value.fields?.[field];
    if (incoming) {
      result.fields[field] = {
        exposure: typeof incoming.exposure === "number" ? incoming.exposure : defaults.fields[field].exposure,
        gamma: typeof incoming.gamma === "number" ? incoming.gamma : defaults.fields[field].gamma,
        weight: typeof incoming.weight === "number" ? incoming.weight : defaults.fields[field].weight
      };
    }
  });
  return result;
};

const formatCouplingKey = (key: keyof CouplingConfig) =>
  key
    .replace(/([A-Z])/g, " $1")
    .replace(/^./, (ch) => ch.toUpperCase());

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
  kind: "frame";
  buffer: ArrayBuffer;
  timestamp: number;
  frameId: number;
  queueDepth: number;
  kernelVersion?: number;
  meta: OpticalFieldMetadata;
  instrumentation: KuramotoInstrumentationSnapshot;
};

type WorkerReadyMessage = { kind: "ready"; width: number; height: number };
type WorkerLogMessage = { kind: "log"; message: string };
type WorkerSimulateResultMessage = {
  kind: "simulateResult";
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

type TelemetryPhase = "frame" | "renderGpu" | "renderCpu" | "kuramoto";

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
  backend: "cpu" | "gpu";
  ts: number;
  metrics: RainbowFrameMetrics;
  kernelVersion: number;
};

const printRgb = (values: [number, number, number]) =>
  `(${values.map((value) => Math.round(value)).join(", ")})`;

declare global {
  interface Window {
    __setFrameProfiler?: (
      enabled: boolean,
      sampleCount?: number,
      label?: string
    ) => void;
    __runFrameRegression?: (
      frameCount?: number
    ) => { maxDelta: number; perFrameMax: number[] };
    __setRenderBackend?: (backend: "gpu" | "cpu") => void;
    __runGpuParityCheck?: () => Promise<
      | {
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
        }
      | null
    >;
    __measureRenderPerformance?: (
      frameCount?: number
    ) =>
      | {
          frameCount: number;
          cpuMs: number;
          gpuMs: number;
          cpuFps: number;
          gpuFps: number;
          throughputGain: number;
        }
      | null;
    __setTelemetryEnabled?: (enabled: boolean) => void;
    __getTelemetryHistory?: () => TelemetryRecord[];
    __getFrameMetrics?: () => FrameMetricsEntry[];
  }
}

const formatBytes = (bytes: number) => {
  if (bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
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
    case "grayBaseColorRims":
      return 1;
    case "grayBaseGrayRims":
      return 2;
    case "colorBaseGrayRims":
      return 3;
    case "colorBaseBlendedRims":
      return 4;
    default:
      return 0;
  }
};

const FIELD_STATUS_STYLES = {
  ok: {
    border: "rgba(34,197,94,0.35)",
    background: "rgba(34,197,94,0.16)",
    color: "#bbf7d0"
  },
  warn: {
    border: "rgba(251,191,36,0.4)",
    background: "rgba(251,191,36,0.18)",
    color: "#fde68a"
  },
  stale: {
    border: "rgba(248,113,113,0.45)",
    background: "rgba(248,113,113,0.2)",
    color: "#fecaca"
  },
  missing: {
    border: "rgba(148,163,184,0.35)",
    background: "rgba(148,163,184,0.16)",
    color: "#e2e8f0"
  }
} as const;

const FIELD_STATUS_LABELS = {
  ok: "fresh",
  warn: "lagging",
  stale: "stale",
  missing: "missing"
} as const;

const surfaceRegionToEnum = (region: SurfaceRegion) => {
  switch (region) {
    case "surfaces":
      return 0;
    case "edges":
      return 1;
    default:
      return 2;
  }
};

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const presetFileInputRef = useRef<HTMLInputElement | null>(null);
  const gpuStateRef = useRef<{ gl: WebGL2RenderingContext; renderer: GpuRenderer } | null>(null);
  const pendingStaticUploadRef = useRef(true);

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
  const [renderBackend, setRenderBackend] = useState<"gpu" | "cpu">("gpu");

  const [imgBitmap, setImgBitmap] = useState<ImageBitmap | null>(null);
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

  const [beta2, setBeta2] = useState(1.1);
  const [jitter, setJitter] = useState(0.5);
  const [sigma, setSigma] = useState(1.4);
  const [microsaccade, setMicrosaccade] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [contrast, setContrast] = useState(1.0);

  const [kernel, setKernel] = useState<KernelSpec>(() => getDefaultKernelSpec());
  const updateKernel = useCallback(
    (patch: Partial<KernelSpec>) =>
      setKernel((prev) => clampKernelSpec({ ...prev, ...patch })),
    []
  );
  useEffect(() => {
    return kernelHub.subscribe((event) => {
      kernelEventRef.current = event;
      kurKernelVersionRef.current = event.version;
      if (!kurSyncRef.current && workerRef.current && workerReadyRef.current) {
        workerRef.current.postMessage({
          kind: "kernelSpec",
          spec: event.spec,
          version: event.version
        });
      }
    });
  }, [kernelHub]);
  useEffect(() => {
    kernelHub.replace(kernel, { source: "ui", force: true });
  }, [kernel, kernelHub]);
  const [dmt, setDmt] = useState(0.0);

  const [thetaMode, setThetaMode] = useState<"gradient" | "global">("gradient");
  const [thetaGlobal, setThetaGlobal] = useState(0);

  const [displayMode, setDisplayMode] = useState<DisplayMode>("color");

  const [phasePin, setPhasePin] = useState(true);
  const [alive, setAlive] = useState(false);
  const [polBins, setPolBins] = useState(16);
  const [normPin, setNormPin] = useState(true);
  const [kurRegime, setKurRegime] = useState<KurRegime>("locked");

  const [surfEnabled, setSurfEnabled] = useState(false);
  const [wallGroup, setWallGroup] = useState<WallpaperGroup>("p4");
  const [nOrient, setNOrient] = useState(4);
  const [surfaceBlend, setSurfaceBlend] = useState(0.35);
  const [warpAmp, setWarpAmp] = useState(1.0);
  const [surfaceRegion, setSurfaceRegion] =
    useState<SurfaceRegion>("surfaces");

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
    volumeDepthToWarp: 0.3
  });
  const setCouplingValue = useCallback(
    (key: keyof CouplingConfig) => (value: number) =>
      setCoupling((prev) => ({ ...prev, [key]: value })),
    []
  );
  const [couplingToggles, setCouplingToggles] = useState<CouplingToggleState>(
    DEFAULT_COUPLING_TOGGLES
  );
  const setCouplingToggle = useCallback(
    (key: keyof CouplingToggleState) => (value: boolean) =>
      setCouplingToggles((prev) => ({ ...prev, [key]: value })),
    []
  );
  const computeCouplingPair = useCallback(
    (override?: CouplingToggleState) => {
      const toggles = override ?? couplingToggles;
      return {
        base: cloneCouplingConfig(coupling),
        effective: applyCouplingToggles(coupling, toggles)
      };
    },
    [coupling, couplingToggles]
  );

  const [composer, setComposer] = useState<ComposerConfig>(() => createDefaultComposerConfig());
  const setComposerFieldValue = useCallback(
    (field: ComposerFieldId, key: "exposure" | "gamma" | "weight") =>
      (value: number) =>
        setComposer((prev) => ({
          ...prev,
          fields: {
            ...prev.fields,
            [field]: {
              ...prev.fields[field],
              [key]: value
            }
          }
        })),
    []
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
  const [presetIndex, setPresetIndex] = useState(0);
  const [telemetryEnabled, setTelemetryEnabled] = useState(false);
  const [telemetrySnapshot, setTelemetrySnapshot] = useState<TelemetrySnapshot | null>(null);
  const [frameLoggingEnabled, setFrameLoggingEnabled] = useState(true);
  const [lastParityResult, setLastParityResult] = useState<ParitySummary | null>(null);
  const [lastPerfResult, setLastPerfResult] = useState<PerformanceSnapshot | null>(null);
  const [recordingStatus, setRecordingStatus] = useState<"idle" | "recording" | "finalizing">("idle");
  const [recordingError, setRecordingError] = useState<string | null>(null);
  const [recordingDownload, setRecordingDownload] = useState<{
    url: string;
    size: number;
    mimeType: string;
    filename: string;
  } | null>(null);
  const [captureSupport, setCaptureSupport] = useState<{
    checked: boolean;
    best: { mimeType: string; container: "mp4" | "webm" } | null;
  }>({ checked: false, best: null });
  const [fieldStatuses, setFieldStatuses] = useState<FieldStatusMap>(() => createInitialStatuses());
  const [rimDebugSnapshot, setRimDebugSnapshot] = useState<RimDebugSnapshot | null>(null);
  const [surfaceDebugSnapshot, setSurfaceDebugSnapshot] = useState<SurfaceDebugSnapshot | null>(null);
  const [phaseDebugSnapshot, setPhaseDebugSnapshot] = useState<PhaseDebugSnapshot | null>(null);
  const [phaseHeatmapEnabled, setPhaseHeatmapEnabled] = useState(false);
  const [phaseHeatmapSnapshot, setPhaseHeatmapSnapshot] = useState<PhaseHeatmapSnapshot | null>(null);
  const [selectedSyntheticCase, setSelectedSyntheticCase] = useState<SyntheticCaseId>("circles");
  const [syntheticBaselines, setSyntheticBaselines] = useState<
    Record<SyntheticCaseId, { metrics: RainbowFrameMetrics; timestamp: number }>
  >({});
  const markFieldFresh = useCallback(
    (kind: FieldKind, resolution: FieldResolution, source: string) => {
      const now = performance.now();
      setFieldStatuses((prev) => {
        const next = setFieldAvailable(prev, kind, resolution, source, now);
        if (!prev[kind].available) {
          const contract = FIELD_CONTRACTS[kind];
          console.info(
            `[fields] ${contract.label} available ${resolution.width}x${resolution.height} via ${source}`
          );
        }
        return next;
      });
    },
    []
  );

  const markFieldGone = useCallback(
    (kind: FieldKind, source: string) => {
      const now = performance.now();
      setFieldStatuses((prev) => {
        if (!prev[kind].available) return prev;
        const contract = FIELD_CONTRACTS[kind];
        console.warn(`[fields] ${contract.label} unavailable via ${source}`);
        return setFieldUnavailable(prev, kind, source, now);
      });
    },
    []
  );

  const stopCaptureStream = useCallback(() => {
    if (captureStreamRef.current) {
      captureStreamRef.current.getTracks().forEach((track) => track.stop());
      captureStreamRef.current = null;
    }
  }, []);

  const recordingBitrate = useMemo(
    () => Math.min(width * height * 24, 80_000_000),
    [width, height]
  );

  const normTargetRef = useRef(0.6);
  const lastObsRef = useRef(0.6);

  const kurSyncRef = useRef(false);
  const kurStateRef = useRef<KuramotoState | null>(null);
  const kurTelemetryRef = useRef<KuramotoTelemetrySnapshot | null>(null);
  const kurIrradianceRef = useRef<IrradianceFrameBuffer | null>(null);
  const kurLogRef = useRef<{ kernelVersion: number; frameId: number }>({
    kernelVersion: -1,
    frameId: -1
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
    null
  );

  const orientationCacheRef = useRef<{
    count: number;
    cos: Float32Array;
    sin: Float32Array;
  }>({
    count: 0,
    cos: new Float32Array(0),
    sin: new Float32Array(0)
  });

  const frameProfilerRef = useRef<FrameProfilerState>({
    enabled: false,
    samples: [],
    maxSamples: 120,
    label: "frame-profiler"
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
            console.warn(
              `[fields] ${contract.label} stale ${change.stalenessMs.toFixed(0)}ms`
            );
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
      kuramoto: 8
    },
    history: [],
    lastLogTs: 0
  });

  const frameLogRef = useRef<{
    windowStart: number;
    frames: number;
  }>({
    windowStart: performance.now(),
    frames: 0
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
          3
        )} anis=${telemetry.kernel.anisotropy.toFixed(3)} chir=${telemetry.kernel.chirality.toFixed(3)}`
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
      const accum = COMPOSER_FIELD_LIST.reduce<Record<ComposerFieldId, { energy: number; blend: number; share: number; weight: number }>>((acc, field) => {
        acc[field] = { energy: 0, blend: 0, share: 0, weight: 0 };
        return acc;
      }, {} as Record<ComposerFieldId, { energy: number; blend: number; share: number; weight: number }>);
      let lastComposer: ComposerTelemetry | null = null;
      history.forEach((entry) => {
        const telemetry = entry.metrics.composer;
        if (!telemetry) return;
        lastComposer = telemetry;
        COMPOSER_FIELD_LIST.forEach((field) => {
          accum[field].energy += telemetry.fields[field].energy;
          accum[field].blend += telemetry.fields[field].blend;
          accum[field].share += telemetry.fields[field].share;
          accum[field].weight = telemetry.fields[field].weight;
        });
      });
      if (!lastComposer) return;
      const count = history.length;
      const snapshot: TelemetrySnapshot = {
        fields: {} as Record<ComposerFieldId, TelemetryFieldSnapshot>,
        coupling: {
          scale: lastComposer.coupling.scale,
          base: cloneCouplingConfig(lastComposer.coupling.base),
          effective: cloneCouplingConfig(lastComposer.coupling.effective)
        },
        updatedAt: Date.now()
      };
      COMPOSER_FIELD_LIST.forEach((field) => {
        snapshot.fields[field] = {
          energy: accum[field].energy / count,
          blend: accum[field].blend / count,
          share: accum[field].share / count,
          weight: accum[field].weight
        };
      });
      setTelemetrySnapshot(snapshot);
    }, 500);
    return () => window.clearInterval(interval);
  }, [telemetryEnabled]);

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
        let state: "ok" | "warn" | "stale" | "missing";
        if (!status.available) {
          state = "missing";
        } else if (status.stale) {
          state = "stale";
        } else if (
          contract.lifetime.expectedMs !== Number.POSITIVE_INFINITY &&
          staleness > contract.lifetime.expectedMs * 1.5
        ) {
          state = "warn";
        } else {
          state = "ok";
        }
        return {
          kind,
          label: contract.label,
          state,
          stalenessMs: staleness,
          resolution: status.resolution
        };
      }),
    [fieldStatuses]
  );

  const ensureKurCpuState = useCallback(() => {
    if (!kurStateRef.current || kurStateRef.current.width !== width || kurStateRef.current.height !== height) {
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
      assertPhaseField(derived, "cpu:init");
      cpuDerivedRef.current = derived;
      gradXRef.current = derived.gradX;
      gradYRef.current = derived.gradY;
      vortRef.current = derived.vort;
      cohRef.current = derived.coh;
      ampRef.current = derived.amp;
      return true;
    }
    assertPhaseField(cpuDerivedRef.current, "cpu:reuse");
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

  const ensureFrameBuffer = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      let buffer = frameBufferRef.current;
      if (
        !buffer ||
        buffer.width !== width ||
        buffer.height !== height
      ) {
        const image = ctx.createImageData(width, height);
        buffer = {
          image,
          data: image.data,
          width,
          height
        };
        frameBufferRef.current = buffer;
      }
      return buffer;
    },
    [width, height]
  );

  const ensureRimDebugBuffers = useCallback(() => {
    const total = width * height;
    let buffers = rimDebugRef.current;
    if (!buffers || buffers.energy.length !== total) {
      buffers = {
        energy: new Float32Array(total),
        hue: new Float32Array(total),
        energyHist: new Uint32Array(RIM_HIST_BINS),
        hueHist: new Uint32Array(RIM_HIST_BINS)
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
          orientationCount
        };
        surfaceDebugRef.current = buffers;
      } else if (buffers) {
        buffers.magnitudeHist.fill(0);
      }
      return buffers ?? null;
    },
    [width, height]
  );

  const updateDebugSnapshots = useCallback(
    (
      commit: boolean,
      rimBuffers: ReturnType<typeof ensureRimDebugBuffers> | null,
      surfaceBuffers: ReturnType<typeof ensureSurfaceDebugBuffers> | null,
      debug: ReturnType<typeof renderRainbowFrame>["debug"]
    ) => {
      if (!commit) return;
      if (showRimDebug && rimBuffers && debug?.rim) {
        setRimDebugSnapshot({
          energyRange: [debug.rim.energyMin, debug.rim.energyMax],
          hueRange: [debug.rim.hueMin, debug.rim.hueMax],
          energyHist: Array.from(rimBuffers.energyHist),
          hueHist: Array.from(rimBuffers.hueHist)
        });
      } else if (!showRimDebug) {
        setRimDebugSnapshot(null);
      }
      if (showSurfaceDebug && surfaceBuffers && debug?.surface) {
        const hist: number[][] = [];
        for (let k = 0; k < surfaceBuffers.orientationCount; k++) {
          const start = k * SURFACE_HIST_BINS;
          const slice = surfaceBuffers.magnitudeHist.slice(
            start,
            start + SURFACE_HIST_BINS
          );
          hist.push(Array.from(slice));
        }
        setSurfaceDebugSnapshot({
          orientationCount: surfaceBuffers.orientationCount,
          magnitudeMax: debug.surface.magnitudeMax,
          magnitudeHist: hist
        });
      } else if (!showSurfaceDebug) {
        setSurfaceDebugSnapshot(null);
      }
    },
    [
      showRimDebug,
      showSurfaceDebug,
      setRimDebugSnapshot,
      setSurfaceDebugSnapshot
    ]
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
          ampHist: Array.from(hist)
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
            data: new Float32Array(targetW * targetH)
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
          max: maxVal
        });
      } else if (!phaseHeatmapEnabled || !phaseField) {
        setPhaseHeatmapSnapshot(null);
      }
    },
    [
      showPhaseDebug,
      phaseHeatmapEnabled,
      setPhaseDebugSnapshot,
      setPhaseHeatmapSnapshot
    ]
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
        sin: new Float32Array(count)
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
    if (renderBackend !== "gpu") return null;
    const canvas = canvasRef.current;
    if (!canvas) return null;
    canvas.width = width;
    canvas.height = height;
    let state = gpuStateRef.current;
    if (!state || state.gl.canvas !== canvas) {
      const gl = canvas.getContext("webgl2", {
        alpha: false,
        antialias: false,
        premultipliedAlpha: false,
        preserveDrawingBuffer: true
      });
      if (!gl) {
        console.warn("[gpu] WebGL2 unavailable; falling back to CPU renderer.");
        setRenderBackend("cpu");
        return null;
      }
      if (state) {
        state.renderer.dispose();
      }
      const renderer = createGpuRenderer(gl);
      state = { gl, renderer };
      gpuStateRef.current = state;
      pendingStaticUploadRef.current = true;
    }
    state.renderer.resize(width, height);
    if (pendingStaticUploadRef.current) {
      refreshGpuStaticTextures();
    }
    return state;
  }, [renderBackend, width, height, refreshGpuStaticTextures]);

  const setFrameProfiler = useCallback(
    (enabled: boolean, sampleCount = 120, label = "frame-profiler") => {
      const profiler = frameProfilerRef.current;
      profiler.enabled = enabled;
      profiler.maxSamples = sampleCount;
      profiler.samples = [];
      profiler.label = label;
      console.log(
        `[frame-profiler] ${enabled ? `collecting ${sampleCount} samples (${label})` : "disabled"}`
      );
    },
    []
  );

  const initKuramotoCpu = useCallback(
    (q: number) => {
      ensureKurCpuState();
      if (!kurStateRef.current || !cpuDerivedRef.current) return;
      initKuramotoState(kurStateRef.current, q, cpuDerivedRef.current);
    },
    [ensureKurCpuState]
  );

  const getKurParams = useCallback((): KuramotoParams => {
    return {
      alphaKur,
      gammaKur,
      omega0,
      K0,
      epsKur,
      fluxX,
      fluxY
    };
  }, [alphaKur, gammaKur, omega0, K0, epsKur, fluxX, fluxY]);

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
        markFieldGone("phase", "cpu-reset");
      }
      markFieldGone("volume", "cpu-reset");
    }
  }, [kurEnabled, qInit, initKuramotoCpu, ensureKurCpuState]);

  useEffect(() => {
    if (!volumeEnabled) {
      volumeFieldRef.current = null;
      volumeStubRef.current = null;
      markFieldGone("volume", "volume-disabled");
      return;
    }
    ensureVolumeState();
    if (volumeStubRef.current) {
      const field = snapshotVolumeStub(volumeStubRef.current);
      assertVolumeField(field, "volume:init");
      volumeFieldRef.current = field;
      markFieldFresh("volume", field.resolution, "volume:stub");
    }
  }, [volumeEnabled, ensureVolumeState, markFieldFresh, markFieldGone]);

  const stepKuramotoCpu = useCallback(
    (dt: number) => {
      if (!kurEnabled) return;
      ensureKurCpuState();
      if (!kurStateRef.current) return;
      const kernelSnapshot = kernelEventRef.current;
      kurAppliedKernelVersionRef.current = kernelSnapshot.version;
      const timestamp = typeof performance !== "undefined" ? performance.now() : Date.now();
      const result = stepKuramotoState(kurStateRef.current, getKurParams(), dt, randn, timestamp, {
        kernel: kernelSnapshot.spec,
        controls: { dmt },
        telemetry: { kernelVersion: kernelSnapshot.version }
      });
      kurTelemetryRef.current = result.telemetry;
      kurIrradianceRef.current = result.irradiance;
      logKurTelemetry(result.telemetry);
    },
    [kurEnabled, ensureKurCpuState, getKurParams, randn, dmt, logKurTelemetry]
  );

  const deriveKurFieldsCpu = useCallback(() => {
    if (!kurEnabled) return;
    ensureKurCpuState();
    if (!kurStateRef.current || !cpuDerivedRef.current) return;
    const kernelSnapshot = kernelEventRef.current;
    deriveKuramotoFieldsCore(kurStateRef.current, cpuDerivedRef.current, {
      kernel: kernelSnapshot.spec,
      controls: { dmt }
    });
    markFieldFresh("phase", cpuDerivedRef.current.resolution, "cpu");
  }, [kurEnabled, ensureKurCpuState, dmt, markFieldFresh]);

  const resetKuramotoField = useCallback(() => {
    initKuramotoCpu(qInit);
    if (!kurSyncRef.current && workerRef.current && workerReadyRef.current) {
      workerRef.current.postMessage({
        kind: "reset",
        qInit
      });
    }
  }, [initKuramotoCpu, qInit]);

  const markKurCustom = useCallback(() => {
    setKurRegime("custom");
  }, [setKurRegime]);

  const applyKurRegime = useCallback(
    (regime: Exclude<KurRegime, "custom">) => {
      const preset = KUR_REGIME_PRESETS[regime];
      setK0(preset.params.K0);
      setAlphaKur(preset.params.alphaKur);
      setGammaKur(preset.params.gammaKur);
      setOmega0(preset.params.omega0);
      setEpsKur(preset.params.epsKur);
      setKurRegime(regime);
      resetKuramotoField();
    },
    [setK0, setAlphaKur, setGammaKur, setOmega0, setEpsKur, resetKuramotoField]
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
      worker.postMessage(
        { kind: "returnBuffer", buffer: frame.buffer },
        [frame.buffer]
      );
    } catch (err) {
      console.debug("[kur-worker] buffer return skipped", err);
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
      assertPhaseField(derived, "worker:frame");
      markFieldFresh("phase", derived.resolution, "worker");
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
        instrumentation: msg.instrumentation
      };
      if (msg.meta && msg.meta.frameId !== msg.frameId) {
        console.warn(
          `[kur-worker] frameId mismatch meta=${msg.meta.frameId} payload=${msg.frameId}`
        );
      }
      if (frame.kernelVersion !== kernelEventRef.current.version) {
        console.debug(
          `[kur-worker] kernel version drift: worker=${frame.kernelVersion} ui=${kernelEventRef.current.version}`
        );
      }
      const lastFrameId = workerLastFrameIdRef.current;
      if (frame.frameId <= lastFrameId) {
        console.debug(
          `[kur-worker] dropping stale frame ${frame.frameId} (last=${lastFrameId})`
        );
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
    [height, width, swapWorkerFrame, releaseFrameToWorker]
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
        case "ready":
          workerReadyRef.current = true;
          if (!kurSyncRef.current && workerRef.current) {
            const snapshot = kernelEventRef.current;
            workerRef.current.postMessage({
              kind: "kernelSpec",
              spec: snapshot.spec,
              version: snapshot.version
            });
          }
          break;
        case "frame":
          handleWorkerFrame(msg);
          break;
      case "log":
        console.log(msg.message);
        break;
      default:
        console.warn("[kur-worker] unknown message", msg);
        break;
      }
    },
    [handleWorkerFrame]
  );

  const startKurWorker = useCallback(() => {
    if (!kurEnabled || kurSyncRef.current) return;
    stopKurWorker();
    clearWorkerData();
    const bufferSize = derivedBufferSize(width, height);
    const buffers = [new ArrayBuffer(bufferSize), new ArrayBuffer(bufferSize)];
    const worker = new Worker(new URL("./kuramotoWorker.ts", import.meta.url), {
      type: "module"
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
        kind: "init",
        width,
        height,
        params: getKurParams(),
        qInit,
        buffers,
        seed: Math.floor(Math.random() * 1e9)
      },
      buffers
    );
  }, [
    clearWorkerData,
    getKurParams,
    handleWorkerMessage,
    height,
    kurEnabled,
    qInit,
    stopKurWorker,
    width
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
    qInit
  ]);

  useEffect(() => {
    if (!kurEnabled || kurSyncRef.current) return;
    const worker = workerRef.current;
    if (!worker || !workerReadyRef.current) return;
    worker.postMessage({
      kind: "updateParams",
      params: getKurParams()
    });
  }, [getKurParams, kurEnabled, kurSync]);

  const renderFrameCore = useCallback(
    (
      out: Uint8ClampedArray,
      tSeconds: number,
      commitObs = true,
      fieldsOverride?: Pick<KurFrameView, "gradX" | "gradY" | "vort" | "coh" | "amp">,
      options?: RenderFrameOptions
    ) => {
      const kernelSnapshot = kernelEventRef.current;
      const kernelSpec = kernelSnapshot.spec;
      const surfaceField = surfaceFieldRef.current;
      if (surfaceField) {
        assertSurfaceField(surfaceField, "cpu:surface");
      }
      const rimField = rimFieldRef.current;
      if (rimField) {
        assertRimField(rimField, "cpu:rim");
      }
      const volumeField = volumeFieldRef.current;
      if (volumeField) {
        assertVolumeField(volumeField, "cpu:volume");
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
            kind: "phase",
            resolution,
            gradX,
            gradY,
            vort,
            coh,
            amp
          };
          assertPhaseField(phaseField, fieldsOverride ? "phase:override" : "phase:cpu");
          markFieldFresh("phase", resolution, fieldsOverride ? "worker" : "cpu");
        }
      }

      const rimDebugRequest = showRimDebug ? ensureRimDebugBuffers() : null;
      const surfaceDebugRequest =
        showSurfaceDebug && orientations.length > 0
          ? ensureSurfaceDebugBuffers(orientations.length)
          : null;

      const { base: couplingBase, effective: couplingEffective } = computeCouplingPair(
        options?.toggles
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
        kurEnabled,
        debug:
          rimDebugRequest || surfaceDebugRequest
            ? {
                rim: rimDebugRequest ?? undefined,
                surface: surfaceDebugRequest ?? undefined
              }
            : undefined,
        composer,
        kurTelemetry: kurTelemetryRef.current ?? undefined
      });
      if (commitObs && result.obsAverage != null) {
        lastObsRef.current = result.obsAverage;
      }
      if (commitObs && result.metrics) {
        metricsRef.current.push({
          backend: "cpu",
          ts: performance.now(),
          metrics: result.metrics,
          kernelVersion: kernelSnapshot.version
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
      kurEnabled,
      volumeEnabled,
      composer,
      computeCouplingPair,
      ensureRimDebugBuffers,
      ensureSurfaceDebugBuffers,
      updateDebugSnapshots,
      updatePhaseDebug,
      basePixelsRef
    ]
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
        metrics = renderFrameCore(scratch, tSeconds, false);
      } catch (error) {
        console.warn("[frame-log] failed to sample metrics", error);
      }
      if (!metrics) {
        console.log(
          `[frame-log] fps=${fps.toFixed(1)} metrics=unavailable heatmap=${phaseHeatmapEnabled ? "on" : "off"}`
        );
        return;
      }
      const rimEnergy = metrics.composer.fields.rim.energy;
      const surfaceEnergy = metrics.composer.fields.surface.energy;
      const cohMean = metrics.gradient.cohMean ?? 0;
      const cohStd = metrics.gradient.cohStd ?? 0;
      console.log(
        `[frame-log] fps=${fps.toFixed(1)} rim=${rimEnergy.toFixed(3)} surface=${surfaceEnergy.toFixed(
          3
        )} |Z|=${cohMean.toFixed(3)}±${cohStd.toFixed(3)} heatmap=${phaseHeatmapEnabled ? "on" : "off"}`
      );
    },
    [frameLoggingEnabled, renderFrameCore, phaseHeatmapEnabled, width, height]
  );

  const drawFrameGpu = useCallback(
    (
      state: { gl: WebGL2RenderingContext; renderer: GpuRenderer },
      tSeconds: number,
      commitObs: boolean
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
        commitObs &&
        (telemetryActive || rimDebugRequest != null || surfaceDebugRequest != null);
      const renderStart = telemetryActive ? performance.now() : 0;

      const ke = kEff(kernelSpec, dmt);
      const effectiveBlend = clamp01(blend + ke.transparency * 0.5);
      const eps = 1e-6;
      const frameGain = normPin
        ? Math.pow(
            (normTargetRef.current + eps) / (lastObsRef.current + eps),
            0.5
          )
        : 1.0;

      const baseOffsets = {
        L: beta2 * (lambdaRef / lambdas.L - 1),
        M: beta2 * (lambdaRef / lambdas.M - 1),
        S: beta2 * (lambdaRef / lambdas.S - 1)
      } as const;

      const jitterPhase = microsaccade ? tSeconds * 6.0 : 0.0;
      const breath = alive
        ? 0.15 * Math.sin(2 * Math.PI * 0.55 * tSeconds)
        : 0.0;

      const rimField = rimFieldRef.current!;
      const { gx, gy, mag } = rimField;
      const gradX = gradXRef.current;
      const gradY = gradYRef.current;
      const vort = vortRef.current;
      const coh = cohRef.current;
      const amp = ampRef.current;
      const volumeField = volumeFieldRef.current;
      if (volumeField) {
        assertVolumeField(volumeField, "gpu:volume-active");
      }
      const surfaceField = surfaceFieldRef.current;
      if (surfaceField) {
        assertSurfaceField(surfaceField, "gpu:surface");
      }
      assertRimField(rimField, "gpu:rim-active");
      const phaseSource = cpuDerivedRef.current;
      const phaseField =
        phaseSource && gradX && gradY && vort && coh && amp
          ? {
              kind: "phase" as const,
              resolution: phaseSource.resolution,
              gradX,
              gradY,
              vort,
              coh,
              amp
            }
          : null;
      if (phaseField) {
        assertPhaseField(phaseField, "gpu:phase-active");
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
              muSum += Math.sin(
                jitterPhase + hash2(xx, yy) * Math.PI * 2
              );
              muCount++;
            }
          }
        }
        muJ = muCount ? muSum / muCount : 0;
      }

      let metricDebug: ReturnType<typeof renderRainbowFrame>["debug"] | null = null;
      if (needsCpuCompositor) {
        if (
          !metricsScratchRef.current ||
          metricsScratchRef.current.length !== width * height * 4
        ) {
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
          kurEnabled,
          debug:
            rimDebugRequest || surfaceDebugRequest
              ? {
                  rim: rimDebugRequest ?? undefined,
                  surface: surfaceDebugRequest ?? undefined
                }
              : undefined,
          composer,
          kurTelemetry: kurTelemetryRef.current ?? undefined
        });
        metricDebug = metricsResult.debug;
        if (telemetryActive) {
          metricsRef.current.push({
            backend: "gpu",
            ts: performance.now(),
            metrics: metricsResult.metrics,
            kernelVersion: kernelSnapshot.version
          });
          if (metricsRef.current.length > 240) {
            metricsRef.current.shift();
          }
          if (metricsResult.obsAverage != null) {
            lastObsRef.current = metricsResult.obsAverage;
          }
        }
      }
      updateDebugSnapshots(commitObs, rimDebugRequest, surfaceDebugRequest, metricDebug ?? undefined);
      updatePhaseDebug(commitObs, phaseField);

      renderer.uploadPhase(phaseField);
      renderer.uploadVolume(volumeField ?? null);

      const couplingScale = 1 + 0.65 * dmt;

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
        thetaMode: thetaMode === "gradient" ? 0 : 1,
        thetaGlobal,
        contrast,
        frameGain,
        rimAlpha,
        rimEnabled,
        warpAmp,
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
        orientations: orientationCache,
        ops: gpuOps,
        center: [cx, cy]
      });

      logFrameMetrics(tSeconds);

      if (telemetryActive) {
        recordTelemetry("renderGpu", performance.now() - renderStart);
      }

    },
    [
      rimFieldRef,
      surfaceFieldRef,
      basePixelsRef,
      dmt,
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
      surfaceBlend,
      surfaceRegion,
      kurEnabled,
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
      logFrameMetrics
    ]
  );

  const advanceVolume = useCallback(
    (dt: number) => {
      if (!volumeEnabled) return;
      ensureVolumeState();
      const stub = volumeStubRef.current;
      if (!stub) return;
      stepVolumeStub(stub, dt);
      const field = snapshotVolumeStub(stub);
      assertVolumeField(field, "volume:stub");
      volumeFieldRef.current = field;
      markFieldFresh("volume", field.resolution, "volume:stub");
    },
    [volumeEnabled, ensureVolumeState, markFieldFresh]
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
              kind: "tick",
              dt,
              timestamp: tSeconds,
              frameId
            });
            workerInflightRef.current = inflight + 1;
          }
        }
        swapWorkerFrame();
      }
      if (teleStart) {
        recordTelemetry("kuramoto", performance.now() - teleStart);
      }
    },
    [
      kurEnabled,
      stepKuramotoCpu,
      deriveKurFieldsCpu,
      swapWorkerFrame,
      recordTelemetry
    ]
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
        recordTelemetry("renderCpu", performance.now() - renderStart);
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
              3
            )}ms over ${count} frames`
          );
          profiler.samples = [];
          profiler.enabled = false;
        }
      }
      ctx.putImageData(buffer.image, 0, 0);
      logFrameMetrics(tSeconds);
      if (frameStart) {
        recordTelemetry("frame", performance.now() - frameStart);
      }
    },
    [
      advanceKuramoto,
      advanceVolume,
      speed,
      ensureFrameBuffer,
      renderFrameCore,
      recordTelemetry,
      logFrameMetrics
    ]
  );

  const runRegressionHarness = useCallback(
    async (frameCount = 10) => {
      if (!rimFieldRef.current || !surfaceFieldRef.current) {
        console.warn(
          "[regression] skipping: surface or rim field not ready."
        );
        return { maxDelta: 0, perFrameMax: [] as number[] };
      }
      if (!kurEnabled) {
        console.warn("[regression] Kuramoto disabled; nothing to compare.");
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
          controls: { dmt }
        });
        deriveKuramotoFieldsCore(cpuState, cpuDerived, {
          kernel: operatorKernel,
          controls: { dmt }
        });
        const buffer = new Uint8ClampedArray(total);
        const tSeconds = i * (1 / 60);
        renderFrameCore(buffer, tSeconds, false, cpuDerived);
        baselineFrames.push(buffer);
      }

      let simBuffers: ArrayBuffer[];
      try {
        simBuffers = await new Promise<ArrayBuffer[]>((resolve, reject) => {
          const worker = new Worker(
            new URL("./kuramotoWorker.ts", import.meta.url),
            { type: "module" }
          );
          worker.onmessage = (event: MessageEvent<WorkerIncomingMessage>) => {
            const msg = event.data;
            if (msg.kind === "simulateResult") {
              worker.terminate();
              resolve(msg.buffers);
            } else if (msg.kind === "log") {
              console.log(msg.message);
            }
          };
          worker.onerror = (error) => {
            worker.terminate();
            reject(error);
          };
          worker.postMessage({
            kind: "simulate",
            frameCount,
            dt,
            params,
            width,
            height,
            qInit,
            seed
          });
        });
      } catch (error) {
        console.error("[regression] worker simulation failed", error);
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
      console.log(
        `[regression] compared ${frameCount} frames, max normalized delta ${maxDelta}`
      );
      console.assert(
        maxDelta < 1e-6,
        `[regression] expected <=1e-6 delta, saw max ${maxDelta}`
      );
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
      dmt
    ]
  );

  const runGpuParityCheck = useCallback(async () => {
    if (!rimFieldRef.current || !basePixelsRef.current || !surfaceFieldRef.current) {
      console.warn("[gpu-regression] base pixels, surface, or rim data unavailable.");
      return null;
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      premultipliedAlpha: false,
      antialias: false,
      preserveDrawingBuffer: true
    });
    if (!gl) {
      console.warn("[gpu-regression] WebGL2 unavailable for parity check.");
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
            kind: "phase" as const,
            resolution: phaseSource.resolution,
            gradX: gradXRef.current,
            gradY: gradYRef.current,
            vort: vortRef.current,
            coh: cohRef.current,
            amp: ampRef.current
          }
        : null;
    if (phaseField) {
      assertPhaseField(phaseField, "gpu:parity-phase");
    }
    renderer.uploadPhase(phaseField);

    const state = { gl, renderer };
    const total = width * height * 4;
    const cpuBuffer = new Uint8ClampedArray(total);
    const gpuBuffer = new Uint8Array(total);
    const pixelCount = width * height;
    const scenes = [
      { label: "scene-A", time: 0.0 },
      { label: "scene-B", time: 0.37 },
      { label: "scene-C", time: 0.73 }
    ];
    const results: {
      label: string;
      mismatched: number;
      percent: number;
      maxDelta: number;
    }[] = [];
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
                kind: "phase" as const,
                resolution: phaseSource.resolution,
                gradX: gradXRef.current,
                gradY: gradYRef.current,
                vort: vortRef.current,
                coh: cohRef.current,
                amp: ampRef.current
              }
            : null;
        if (phaseUpload) {
          assertPhaseField(phaseUpload, "gpu:parity-phase-loop");
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
            ? ([
                worstIndex % width,
                Math.floor(worstIndex / width)
              ] as [number, number])
            : ([0, 0] as [number, number]);
        if (maxDelta > 1) {
          console.warn(
            `[gpu-regression] ${scene.label} worst Δ${maxDelta.toFixed(
              2
            )} at (${coord[0]},${coord[1]}) CPU ${printRgb(
              worstCpu
            )} GPU ${printRgb(worstGpu)}`
          );
        }
        results.push({
          label: scene.label,
          mismatched,
          percent: (mismatched / pixelCount) * 100,
          maxDelta,
          maxCoord: coord,
          cpuColor: worstCpu,
          gpuColor: worstGpu
        });
      }
    } finally {
      lastObsRef.current = prevLastObs;
      renderer.dispose();
    }

    return {
      scenes: results,
      tolerancePercent: 0.5
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
    drawFrameGpu
  ]);

  const measureRenderPerformance = useCallback(
    (frameCount = 60) => {
      if (!rimFieldRef.current || !basePixelsRef.current || !surfaceFieldRef.current) {
        console.warn("[perf] surface or rim field unavailable.");
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
        console.warn("[perf] GPU renderer unavailable.");
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
        throughputGain: cpuMs / gpuMs
      };
    },
    [rimFieldRef, surfaceFieldRef, basePixelsRef, width, height, renderFrameCore, ensureGpuRenderer, drawFrameGpu]
  );

  const handleParityCheck = useCallback(async () => {
    const result = await runGpuParityCheck();
    if (!result) {
      setLastParityResult(null);
      return;
    }
    setLastParityResult({
      ...result,
      timestamp: Date.now()
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
        timestamp: Date.now()
      });
    },
    [measureRenderPerformance]
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
      setRenderBackend(useGpu ? "gpu" : "cpu");
    },
    [setRenderBackend]
  );

  const parityDisplay = useMemo(() => {
    if (!lastParityResult || lastParityResult.scenes.length === 0) return null;
    const worst = lastParityResult.scenes.reduce(
      (max, scene) => (scene.percent > max.percent ? scene : max),
      lastParityResult.scenes[0]
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
      throughputGain: lastPerfResult.throughputGain
    };
  }, [lastPerfResult]);

  const isRecording = recordingStatus === "recording";
  const isFinalizing = recordingStatus === "finalizing";
  const mp4Unsupported =
    captureSupport.checked && captureSupport.best?.container !== "mp4";
  const captureReady =
    captureSupport.checked &&
    !!captureSupport.best &&
    captureSupport.best.container === "mp4";
  const captureButtonDisabled = isFinalizing || (!isRecording && !captureReady);
  const recordingBitrateMbps = useMemo(
    () => (recordingBitrate / 1_000_000).toFixed(1),
    [recordingBitrate]
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const mediaRecorderCtor = (window as typeof window & {
      MediaRecorder?: typeof MediaRecorder;
    }).MediaRecorder;
    if (!mediaRecorderCtor || typeof mediaRecorderCtor.isTypeSupported !== "function") {
      setCaptureSupport({ checked: true, best: null });
      return;
    }
    const candidates = [
      { mimeType: "video/mp4;codecs=avc1.4d402a", container: "mp4" as const },
      { mimeType: "video/mp4;codecs=avc1.4d401e", container: "mp4" as const },
      { mimeType: "video/mp4;codecs=h264", container: "mp4" as const },
      { mimeType: "video/mp4", container: "mp4" as const },
      { mimeType: "video/webm;codecs=vp9", container: "webm" as const },
      { mimeType: "video/webm;codecs=vp8", container: "webm" as const },
      { mimeType: "video/webm", container: "webm" as const }
    ];
    const supported = candidates.find((candidate) => {
      try {
        return mediaRecorderCtor.isTypeSupported(candidate.mimeType);
      } catch {
        return false;
      }
    });
    setCaptureSupport({ checked: true, best: supported ?? null });
  }, [setCaptureSupport]);

  const startMp4Capture = useCallback(() => {
    setRecordingError(null);
    if (recordingStatus !== "idle") {
      return;
    }
    if (!captureSupport.checked) {
      setRecordingError("Detecting capture support, please try again in a moment.");
      return;
    }
    if (!captureSupport.best) {
      setRecordingError("MediaRecorder is unavailable in this browser.");
      return;
    }
    if (captureSupport.best.container !== "mp4") {
      setRecordingError(
        "This browser cannot record MP4 directly. Try Safari 17+ or export WebM and convert with ffmpeg."
      );
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) {
      setRecordingError("Canvas is not ready yet.");
      return;
    }
    if (typeof canvas.captureStream !== "function") {
      setRecordingError("canvas.captureStream() is not supported in this environment.");
      return;
    }
    const mediaRecorderCtor = (window as typeof window & {
      MediaRecorder?: typeof MediaRecorder;
    }).MediaRecorder;
    if (!mediaRecorderCtor) {
      setRecordingError("MediaRecorder constructor is missing on window.");
      return;
    }
    if (recordedUrlRef.current) {
      URL.revokeObjectURL(recordedUrlRef.current);
      recordedUrlRef.current = null;
    }
    setRecordingDownload(null);
    recordingChunksRef.current = [];
    recordingMimeTypeRef.current = captureSupport.best.mimeType;
    try {
      const stream = canvas.captureStream(60);
      captureStreamRef.current = stream;
      const track = stream.getVideoTracks()[0];
      if (track && typeof track.applyConstraints === "function") {
        track
          .applyConstraints({
            frameRate: { ideal: 60, max: 60 }
          })
          .catch(() => {
            /* ignore */
          });
      }
      const recorder = new mediaRecorderCtor(stream, {
        mimeType: captureSupport.best.mimeType,
        videoBitsPerSecond: recordingBitrate
      });
      recorderRef.current = recorder;
      setRecordingStatus("recording");
      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          recordingChunksRef.current.push(event.data);
        }
      };
      recorder.onerror = (event) => {
        const err = (event as { error?: DOMException }).error;
        const message = err?.message ?? "Recorder error.";
        recordingChunksRef.current = [];
        recordingMimeTypeRef.current = null;
        stopCaptureStream();
        recorderRef.current = null;
        setRecordingStatus("idle");
        setRecordingError(message);
      };
      recorder.onstop = () => {
        const chunks = recordingChunksRef.current;
        recordingChunksRef.current = [];
        const mimeType =
          recordingMimeTypeRef.current ?? captureSupport.best?.mimeType ?? "video/mp4";
        recordingMimeTypeRef.current = null;
        stopCaptureStream();
        recorderRef.current = null;
        if (!chunks.length) {
          setRecordingStatus("idle");
          setRecordingError("Recorder produced no data.");
          return;
        }
        const blob = new Blob(chunks, { type: mimeType });
        const safeDate = new Date().toISOString().replace(/[:.]/g, "-");
        const extension = mimeType.includes("mp4") ? "mp4" : mimeType.includes("webm") ? "webm" : "mp4";
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
          filename
        });
        setRecordingStatus("idle");
      };
      recorder.start(1000);
    } catch (error) {
      recordingChunksRef.current = [];
      recordingMimeTypeRef.current = null;
      stopCaptureStream();
      recorderRef.current = null;
      setRecordingStatus("idle");
      setRecordingError(error instanceof Error ? error.message : String(error));
    }
  }, [
    captureSupport,
    canvasRef,
    recordingBitrate,
    recordingStatus,
    setRecordingDownload,
    stopCaptureStream
  ]);

  const stopMp4Capture = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder) {
      return;
    }
    if (recorder.state === "inactive") {
      stopCaptureStream();
      recorderRef.current = null;
      return;
    }
    setRecordingStatus("finalizing");
    try {
      recorder.stop();
    } catch (error) {
      stopCaptureStream();
      recorderRef.current = null;
      recordingChunksRef.current = [];
      recordingMimeTypeRef.current = null;
      setRecordingStatus("idle");
      setRecordingError(error instanceof Error ? error.message : String(error));
    }
  }, [stopCaptureStream]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const w = window as any;
    w.__setFrameProfiler = setFrameProfiler;
    w.__runFrameRegression = runRegressionHarness;
    w.__runGpuParityCheck = runGpuParityCheck;
    w.__measureRenderPerformance = measureRenderPerformance;
    const setBackend = (backend: "gpu" | "cpu") => {
      handleRendererToggle(backend === "gpu");
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
    setTelemetryEnabled
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let anim = true;
    let frameId = 0;
    const start = performance.now();

    if (renderBackend === "gpu") {
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
          recordTelemetry("frame", performance.now() - frameStart);
        }
        frameId = requestAnimationFrame(render);
      };
      render();
      return () => {
        anim = false;
        cancelAnimationFrame(frameId);
      };
    }

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
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
    recordTelemetry
  ]);

  useEffect(() => {
    return () => {
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") {
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

  const ingestImageData = useCallback(
    (image: ImageData, source: string) => {
      const newW = image.width;
      const newH = image.height;
      setWidth(newW);
      setHeight(newH);
      basePixelsRef.current = image;
      const surfaceField = describeImageData(image);
      assertSurfaceField(surfaceField, "io:surface");
      surfaceFieldRef.current = surfaceField;
      markFieldFresh("surface", surfaceField.resolution, source);

      const rimField = computeEdgeField({
        data: image.data,
        width: newW,
        height: newH
      });
      assertRimField(rimField, "io:rim");
      rimFieldRef.current = rimField;
      markFieldFresh("rim", rimField.resolution, source);
      pendingStaticUploadRef.current = true;
      const gpuState = gpuStateRef.current;
      if (gpuState) {
        gpuState.renderer.resize(newW, newH);
      }
      refreshGpuStaticTextures();
      normTargetRef.current = 0.6;
      lastObsRef.current = 0.6;
    },
    [markFieldFresh, refreshGpuStaticTextures]
  );

  const loadSyntheticCase = useCallback(
    (caseId: SyntheticCaseId) => {
      const synthetic = SYNTHETIC_CASES.find((entry) => entry.id === caseId);
      if (!synthetic) {
        console.warn(`[synthetic] unknown case ${caseId}`);
        return;
      }
      setSelectedSyntheticCase(caseId);
      const { width: defaultW, height: defaultH } = DEFAULT_SYNTHETIC_SIZE;
      const image = synthetic.generate(defaultW, defaultH);
      ingestImageData(image, `dev:${caseId}`);
      if (kurEnabled) {
        deriveKurFieldsCpu();
      }
      if (!metricsScratchRef.current || metricsScratchRef.current.length !== defaultW * defaultH * 4) {
        metricsScratchRef.current = new Uint8ClampedArray(defaultW * defaultH * 4);
      }
      const scratch = metricsScratchRef.current;
      const metrics = renderFrameCore(scratch, 0, false);
      if (metrics) {
        setSyntheticBaselines((prev) => ({
          ...prev,
          [caseId]: {
            metrics,
            timestamp: Date.now()
          }
        }));
        const rimEnergy = metrics.composer.fields.rim.energy.toFixed(3);
        const surfaceEnergy = metrics.composer.fields.surface.energy.toFixed(3);
        const cohMean = metrics.gradient.cohMean.toFixed(3);
        console.log(
          `[synthetic] ${synthetic.label} rim=${rimEnergy} surface=${surfaceEnergy} |Z|=${cohMean}`
        );
      }
    },
    [deriveKurFieldsCpu, ingestImageData, kurEnabled, renderFrameCore, setSelectedSyntheticCase]
  );

  const exportCouplingDiff = useCallback(
    (branch: "rimToSurface" | "surfaceToRim") => {
      if (typeof document === "undefined") {
        console.warn("[coupling-diff] document unavailable for export");
        return;
      }
      if (!basePixelsRef.current || !rimFieldRef.current) {
        console.warn("[coupling-diff] base image or rim field not ready");
        return;
      }
      const total = width * height * 4;
      const currentBuffer = new Uint8ClampedArray(total);
      const toggledBuffer = new Uint8ClampedArray(total);
      const override: CouplingToggleState = {
        rimToSurface: branch === "rimToSurface" ? false : couplingToggles.rimToSurface,
        surfaceToRim: branch === "surfaceToRim" ? false : couplingToggles.surfaceToRim
      };
      const baselineMetrics = renderFrameCore(currentBuffer, 0, false);
      const toggledMetrics = renderFrameCore(toggledBuffer, 0, false, undefined, {
        toggles: override
      });
      if (!baselineMetrics || !toggledMetrics) {
        console.warn("[coupling-diff] unable to compute frame metrics");
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
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        console.warn("[coupling-diff] failed to create canvas context");
        return;
      }
      const diffImage = new ImageData(diff, width, height);
      ctx.putImageData(diffImage, 0, 0);
      const url = canvas.toDataURL("image/png");
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `coupling-diff-${branch}-${timestamp}.png`;
      anchor.click();
      console.log(
        `[coupling-diff] ${branch} maxΔ=${maxDelta.toFixed(1)} rim=${baselineMetrics.composer.fields.rim.energy.toFixed(3)}→${toggledMetrics.composer.fields.rim.energy.toFixed(3)} surface=${baselineMetrics.composer.fields.surface.energy.toFixed(3)}→${toggledMetrics.composer.fields.surface.energy.toFixed(3)}`
      );
    },
    [couplingToggles, renderFrameCore, width, height]
  );

  const runSyntheticDeck = useCallback(() => {
    SYNTHETIC_CASES.forEach((entry) => loadSyntheticCase(entry.id));
  }, [loadSyntheticCase]);

  const onFile = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      const bitmap = await createImageBitmap(file);
      const maxDim = 1000;
      const scale = Math.min(
        1,
        maxDim / Math.max(bitmap.width, bitmap.height)
      );
      const newW = Math.max(1, Math.round(bitmap.width * scale));
      const newH = Math.max(1, Math.round(bitmap.height * scale));
      setWidth(newW);
      setHeight(newH);
      setImgBitmap(bitmap);

      const off = document.createElement("canvas");
      off.width = newW;
      off.height = newH;
      const octx = off.getContext("2d", { willReadFrequently: true });
      if (!octx) return;
      octx.drawImage(bitmap, 0, 0, newW, newH);
      const img = octx.getImageData(0, 0, newW, newH);
      ingestImageData(img, "io:file");
    },
    [ingestImageData]
  );

  const applyPreset = useCallback(
    (preset: Preset) => {
      const v = preset.params;
      pendingStaticUploadRef.current = true;
      setRenderBackend("gpu");
      setVolumeEnabled(false);
      setEdgeThreshold(v.edgeThreshold);
      setBlend(v.blend);
      setKernel(createKernelSpec(v.kernel));
      setDmt(v.dmt);
      setThetaMode(v.thetaMode);
      setThetaGlobal(v.thetaGlobal);
      setBeta2(v.beta2);
      setJitter(v.jitter);
      setSigma(v.sigma);
      setMicrosaccade(v.microsaccade);
      setSpeed(v.speed);
      setContrast(v.contrast);
      setCoupling(cloneCouplingConfig(preset.coupling));
      setComposer(cloneComposerConfig(preset.composer));
      setCouplingToggles(
        preset.couplingToggles
          ? cloneCouplingToggles(preset.couplingToggles)
          : DEFAULT_COUPLING_TOGGLES
      );
    },
    []
  );

  const buildCurrentPreset = useCallback((): Preset => ({
    name: PRESETS[presetIndex]?.name ?? "Custom snapshot",
    params: {
      edgeThreshold,
      blend,
      kernel: kernelSpecToJSON(kernel),
      dmt,
      thetaMode,
      thetaGlobal,
      beta2,
      jitter,
      sigma,
      microsaccade,
      speed,
      contrast
    },
    coupling: cloneCouplingConfig(coupling),
    composer: cloneComposerConfig(composer),
    couplingToggles: cloneCouplingToggles(couplingToggles)
  }), [
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
    coupling,
    composer,
    couplingToggles
  ]);

  const handlePresetExport = useCallback(() => {
    const preset = buildCurrentPreset();
    const payload = { version: 2, preset };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const downloadName = `${preset.name.toLowerCase().replace(/[^a-z0-9]+/g, "-") || "preset"}.json`;
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = downloadName;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [buildCurrentPreset]);

  const handlePresetImport = useCallback(() => {
    const input = presetFileInputRef.current;
    if (!input) return;
    input.value = "";
    input.click();
  }, []);

  const handlePresetImportFile = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = "";
      if (!file) return;
      try {
        const text = await file.text();
        const parsed = JSON.parse(text);
        const payload = parsed?.preset ?? parsed;
        if (!payload?.params) {
          console.error("[preset] invalid payload");
          return;
        }
        const importedPreset: Preset = {
          name: typeof payload.name === "string" ? payload.name : "Imported preset",
          params: {
            edgeThreshold: typeof payload.params.edgeThreshold === "number" ? payload.params.edgeThreshold : edgeThreshold,
            blend: typeof payload.params.blend === "number" ? payload.params.blend : blend,
            kernel: createKernelSpec({
              gain: payload.params.kernel?.gain ?? kernel.gain,
              k0: payload.params.kernel?.k0 ?? kernel.k0,
              Q: payload.params.kernel?.Q ?? kernel.Q,
              anisotropy: payload.params.kernel?.anisotropy ?? kernel.anisotropy,
              chirality: payload.params.kernel?.chirality ?? kernel.chirality,
              transparency: payload.params.kernel?.transparency ?? kernel.transparency
            }),
            dmt: typeof payload.params.dmt === "number" ? payload.params.dmt : dmt,
            thetaMode: payload.params.thetaMode === "global" ? "global" : "gradient",
            thetaGlobal: typeof payload.params.thetaGlobal === "number" ? payload.params.thetaGlobal : thetaGlobal,
            beta2: typeof payload.params.beta2 === "number" ? payload.params.beta2 : beta2,
            jitter: typeof payload.params.jitter === "number" ? payload.params.jitter : jitter,
            sigma: typeof payload.params.sigma === "number" ? payload.params.sigma : sigma,
            microsaccade: payload.params.microsaccade ?? microsaccade,
            speed: typeof payload.params.speed === "number" ? payload.params.speed : speed,
            contrast: typeof payload.params.contrast === "number" ? payload.params.contrast : contrast
          },
          coupling: sanitizeCouplingConfig(payload.coupling, coupling),
          composer: sanitizeComposerImport(payload.composer),
          couplingToggles: {
            rimToSurface:
              payload.couplingToggles?.rimToSurface !== undefined
                ? Boolean(payload.couplingToggles.rimToSurface)
                : true,
            surfaceToRim:
              payload.couplingToggles?.surfaceToRim !== undefined
                ? Boolean(payload.couplingToggles.surfaceToRim)
                : true
          }
        };
        applyPreset(importedPreset);
      } catch (error) {
        console.error("[preset] failed to import", error);
      }
    },
    [
      applyPreset,
      coupling,
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
      contrast
    ]
  );

  useEffect(() => {
    applyPreset(PRESETS[presetIndex]);
  }, [presetIndex, applyPreset]);

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
        height: "100vh",
        padding: "2rem",
        display: "flex",
        flexDirection: "column",
        gap: "1.5rem",
        overflow: "hidden"
      }}
    >
      <div>
        <h1 style={{ margin: 0, fontSize: "2.2rem" }}>
          Rainbow Perimeter Lab
        </h1>
        <p style={{ maxWidth: "52rem", color: "#94a3b8" }}>
          Upload a photo, apply the Rainbow Rims preset, then fine tune the
          kernel, DMT gain, surface wallpaper morph, and Kuramoto coupling
          to explore the hallucinatory perimeter lines described in the
          design brief.
        </p>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(22rem, 27rem) 1fr",
          gap: "1.5rem",
          alignItems: "stretch",
          flex: 1,
          minHeight: 0
        }}
      >
        <div
          className="panel"
          style={{
            gap: "1.25rem",
            overflowY: "auto",
            height: "100%",
            minHeight: 0,
            paddingRight: "1.25rem"
          }}
        >
          <section className="panel">
            <h2>Image & Preset</h2>
            <div className="control">
              <label htmlFor="file-input">Upload image</label>
              <input
                id="file-input"
                type="file"
                accept="image/*"
                onChange={onFile}
              />
            </div>
            <SelectControl
              label="Preset"
              value={presetIndex.toString()}
              onChange={(v) => setPresetIndex(parseInt(v, 10))}
              options={PRESETS.map((p, i) => ({
                value: i.toString(),
                label: p.name
              }))}
            />
            <button
              onClick={() => applyPreset(PRESETS[presetIndex])}
              style={{
                padding: "0.5rem 0.75rem",
                borderRadius: "0.6rem",
                border: "1px solid rgba(148,163,184,0.35)",
                background: "rgba(14, 116, 144, 0.2)",
                color: "#f8fafc",
                cursor: "pointer"
              }}
            >
              Apply preset
            </button>
            <div
              style={{
                display: "flex",
                gap: "0.5rem",
                flexWrap: "wrap"
              }}
            >
              <button
                onClick={handlePresetExport}
                style={{
                  padding: "0.4rem 0.65rem",
                  borderRadius: "0.55rem",
                  border: "1px solid rgba(148,163,184,0.35)",
                  background: "rgba(15,118,110,0.18)",
                  color: "#e0f2fe",
                  cursor: "pointer"
                }}
              >
                Export preset
              </button>
              <button
                onClick={handlePresetImport}
                style={{
                  padding: "0.4rem 0.65rem",
                  borderRadius: "0.55rem",
                  border: "1px solid rgba(148,163,184,0.35)",
                  background: "rgba(30,64,175,0.18)",
                  color: "#e0e7ff",
                  cursor: "pointer"
                }}
              >
                Import preset
              </button>
              <input
                ref={presetFileInputRef}
                type="file"
                accept="application/json"
                style={{ display: "none" }}
                onChange={handlePresetImportFile}
              />
            </div>
          </section>

          <section className="panel">
            <h2>Diagnostics</h2>
            <ToggleControl
              label="Use GPU renderer"
              value={renderBackend === "gpu"}
              onChange={handleRendererToggle}
            />
            <ToggleControl
              label="Telemetry logging"
              value={telemetryEnabled}
              onChange={setTelemetryEnabled}
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
                  padding: "0.5rem 0.75rem",
                  borderRadius: "0.6rem",
                  border: "1px solid rgba(148,163,184,0.35)",
                  background: "rgba(147, 197, 253, 0.18)",
                  color: "#f8fafc",
                  cursor: "pointer"
                }}
              >
                Run GPU parity check
              </button>
              {parityDisplay && (
                <small
                  style={{
                    display: "block",
                    marginTop: "0.4rem",
                    color: parityDisplay.within ? "#38bdf8" : "#f87171"
                  }}
                >
                  worst {parityDisplay.worst.label}: Δ<span style={{ fontFamily: "monospace" }}>
                    {parityDisplay.worst.maxDelta.toFixed(2)}
                  </span>{" "}
                  | {parityDisplay.worst.percent.toFixed(2)}% &lt;= {parityDisplay.tolerance.toFixed(2)}%?{" "}
                  {parityDisplay.within ? "OK" : "Check reference"}
                  {!parityDisplay.within && (
                    <>
                      <br />
                      <span>
                        pixel ({parityDisplay.worst.maxCoord[0]}, {parityDisplay.worst.maxCoord[1]}) CPU{" "}
                        {printRgb(parityDisplay.worst.cpuColor)} vs GPU{" "}
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
                  padding: "0.5rem 0.75rem",
                  borderRadius: "0.6rem",
                  border: "1px solid rgba(148,163,184,0.35)",
                  background: "rgba(94, 234, 212, 0.18)",
                  color: "#f8fafc",
                  cursor: "pointer"
                }}
              >
                Measure render throughput (120 frames)
              </button>
              {perfDisplay && (
                <small
                  style={{
                    display: "block",
                    marginTop: "0.4rem",
                    color: "#94a3b8"
                  }}
                >
                  GPU {perfDisplay.gpuFps.toFixed(0)} fps (~{perfDisplay.throughputGain.toFixed(1)}× vs{" "}
                  {perfDisplay.cpuMs.toFixed(1)} ms CPU)
                </small>
              )}
            </div>
            {(showRimDebug && rimDebugSnapshot) ||
            (showSurfaceDebug && surfaceDebugSnapshot) ||
            (showPhaseDebug && phaseDebugSnapshot) ? (
              <div
                className="control"
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.75rem",
                  background: "rgba(15,23,42,0.45)",
                  borderRadius: "0.85rem",
                  padding: "0.75rem"
                }}
              >
                {showRimDebug && rimDebugSnapshot && (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
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
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        fontSize: "0.75rem",
                        color: "#cbd5f5"
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
                marginTop: "0.75rem",
                display: "grid",
                gap: "0.4rem"
              }}
            >
              {fieldStatusEntries.map((entry) => {
                const palette = FIELD_STATUS_STYLES[entry.state];
                const statusLabel = FIELD_STATUS_LABELS[entry.state];
                const resolutionText = entry.resolution
                  ? `${entry.resolution.width}×${entry.resolution.height}`
                  : "—";
                const stalenessText =
                  entry.state === "missing"
                    ? "offline"
                    : entry.stalenessMs === Number.POSITIVE_INFINITY
                    ? "idle"
                    : `${entry.stalenessMs.toFixed(0)}ms`;
                return (
                  <div
                    key={entry.kind}
                    style={{
                      border: `1px solid ${palette.border}`,
                      background: palette.background,
                      color: palette.color,
                      borderRadius: "0.65rem",
                      padding: "0.45rem 0.6rem",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "baseline"
                    }}
                  >
                    <div style={{ fontWeight: 600 }}>{entry.label}</div>
                    <div style={{ fontSize: "0.75rem", fontFamily: "monospace" }}>
                      {statusLabel} · {resolutionText} · {stalenessText}
                    </div>
                  </div>
                );
              })}
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
                display: "flex",
                gap: "0.5rem",
                flexWrap: "wrap"
              }}
            >
              <button
                onClick={() => loadSyntheticCase(selectedSyntheticCase)}
                style={{
                  padding: "0.4rem 0.65rem",
                  borderRadius: "0.55rem",
                  border: "1px solid rgba(148,163,184,0.35)",
                  background: "rgba(30, 64, 175, 0.18)",
                  color: "#e0e7ff",
                  cursor: "pointer"
                }}
              >
                Load selected case
              </button>
              <button
                onClick={runSyntheticDeck}
                style={{
                  padding: "0.4rem 0.65rem",
                  borderRadius: "0.55rem",
                  border: "1px solid rgba(148,163,184,0.35)",
                  background: "rgba(59, 130, 246, 0.2)",
                  color: "#e0f2fe",
                  cursor: "pointer"
                }}
              >
                Run full deck
              </button>
            </div>
            <small style={{ display: "block", color: "#94a3b8", marginTop: "0.35rem" }}>
              Records rim, surface, and |Z| baselines for analytics.
            </small>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "0.5rem",
                marginTop: "0.75rem"
              }}
            >
              {SYNTHETIC_CASES.map((entry) => {
                const baseline = syntheticBaselines[entry.id];
                return (
                  <div
                    key={entry.id}
                    style={{
                      border: "1px solid rgba(148,163,184,0.3)",
                      borderRadius: "0.65rem",
                      background: "rgba(15,23,42,0.4)",
                      padding: "0.55rem 0.65rem"
                    }}
                  >
                    <div style={{ fontWeight: 600, color: "#e2e8f0" }}>{entry.label}</div>
                    <div style={{ fontSize: "0.75rem", color: "#94a3b8", marginTop: "0.25rem" }}>
                      {entry.description}
                    </div>
                    {baseline ? (
                      <div
                        style={{
                          marginTop: "0.4rem",
                          fontFamily: "monospace",
                          fontSize: "0.8rem",
                          color: "#a5b4fc"
                        }}
                      >
                        rim {baseline.metrics.composer.fields.rim.energy.toFixed(3)} · surface {baseline.metrics.composer.fields.surface.energy.toFixed(3)} · |Z| {baseline.metrics.gradient.cohMean.toFixed(3)} · {new Date(baseline.timestamp).toLocaleTimeString()}
                      </div>
                    ) : (
                      <div
                        style={{
                          marginTop: "0.4rem",
                          fontFamily: "monospace",
                          fontSize: "0.8rem",
                          color: "#64748b"
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
              <label>MP4 capture</label>
              <div className="control-row">
                <button
                  onClick={() =>
                    isRecording ? stopMp4Capture() : startMp4Capture()
                  }
                  disabled={captureButtonDisabled}
                  style={{
                    padding: "0.5rem 0.75rem",
                    borderRadius: "0.6rem",
                    border: "1px solid rgba(148,163,184,0.35)",
                    background: isRecording
                      ? "rgba(248, 113, 113, 0.25)"
                      : "rgba(59, 130, 246, 0.25)",
                    color: "#f8fafc",
                    cursor: captureButtonDisabled ? "not-allowed" : "pointer",
                    opacity: captureButtonDisabled ? 0.6 : 1
                  }}
                >
                  {isRecording
                    ? "Stop capture"
                    : isFinalizing
                    ? "Finalizing…"
                    : "Start MP4 capture"}
                </button>
                {(isRecording || isFinalizing) && (
                  <span style={{ color: "#fbbf24" }}>
                    {isRecording ? "Recording…" : "Finalizing…"}
                  </span>
                )}
              </div>
              {!captureSupport.checked && (
                <small style={{ color: "#94a3b8" }}>
                  Checking browser capture support…
                </small>
              )}
              <small style={{ color: "#94a3b8" }}>
                Canvas {width}×{height}px · 60 fps · target {recordingBitrateMbps} Mbps
              </small>
              {recordingError && (
                <small style={{ color: "#f87171" }}>{recordingError}</small>
              )}
              {mp4Unsupported && (
                <small style={{ color: "#f97316" }}>
                  MP4 capture unsupported (best available:{" "}
                  {captureSupport.best ? captureSupport.best.mimeType : "none"}).
                </small>
              )}
              {recordingDownload && (
                <small style={{ color: "#38bdf8" }}>
                  MP4 ready:{" "}
                  <a
                    href={recordingDownload.url}
                    download={recordingDownload.filename}
                    style={{ color: "#38bdf8" }}
                  >
                    Download ({formatBytes(recordingDownload.size)})
                  </a>
                </small>
              )}
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
            <ToggleControl
              label="Zero-mean pin"
              value={phasePin}
              onChange={setPhasePin}
            />
            <ToggleControl
              label="Alive microbreath"
              value={alive}
              onChange={setAlive}
            />
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
            <SliderControl
              label="DMT Gain"
              value={dmt}
              min={0}
              max={1}
              step={0.01}
              onChange={setDmt}
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
                { value: "off", label: "Off" },
                { value: "p2", label: "p2" },
                { value: "p4", label: "p4" },
                { value: "p6", label: "p6" },
                { value: "pmm", label: "pmm" },
                { value: "p4m", label: "p4m" }
              ]}
            />
            <SelectControl
              label="Region"
              value={surfaceRegion}
              onChange={(v) => setSurfaceRegion(v as SurfaceRegion)}
              options={[
                { value: "surfaces", label: "Surfaces" },
                { value: "edges", label: "Edges" },
                { value: "both", label: "Both" }
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
              onChange={setCouplingToggle("rimToSurface")}
            />
            <ToggleControl
              label="Enable surface → rims coupling"
              value={couplingToggles.surfaceToRim}
              onChange={setCouplingToggle("surfaceToRim")}
            />
            <h3>Rims → Surface</h3>
            <SliderControl
              label="Energy → surface blend"
              value={coupling.rimToSurfaceBlend}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue("rimToSurfaceBlend")}
              format={(v) => v.toFixed(2)}
            />
            <SliderControl
              label="Tangent alignment weight"
              value={coupling.rimToSurfaceAlign}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue("rimToSurfaceAlign")}
              format={(v) => v.toFixed(2)}
            />
            <h3>Surface → Rims</h3>
            <SliderControl
              label="Warp gradient → offset bias"
              value={coupling.surfaceToRimOffset}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue("surfaceToRimOffset")}
              format={(v) => v.toFixed(2)}
            />
            <SliderControl
              label="Warp gradient → sigma thinning"
              value={coupling.surfaceToRimSigma}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue("surfaceToRimSigma")}
              format={(v) => v.toFixed(2)}
            />
            <SliderControl
              label="Lattice phase → hue bias"
              value={coupling.surfaceToRimHue}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue("surfaceToRimHue")}
              format={(v) => v.toFixed(2)}
            />
            <div
              style={{
                display: "flex",
                gap: "0.5rem",
                flexWrap: "wrap",
                margin: "0.5rem 0 1rem"
              }}
            >
              <button
                onClick={() => exportCouplingDiff("rimToSurface")}
                style={{
                  padding: "0.4rem 0.65rem",
                  borderRadius: "0.55rem",
                  border: "1px solid rgba(148,163,184,0.35)",
                  background: "rgba(59, 130, 246, 0.18)",
                  color: "#e0f2fe",
                  cursor: "pointer"
                }}
              >
                Export rim→surface diff
              </button>
              <button
                onClick={() => exportCouplingDiff("surfaceToRim")}
                style={{
                  padding: "0.4rem 0.65rem",
                  borderRadius: "0.55rem",
                  border: "1px solid rgba(148,163,184,0.35)",
                  background: "rgba(14, 165, 233, 0.18)",
                  color: "#cffafe",
                  cursor: "pointer"
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
              onChange={setCouplingValue("kurToTransparency")}
              format={(v) => v.toFixed(2)}
            />
            <SliderControl
              label="∇θ → orientation blend"
              value={coupling.kurToOrientation}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue("kurToOrientation")}
              format={(v) => v.toFixed(2)}
            />
            <SliderControl
              label="Vorticity → chirality"
              value={coupling.kurToChirality}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue("kurToChirality")}
              format={(v) => v.toFixed(2)}
            />
            <h3>Volume → 2D</h3>
            <SliderControl
              label="Phase → rim hue"
              value={coupling.volumePhaseToHue}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue("volumePhaseToHue")}
              format={(v) => v.toFixed(2)}
            />
            <SliderControl
              label="Depth grad → warp amp"
              value={coupling.volumeDepthToWarp}
              min={0}
              max={1}
              step={0.01}
              onChange={setCouplingValue("volumeDepthToWarp")}
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
                { value: "auto", label: "Auto" },
                { value: "rimBias", label: "Rim bias" },
                { value: "surfaceBias", label: "Surface bias" }
              ]}
            />
            <SelectControl
              label="Solver regime"
              value={composer.solverRegime}
              onChange={(value) => handleComposerSolver(value as SolverRegime)}
              options={[
                { value: "balanced", label: "Balanced" },
                { value: "rimLocked", label: "Rim locked" },
                { value: "surfaceLocked", label: "Surface locked" }
              ]}
            />
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "1rem"
              }}
            >
              {COMPOSER_FIELD_LIST.map((field) => {
                const cfg = composer.fields[field];
                const label = COMPOSER_FIELD_LABELS[field];
                return (
                  <div key={field} style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    <h3 style={{ margin: 0 }}>{label}</h3>
                    <SliderControl
                      label="Exposure"
                      value={cfg.exposure}
                      min={0}
                      max={4}
                      step={0.05}
                      onChange={setComposerFieldValue(field, "exposure")}
                      format={(v) => v.toFixed(2)}
                    />
                    <SliderControl
                      label="Gamma"
                      value={cfg.gamma}
                      min={0.2}
                      max={3}
                      step={0.05}
                      onChange={setComposerFieldValue(field, "gamma")}
                      format={(v) => v.toFixed(2)}
                    />
                    <SliderControl
                      label="Weight"
                      value={cfg.weight}
                      min={0}
                      max={2.5}
                      step={0.05}
                      onChange={setComposerFieldValue(field, "weight")}
                      format={(v) => v.toFixed(2)}
                    />
                  </div>
                );
              })}
            </div>
          </section>

          <section className="panel">
            <h2>Kuramoto Field</h2>
            <ToggleControl
              label="Enable OA field"
              value={kurEnabled}
              onChange={setKurEnabled}
            />
            <ToggleControl
              label="Sync to main thread"
              value={kurSync}
              onChange={setKurSync}
            />
            <SelectControl
              label="Regime"
              value={kurRegime}
              onChange={(value) => {
                const next = value as KurRegime;
                if (next === "custom") {
                  setKurRegime("custom");
                } else {
                  applyKurRegime(next);
                }
              }}
              options={[
                { value: "locked", label: KUR_REGIME_PRESETS.locked.label },
                { value: "highEnergy", label: KUR_REGIME_PRESETS.highEnergy.label },
                { value: "chaotic", label: KUR_REGIME_PRESETS.chaotic.label },
                { value: "custom", label: "Custom (manual)" }
              ]}
            />
            {kurRegime !== "custom" && (
              <small style={{ display: "block", color: "#94a3b8", marginTop: "0.35rem" }}>
                {KUR_REGIME_PRESETS[kurRegime as Exclude<KurRegime, "custom">].description}
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
                padding: "0.5rem 0.75rem",
                borderRadius: "0.6rem",
                border: "1px solid rgba(148,163,184,0.35)",
                background: "rgba(30, 64, 175, 0.25)",
                color: "#f8fafc",
                cursor: "pointer"
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
                { value: "color", label: "Color base + color rims" },
                {
                  value: "grayBaseColorRims",
                  label: "Gray base + color rims"
                },
                {
                  value: "grayBaseGrayRims",
                  label: "Gray base + gray rims"
                },
                {
                  value: "colorBaseGrayRims",
                  label: "Color base + gray rims"
                },
                {
                  value: "colorBaseBlendedRims",
                  label: "Color base + blended rims"
                }
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
              onChange={setNormPin}
            />
          </section>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "1rem",
            overflowY: "auto",
            height: "100%",
            minHeight: 0,
            paddingRight: "0.75rem"
          }}
        >
          <div
            style={{
              position: "relative",
              width: "100%",
              maxWidth: `${Math.min(width, 1000)}px`,
              maxHeight: `${Math.min(height, 1000)}px`,
              aspectRatio: `${width} / ${height}`,
              borderRadius: "1.25rem",
              border: "1px solid rgba(148,163,184,0.2)",
              boxShadow: "0 20px 60px rgba(15,23,42,0.65)",
              overflow: "hidden"
            }}
          >
            <canvas
              ref={canvasRef}
              width={width}
              height={height}
              style={{
                width: "100%",
                height: "100%",
                display: "block"
              }}
            />
            <div
              style={{
                position: "absolute",
                top: "0.75rem",
                left: "0.75rem",
                padding: "0.25rem 0.6rem",
                borderRadius: "9999px",
                background: "rgba(15,23,42,0.65)",
                color: "#e2e8f0",
                fontSize: "0.75rem",
                letterSpacing: "0.02em",
                fontWeight: 500,
                pointerEvents: "none"
              }}
            >
              {`Flux (φₓ=${fluxX.toFixed(2)}, φ_y=${fluxY.toFixed(2)})`}
            </div>
          </div>
          <p style={{ color: "#64748b", margin: 0 }}>
            Tip: enable the OA field once you have rims dialed in, then
            gradually increase K₀ and lower ε to let the oscillator field
            steer both rims and surface morph in a coherent way.
          </p>
          <section className="panel">
            <h2>Telemetry</h2>
            {telemetrySnapshot ? (
              <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(11rem, 1fr))",
                    gap: "0.75rem"
                  }}
                >
                  {COMPOSER_FIELD_LIST.map((field) => {
                    const data = telemetrySnapshot.fields[field];
                    return (
                      <div
                        key={field}
                        style={{
                          background: "rgba(15,23,42,0.45)",
                          borderRadius: "0.75rem",
                          padding: "0.75rem",
                          display: "flex",
                          flexDirection: "column",
                          gap: "0.35rem"
                        }}
                      >
                        <strong style={{ color: "#e2e8f0" }}>{COMPOSER_FIELD_LABELS[field]}</strong>
                        <span style={{ fontSize: "0.8rem", color: "#a5b4fc" }}>
                          Energy {data.energy.toFixed(3)}
                        </span>
                        <span style={{ fontSize: "0.8rem", color: "#cbd5f5" }}>
                          Blend {(data.blend * 100).toFixed(1)}%
                        </span>
                        <span style={{ fontSize: "0.8rem", color: "#cbd5f5" }}>
                          Share {(data.share * 100).toFixed(1)}%
                        </span>
                        <span style={{ fontSize: "0.8rem", color: "#94a3b8" }}>
                          Weight {data.weight.toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <div
                  style={{
                    background: "rgba(15,23,42,0.45)",
                    borderRadius: "0.75rem",
                    padding: "0.75rem",
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.5rem"
                  }}
                >
                  <div style={{ color: "#e2e8f0", fontWeight: 600 }}>
                    Coupling scale {telemetrySnapshot.coupling.scale.toFixed(2)}
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fit, minmax(13rem, 1fr))",
                      gap: "0.5rem"
                    }}
                  >
                    {Object.entries(telemetrySnapshot.coupling.effective).map(([key, value]) => (
                      <div key={key} style={{ fontSize: "0.75rem", color: "#cbd5f5" }}>
                        <strong style={{ color: "#facc15" }}>{formatCouplingKey(key as keyof CouplingConfig)}</strong>
                        : {value.toFixed(2)}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <p style={{ color: "#94a3b8", margin: 0 }}>
                {telemetryEnabled ? "Collecting telemetry…" : "Telemetry disabled."}
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

type TelemetrySnapshot = {
  fields: Record<ComposerFieldId, TelemetryFieldSnapshot>;
  coupling: {
    scale: number;
    base: CouplingConfig;
    effective: CouplingConfig;
  };
  updatedAt: number;
};

const HistogramPanel = ({
  title,
  bins,
  defaultColor = "#38bdf8",
  colorForBin,
  rangeLabel
}: HistogramPanelProps) => {
  const max = bins.reduce((acc, value) => (value > acc ? value : acc), 0);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.45rem" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: "0.75rem",
          color: "#cbd5f5"
        }}
      >
        <span>{title}</span>
        {rangeLabel ? <span>{rangeLabel}</span> : null}
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${bins.length}, 1fr)`,
          alignItems: "end",
          gap: "2px",
          height: "64px",
          background: "rgba(15,23,42,0.45)",
          borderRadius: "0.6rem",
          padding: "6px"
        }}
      >
        {bins.map((value, idx) => {
          const normalized = max > 0 ? value / max : 0;
          const color = colorForBin
            ? colorForBin(idx, value, max)
            : defaultColor;
          return (
            <span
              key={idx}
              style={{
                display: "block",
                width: "100%",
                height: `${Math.max(normalized * 100, 2)}%`,
                background: color,
                borderRadius: "0.35rem 0.35rem 0 0"
              }}
            />
          );
        })}
      </div>
    </div>
  );
};

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
    const ctx = canvas.getContext("2d");
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
    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: "0.75rem",
          color: "#cbd5f5"
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
          width: "160px",
          height: "160px",
          borderRadius: "0.6rem",
          border: "1px solid rgba(148,163,184,0.35)",
          background: "rgba(15,23,42,0.35)"
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
  format
}: SliderProps) {
  return (
    <div className="control">
      <label>{label}</label>
      <div className="control-row">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
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
};

function ToggleControl({ label, value, onChange }: ToggleProps) {
  return (
    <div className="control">
      <label>{label}</label>
      <div className="control-row">
        <input
          type="checkbox"
          checked={value}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span>{value ? "On" : "Off"}</span>
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
          padding: "0.45rem 0.6rem",
          borderRadius: "0.6rem",
          border: "1px solid rgba(148,163,184,0.35)",
          background: "rgba(15,23,42,0.7)",
          color: "#e2e8f0"
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
