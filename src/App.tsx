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
  type KuramotoDerived,
  type KuramotoParams,
  type KuramotoState
} from "./kuramotoCore";
import {
  createGpuRenderer,
  type EdgeTextures,
  type GpuRenderer
} from "./gpuRenderer";

type KernelSpec = {
  gain: number;
  k0: number;
  Q: number;
  anisotropy: number;
  chirality: number;
  transparency: number;
};

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
};

const DMT_SENS = {
  g1: 0.6,
  k1: 0.35,
  q1: 0.5,
  a1: 0.7,
  c1: 0.6,
  t1: 0.5
} as const;

const PRESETS: Preset[] = [
  {
    name: "Rainbow Rims + DMT Kernel Effects",
    params: {
      edgeThreshold: 0.08,
      blend: 0.39,
      kernel: {
        gain: 3.0,
        k0: 0.2,
        Q: 4.6,
        anisotropy: 0.95,
        chirality: 1.46,
        transparency: 0.28
      },
      dmt: 0.2,
      thetaMode: "gradient",
      thetaGlobal: 0,
      beta2: 1.9,
      jitter: 1.16,
      sigma: 4.0,
      microsaccade: true,
      speed: 1.32,
      contrast: 1.62
    }
  }
];

type SurfaceRegion = "surfaces" | "edges" | "both";
type WallpaperGroup = "off" | "p2" | "p4" | "p6" | "pmm" | "p4m";
type DisplayMode =
  | "color"
  | "grayBaseColorRims"
  | "grayBaseGrayRims"
  | "colorBaseGrayRims";

type Op =
  | { kind: "rot"; angle: number }
  | { kind: "mirrorX" }
  | { kind: "mirrorY" }
  | { kind: "diag1" }
  | { kind: "diag2" };

type EdgeField = {
  gx: Float32Array;
  gy: Float32Array;
  mag: Float32Array;
};

type FrameProfilerState = {
  enabled: boolean;
  samples: number[];
  maxSamples: number;
  label: string;
};

type KurFrameView = {
  buffer: ArrayBuffer;
  gradX: Float32Array;
  gradY: Float32Array;
  vort: Float32Array;
  coh: Float32Array;
  timestamp: number;
  frameId: number;
};

type WorkerFrameMessage = {
  kind: "frame";
  buffer: ArrayBuffer;
  timestamp: number;
  frameId: number;
  queueDepth: number;
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
          }[];
          tolerancePercent: number;
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
  }
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const clamp01 = (v: number) => clamp(v, 0, 1);
const gauss = (x: number, s: number) => Math.exp(-(x * x) / (2 * s * s + 1e-9));
const luma01 = (r: number, g: number, b: number) =>
  clamp01(0.2126 * r + 0.7152 * g + 0.0722 * b);
const formatBytes = (bytes: number) => {
  if (bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const idx = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / Math.pow(1024, idx);
  const precision = value >= 100 || idx === 0 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(precision)} ${units[idx]}`;
};

const hash2 = (x: number, y: number) => {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
};

const kEff = (k: KernelSpec, d: number): KernelSpec => ({
  gain: k.gain * (1 + DMT_SENS.g1 * d),
  k0: k.k0 * (1 + DMT_SENS.k1 * d),
  Q: k.Q * (1 + DMT_SENS.q1 * d),
  anisotropy: k.anisotropy + DMT_SENS.a1 * d,
  chirality: k.chirality + DMT_SENS.c1 * d,
  transparency: k.transparency + DMT_SENS.t1 * d
});

const groupOps = (kind: WallpaperGroup): Op[] => {
  switch (kind) {
    case "p2":
      return [
        { kind: "rot", angle: 0 },
        { kind: "rot", angle: Math.PI }
      ];
    case "p4":
      return [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2].map((angle) => ({
        kind: "rot",
        angle
      }));
    case "p6":
      return Array.from({ length: 6 }, (_, k) => ({
        kind: "rot",
        angle: k * (Math.PI / 3)
      }));
    case "pmm":
      return [
        { kind: "rot", angle: 0 },
        { kind: "rot", angle: Math.PI },
        { kind: "mirrorX" },
        { kind: "mirrorY" }
      ];
    case "p4m":
      return [
        { kind: "rot", angle: 0 },
        { kind: "rot", angle: Math.PI / 2 },
        { kind: "rot", angle: Math.PI },
        { kind: "rot", angle: (3 * Math.PI) / 2 },
        { kind: "mirrorX" },
        { kind: "mirrorY" },
        { kind: "diag1" },
        { kind: "diag2" }
      ];
    default:
      return [{ kind: "rot", angle: 0 }];
  }
};

const applyOp = (op: Op, x: number, y: number, cx: number, cy: number) => {
  const dx = x - cx;
  const dy = y - cy;
  switch (op.kind) {
    case "rot": {
      const c = Math.cos(op.angle);
      const s = Math.sin(op.angle);
      return { x: cx + c * dx - s * dy, y: cy + s * dx + c * dy };
    }
    case "mirrorX":
      return { x: 2 * cx - x, y };
    case "mirrorY":
      return { x, y: 2 * cy - y };
    case "diag1":
      return { x: cx + dy, y: cy + dx };
    case "diag2":
      return { x: cx - dy, y: cy - dx };
    default:
      return { x, y };
  }
};

const sampleScalar = (
  arr: Float32Array,
  x: number,
  y: number,
  W: number,
  H: number
) => {
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

const sampleRGB = (
  data: Uint8ClampedArray,
  x: number,
  y: number,
  W: number,
  H: number
) => {
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
    B: mix(b0, b1, fy)
  };
};

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

const toGpuOps = (ops: Op[]): { kind: number; angle: number }[] =>
  ops.map((op) => {
    switch (op.kind) {
      case "rot":
        return { kind: 0, angle: op.angle };
      case "mirrorX":
        return { kind: 1, angle: 0 };
      case "mirrorY":
        return { kind: 2, angle: 0 };
      case "diag1":
        return { kind: 3, angle: 0 };
      case "diag2":
        return { kind: 4, angle: 0 };
      default:
        return { kind: 0, angle: 0 };
    }
  });

const displayModeToEnum = (mode: DisplayMode) => {
  switch (mode) {
    case "grayBaseColorRims":
      return 1;
    case "grayBaseGrayRims":
      return 2;
    case "colorBaseGrayRims":
      return 3;
    default:
      return 0;
  }
};

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
  const gpuStateRef = useRef<{ gl: WebGL2RenderingContext; renderer: GpuRenderer } | null>(null);
  const pendingStaticUploadRef = useRef(true);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const captureStreamRef = useRef<MediaStream | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const recordedUrlRef = useRef<string | null>(null);
  const recordingMimeTypeRef = useRef<string | null>(null);

  const [width, setWidth] = useState(720);
  const [height, setHeight] = useState(480);
  const [renderBackend, setRenderBackend] = useState<"gpu" | "cpu">("gpu");

  const [imgBitmap, setImgBitmap] = useState<ImageBitmap | null>(null);
  const basePixelsRef = useRef<ImageData | null>(null);
  const edgeDataRef = useRef<EdgeField | null>(null);

  const [edgeThreshold, setEdgeThreshold] = useState(0.22);
  const [blend, setBlend] = useState(0.65);
  const [rimAlpha, setRimAlpha] = useState(1.0);

  const [beta2, setBeta2] = useState(1.1);
  const [jitter, setJitter] = useState(0.5);
  const [sigma, setSigma] = useState(1.4);
  const [microsaccade, setMicrosaccade] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [contrast, setContrast] = useState(1.0);

  const [kernel, setKernel] = useState<KernelSpec>({
    gain: 1.0,
    k0: 0.08,
    Q: 2.2,
    anisotropy: 0.6,
    chirality: 0.4,
    transparency: 0.2
  });
  const [dmt, setDmt] = useState(0.0);

  const [thetaMode, setThetaMode] = useState<"gradient" | "global">("gradient");
  const [thetaGlobal, setThetaGlobal] = useState(0);

  const [displayMode, setDisplayMode] = useState<DisplayMode>("color");

  const [phasePin, setPhasePin] = useState(true);
  const [alive, setAlive] = useState(false);
  const [polBins, setPolBins] = useState(16);
  const [normPin, setNormPin] = useState(true);

  const [surfEnabled, setSurfEnabled] = useState(false);
  const [wallGroup, setWallGroup] = useState<WallpaperGroup>("p4");
  const [nOrient, setNOrient] = useState(4);
  const [surfaceBlend, setSurfaceBlend] = useState(0.35);
  const [warpAmp, setWarpAmp] = useState(1.0);
  const [surfaceRegion, setSurfaceRegion] =
    useState<SurfaceRegion>("surfaces");

  const [coupleE2S, setCoupleE2S] = useState(true);
  const [etaAmp, setEtaAmp] = useState(0.6);

  const [coupleS2E, setCoupleS2E] = useState(true);
  const [gammaOff, setGammaOff] = useState(0.2);
  const [gammaSigma, setGammaSigma] = useState(0.35);
  const [alphaPol, setAlphaPol] = useState(0.25);
  const [kSigma, setKSigma] = useState(0.8);

  const [kurEnabled, setKurEnabled] = useState(false);
  const [kurSync, setKurSync] = useState(false);
  const [K0, setK0] = useState(0.6);
  const [alphaKur, setAlphaKur] = useState(0.2);
  const [gammaKur, setGammaKur] = useState(0.15);
  const [omega0, setOmega0] = useState(0.0);
  const [epsKur, setEpsKur] = useState(0.002);
  const [qInit, setQInit] = useState(1);
  const [presetIndex, setPresetIndex] = useState(0);
  const [telemetryEnabled, setTelemetryEnabled] = useState(true);
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
  const cpuDerivedRef = useRef<KuramotoDerived | null>(null);
  const cpuDerivedBufferRef = useRef<ArrayBuffer | null>(null);
  const gradXRef = useRef<Float32Array | null>(null);
  const gradYRef = useRef<Float32Array | null>(null);
  const vortRef = useRef<Float32Array | null>(null);
  const cohRef = useRef<Float32Array | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const workerReadyRef = useRef(false);
  const workerInflightRef = useRef(0);
  const workerNextFrameIdRef = useRef(0);
  const workerPendingFramesRef = useRef<KurFrameView[]>([]);
  const workerActiveFrameRef = useRef<KurFrameView | null>(null);

  const frameBufferRef = useRef<{
    image: ImageData;
    data: Uint8ClampedArray;
    width: number;
    height: number;
  } | null>(null);

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

  const telemetryRef = useRef<{
    enabled: boolean;
    thresholds: Record<TelemetryPhase, number>;
    history: TelemetryRecord[];
    lastLogTs: number;
  }>({
    enabled: true,
    thresholds: {
      frame: 28,
      renderGpu: 10,
      renderCpu: 20,
      kuramoto: 8
    },
    history: [],
    lastLogTs: 0
  });

  const randn = useRandN();

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
  }, [telemetryEnabled]);

  const orientations = useMemo(() => {
    const N = clamp(Math.round(nOrient), 2, 8);
    return Array.from({ length: N }, (_, j) => (j * 2 * Math.PI) / N);
  }, [nOrient]);

  const lambdas = useMemo(() => ({ L: 560, M: 530, S: 420 }), []);
  const lambdaRef = 520;

  const ensureKurCpuState = useCallback(() => {
    if (!kurStateRef.current || kurStateRef.current.width !== width || kurStateRef.current.height !== height) {
      kurStateRef.current = createKuramotoState(width, height);
    }
    const expected = width * height;
    if (
      !cpuDerivedRef.current ||
      cpuDerivedRef.current.gradX.length !== expected
    ) {
      const buffer = new ArrayBuffer(derivedBufferSize(width, height));
      cpuDerivedBufferRef.current = buffer;
      const derived = createDerivedViews(buffer, width, height);
      cpuDerivedRef.current = derived;
      gradXRef.current = derived.gradX;
      gradYRef.current = derived.gradY;
      vortRef.current = derived.vort;
      cohRef.current = derived.coh;
      return true;
    }
    gradXRef.current = cpuDerivedRef.current.gradX;
    gradYRef.current = cpuDerivedRef.current.gradY;
    vortRef.current = cpuDerivedRef.current.vort;
    cohRef.current = cpuDerivedRef.current.coh;
    return false;
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
    if (edgeDataRef.current) {
      const edge = edgeDataRef.current as EdgeTextures;
      renderer.uploadEdge(edge);
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
      epsKur
    };
  }, [alphaKur, gammaKur, omega0, K0, epsKur]);

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
      }
    }
  }, [kurEnabled, qInit, initKuramotoCpu, ensureKurCpuState]);

  const stepKuramotoCpu = useCallback(
    (dt: number) => {
      if (!kurEnabled) return;
      ensureKurCpuState();
      if (!kurStateRef.current) return;
      stepKuramotoState(kurStateRef.current, getKurParams(), dt, randn);
    },
    [kurEnabled, ensureKurCpuState, getKurParams, randn]
  );

  const deriveKurFieldsCpu = useCallback(() => {
    if (!kurEnabled) return;
    ensureKurCpuState();
    if (!kurStateRef.current || !cpuDerivedRef.current) return;
    deriveKuramotoFieldsCore(kurStateRef.current, cpuDerivedRef.current);
  }, [kurEnabled, ensureKurCpuState]);

  useEffect(() => {
    kurSyncRef.current = kurSync;
  }, [kurSync]);

  const clearWorkerData = useCallback(() => {
    workerPendingFramesRef.current = [];
    workerActiveFrameRef.current = null;
    workerInflightRef.current = 0;
    workerNextFrameIdRef.current = 0;
    gradXRef.current = null;
    gradYRef.current = null;
    vortRef.current = null;
    cohRef.current = null;
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
    gradXRef.current = next.gradX;
    gradYRef.current = next.gradY;
    vortRef.current = next.vort;
    cohRef.current = next.coh;
    if (prev) {
      releaseFrameToWorker(prev);
    }
  }, [releaseFrameToWorker]);

  const handleWorkerFrame = useCallback(
    (msg: WorkerFrameMessage) => {
      if (!workerRef.current) return;
      workerInflightRef.current = Math.max(workerInflightRef.current - 1, 0);
      const derived = createDerivedViews(msg.buffer, width, height);
      const frame: KurFrameView = {
        buffer: msg.buffer,
        gradX: derived.gradX,
        gradY: derived.gradY,
        vort: derived.vort,
        coh: derived.coh,
        timestamp: msg.timestamp,
        frameId: msg.frameId
      };
      workerPendingFramesRef.current.push(frame);
      if (!kurSyncRef.current) {
        swapWorkerFrame();
      }
    },
    [height, width, swapWorkerFrame]
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

  const wallpaperAt = useCallback(
    (
      xp: number,
      yp: number,
      cosA: Float32Array,
      sinA: Float32Array,
      ke: KernelSpec,
      tSeconds: number
    ) => {
      const N = cosA.length;
      const twoPI = 2 * Math.PI;
      let gx = 0;
      let gy = 0;
      for (let j = 0; j < N; j++) {
        const phase =
          ke.chirality * (j / N) +
          (alive ? 0.2 * Math.sin(twoPI * 0.3 * tSeconds + j) : 0);
        const s = xp * cosA[j] + yp * sinA[j];
        const arg = twoPI * ke.k0 * s + phase;
        const d = -twoPI * ke.k0 * Math.sin(arg);
        gx += d * cosA[j];
        gy += d * sinA[j];
      }
      const inv = N > 0 ? 1 / N : 1;
      return { gx: gx * inv, gy: gy * inv };
    },
    [alive]
  );

  const renderFrameCore = useCallback(
    (
      out: Uint8ClampedArray,
      tSeconds: number,
      commitObs = true,
      fieldsOverride?: Pick<KurFrameView, "gradX" | "gradY" | "vort" | "coh">
    ) => {
      const gradX = fieldsOverride?.gradX ?? gradXRef.current;
      const gradY = fieldsOverride?.gradY ?? gradYRef.current;
      const vort = fieldsOverride?.vort ?? vortRef.current;
      const coh = fieldsOverride?.coh ?? cohRef.current;
      const ke = kEff(kernel, dmt);
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

      if (imgBitmap && edgeDataRef.current && basePixelsRef.current) {
        const { gx, gy, mag } = edgeDataRef.current;
        const base = basePixelsRef.current.data;

        let muJ = 0;
        let cnt = 0;
        if (phasePin && microsaccade) {
          const stride = 8;
          for (let yy = 0; yy < height; yy += stride) {
            for (let xx = 0; xx < width; xx += stride) {
              const idx = yy * width + xx;
              if (mag[idx] >= edgeThreshold) {
                muJ += Math.sin(
                  jitterPhase + hash2(xx, yy) * Math.PI * 2
                );
                cnt++;
              }
            }
          }
          muJ = cnt ? muJ / cnt : 0;
        }

        const ops = groupOps(wallGroup);
        const opsCount = ops.length;
        const useWallpaper = surfEnabled || coupleS2E;
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
        let obsSum = 0;
        let obsCount = 0;

        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const p = y * width + x;
            const i = p * 4;
            out[i + 0] = base[i + 0];
            out[i + 1] = base[i + 1];
            out[i + 2] = base[i + 2];
            out[i + 3] = 255;

            if (
              displayMode === "grayBaseColorRims" ||
              displayMode === "grayBaseGrayRims"
            ) {
              const yb = Math.floor(
                0.2126 * out[i + 0] +
                  0.7152 * out[i + 1] +
                  0.0722 * out[i + 2]
              );
              out[i + 0] = yb;
              out[i + 1] = yb;
              out[i + 2] = yb;
            }

            let gxT0 = 0;
            let gyT0 = 0;
            if (kurEnabled && gradX && gradY) {
              gxT0 = gradX[p];
              gyT0 = gradY[p];
            } else if (useWallpaper) {
              for (let k = 0; k < opsCount; k++) {
                const pt = applyOp(ops[k], x, y, cx, cy);
                const r = wallpaperAt(
                  pt.x - cx,
                  pt.y - cy,
                  cosA,
                  sinA,
                  ke,
                  tSeconds
                );
                gxT0 += r.gx;
                gyT0 += r.gy;
              }
              const inv = opsCount > 0 ? 1 / opsCount : 1;
              gxT0 *= inv;
              gyT0 *= inv;
            }

            const magVal = mag[p];
            let rimEnergy = 0;

            if (magVal >= edgeThreshold) {
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
                thetaMode === "gradient"
                  ? polBins > 0
                    ? Math.round((thetaRaw / TAU) * polBins) * (TAU / polBins)
                    : thetaRaw
                  : thetaGlobal;

              const gradNorm0 = Math.hypot(gxT0, gyT0);
              let thetaUse = thetaEdge;
              if (coupleS2E && gradNorm0 > 1e-6) {
                const ex = Math.cos(thetaEdge);
                const ey = Math.sin(thetaEdge);
                const sx = gxT0 / gradNorm0;
                const sy = gyT0 / gradNorm0;
                const vx = (1 - alphaPol) * ex + alphaPol * sx;
                const vy = (1 - alphaPol) * ey + alphaPol * sy;
                let tBlend = Math.atan2(vy, vx);
                if (polBins > 0) {
                  tBlend =
                    Math.round((tBlend / TAU) * polBins) *
                    (TAU / polBins);
                }
                thetaUse = tBlend;
              }

              const delta = ke.anisotropy * 0.9;
              const rho = ke.chirality * 0.75;
              const thetaEff = thetaUse + rho * tSeconds;
              const polL =
                0.5 * (1 + Math.cos(delta) * Math.cos(2 * thetaEff));
              const polM =
                0.5 *
                (1 + Math.cos(delta) * Math.cos(2 * (thetaEff + 0.3)));
              const polS =
                0.5 *
                (1 + Math.cos(delta) * Math.cos(2 * (thetaEff + 0.6)));

              const rawJ = Math.sin(jitterPhase + hash2(x, y) * Math.PI * 2);
              const localJ =
                jitter *
                (microsaccade ? (phasePin ? rawJ - muJ : rawJ) : 0);

              const gradNorm = Math.hypot(gxT0, gyT0);
              const bias =
                coupleS2E && gradNorm > 1e-6
                  ? gammaOff * ((gxT0 * nx + gyT0 * ny) / gradNorm)
                  : 0;
              const Esurf = coupleS2E ? clamp01(gradNorm * kSigma) : 0;
              const sigmaEff = coupleS2E
                ? Math.max(0.35, sigma * (1 - gammaSigma * Esurf))
                : sigma;

              const offL = baseOffsets.L + localJ * 0.35 + bias;
              const offM = baseOffsets.M + localJ * 0.5 + bias;
              const offS = baseOffsets.S + localJ * 0.8 + bias;

              const pL = sampleScalar(
                mag,
                x + (offL + breath) * nx,
                y + (offL + breath) * ny,
                width,
                height
              );
              const pM = sampleScalar(
                mag,
                x + (offM + breath) * nx,
                y + (offM + breath) * ny,
                width,
                height
              );
              const pS = sampleScalar(
                mag,
                x + (offS + breath) * nx,
                y + (offS + breath) * ny,
                width,
                height
              );

              const gL = gauss(offL, sigmaEff) * ke.gain;
              const gM = gauss(offM, sigmaEff) * ke.gain;
              const gS = gauss(offS, sigmaEff) * ke.gain;

              const QQ = 1 + 0.5 * ke.Q;
              const modL = Math.pow(
                0.5 * (1 + Math.cos(2 * Math.PI * ke.k0 * offL)),
                QQ
              );
              const modM = Math.pow(
                0.5 * (1 + Math.cos(2 * Math.PI * ke.k0 * offM)),
                QQ
              );
              const modS = Math.pow(
                0.5 * (1 + Math.cos(2 * Math.PI * ke.k0 * offS)),
                QQ
              );

              const chiPhase =
                2 * Math.PI * ke.k0 * (x * tx + y * ty) * 0.002;
              const chBase =
                ke.chirality +
                (kurEnabled && vort
                  ? clamp(vort[p] * 0.5, -1, 1)
                  : 0);
              const chiL = 0.5 + 0.5 * Math.sin(chiPhase) * chBase;
              const chiM = 0.5 + 0.5 * Math.sin(chiPhase + 0.8) * chBase;
              const chiS = 0.5 + 0.5 * Math.sin(chiPhase + 1.6) * chBase;

              const cont = contrast * frameGain;
              const Lc = pL * gL * modL * chiL * polL * cont;
              const Mc = pM * gM * modM * chiM * polM * cont;
              const Sc = pS * gS * modS * chiS * polS * cont;

              rimEnergy = (Lc + Mc + Sc) / Math.max(1e-6, cont);

              let R =
                4.4679 * Lc + -3.5873 * Mc + 0.1193 * Sc;
              let G =
                -1.2186 * Lc + 2.3809 * Mc + -0.1624 * Sc;
              let B =
                0.0497 * Lc + -0.2439 * Mc + 1.2045 * Sc;
              R = clamp01(R);
              G = clamp01(G);
              B = clamp01(B);
              if (
                displayMode === "grayBaseGrayRims" ||
                displayMode === "colorBaseGrayRims"
              ) {
                const yr = luma01(R, G, B);
                R = yr;
                G = yr;
                B = yr;
              }
              R *= rimAlpha;
              G *= rimAlpha;
              B *= rimAlpha;

              out[i + 0] = Math.floor(
                out[i + 0] * (1 - effectiveBlend) + R * 255 * effectiveBlend
              );
              out[i + 1] = Math.floor(
                out[i + 1] * (1 - effectiveBlend) + G * 255 * effectiveBlend
              );
              out[i + 2] = Math.floor(
                out[i + 2] * (1 - effectiveBlend) + B * 255 * effectiveBlend
              );

              if ((x & 7) === 0 && (y & 7) === 0) {
                obsSum += (pL + pM + pS) / 3;
                obsCount++;
              }
            }

            if (surfEnabled) {
              let mask = 1.0;
              if (surfaceRegion === "surfaces") {
                mask = clamp01(
                  (edgeThreshold - magVal) /
                    Math.max(1e-6, edgeThreshold)
                );
              } else if (surfaceRegion === "edges") {
                mask = clamp01(
                  (magVal - edgeThreshold) / Math.max(1e-6, 1 - edgeThreshold)
                );
              }
              if (mask > 1e-3) {
                let gxSurf = gxT0;
                let gySurf = gyT0;
                if (coupleE2S && magVal >= edgeThreshold) {
                  const tx = -(gy[p] / (Math.hypot(gx[p], gy[p]) + 1e-8));
                  const ty = gx[p] / (Math.hypot(gx[p], gy[p]) + 1e-8);
                  const dot =
                    (gxT0 * tx + gyT0 * ty) /
                    (Math.hypot(gxT0, gyT0) * Math.hypot(tx, ty) + 1e-6);
                  const align = Math.pow(clamp01((dot + 1) * 0.5), alphaPol);
                  gxSurf = (1 - align) * gxSurf + align * tx;
                  gySurf = (1 - align) * gySurf + align * ty;
                }
                const dirNorm = Math.hypot(gxSurf, gySurf) + 1e-6;
                const dirW =
                  1 +
                  0.5 *
                    ke.anisotropy *
                    Math.cos(2 * Math.atan2(gySurf, gxSurf));
                const wx = warpAmp * (gxSurf / dirNorm) * dirW;
                const wy = warpAmp * (gySurf / dirNorm) * dirW;
                const sample = sampleRGB(
                  base,
                  x + wx,
                  y + wy,
                  width,
                  height
                );
                let rW = sample.R / 255;
                let gW = sample.G / 255;
                let bW = sample.B / 255;
                if (displayMode === "grayBaseGrayRims") {
                  const yy = luma01(rW, gW, bW);
                  rW = yy;
                  gW = yy;
                  bW = yy;
                }
                let sb = surfaceBlend * mask;
                if (coupleE2S) {
                  sb *= 1 + etaAmp * clamp01(rimEnergy);
                }
                if (kurEnabled && coh) {
                  sb *= 0.5 + 0.5 * coh[p];
                }
                sb = clamp01(sb);
                out[i + 0] = Math.floor(out[i + 0] * (1 - sb) + rW * 255 * sb);
                out[i + 1] = Math.floor(out[i + 1] * (1 - sb) + gW * 255 * sb);
                out[i + 2] = Math.floor(out[i + 2] * (1 - sb) + bW * 255 * sb);
              }
            }
          }
        }

        if (commitObs) {
          const obs = obsCount ? obsSum / obsCount : lastObsRef.current;
          lastObsRef.current = clamp(obs, 0.001, 10);
        }
      } else {
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            const n = Math.sin((x / width) * Math.PI * 6 + tSeconds);
            const m = Math.sin((y / height) * Math.PI * 6 - tSeconds * 0.7);
            const v = clamp01(0.5 + 0.5 * n * m);
            out[i + 0] = Math.floor(v * 255);
            out[i + 1] = Math.floor((0.4 + 0.6 * v) * 180);
            out[i + 2] = Math.floor((1 - v) * 255);
            out[i + 3] = 255;
          }
        }
      }
    },
    [
      kernel,
      dmt,
      blend,
      normPin,
      lambdas,
      lambdaRef,
      beta2,
      microsaccade,
      alive,
      imgBitmap,
      edgeDataRef,
      basePixelsRef,
      phasePin,
      height,
      width,
      edgeThreshold,
      wallGroup,
      surfEnabled,
      coupleS2E,
      orientations,
      getOrientationCache,
      wallpaperAt,
      thetaMode,
      thetaGlobal,
      polBins,
      jitter,
      gammaOff,
      kSigma,
      gammaSigma,
      sigma,
      contrast,
      rimAlpha,
      displayMode,
      surfaceBlend,
      surfaceRegion,
      warpAmp,
      coupleE2S,
      etaAmp,
      kurEnabled,
      vortRef,
      gradXRef,
      gradYRef,
      cohRef,
      alphaPol
    ]
  );

  const drawFrameGpu = useCallback(
    (
      state: { gl: WebGL2RenderingContext; renderer: GpuRenderer },
      tSeconds: number,
      commitObs: boolean
    ) => {
      const { gl, renderer } = state;
      if (!edgeDataRef.current || !basePixelsRef.current) {
        gl.clear(gl.COLOR_BUFFER_BIT);
        return;
      }

      const telemetryActive = telemetryRef.current.enabled && commitObs;
      const renderStart = telemetryActive ? performance.now() : 0;

      const ke = kEff(kernel, dmt);
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

      const { gx, gy, mag } = edgeDataRef.current;
      const gradX = gradXRef.current;
      const gradY = gradYRef.current;
      const vort = vortRef.current;
      const coh = cohRef.current;

      const ops = groupOps(wallGroup);
      const gpuOps = toGpuOps(ops);
      const useWallpaper = surfEnabled || coupleS2E;
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

      let obsSum = 0;
      let obsCount = 0;
      if (commitObs) {
        for (let y = 0; y < height; y += 8) {
          for (let x = 0; x < width; x += 8) {
            const idx = y * width + x;
            if (mag[idx] < edgeThreshold) continue;

            let nx = gx[idx];
            let ny = gy[idx];
            const nlen = Math.hypot(nx, ny) + 1e-8;
            nx /= nlen;
            ny /= nlen;

            let gxT0 = 0;
            let gyT0 = 0;
            if (kurEnabled && gradX && gradY) {
              gxT0 = gradX[idx];
              gyT0 = gradY[idx];
            } else if (!kurEnabled && useWallpaper) {
              for (let k = 0; k < ops.length; k++) {
                const pt = applyOp(ops[k], x, y, cx, cy);
                const r = wallpaperAt(
                  pt.x - cx,
                  pt.y - cy,
                  cosA,
                  sinA,
                  ke,
                  tSeconds
                );
                gxT0 += r.gx;
                gyT0 += r.gy;
              }
              const inv = ops.length > 0 ? 1 / ops.length : 1;
              gxT0 *= inv;
              gyT0 *= inv;
            }

            const gradNorm = Math.hypot(gxT0, gyT0);
            const rawJ = Math.sin(
              jitterPhase + hash2(x, y) * Math.PI * 2
            );
            const localJ =
              jitter *
              (microsaccade ? (phasePin ? rawJ - muJ : rawJ) : 0);
            const bias =
              coupleS2E && gradNorm > 1e-6
                ? gammaOff * ((gxT0 * nx + gyT0 * ny) / gradNorm)
                : 0;
            const Esurf = coupleS2E ? clamp01(gradNorm * kSigma) : 0;
            const sigmaEff = coupleS2E
              ? Math.max(0.35, sigma * (1 - gammaSigma * Esurf))
              : sigma;

            const offL = baseOffsets.L + localJ * 0.35 + bias;
            const offM = baseOffsets.M + localJ * 0.5 + bias;
            const offS = baseOffsets.S + localJ * 0.8 + bias;

            const pL = sampleScalar(
              mag,
              x + (offL + breath) * nx,
              y + (offL + breath) * ny,
              width,
              height
            );
            const pM = sampleScalar(
              mag,
              x + (offM + breath) * nx,
              y + (offM + breath) * ny,
              width,
              height
            );
            const pS = sampleScalar(
              mag,
              x + (offS + breath) * nx,
              y + (offS + breath) * ny,
              width,
              height
            );
            obsSum += (pL + pM + pS) / 3;
            obsCount++;
          }
        }
      }

      renderer.uploadKur({
        gradX,
        gradY,
        vort,
        coh
      });

      renderer.render({
        time: tSeconds,
        edgeThreshold,
        effectiveBlend,
        displayMode: displayModeToEnum(displayMode),
        baseOffsets: [baseOffsets.L, baseOffsets.M, baseOffsets.S],
        sigma,
        sigmaMin: coupleS2E ? 0.35 : sigma,
        jitter,
        jitterPhase,
        breath,
        muJ,
        phasePin,
        microsaccade,
        polBins,
        thetaMode: thetaMode === "gradient" ? 0 : 1,
        thetaGlobal,
        coupleS2E,
        alphaPol,
        gammaOff,
        kSigma,
        gammaSigma,
        contrast,
        frameGain,
        rimAlpha,
        warpAmp,
        surfaceBlend,
        surfaceRegion: surfaceRegionToEnum(surfaceRegion),
        surfEnabled,
        coupleE2S,
        etaAmp,
        kurEnabled,
        useWallpaper: !kurEnabled && useWallpaper,
        kernel: ke,
        alive,
        orientations: orientationCache,
        ops: gpuOps,
        center: [cx, cy]
      });

      if (telemetryActive) {
        recordTelemetry("renderGpu", performance.now() - renderStart);
      }

      if (commitObs) {
        const obs = obsCount ? obsSum / obsCount : lastObsRef.current;
        lastObsRef.current = clamp(obs, 0.001, 10);
      }
    },
    [
      edgeDataRef,
      basePixelsRef,
      kernel,
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
      coupleS2E,
      thetaMode,
      thetaGlobal,
      polBins,
      jitter,
      gammaOff,
      kSigma,
      gammaSigma,
      sigma,
      contrast,
      rimAlpha,
      warpAmp,
      surfaceBlend,
      surfaceRegion,
      coupleE2S,
      etaAmp,
      kurEnabled,
      alphaPol,
      width,
      height,
      edgeThreshold,
      wallpaperAt,
      recordTelemetry
    ]
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
      if (frameStart) {
        recordTelemetry("frame", performance.now() - frameStart);
      }
    },
    [
      advanceKuramoto,
      speed,
      ensureFrameBuffer,
      renderFrameCore,
      recordTelemetry
    ]
  );

  const runRegressionHarness = useCallback(
    async (frameCount = 10) => {
      if (!edgeDataRef.current || !basePixelsRef.current) {
        console.warn(
          "[regression] skipping: base pixels or edge data not ready."
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

      const cpuState = createKuramotoState(width, height);
      const cpuBuffer = new ArrayBuffer(derivedBufferSize(width, height));
      const cpuDerived = createDerivedViews(cpuBuffer, width, height);
      const cpuRand = createNormalGenerator(seed);
      initKuramotoState(cpuState, qInit, cpuDerived);

      const baselineFrames: Uint8ClampedArray[] = [];
      for (let i = 0; i < frameCount; i++) {
        stepKuramotoState(cpuState, params, dt, cpuRand);
        deriveKuramotoFieldsCore(cpuState, cpuDerived);
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
      edgeDataRef,
      basePixelsRef,
      kurEnabled,
      width,
      height,
      renderFrameCore,
      getKurParams,
      speed,
      qInit
    ]
  );

  const runGpuParityCheck = useCallback(async () => {
    if (!edgeDataRef.current || !basePixelsRef.current) {
      console.warn("[gpu-regression] base pixels or edge data unavailable.");
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
    renderer.uploadEdge(edgeDataRef.current);
    renderer.uploadKur({
      gradX: gradXRef.current,
      gradY: gradYRef.current,
      vort: vortRef.current,
      coh: cohRef.current
    });

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
        renderer.uploadKur({
          gradX: gradXRef.current,
          gradY: gradYRef.current,
          vort: vortRef.current,
          coh: cohRef.current
        });
        renderFrameCore(cpuBuffer, scene.time, false);
        drawFrameGpu(state, scene.time, false);
        gl.finish();
        renderer.readPixels(gpuBuffer);
        let mismatched = 0;
        let maxDelta = 0;
        for (let i = 0; i < pixelCount; i++) {
          const idx = i * 4;
          const dr = Math.abs(cpuBuffer[idx] - gpuBuffer[idx]);
          const dg = Math.abs(cpuBuffer[idx + 1] - gpuBuffer[idx + 1]);
          const db = Math.abs(cpuBuffer[idx + 2] - gpuBuffer[idx + 2]);
          const delta = Math.max(dr, dg, db);
          if (delta > maxDelta) maxDelta = delta;
          if (delta > 1) mismatched++;
        }
        results.push({
          label: scene.label,
          mismatched,
          percent: (mismatched / pixelCount) * 100,
          maxDelta
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
    edgeDataRef,
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
      if (!edgeDataRef.current || !basePixelsRef.current) {
        console.warn("[perf] base pixels or edge data unavailable.");
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
    [edgeDataRef, basePixelsRef, width, height, renderFrameCore, ensureGpuRenderer, drawFrameGpu]
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
      basePixelsRef.current = img;

      const gray = new Float32Array(newW * newH);
      const d = img.data;
      for (let y = 0; y < newH; y++) {
        for (let x = 0; x < newW; x++) {
          const i = (y * newW + x) * 4;
          gray[y * newW + x] =
            (0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]) /
            255;
        }
      }
      const gx = new Float32Array(newW * newH);
      const gy = new Float32Array(newW * newH);
      const idx = (ix: number, iy: number) => iy * newW + ix;
      for (let y = 1; y < newH - 1; y++) {
        for (let x = 1; x < newW - 1; x++) {
          let sx = 0;
          let sy = 0;
          const kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
          const ky = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
          let k = 0;
          for (let j = -1; j <= 1; j++) {
            for (let i2 = -1; i2 <= 1; i2++) {
              const v = gray[idx(x + i2, y + j)];
              sx += v * kx[k];
              sy += v * ky[k];
              k++;
            }
          }
          gx[idx(x, y)] = sx;
          gy[idx(x, y)] = sy;
        }
      }
      const mag = new Float32Array(newW * newH);
      let maxMag = 1e-6;
      for (let i = 0; i < mag.length; i++) {
        const m = Math.hypot(gx[i], gy[i]);
        mag[i] = m;
        if (m > maxMag) maxMag = m;
      }
      const inv = 1 / maxMag;
      for (let i = 0; i < mag.length; i++) {
        mag[i] *= inv;
      }
      edgeDataRef.current = { gx, gy, mag };
      pendingStaticUploadRef.current = true;
      refreshGpuStaticTextures();
      normTargetRef.current = 0.6;
      lastObsRef.current = 0.6;
    },
    [refreshGpuStaticTextures]
  );

  const applyPreset = useCallback(
    (preset: Preset) => {
      const v = preset.params;
      setEdgeThreshold(v.edgeThreshold);
      setBlend(v.blend);
      setKernel({ ...v.kernel });
      setDmt(v.dmt);
      setThetaMode(v.thetaMode);
      setThetaGlobal(v.thetaGlobal);
      setBeta2(v.beta2);
      setJitter(v.jitter);
      setSigma(v.sigma);
      setMicrosaccade(v.microsaccade);
      setSpeed(v.speed);
      setContrast(v.contrast);
    },
    []
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

  const resetKuramotoField = useCallback(() => {
    initKuramotoCpu(qInit);
    if (!kurSyncRef.current && workerRef.current && workerReadyRef.current) {
      workerRef.current.postMessage({
        kind: "reset",
        qInit
      });
    }
  }, [initKuramotoCpu, qInit]);

  return (
    <main
      style={{
        minHeight: "100vh",
        padding: "2rem",
        display: "flex",
        flexDirection: "column",
        gap: "1.5rem"
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
          alignItems: "start"
        }}
      >
        <div className="panel" style={{ gap: "1.25rem" }}>
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
              onChange={(v) =>
                setKernel((prev) => ({
                  ...prev,
                  gain: v
                }))
              }
            />
            <SliderControl
              label="Spatial Frequency k₀"
              value={kernel.k0}
              min={0.01}
              max={0.4}
              step={0.01}
              onChange={(v) =>
                setKernel((prev) => ({
                  ...prev,
                  k0: v
                }))
              }
            />
            <SliderControl
              label="Sharpness Q"
              value={kernel.Q}
              min={0.5}
              max={8}
              step={0.05}
              onChange={(v) =>
                setKernel((prev) => ({
                  ...prev,
                  Q: v
                }))
              }
            />
            <SliderControl
              label="Anisotropy"
              value={kernel.anisotropy}
              min={0}
              max={1.5}
              step={0.05}
              onChange={(v) =>
                setKernel((prev) => ({
                  ...prev,
                  anisotropy: v
                }))
              }
            />
            <SliderControl
              label="Chirality"
              value={kernel.chirality}
              min={0}
              max={2}
              step={0.05}
              onChange={(v) =>
                setKernel((prev) => ({
                  ...prev,
                  chirality: v
                }))
              }
            />
            <SliderControl
              label="Transparency"
              value={kernel.transparency}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) =>
                setKernel((prev) => ({
                  ...prev,
                  transparency: v
                }))
              }
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
            <h2>Coupling</h2>
            <ToggleControl
              label="Edges → Surface"
              value={coupleE2S}
              onChange={setCoupleE2S}
            />
            <SliderControl
              label="Rim gain → Surface"
              value={etaAmp}
              min={0}
              max={2}
              step={0.05}
              onChange={setEtaAmp}
            />
            <ToggleControl
              label="Surface → Edges"
              value={coupleS2E}
              onChange={setCoupleS2E}
            />
            <SliderControl
              label="Offset bias γₒff"
              value={gammaOff}
              min={-1}
              max={1}
              step={0.05}
              onChange={setGammaOff}
            />
            <SliderControl
              label="Sigma scaling γσ"
              value={gammaSigma}
              min={0}
              max={1.5}
              step={0.05}
              onChange={setGammaSigma}
            />
            <SliderControl
              label="Polarization blend αₚₒₗ"
              value={alphaPol}
              min={0}
              max={1}
              step={0.05}
              onChange={setAlphaPol}
            />
            <SliderControl
              label="Warp → Sharpness kσ"
              value={kSigma}
              min={0}
              max={2}
              step={0.05}
              onChange={setKSigma}
            />
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
            <SliderControl
              label="Coupling K₀"
              value={K0}
              min={0}
              max={2}
              step={0.05}
              onChange={setK0}
            />
            <SliderControl
              label="Phase lag α"
              value={alphaKur}
              min={-Math.PI}
              max={Math.PI}
              step={0.05}
              onChange={setAlphaKur}
            />
            <SliderControl
              label="Line width γ"
              value={gammaKur}
              min={0}
              max={1}
              step={0.02}
              onChange={setGammaKur}
            />
            <SliderControl
              label="Mean freq ω₀"
              value={omega0}
              min={-2}
              max={2}
              step={0.05}
              onChange={setOmega0}
            />
            <SliderControl
              label="Noise ε"
              value={epsKur}
              min={0}
              max={0.02}
              step={0.0005}
              onChange={setEpsKur}
              format={(v) => v.toFixed(4)}
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
            gap: "1rem"
          }}
        >
          <canvas
            ref={canvasRef}
            width={width}
            height={height}
            style={{
              width: "100%",
              height: "auto",
              maxWidth: `${Math.min(width, 1000)}px`,
              maxHeight: `${Math.min(height, 1000)}px`,
              aspectRatio: `${width} / ${height}`,
              borderRadius: "1.25rem",
              border: "1px solid rgba(148,163,184,0.2)",
              boxShadow: "0 20px 60px rgba(15,23,42,0.65)"
            }}
          />
          <p style={{ color: "#64748b", margin: 0 }}>
            Tip: enable the OA field once you have rims dialed in, then
            gradually increase K₀ and lower ε to let the oscillator field
            steer both rims and surface morph in a coherent way.
          </p>
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
