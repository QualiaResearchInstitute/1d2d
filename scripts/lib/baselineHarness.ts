import { join } from 'node:path';

import { computeEdgeField, type ImageBuffer } from '../../src/pipeline/edgeDetection.js';
import {
  renderRainbowFrame,
  type RainbowFrameInput,
  type RainbowFrameMetrics,
  type CouplingConfig,
} from '../../src/pipeline/rainbowFrame.js';
import {
  createDefaultSu7RuntimeParams,
  type GateListSnapshot,
} from '../../src/pipeline/su7/types.js';
import { createSu7GateListSnapshot } from '../../src/pipeline/su7/math.js';
import {
  createKuramotoState,
  createDerivedViews,
  derivedBufferSize,
  initKuramotoState,
  stepKuramotoState,
  deriveKuramotoFields,
  createNormalGenerator,
  snapshotVolumeField,
  type KuramotoParams,
} from '../../src/kuramotoCore.js';
import { createKernelSpec, type KernelSpec } from '../../src/kernel/kernelSpec.js';

type KuramotoDerived = ReturnType<typeof createDerivedViews>;

export const BASELINE_DIM = { width: 256, height: 256 } as const;

export const CANONICAL_KERNEL: KernelSpec = createKernelSpec({
  gain: 3,
  k0: 0.2,
  Q: 4.6,
  anisotropy: 0.95,
  chirality: 1.46,
  transparency: 0.28,
});

export const CANONICAL_PARAMS = {
  edgeThreshold: 0.08,
  blend: 0.39,
  dmt: 0.2,
  arousal: 0.35,
  thetaMode: 'gradient' as const,
  thetaGlobal: 0,
  beta2: 1.9,
  jitter: 1.16,
  sigma: 4,
  microsaccade: true,
  speed: 1.32,
  contrast: 1.62,
};

export const CANONICAL_KUR_PARAMS: KuramotoParams = {
  alphaKur: 0.2,
  gammaKur: 0.15,
  omega0: 0,
  K0: 0.6,
  epsKur: 0.002,
  fluxX: 0,
  fluxY: 0,
  smallWorldWeight: 0,
  p_sw: 0,
};

const CANONICAL_COUPLING: CouplingConfig = {
  rimToSurfaceBlend: 0.28,
  rimToSurfaceAlign: 0.35,
  surfaceToRimOffset: 0.28,
  surfaceToRimSigma: 0.35,
  surfaceToRimHue: 0.4,
  kurToTransparency: 0.25,
  kurToOrientation: 0.32,
  kurToChirality: 0.35,
  volumePhaseToHue: 0.52,
  volumeDepthToWarp: 0.48,
};

const buildBaseImage = (width: number, height: number): ImageBuffer => {
  const data = new Uint8ClampedArray(width * height * 4);
  const encoder = (nx: number, ny: number) => {
    const r = Math.hypot(nx, ny);
    const hue = (Math.atan2(ny, nx) / (2 * Math.PI) + 1) % 1;
    const sat = 0.8;
    const val = 0.5 + 0.5 * Math.cos(r * Math.PI * 0.75);
    const chroma = val * sat;
    const hPrime = hue * 6;
    const xComp = chroma * (1 - Math.abs((hPrime % 2) - 1));
    let r1 = 0;
    let g1 = 0;
    let b1 = 0;
    if (hPrime >= 0 && hPrime < 1) {
      r1 = chroma;
      g1 = xComp;
    } else if (hPrime < 2) {
      r1 = xComp;
      g1 = chroma;
    } else if (hPrime < 3) {
      g1 = chroma;
      b1 = xComp;
    } else if (hPrime < 4) {
      g1 = xComp;
      b1 = chroma;
    } else if (hPrime < 5) {
      r1 = xComp;
      b1 = chroma;
    } else {
      r1 = chroma;
      b1 = xComp;
    }
    const m = val - chroma / 2;
    return {
      R: Math.round((r1 + m) * 255),
      G: Math.round((g1 + m) * 255),
      B: Math.round((b1 + m) * 255),
    };
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const nx = (x / (width - 1)) * 2 - 1;
      const ny = (y / (height - 1)) * 2 - 1;
      const { R, G, B } = encoder(nx, ny);
      data[idx + 0] = R;
      data[idx + 1] = G;
      data[idx + 2] = B;
      data[idx + 3] = 255;
    }
  }

  return { data, width, height };
};

const simulateKuramoto = (
  width: number,
  height: number,
  steps: number,
  dt: number,
): { phase: KuramotoDerived; volume: ReturnType<typeof snapshotVolumeField> } => {
  const state = createKuramotoState(width, height);
  const buffer = new ArrayBuffer(derivedBufferSize(width, height));
  const derived = createDerivedViews(buffer, width, height);
  initKuramotoState(state, 1, derived);
  const randn = createNormalGenerator(1234);
  for (let i = 0; i < steps; i++) {
    stepKuramotoState(state, CANONICAL_KUR_PARAMS, dt, randn, dt * (i + 1), {
      kernel: CANONICAL_KERNEL,
      controls: { dmt: CANONICAL_PARAMS.dmt },
    });
  }
  deriveKuramotoFields(state, derived, {
    kernel: CANONICAL_KERNEL,
    controls: { dmt: CANONICAL_PARAMS.dmt },
  });
  const volume = snapshotVolumeField(state);
  return { phase: derived, volume };
};

export type BaselineResult = {
  pixels: Uint8ClampedArray;
  metrics: RainbowFrameMetrics;
  obsAverage: number | null;
  gateSnapshot: GateListSnapshot;
};

export type BaselineFrameOptions =
  | number
  | {
      timeSeconds?: number;
      dimensions?: { width: number; height: number };
    };

export const generateBaselineFrame = (options: BaselineFrameOptions = 0): BaselineResult => {
  const timeSeconds = typeof options === 'number' ? options : (options.timeSeconds ?? 0);
  const dimensions =
    typeof options === 'number' ? BASELINE_DIM : (options.dimensions ?? BASELINE_DIM);
  const { width, height } = dimensions;
  const baseImage = buildBaseImage(width, height);
  const edgeField = computeEdgeField(baseImage);
  const { phase: phaseField, volume: volumeField } = simulateKuramoto(width, height, 180, 0.016);

  const out = new Uint8ClampedArray(width * height * 4);
  const orientations = Array.from({ length: 4 }, (_, j) => (j * 2 * Math.PI) / 4);
  const su7Params = createDefaultSu7RuntimeParams();

  const frameInput = {
    width,
    height,
    timeSeconds,
    out,
    surface: {
      kind: 'surface',
      resolution: edgeField.resolution,
      rgba: baseImage.data,
    },
    rim: edgeField,
    phase: phaseField,
    volume: volumeField,
    kernel: CANONICAL_KERNEL,
    dmt: CANONICAL_PARAMS.dmt,
    arousal: CANONICAL_PARAMS.arousal,
    blend: CANONICAL_PARAMS.blend,
    normPin: true,
    normTarget: 0.6,
    lastObs: 0.6,
    lambdaRef: 520,
    lambdas: { L: 560, M: 530, S: 420 },
    beta2: CANONICAL_PARAMS.beta2,
    microsaccade: CANONICAL_PARAMS.microsaccade,
    alive: false,
    phasePin: true,
    edgeThreshold: CANONICAL_PARAMS.edgeThreshold,
    wallpaperGroup: 'p4',
    surfEnabled: false,
    coupleS2E: true,
    coupleE2S: true,
    orientationAngles: orientations,
    thetaMode: CANONICAL_PARAMS.thetaMode,
    thetaGlobal: CANONICAL_PARAMS.thetaGlobal,
    polBins: 16,
    jitter: CANONICAL_PARAMS.jitter,
    gammaOff: 0.2,
    kSigma: 0.8,
    gammaSigma: 0.35,
    sigma: CANONICAL_PARAMS.sigma,
    contrast: CANONICAL_PARAMS.contrast,
    rimAlpha: 1,
    displayMode: 'color',
    surfaceBlend: 0.35,
    surfaceRegion: 'surfaces',
    warpAmp: 1,
    curvatureStrength: 0,
    curvatureMode: 'poincare',
    etaAmp: 0.6,
    kurEnabled: true,
    rimEnabled: true,
    alphaPol: 0.25,
    coupling: CANONICAL_COUPLING,
    couplingBase: CANONICAL_COUPLING,
    su7: su7Params,
  } as RainbowFrameInput;

  const result = renderRainbowFrame(frameInput);
  const gateSnapshot = createSu7GateListSnapshot(su7Params);

  return {
    pixels: out,
    metrics: result.metrics,
    obsAverage: result.obsAverage,
    gateSnapshot,
  };
};

export const encodePpm = (
  pixels: Uint8ClampedArray,
  width: number = BASELINE_DIM.width,
  height: number = BASELINE_DIM.height,
) => {
  const header = `P6\n${width} ${height}\n255\n`;
  const headerBytes = new TextEncoder().encode(header);
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < pixels.length; i += 4, j += 3) {
    rgb[j + 0] = pixels[i + 0];
    rgb[j + 1] = pixels[i + 1];
    rgb[j + 2] = pixels[i + 2];
  }
  const combined = new Uint8Array(headerBytes.length + rgb.length);
  combined.set(headerBytes, 0);
  combined.set(rgb, headerBytes.length);
  return combined;
};

export const downsamplePixels = (
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  targetWidth: number,
  targetHeight: number,
): Uint8ClampedArray => {
  const output = new Uint8ClampedArray(targetWidth * targetHeight * 4);
  const scaleX = width / targetWidth;
  const scaleY = height / targetHeight;

  for (let ty = 0; ty < targetHeight; ty++) {
    const yStart = Math.floor(ty * scaleY);
    const yEnd = Math.max(yStart + 1, Math.min(Math.floor((ty + 1) * scaleY), height));
    for (let tx = 0; tx < targetWidth; tx++) {
      const xStart = Math.floor(tx * scaleX);
      const xEnd = Math.max(xStart + 1, Math.min(Math.floor((tx + 1) * scaleX), width));
      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;
      let count = 0;
      for (let y = yStart; y < yEnd; y++) {
        for (let x = xStart; x < xEnd; x++) {
          const idx = (y * width + x) * 4;
          r += pixels[idx + 0];
          g += pixels[idx + 1];
          b += pixels[idx + 2];
          a += pixels[idx + 3];
          count++;
        }
      }
      const outIdx = (ty * targetWidth + tx) * 4;
      const denom = count > 0 ? count : 1;
      output[outIdx + 0] = Math.round(r / denom);
      output[outIdx + 1] = Math.round(g / denom);
      output[outIdx + 2] = Math.round(b / denom);
      output[outIdx + 3] = Math.round(a / denom);
    }
  }

  return output;
};

export const baselinePaths = (root: string) => ({
  metrics: join(root, 'metrics', 'canonical.json'),
  render: join(root, 'renders', 'canonical.ppm'),
  preset: join(root, 'metrics', 'su7-preset.json'),
  presetHash: join(root, 'metrics', 'su7-preset.hash'),
  renderHash: join(root, 'renders', 'canonical.hash'),
  thumbnail: join(root, 'renders', 'canonical-64.ppm'),
  thumbnailHash: join(root, 'renders', 'canonical-64.hash'),
});
