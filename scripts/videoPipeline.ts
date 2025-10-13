#!/usr/bin/env node
/// <reference types="node" />

declare const Buffer: any;

import { spawn } from 'node:child_process';
import * as fsPromises from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { computeEdgeField, type ImageBuffer } from '../src/pipeline/edgeDetection.js';
import {
  renderRainbowFrame,
  type CouplingConfig,
  type DisplayMode,
  type SurfaceRegion,
  type WallpaperGroup,
  type CurvatureMode,
} from '../src/pipeline/rainbowFrame.js';
import { createDefaultComposerConfig } from '../src/pipeline/rainbowFrame.js';
import { createDefaultSu7RuntimeParams } from '../src/pipeline/su7/types.js';
import {
  createKernelSpec,
  getDefaultKernelSpec,
  type KernelSpec,
} from '../src/kernel/kernelSpec.js';
import {
  createKuramotoState,
  createDerivedViews,
  derivedBufferSize,
  initKuramotoState,
  deriveKuramotoFields,
  stepKuramotoState,
  createNormalGenerator,
  type KuramotoParams,
  type PhaseField,
} from '../src/kuramotoCore.js';
import { TimelinePlayer, deriveSeedFromHash, type Timeline } from '../src/timeline/index.js';
import { parseRationalFps } from './videoUtils.js';
import { OfflineRenderer } from './lib/offlineRenderer.js';
import type { PerformanceSnapshot } from './lib/performanceWatchdog.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = resolve(__filename, '..');

const FALLBACK_TIMELINE_HASH = '0000000000000000000000000000000000000000000000000000000000000000';
const LAMBDA_REF = 520;
const LAMBDAS = { L: 560, M: 530, S: 420 } as const;
const ORIENTATION_ANGLES = [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2];
const UTF8_DECODER = new TextDecoder('utf-8');

const DEFAULT_COUPLING: CouplingConfig = {
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
};

const DEFAULT_KUR_PARAMS: KuramotoParams = {
  alphaKur: 0.2,
  gammaKur: 0.15,
  omega0: 0,
  K0: 0.6,
  epsKur: 0.002,
  fluxX: 0,
  fluxY: 0,
  smallWorldWeight: 0,
  p_sw: 0,
  smallWorldEnabled: false,
  smallWorldDegree: 12,
  smallWorldSeed: 1337,
};

const OFFLINE_RENDER_BUDGETS = {
  frameMs: 75,
  rssMb: 4096,
  heapMb: 2048,
  cpuPercent: 1200,
} as const;

const OFFLINE_TILE_HEIGHT = 128;
const OFFLINE_WATCHDOG_TOLERANCE = 0.2;

const DEFAULTS = {
  edgeThreshold: 0.22,
  blend: 0.65,
  dmt: 0,
  arousal: 0.3,
  normPin: true,
  beta2: 1.1,
  microsaccade: true,
  alive: false,
  phasePin: true,
  jitter: 0.5,
  sigma: 1.4,
  contrast: 1.0,
  rimAlpha: 1.0,
  rimEnabled: true,
  surfaceBlend: 0.35,
  surfaceRegion: 'surfaces' as SurfaceRegion,
  warpAmp: 1.0,
  curvatureStrength: 0,
  curvatureMode: 'poincare' as CurvatureMode,
  displayMode: 'color' as DisplayMode,
  wallGroup: 'p4' as WallpaperGroup,
  thetaMode: 'gradient' as 'gradient' | 'global',
  thetaGlobal: 0,
  polBins: 16,
  kurEnabled: false,
};

type CliOptions = {
  input: string;
  output: string;
  timelinePath?: string;
  ffmpeg: string;
  ffprobe: string;
  fpsOverride?: number;
  keepTemp: boolean;
  useKuramoto: boolean;
};

type VideoProbe = {
  width: number;
  height: number;
  fps: number;
  frameCount?: number;
};

const parseArgs = (argv: string[]): CliOptions => {
  const options: CliOptions = {
    ffmpeg: 'ffmpeg',
    ffprobe: 'ffprobe',
    keepTemp: false,
    useKuramoto: false,
    input: '',
    output: '',
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case '--input':
      case '-i': {
        options.input = argv[++i] ?? '';
        break;
      }
      case '--output':
      case '-o': {
        options.output = argv[++i] ?? '';
        break;
      }
      case '--timeline':
      case '-t': {
        options.timelinePath = argv[++i];
        break;
      }
      case '--ffmpeg': {
        options.ffmpeg = argv[++i] ?? 'ffmpeg';
        break;
      }
      case '--ffprobe': {
        options.ffprobe = argv[++i] ?? 'ffprobe';
        break;
      }
      case '--fps': {
        const value = Number(argv[++i]);
        if (!Number.isFinite(value) || value <= 0) {
          throw new Error(`Invalid --fps value: ${argv[i]}`);
        }
        options.fpsOverride = value;
        break;
      }
      case '--keep-temp': {
        options.keepTemp = true;
        break;
      }
      case '--kur':
      case '--use-kuramoto': {
        options.useKuramoto = true;
        break;
      }
      default: {
        if (arg.startsWith('-')) {
          throw new Error(`Unknown option: ${arg}`);
        }
      }
    }
  }
  if (!options.input) {
    throw new Error('Missing --input path');
  }
  if (!options.output) {
    throw new Error('Missing --output path');
  }
  return options;
};

const concatChunks = (chunks: Uint8Array[]): Buffer => {
  if (chunks.length === 0) {
    return Buffer.alloc(0);
  }
  if (chunks.length === 1) {
    return Buffer.from(chunks[0]);
  }
  const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const out = Buffer.allocUnsafe(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
};

const runCommand = (
  command: string,
  args: string[],
  stdin?: Uint8Array,
  options: { cwd?: string } = {},
): Promise<{ stdout: Buffer; stderr: Buffer }> =>
  new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd ?? process.cwd(),
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    const stdoutChunks: Uint8Array[] = [];
    const stderrChunks: Uint8Array[] = [];
    child.stdout.on('data', (chunk: Buffer) => stdoutChunks.push(chunk));
    child.stderr.on('data', (chunk: Buffer) => stderrChunks.push(chunk));
    child.on('error', reject);
    child.on('close', (code) => {
      if (code && code !== 0) {
        const stderr = UTF8_DECODER.decode(concatChunks(stderrChunks));
        reject(new Error(`${command} exited with code ${code}\n${stderr}`));
        return;
      }
      resolve({
        stdout: concatChunks(stdoutChunks),
        stderr: concatChunks(stderrChunks),
      });
    });
    if (stdin && stdin.byteLength > 0) {
      child.stdin.write(stdin);
    }
    child.stdin.end();
  });

const probeVideo = async (ffprobe: string, input: string): Promise<VideoProbe> => {
  const args = [
    '-v',
    'error',
    '-select_streams',
    'v:0',
    '-show_entries',
    'stream=width,height,r_frame_rate,nb_frames',
    '-of',
    'json',
    input,
  ];
  const { stdout } = await runCommand(ffprobe, args);
  const payload = JSON.parse(UTF8_DECODER.decode(stdout)) as {
    streams: Array<{
      width: number;
      height: number;
      r_frame_rate: string;
      nb_frames?: string;
    }>;
  };
  if (!payload.streams || payload.streams.length === 0) {
    throw new Error(`ffprobe returned no video streams for ${input}`);
  }
  const stream = payload.streams[0]!;
  const fps = parseRationalFps(stream.r_frame_rate);
  const frameCount =
    stream.nb_frames && Number.isFinite(Number(stream.nb_frames))
      ? Number(stream.nb_frames)
      : undefined;
  return { width: stream.width, height: stream.height, fps, frameCount };
};

const decodeFrameToRgba = async (
  ffmpeg: string,
  framePath: string,
  width: number,
  height: number,
): Promise<Uint8Array> => {
  const args = ['-v', 'error', '-i', framePath, '-f', 'rawvideo', '-pix_fmt', 'rgba', '-'];
  const { stdout } = await runCommand(ffmpeg, args);
  if (stdout.length !== width * height * 4) {
    throw new Error(
      `Decoded frame size mismatch for ${framePath}; expected ${width * height * 4} bytes, received ${stdout.length}`,
    );
  }
  return stdout;
};

const encodeRgbaToPng = async (
  ffmpeg: string,
  rgba: ArrayBufferView | Buffer,
  width: number,
  height: number,
  outputPath: string,
  options: { bitDepth?: 8 | 10 | 16 } = {},
): Promise<void> => {
  const bitDepth = options.bitDepth ?? 8;
  const pixFmt = bitDepth > 8 ? 'rgba64le' : 'rgba';
  const expectedBytes = width * height * 4 * (bitDepth > 8 ? 2 : 1);
  const buffer = Buffer.isBuffer(rgba)
    ? rgba
    : Buffer.from(rgba.buffer, rgba.byteOffset ?? 0, rgba.byteLength);
  if (buffer.byteLength !== expectedBytes) {
    throw new Error(
      `encodeRgbaToPng expected ${expectedBytes} bytes for ${width}x${height} (bitDepth=${bitDepth}), received ${buffer.byteLength}`,
    );
  }
  const args = [
    '-v',
    'error',
    '-y',
    '-f',
    'rawvideo',
    '-pix_fmt',
    pixFmt,
    '-s',
    `${width}x${height}`,
    '-i',
    '-',
    outputPath,
  ];
  await runCommand(ffmpeg, args, buffer);
};

const encodeVideoFromPng = async (
  ffmpeg: string,
  inputPattern: string,
  fps: number,
  outputPath: string,
): Promise<void> => {
  const args = [
    '-v',
    'error',
    '-y',
    '-framerate',
    fps.toString(),
    '-i',
    inputPattern,
    '-c:v',
    selectCodec(outputPath),
    '-pix_fmt',
    selectPixelFormat(outputPath),
    outputPath,
  ];
  await runCommand(ffmpeg, args);
};

const selectCodec = (outputPath: string): string => {
  const lower = outputPath.toLowerCase();
  if (lower.endsWith('.mov')) {
    return 'prores_ks';
  }
  if (lower.endsWith('.png')) {
    return 'png';
  }
  return 'libx264';
};

const selectPixelFormat = (outputPath: string): string => {
  const lower = outputPath.toLowerCase();
  if (lower.endsWith('.mov')) {
    return 'yuva444p10le';
  }
  if (lower.endsWith('.png')) {
    return 'rgba';
  }
  return 'yuv420p';
};

const ensureTempDirs = async () => {
  const root = await (fsPromises as any).mkdtemp(join(tmpdir(), 'rainbow-video-'));
  const decodeDir = join(root, 'decoded');
  const processedDir = join(root, 'processed');
  await (fsPromises as any).mkdir(decodeDir, { recursive: true });
  await (fsPromises as any).mkdir(processedDir, { recursive: true });
  return { root, decodeDir, processedDir };
};

type FrameParameters = {
  edgeThreshold: number;
  blend: number;
  dmt: number;
  arousal: number;
  normPin: boolean;
  beta2: number;
  microsaccade: boolean;
  alive: boolean;
  phasePin: boolean;
  jitter: number;
  sigma: number;
  contrast: number;
  rimAlpha: number;
  rimEnabled: boolean;
  surfaceBlend: number;
  surfaceRegion: SurfaceRegion;
  warpAmp: number;
  curvatureStrength: number;
  curvatureMode: CurvatureMode;
  displayMode: DisplayMode;
  wallGroup: WallpaperGroup;
  thetaMode: 'gradient' | 'global';
  thetaGlobal: number;
  polBins: number;
  kurEnabled: boolean;
};

type OfflineProcessSummary = {
  frameCount: number;
  performance: PerformanceSnapshot;
  su7: {
    frames: number;
    determinantDriftMax: number;
    determinantDriftMean: number;
    unitaryErrorMax: number;
    unitaryErrorMean: number;
    normDeltaMax: number;
    normDeltaMean: number;
    projectorEnergyMax: number;
    projectorEnergyMean: number;
    geodesicFallbacks: number;
  };
};

const resolveFrameParameters = (
  evaluation: ReturnType<TimelinePlayer['evaluateAtFrame']> | null,
  fallback: FrameParameters,
): FrameParameters => {
  const values = evaluation?.values ?? {};
  const getNumber = (id: string, defaultValue: number): number => {
    const value = values[id];
    return typeof value === 'number' && Number.isFinite(value) ? value : defaultValue;
  };
  const getBoolean = (id: string, defaultValue: boolean): boolean => {
    const value = values[id];
    return typeof value === 'boolean' ? value : defaultValue;
  };
  const getString = <T extends string>(id: string, defaultValue: T): T => {
    const value = values[id];
    return typeof value === 'string' ? (value as T) : defaultValue;
  };
  return {
    edgeThreshold: getNumber('edgeThreshold', fallback.edgeThreshold),
    blend: getNumber('blend', fallback.blend),
    dmt: getNumber('dmt', fallback.dmt),
    arousal: getNumber('arousal', fallback.arousal),
    normPin: getBoolean('normPin', fallback.normPin),
    beta2: getNumber('beta2', fallback.beta2),
    microsaccade: getBoolean('microsaccade', fallback.microsaccade),
    alive: getBoolean('alive', fallback.alive),
    phasePin: getBoolean('phasePin', fallback.phasePin),
    jitter: getNumber('jitter', fallback.jitter),
    sigma: getNumber('sigma', fallback.sigma),
    contrast: getNumber('contrast', fallback.contrast),
    rimAlpha: getNumber('rimAlpha', fallback.rimAlpha),
    rimEnabled: getBoolean('rimEnabled', fallback.rimEnabled),
    surfaceBlend: getNumber('surfaceBlend', fallback.surfaceBlend),
    surfaceRegion: getString('surfaceRegion', fallback.surfaceRegion),
    warpAmp: getNumber('warpAmp', fallback.warpAmp),
    curvatureStrength: getNumber('curvatureStrength', fallback.curvatureStrength),
    curvatureMode: getString('curvatureMode', fallback.curvatureMode),
    displayMode: getString('displayMode', fallback.displayMode),
    wallGroup: getString('wallGroup', fallback.wallGroup),
    thetaMode: getString<'gradient' | 'global'>('thetaMode', fallback.thetaMode),
    thetaGlobal: getNumber('thetaGlobal', fallback.thetaGlobal),
    polBins: Math.max(1, Math.round(getNumber('polBins', fallback.polBins))),
    kurEnabled: getBoolean('kurEnabled', fallback.kurEnabled),
  };
};

const decodeTimeline = async (path?: string): Promise<TimelinePlayer | null> => {
  if (!path) return null;
  const raw = await fsPromises.readFile(path, 'utf8');
  const parsed = JSON.parse(raw) as Timeline;
  return new TimelinePlayer(parsed);
};

const buildKernel = (
  timeline: ReturnType<TimelinePlayer['evaluateAtFrame']> | null,
): KernelSpec => {
  const values = timeline?.values ?? {};
  const hasKernelOverrides =
    typeof values.kernelGain === 'number' ||
    typeof values.kernelK0 === 'number' ||
    typeof values.kernelQ === 'number' ||
    typeof values.kernelAnisotropy === 'number' ||
    typeof values.kernelChirality === 'number' ||
    typeof values.kernelTransparency === 'number';
  if (!hasKernelOverrides) {
    return getDefaultKernelSpec();
  }
  return createKernelSpec({
    gain:
      typeof values.kernelGain === 'number' && Number.isFinite(values.kernelGain)
        ? values.kernelGain
        : getDefaultKernelSpec().gain,
    k0:
      typeof values.kernelK0 === 'number' && Number.isFinite(values.kernelK0)
        ? values.kernelK0
        : getDefaultKernelSpec().k0,
    Q:
      typeof values.kernelQ === 'number' && Number.isFinite(values.kernelQ)
        ? values.kernelQ
        : getDefaultKernelSpec().Q,
    anisotropy:
      typeof values.kernelAnisotropy === 'number' && Number.isFinite(values.kernelAnisotropy)
        ? values.kernelAnisotropy
        : getDefaultKernelSpec().anisotropy,
    chirality:
      typeof values.kernelChirality === 'number' && Number.isFinite(values.kernelChirality)
        ? values.kernelChirality
        : getDefaultKernelSpec().chirality,
    transparency:
      typeof values.kernelTransparency === 'number' && Number.isFinite(values.kernelTransparency)
        ? values.kernelTransparency
        : getDefaultKernelSpec().transparency,
  });
};

const processFrames = async (
  options: CliOptions,
  info: VideoProbe,
  decodeDir: string,
  processedDir: string,
  timeline: TimelinePlayer | null,
): Promise<OfflineProcessSummary> => {
  const files = (await fsPromises.readdir(decodeDir))
    .filter((name) => name.toLowerCase().endsWith('.png'))
    .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  if (files.length === 0) {
    throw new Error('No decoded frames found to process.');
  }

  const composer = createDefaultComposerConfig();
  const su7Params = createDefaultSu7RuntimeParams();
  let kernel = getDefaultKernelSpec();
  let lastObs = 1;
  const normTarget = 1;
  const timelineHash = timeline?.hash ?? FALLBACK_TIMELINE_HASH;
  const tileHeight = Math.min(OFFLINE_TILE_HEIGHT, info.height);
  const renderer = new OfflineRenderer({
    width: info.width,
    height: info.height,
    tileHeight,
    budgets: { ...OFFLINE_RENDER_BUDGETS },
    watchdog: {
      tolerance: OFFLINE_WATCHDOG_TOLERANCE,
      historySize: Math.min(files.length, 360),
    },
  });

  const su7Accumulator = {
    frames: 0,
    determinantDriftMax: 0,
    determinantDriftSum: 0,
    unitaryErrorMax: 0,
    unitaryErrorSum: 0,
    normDeltaMax: 0,
    normDeltaMeanSum: 0,
    projectorEnergyMax: 0,
    projectorEnergySum: 0,
    geodesicFallbacks: 0,
  };

  let kurState: ReturnType<typeof createKuramotoState> | null = null;
  let kurDerivedBuffer: ArrayBuffer | null = null;
  let kurDerived: ReturnType<typeof createDerivedViews> | null = null;
  const kurParams: KuramotoParams = { ...DEFAULT_KUR_PARAMS };
  const frameDt = 1 / info.fps;
  const clampSeed = (seed: number) => seed >>> 0;

  for (let index = 0; index < files.length; index++) {
    const filename = files[index]!;
    const sourcePath = join(decodeDir, filename);
    const rgbaRaw = await decodeFrameToRgba(options.ffmpeg, sourcePath, info.width, info.height);
    const rgbaClamped = new Uint8ClampedArray(
      rgbaRaw.buffer,
      rgbaRaw.byteOffset,
      rgbaRaw.byteLength,
    );
    const image: ImageBuffer = { data: rgbaClamped, width: info.width, height: info.height };
    const rim = computeEdgeField(image);

    const evaluation = timeline ? timeline.evaluateAtFrame(index) : null;
    const frameParams = resolveFrameParameters(evaluation, DEFAULTS);
    kernel = buildKernel(evaluation);

    const surface = {
      kind: 'surface' as const,
      resolution: rim.resolution,
      rgba: rgbaClamped,
    };

    let phaseField: PhaseField | null = null;
    if (frameParams.kurEnabled || options.useKuramoto) {
      if (!kurState) {
        kurState = createKuramotoState(info.width, info.height);
        const buffer = new ArrayBuffer(derivedBufferSize(info.width, info.height));
        kurDerivedBuffer = buffer;
        kurDerived = createDerivedViews(buffer, info.width, info.height);
        initKuramotoState(kurState, 1, kurDerived);
        deriveKuramotoFields(kurState, kurDerived, {
          kernel,
          controls: { dmt: frameParams.dmt },
          params: kurParams,
        });
      }
      if (kurState && kurDerived) {
        const seed =
          timeline && timeline.hasScopedSeed('kuramotoNoise')
            ? timeline.getSeedAtFrame('kuramotoNoise', index)
            : clampSeed(deriveSeedFromHash(timelineHash, 'kuramotoNoise', index));
        const randn = createNormalGenerator(seed);
        stepKuramotoState(kurState, kurParams, frameDt, randn, (index + 1) * frameDt, {
          kernel,
          controls: { dmt: frameParams.dmt },
        });
        deriveKuramotoFields(kurState, kurDerived, {
          kernel,
          controls: { dmt: frameParams.dmt },
        });
        phaseField = {
          kind: 'phase',
          resolution: kurDerived.resolution,
          gradX: kurDerived.gradX,
          gradY: kurDerived.gradY,
          vort: kurDerived.vort,
          coh: kurDerived.coh,
          amp: kurDerived.amp,
        };
      }
    } else {
      kurState = null;
      kurDerived = null;
      kurDerivedBuffer = null;
    }

    const timeSeconds = index * frameDt;
    const frameResult = await renderer.renderFrame({ frameIndex: index }, (outBuffer) =>
      renderRainbowFrame({
        width: info.width,
        height: info.height,
        timeSeconds,
        out: outBuffer,
        surface,
        rim,
        phase: phaseField,
        volume: null,
        kernel,
        dmt: frameParams.dmt,
        arousal: frameParams.arousal,
        blend: frameParams.blend,
        normPin: frameParams.normPin,
        normTarget,
        lastObs,
        lambdaRef: LAMBDA_REF,
        lambdas: LAMBDAS,
        beta2: frameParams.beta2,
        microsaccade: frameParams.microsaccade,
        alive: frameParams.alive,
        phasePin: frameParams.phasePin,
        edgeThreshold: frameParams.edgeThreshold,
        wallpaperGroup: frameParams.wallGroup,
        surfEnabled: false,
        orientationAngles: ORIENTATION_ANGLES,
        thetaMode: frameParams.thetaMode,
        thetaGlobal: frameParams.thetaGlobal,
        polBins: frameParams.polBins,
        jitter: frameParams.jitter,
        coupling: DEFAULT_COUPLING,
        couplingBase: DEFAULT_COUPLING,
        sigma: frameParams.sigma,
        contrast: frameParams.contrast,
        rimAlpha: frameParams.rimAlpha,
        rimEnabled: frameParams.rimEnabled,
        displayMode: frameParams.displayMode,
        surfaceBlend: frameParams.surfaceBlend,
        surfaceRegion: frameParams.surfaceRegion,
        warpAmp: frameParams.warpAmp,
        curvatureStrength: frameParams.curvatureStrength,
        curvatureMode: frameParams.curvatureMode,
        hyperbolicAtlas: null,
        kurEnabled: frameParams.kurEnabled || options.useKuramoto,
        su7: su7Params,
        composer,
        guardrailOptions: { emitGuardrailEvents: false },
      }),
    );

    const rainbow = frameResult.rainbow;
    if (rainbow.obsAverage != null && Number.isFinite(rainbow.obsAverage)) {
      lastObs = rainbow.obsAverage;
    }
    const su = rainbow.metrics.su7;
    const driftAbs = Math.abs(su.determinantDrift);
    su7Accumulator.frames += 1;
    su7Accumulator.determinantDriftMax = Math.max(su7Accumulator.determinantDriftMax, driftAbs);
    su7Accumulator.determinantDriftSum += driftAbs;
    su7Accumulator.unitaryErrorMax = Math.max(su7Accumulator.unitaryErrorMax, su.unitaryError);
    su7Accumulator.unitaryErrorSum += su.unitaryError;
    su7Accumulator.normDeltaMax = Math.max(su7Accumulator.normDeltaMax, su.normDeltaMax);
    su7Accumulator.normDeltaMeanSum += su.normDeltaMean;
    su7Accumulator.projectorEnergyMax = Math.max(
      su7Accumulator.projectorEnergyMax,
      su.projectorEnergy,
    );
    su7Accumulator.projectorEnergySum += su.projectorEnergy;
    su7Accumulator.geodesicFallbacks += su.geodesicFallbacks;

    const outputPath = join(processedDir, filename);
    const frameBuffer = Buffer.from(
      frameResult.output10Bit.buffer,
      frameResult.output10Bit.byteOffset ?? 0,
      frameResult.output10Bit.byteLength,
    );
    await encodeRgbaToPng(options.ffmpeg, frameBuffer, info.width, info.height, outputPath, {
      bitDepth: 10,
    });
  }

  const performance = renderer.getPerformanceSnapshot();
  const frames = su7Accumulator.frames || files.length;

  return {
    frameCount: files.length,
    performance,
    su7: {
      frames,
      determinantDriftMax: su7Accumulator.determinantDriftMax,
      determinantDriftMean: frames ? su7Accumulator.determinantDriftSum / frames : 0,
      unitaryErrorMax: su7Accumulator.unitaryErrorMax,
      unitaryErrorMean: frames ? su7Accumulator.unitaryErrorSum / frames : 0,
      normDeltaMax: su7Accumulator.normDeltaMax,
      normDeltaMean: frames ? su7Accumulator.normDeltaMeanSum / frames : 0,
      projectorEnergyMax: su7Accumulator.projectorEnergyMax,
      projectorEnergyMean: frames ? su7Accumulator.projectorEnergySum / frames : 0,
      geodesicFallbacks: su7Accumulator.geodesicFallbacks,
    },
  };
};

const writeMetadata = async (
  outputPath: string,
  info: VideoProbe,
  timeline: TimelinePlayer | null,
  summary: OfflineProcessSummary,
) => {
  const performanceHistory = summary.performance.history.map((sample) => ({
    frameIndex: sample.frameIndex,
    frameMs: Number(sample.frameMs.toFixed(3)),
    rssMb: Number(sample.rssMb.toFixed(2)),
    heapMb: Number(sample.heapMb.toFixed(2)),
    cpuPercent: Number(sample.cpuPercent.toFixed(1)),
  }));
  const metadata = {
    source: basename(outputPath),
    width: info.width,
    height: info.height,
    fps: info.fps,
    frames: summary.frameCount,
    timeline: timeline
      ? {
          hash: timeline.hash,
          fps: timeline.fps,
          durationFrames: timeline.durationFrames,
        }
      : null,
    performance: {
      budgets: OFFLINE_RENDER_BUDGETS,
      frames: summary.performance.frames,
      frameMsAvg: Number(summary.performance.frameMsAvg.toFixed(3)),
      frameMsMax: Number(summary.performance.frameMsMax.toFixed(3)),
      rssMbMax: Number(summary.performance.rssMbMax.toFixed(1)),
      heapMbMax: Number(summary.performance.heapMbMax.toFixed(1)),
      cpuPercentAvg: Number(summary.performance.cpuPercentAvg.toFixed(2)),
      cpuPercentMax: Number(summary.performance.cpuPercentMax.toFixed(2)),
      lastSample: summary.performance.lastSample,
      history: performanceHistory,
      violations: summary.performance.violations,
    },
    su7: {
      frames: summary.su7.frames,
      determinantDriftMax: summary.su7.determinantDriftMax,
      determinantDriftMean: summary.su7.determinantDriftMean,
      unitaryErrorMax: summary.su7.unitaryErrorMax,
      unitaryErrorMean: summary.su7.unitaryErrorMean,
      normDeltaMax: summary.su7.normDeltaMax,
      normDeltaMean: summary.su7.normDeltaMean,
      projectorEnergyMax: summary.su7.projectorEnergyMax,
      projectorEnergyMean: summary.su7.projectorEnergyMean,
      geodesicFallbacks: summary.su7.geodesicFallbacks,
    },
  };
  const metaPath = `${outputPath}.meta.json`;
  await fsPromises.writeFile(metaPath, `${JSON.stringify(metadata, null, 2)}\n`, 'utf8');
};

const main = async () => {
  const options = parseArgs(process.argv.slice(2));
  const resolvedInput = resolve(process.cwd(), options.input);
  const resolvedOutput = resolve(process.cwd(), options.output);

  const videoInfo = await probeVideo(options.ffprobe, resolvedInput);
  const fps = options.fpsOverride ?? videoInfo.fps;
  const { root, decodeDir, processedDir } = await ensureTempDirs();
  try {
    await runCommand(options.ffmpeg, [
      '-v',
      'error',
      '-y',
      '-i',
      resolvedInput,
      '-vf',
      'format=rgba',
      join(decodeDir, 'frame_%05d.png'),
    ]);

    const timeline = await decodeTimeline(options.timelinePath);
    if (timeline && Math.abs(timeline.fps - fps) > 1e-3) {
      console.warn(
        `[timeline] fps mismatch – timeline ${timeline.fps.toFixed(
          3,
        )} vs video ${fps.toFixed(3)}; using video fps`,
      );
    }

    const summary = await processFrames(
      options,
      { ...videoInfo, fps },
      decodeDir,
      processedDir,
      timeline,
    );

    const processedFrames = (await fsPromises.readdir(processedDir)).filter((name) =>
      name.toLowerCase().endsWith('.png'),
    );
    if (processedFrames.length !== summary.frameCount) {
      console.warn(
        `[offline] processed frame count mismatch: files=${processedFrames.length} summary=${summary.frameCount}`,
      );
    }
    const pattern = join(processedDir, 'frame_%05d.png');
    await encodeVideoFromPng(options.ffmpeg, pattern, fps, resolvedOutput);
    await writeMetadata(resolvedOutput, { ...videoInfo, fps }, timeline, summary);
    console.log(
      `[video] wrote ${summary.frameCount} 10-bit frames → ${resolvedOutput} (${videoInfo.width}×${videoInfo.height} @ ${fps.toFixed(
        3,
      )} fps; avg frame ${summary.performance.frameMsAvg.toFixed(2)}ms, max ${summary.performance.frameMsMax.toFixed(
        2,
      )}ms)`,
    );
    if (summary.performance.violations.length > 0) {
      console.warn(
        `[watchdog] ${summary.performance.violations.length} budget violation(s) recorded – see metadata for details.`,
      );
    } else {
      console.log('[watchdog] budgets respected (no violations recorded).');
    }
    console.log(
      `[su7] determinant drift max ${summary.su7.determinantDriftMax.toExponential(
        2,
      )}, mean ${summary.su7.determinantDriftMean.toExponential(2)}; geodesic fallbacks ${summary.su7.geodesicFallbacks}`,
    );
  } finally {
    if (!options.keepTemp) {
      await fsPromises.rm(root, { recursive: true, force: true });
    } else {
      console.warn(`[video] keeping temporary directory ${root}`);
    }
  }
};

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
