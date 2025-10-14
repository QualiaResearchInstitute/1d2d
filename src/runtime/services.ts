import { mkdtemp, mkdir, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { runMediaPipeline } from '../media/mediaPipeline.js';
import type { ImageBuffer } from '../pipeline/edgeDetection.js';
import {
  renderRainbowFrame,
  type RainbowFrameResult,
  type PhaseField,
  type SurfaceField,
} from '../pipeline/rainbowFrame.js';
import {
  createKuramotoState,
  createDerivedViews,
  createNormalGenerator,
  deriveKuramotoFields,
  derivedBufferSize,
  initKuramotoState,
  stepKuramotoState,
  type KuramotoParams,
} from '../kuramotoCore.js';
import { makeResolution, type RimField } from '../fields/contracts.js';

import {
  DEFAULT_COUPLING,
  getDefaultSu7,
  loadManifest,
  resolvePresetConfig,
  type FrameParameters,
  type ResolvedPresetConfig,
} from '../cli/utils/preset.js';
import { decodeFrame, encodeImage, probeMedia } from '../cli/utils/ffmpeg.js';
import { runCommand } from '../cli/utils/exec.js';

const ORIENTATION_ANGLES = [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2];
const LAMBDA_REF = 520;
const LAMBDAS = { L: 560, M: 530, S: 420 } as const;

export type RuntimeConfigResult = {
  config: ResolvedPresetConfig;
  manifestPath?: string;
  presetId?: string;
};

export const resolveRuntimeConfig = async (
  manifestPath?: string,
  presetId?: string,
): Promise<RuntimeConfigResult> => {
  if (!manifestPath) {
    const defaults = resolvePresetConfig(
      {
        schemaVersion: '1.0.0',
        metadata: {
          name: 'Default',
        },
        nodes: [],
        links: [],
        controls: undefined,
      },
      undefined,
    );
    return { config: defaults };
  }
  const { manifest } = await loadManifest(manifestPath);
  const resolved = resolvePresetConfig(manifest, presetId);
  return { config: resolved, manifestPath, presetId };
};

const cloneComposer = (config: ResolvedPresetConfig, frame: FrameParameters) => ({
  ...config.composer,
  fields: {
    surface: {
      ...config.composer.fields.surface,
      weight: frame.surfaceBlend,
    },
    rim: {
      ...config.composer.fields.rim,
      weight: frame.rimEnabled ? config.composer.fields.rim.weight : 0,
    },
    kur: {
      ...config.composer.fields.kur,
      weight: frame.kurEnabled ? config.composer.fields.kur.weight : 0,
    },
    volume: {
      ...config.composer.fields.volume,
    },
  },
});

const buildPhaseFieldFromKuramoto = (
  derived: ReturnType<typeof createDerivedViews>,
  width: number,
  height: number,
): PhaseField => ({
  kind: 'phase',
  resolution: makeResolution(width, height),
  gradX: derived.gradX,
  gradY: derived.gradY,
  vort: derived.vort,
  coh: derived.coh,
  amp: derived.amp,
});

export type ApplyOptions = {
  input: string;
  output: string;
  ffmpeg: string;
  ffprobe: string;
  manifest?: string;
  preset?: string;
  bitDepth: 8 | 10 | 16;
};

export type ApplyResult = {
  output: string;
  width: number;
  height: number;
  manifest: string | null;
  preset: string | null;
  telemetry: ReturnType<typeof runMediaPipeline>['telemetry'];
  metrics: {
    rimMean: number;
    warpMean: number;
    coherenceMean: number;
    indraIndex: number;
  };
};

export const applyFrame = async (options: ApplyOptions): Promise<ApplyResult> => {
  const probe = await probeMedia(options.ffprobe, options.input);
  const decoded = await decodeFrame(options.ffmpeg, options.input, probe.width, probe.height);
  const pixelData = Uint8ClampedArray.from(decoded);
  const image: ImageBuffer = {
    data: pixelData,
    width: probe.width,
    height: probe.height,
  };

  const { config, manifestPath, presetId } = await resolveRuntimeConfig(
    options.manifest,
    options.preset,
  );
  const pipeline = runMediaPipeline(image, {
    kuramoto: config.frame.kurEnabled
      ? { enabled: true, params: config.kuramoto, steps: 16, dt: 0.016, seed: 1337 }
      : { enabled: false },
  });
  const rim: RimField = pipeline.rim;
  const phaseField: PhaseField =
    (config.frame.kurEnabled && pipeline.kuramoto?.phase) || pipeline.phase.field;

  const surface: SurfaceField = {
    kind: 'surface',
    resolution: makeResolution(image.width, image.height),
    rgba: pixelData,
  };

  const frame = config.frame;
  const composer = cloneComposer(config, frame);
  const outBuffer = new Uint8ClampedArray(image.width * image.height * 4);
  const rainbow = renderRainbowFrame({
    width: image.width,
    height: image.height,
    timeSeconds: 0,
    out: outBuffer,
    surface,
    rim,
    phase: phaseField,
    volume: null,
    kernel: config.kernel,
    dmt: frame.dmt,
    arousal: frame.arousal,
    blend: frame.blend,
    normPin: frame.normPin,
    normTarget: 1,
    lastObs: 1,
    lambdaRef: LAMBDA_REF,
    lambdas: LAMBDAS,
    beta2: frame.beta2,
    microsaccade: frame.microsaccade,
    alive: frame.alive,
    phasePin: frame.phasePin,
    edgeThreshold: frame.edgeThreshold,
    wallpaperGroup: frame.wallGroup,
    surfEnabled: frame.surfaceBlend > 0,
    orientationAngles: ORIENTATION_ANGLES,
    thetaMode: frame.thetaMode,
    thetaGlobal: frame.thetaGlobal,
    polBins: frame.polBins,
    jitter: frame.jitter,
    coupling: config.coupling,
    couplingBase: DEFAULT_COUPLING,
    sigma: frame.sigma,
    contrast: frame.contrast,
    rimAlpha: frame.rimAlpha,
    rimEnabled: frame.rimEnabled,
    displayMode: frame.displayMode,
    surfaceBlend: frame.surfaceBlend,
    surfaceRegion: frame.surfaceRegion,
    warpAmp: frame.warpAmp,
    curvatureStrength: frame.curvatureStrength,
    curvatureMode: frame.curvatureMode,
    hyperbolicAtlas: null,
    kurEnabled: frame.kurEnabled,
    su7: getDefaultSu7(),
    composer,
    attentionHooks: undefined,
    guardrailOptions: { emitGuardrailEvents: false },
    fluxOverlay: null,
  });

  await encodeImage(options.ffmpeg, outBuffer, image.width, image.height, options.output, {
    bitDepth: options.bitDepth,
  });

  return {
    output: resolve(process.cwd(), options.output),
    width: image.width,
    height: image.height,
    manifest: manifestPath ?? null,
    preset: presetId ?? null,
    telemetry: pipeline.telemetry,
    metrics: {
      rimMean: rainbow.metrics.rim.mean,
      warpMean: rainbow.metrics.warp.mean,
      coherenceMean: rainbow.metrics.gradient.cohMean,
      indraIndex: rainbow.metrics.qualia.indraIndex,
    },
  };
};

export type SimulationOptions = {
  input: string;
  ffmpeg: string;
  ffprobe: string;
  manifest?: string;
  preset?: string;
  frames: number;
  dt: number;
  seed: number;
};

export type SimulationSummary = {
  frames: number;
  dt: number;
  manifest: string | null;
  preset: string | null;
  metrics: {
    rimMean: number;
    cohMean: number;
    indraIndex: number;
  };
};

export const simulate = async (options: SimulationOptions): Promise<SimulationSummary> => {
  const probe = await probeMedia(options.ffprobe, options.input);
  const decoded = await decodeFrame(options.ffmpeg, options.input, probe.width, probe.height);
  const pixelData = Uint8ClampedArray.from(decoded);
  const image: ImageBuffer = { data: pixelData, width: probe.width, height: probe.height };
  const { config, manifestPath, presetId } = await resolveRuntimeConfig(
    options.manifest,
    options.preset,
  );
  const { results } = await simulateFramesInternal(config, image, {
    frames: options.frames,
    dt: options.dt,
    seed: options.seed,
  });
  const summary = results.reduce(
    (acc, entry) => {
      acc.rimMean += entry.metrics.rim.mean;
      acc.cohMean += entry.metrics.gradient.cohMean;
      acc.indraIndex += entry.metrics.qualia.indraIndex;
      acc.framesProcessed += 1;
      return acc;
    },
    { rimMean: 0, cohMean: 0, indraIndex: 0, framesProcessed: 0 },
  );
  if (summary.framesProcessed > 0) {
    summary.rimMean /= summary.framesProcessed;
    summary.cohMean /= summary.framesProcessed;
    summary.indraIndex /= summary.framesProcessed;
  }
  return {
    frames: options.frames,
    dt: options.dt,
    manifest: manifestPath ?? null,
    preset: presetId ?? null,
    metrics: {
      rimMean: summary.rimMean,
      cohMean: summary.cohMean,
      indraIndex: summary.indraIndex,
    },
  };
};

type SimulationInternalOptions = {
  frames: number;
  dt: number;
  seed: number;
};

const simulateFramesInternal = async (
  config: ResolvedPresetConfig,
  image: ImageBuffer,
  options: SimulationInternalOptions,
): Promise<{ frames: number; results: RainbowFrameResult[] }> => {
  const rim = runMediaPipeline(image, { kuramoto: { enabled: false } }).rim;
  const surface: SurfaceField = {
    kind: 'surface',
    resolution: makeResolution(image.width, image.height),
    rgba: image.data,
  };
  const frame = config.frame;
  let phaseField: PhaseField | null = null;
  let kurState: ReturnType<typeof createKuramotoState> | null = null;
  let derived: ReturnType<typeof createDerivedViews> | null = null;
  let derivedBuffer: ArrayBuffer | null = null;
  if (frame.kurEnabled) {
    kurState = createKuramotoState(image.width, image.height);
    derivedBuffer = new ArrayBuffer(derivedBufferSize(image.width, image.height));
    derived = createDerivedViews(derivedBuffer, image.width, image.height);
    initKuramotoState(kurState, 1, derived);
    deriveKuramotoFields(kurState, derived, { params: config.kuramoto });
    phaseField = buildPhaseFieldFromKuramoto(derived, image.width, image.height);
  } else {
    phaseField = runMediaPipeline(image, { kuramoto: { enabled: false } }).phase.field;
  }

  const results: RainbowFrameResult[] = [];
  const composerBase = cloneComposer(config, frame);
  let lastObs = 1;
  const randSeed = options.seed >>> 0;

  for (let frameIndex = 0; frameIndex < options.frames; frameIndex++) {
    if (frame.kurEnabled && kurState && derived) {
      const randn = createNormalGenerator(randSeed + frameIndex);
      stepKuramotoState(
        kurState,
        config.kuramoto,
        options.dt,
        randn,
        (frameIndex + 1) * options.dt,
        { params: config.kuramoto },
      );
      deriveKuramotoFields(kurState, derived, { params: config.kuramoto });
      phaseField = buildPhaseFieldFromKuramoto(derived, image.width, image.height);
    }
    const out = new Uint8ClampedArray(image.width * image.height * 4);
    const rainbow = renderRainbowFrame({
      width: image.width,
      height: image.height,
      timeSeconds: frameIndex * options.dt,
      out,
      surface,
      rim,
      phase: phaseField,
      volume: null,
      kernel: config.kernel,
      dmt: frame.dmt,
      arousal: frame.arousal,
      blend: frame.blend,
      normPin: frame.normPin,
      normTarget: 1,
      lastObs,
      lambdaRef: LAMBDA_REF,
      lambdas: LAMBDAS,
      beta2: frame.beta2,
      microsaccade: frame.microsaccade,
      alive: frame.alive,
      phasePin: frame.phasePin,
      edgeThreshold: frame.edgeThreshold,
      wallpaperGroup: frame.wallGroup,
      surfEnabled: frame.surfaceBlend > 0,
      orientationAngles: ORIENTATION_ANGLES,
      thetaMode: frame.thetaMode,
      thetaGlobal: frame.thetaGlobal,
      polBins: frame.polBins,
      jitter: frame.jitter,
      coupling: config.coupling,
      couplingBase: DEFAULT_COUPLING,
      sigma: frame.sigma,
      contrast: frame.contrast,
      rimAlpha: frame.rimAlpha,
      rimEnabled: frame.rimEnabled,
      displayMode: frame.displayMode,
      surfaceBlend: frame.surfaceBlend,
      surfaceRegion: frame.surfaceRegion,
      warpAmp: frame.warpAmp,
      curvatureStrength: frame.curvatureStrength,
      curvatureMode: frame.curvatureMode,
      hyperbolicAtlas: null,
      kurEnabled: frame.kurEnabled,
      su7: getDefaultSu7(),
      composer: composerBase,
      guardrailOptions: { emitGuardrailEvents: false },
      fluxOverlay: null,
    });
    if (rainbow.obsAverage != null && Number.isFinite(rainbow.obsAverage)) {
      lastObs = rainbow.obsAverage;
    }
    results.push(rainbow);
  }

  return { frames: options.frames, results };
};

export type CaptureOptions = {
  input: string;
  output: string;
  ffmpeg: string;
  ffprobe: string;
  manifest?: string;
  preset?: string;
  framesLimit?: number;
  keepTemp: boolean;
};

export type CaptureSummary = {
  frames: number;
  width: number;
  height: number;
  fps: number | null;
  durationSeconds: number;
  output: string;
};

export const captureVideo = async (options: CaptureOptions): Promise<CaptureSummary> => {
  const info = await probeMedia(options.ffprobe, options.input);
  const tempRoot = await mkdtemp(join(tmpdir(), 'indra-capture-'));
  const decodeDir = join(tempRoot, 'decoded');
  const processedDir = join(tempRoot, 'processed');
  await mkdir(decodeDir, { recursive: true });
  await mkdir(processedDir, { recursive: true });

  try {
    const pattern = join(decodeDir, 'frame_%05d.png');
    await runCommand(options.ffmpeg, ['-v', 'error', '-i', options.input, pattern]);
    const frames = (await readdir(decodeDir))
      .filter((name) => name.toLowerCase().endsWith('.png'))
      .sort();
    if (frames.length === 0) {
      throw new Error('No frames decoded from input video.');
    }

    const { config } = await resolveRuntimeConfig(options.manifest, options.preset);
    let processedCount = 0;
    const start = Date.now();

    for (let index = 0; index < frames.length; index++) {
      if (options.framesLimit && processedCount >= options.framesLimit) {
        break;
      }
      const frameName = frames[index]!;
      const framePath = join(decodeDir, frameName);
      const decoded = await decodeFrame(options.ffmpeg, framePath, info.width, info.height);
      const data = Uint8ClampedArray.from(decoded);
      const image: ImageBuffer = { data, width: info.width, height: info.height };
      const pipeline = runMediaPipeline(image, {
        kuramoto: config.frame.kurEnabled
          ? { enabled: true, params: config.kuramoto, steps: 16, dt: 0.016, seed: 1337 + index }
          : { enabled: false },
      });
      const surface: SurfaceField = {
        kind: 'surface',
        resolution: makeResolution(info.width, info.height),
        rgba: data,
      };
      const rim = pipeline.rim;
      const phase: PhaseField =
        (config.frame.kurEnabled && pipeline.kuramoto?.phase) || pipeline.phase.field;
      const composer = cloneComposer(config, config.frame);
      const out = new Uint8ClampedArray(info.width * info.height * 4);
      renderRainbowFrame({
        width: info.width,
        height: info.height,
        timeSeconds: index / Math.max(info.fps ?? 60, 1),
        out,
        surface,
        rim,
        phase,
        volume: null,
        kernel: config.kernel,
        dmt: config.frame.dmt,
        arousal: config.frame.arousal,
        blend: config.frame.blend,
        normPin: config.frame.normPin,
        normTarget: 1,
        lastObs: 1,
        lambdaRef: LAMBDA_REF,
        lambdas: LAMBDAS,
        beta2: config.frame.beta2,
        microsaccade: config.frame.microsaccade,
        alive: config.frame.alive,
        phasePin: config.frame.phasePin,
        edgeThreshold: config.frame.edgeThreshold,
        wallpaperGroup: config.frame.wallGroup,
        surfEnabled: config.frame.surfaceBlend > 0,
        orientationAngles: ORIENTATION_ANGLES,
        thetaMode: config.frame.thetaMode,
        thetaGlobal: config.frame.thetaGlobal,
        polBins: config.frame.polBins,
        jitter: config.frame.jitter,
        coupling: config.coupling,
        couplingBase: DEFAULT_COUPLING,
        sigma: config.frame.sigma,
        contrast: config.frame.contrast,
        rimAlpha: config.frame.rimAlpha,
        rimEnabled: config.frame.rimEnabled,
        displayMode: config.frame.displayMode,
        surfaceBlend: config.frame.surfaceBlend,
        surfaceRegion: config.frame.surfaceRegion,
        warpAmp: config.frame.warpAmp,
        curvatureStrength: config.frame.curvatureStrength,
        curvatureMode: config.frame.curvatureMode,
        hyperbolicAtlas: null,
        kurEnabled: config.frame.kurEnabled,
        su7: getDefaultSu7(),
        composer,
        guardrailOptions: { emitGuardrailEvents: false },
        fluxOverlay: null,
      });
      const outputFrame = join(processedDir, frameName);
      await encodeImage(options.ffmpeg, out, info.width, info.height, outputFrame, { bitDepth: 8 });
      processedCount += 1;
    }

    const processedPattern = join(processedDir, 'frame_%05d.png');
    await runCommand(options.ffmpeg, [
      '-v',
      'error',
      '-y',
      '-framerate',
      (info.fps ?? 60).toString(),
      '-i',
      processedPattern,
      '-c:v',
      options.output.toLowerCase().endsWith('.png') ? 'png' : 'libx264',
      '-pix_fmt',
      options.output.toLowerCase().endsWith('.mov') ? 'yuva444p10le' : 'yuv420p',
      options.output,
    ]);

    const elapsed = (Date.now() - start) / 1000;
    return {
      frames: processedCount,
      width: info.width,
      height: info.height,
      fps: info.fps ?? null,
      durationSeconds: elapsed,
      output: resolve(process.cwd(), options.output),
    };
  } finally {
    if (!options.keepTemp) {
      await rm(tempRoot, { recursive: true, force: true });
    }
  }
};
