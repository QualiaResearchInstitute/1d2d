import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { createDefaultComposerConfig } from '../../pipeline/rainbowFrame.js';
import {
  type ComposerConfig,
  type CouplingConfig,
  type CurvatureMode,
  type DisplayMode,
  type SurfaceRegion,
  type WallpaperGroup,
  createDefaultSu7RuntimeParams,
} from '../../pipeline/rainbowFrame.js';
import {
  createKernelSpec,
  getDefaultKernelSpec,
  type KernelSpec,
} from '../../kernel/kernelSpec.js';
import type { KuramotoParams } from '../../kuramotoCore.js';
import { loadManifestFromJson, type ManifestLoadResultSuccess } from '../../manifest/loader.js';
import type { ManifestPreset, SceneManifest } from '../../manifest/types.js';

export type FrameParameters = {
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

export type ResolvedPresetConfig = {
  frame: FrameParameters;
  kernel: KernelSpec;
  coupling: CouplingConfig;
  composer: ComposerConfig;
  kuramoto: KuramotoParams;
};

export const DEFAULT_COUPLING: CouplingConfig = {
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

const DEFAULT_FRAME_PARAMS: FrameParameters = {
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
  surfaceRegion: 'surfaces',
  warpAmp: 1.0,
  curvatureStrength: 0,
  curvatureMode: 'poincare',
  displayMode: 'color',
  wallGroup: 'p4',
  thetaMode: 'gradient',
  thetaGlobal: 0,
  polBins: 16,
  kurEnabled: false,
};

const DEFAULT_KURAMOTO: KuramotoParams = {
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

const coerceNumber = (value: unknown, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

const coerceBoolean = (value: unknown, fallback: boolean): boolean =>
  typeof value === 'boolean' ? value : fallback;

const coerceString = <T extends string>(value: unknown, fallback: T): T =>
  typeof value === 'string' ? (value as T) : fallback;

const resolvePresetPanel = (preset: ManifestPreset | undefined, panelId: string) => {
  const panels = preset?.panels;
  if (!panels || typeof panels !== 'object') return undefined;
  const panel = (panels as Record<string, unknown>)[panelId];
  return (
    (panel && typeof panel === 'object' ? (panel as Record<string, unknown>) : undefined) ??
    undefined
  );
};

const getNumber = (panel: Record<string, unknown> | undefined, key: string, fallback: number) =>
  panel && key in panel
    ? coerceNumber((panel as Record<string, unknown>)[key], fallback)
    : fallback;

const getBooleanFromPanel = (
  panel: Record<string, unknown> | undefined,
  key: string,
  fallback: boolean,
) => (panel && key in panel ? coerceBoolean(panel[key], fallback) : fallback);

const getStringFromPanel = <T extends string>(
  panel: Record<string, unknown> | undefined,
  key: string,
  fallback: T,
) => (panel && key in panel ? coerceString(panel[key], fallback) : fallback);

const applyFrameOverrides = (
  base: FrameParameters,
  preset: ManifestPreset | undefined,
): FrameParameters => {
  const tracer = resolvePresetPanel(preset, 'tracer');
  const compositor = resolvePresetPanel(preset, 'compositor');
  const hyperbolic = resolvePresetPanel(preset, 'hyperbolic');
  const display = resolvePresetPanel(preset, 'display');
  const output = resolvePresetPanel(preset, 'output');
  const kuramoto = resolvePresetPanel(preset, 'kuramoto');

  return {
    edgeThreshold: getNumber(tracer, 'edgeThreshold', base.edgeThreshold),
    blend: getNumber(compositor, 'blend', base.blend),
    dmt: getNumber(resolvePresetPanel(preset, 'dmt'), 'value', base.dmt),
    arousal: getNumber(compositor, 'arousal', base.arousal),
    normPin: getBooleanFromPanel(tracer, 'normPin', base.normPin),
    beta2: getNumber(tracer, 'beta2', base.beta2),
    microsaccade: getBooleanFromPanel(tracer, 'microsaccade', base.microsaccade),
    alive: getBooleanFromPanel(tracer, 'alive', base.alive),
    phasePin: getBooleanFromPanel(tracer, 'phasePin', base.phasePin),
    jitter: getNumber(tracer, 'jitter', base.jitter),
    sigma: getNumber(tracer, 'sigma', base.sigma),
    contrast: getNumber(compositor, 'contrast', base.contrast),
    rimAlpha: getNumber(display ?? output, 'rimAlpha', base.rimAlpha),
    rimEnabled: getBooleanFromPanel(display ?? output, 'rimEnabled', base.rimEnabled),
    surfaceBlend: getNumber(compositor, 'surfaceBlend', base.surfaceBlend),
    surfaceRegion: getStringFromPanel(hyperbolic, 'surfaceRegion', base.surfaceRegion),
    warpAmp: getNumber(hyperbolic, 'warpAmp', base.warpAmp),
    curvatureStrength: getNumber(hyperbolic, 'curvatureStrength', base.curvatureStrength),
    curvatureMode: getStringFromPanel(hyperbolic, 'curvatureMode', base.curvatureMode),
    displayMode: getStringFromPanel(display ?? output, 'displayMode', base.displayMode),
    wallGroup: getStringFromPanel(hyperbolic, 'wallpaperGroup', base.wallGroup),
    thetaMode: getStringFromPanel<'gradient' | 'global'>(tracer, 'thetaMode', base.thetaMode),
    thetaGlobal: getNumber(tracer, 'thetaGlobal', base.thetaGlobal),
    polBins: Math.max(1, Math.round(getNumber(tracer, 'polBins', base.polBins))),
    kurEnabled: getBooleanFromPanel(kuramoto, 'kurEnabled', base.kurEnabled),
  };
};

const applyKernelOverrides = (preset: ManifestPreset | undefined): KernelSpec => {
  const defaults = getDefaultKernelSpec();
  const optics = resolvePresetPanel(preset, 'optics');
  if (!optics) return defaults;
  return createKernelSpec({
    gain: coerceNumber(optics.gain, defaults.gain),
    k0: coerceNumber(optics.k0, defaults.k0),
    Q: coerceNumber(optics.Q ?? optics.kq ?? defaults.Q, defaults.Q),
    anisotropy: coerceNumber(optics.anisotropy, defaults.anisotropy),
    chirality: coerceNumber(optics.chirality, defaults.chirality),
    transparency: coerceNumber(optics.transparency, defaults.transparency),
  });
};

const applyKuramotoOverrides = (preset: ManifestPreset | undefined): KuramotoParams => {
  const panel = resolvePresetPanel(preset, 'kuramoto');
  const base = { ...DEFAULT_KURAMOTO };
  if (!panel) return base;
  return {
    alphaKur: coerceNumber(panel.alphaKur, base.alphaKur),
    gammaKur: coerceNumber(panel.gammaKur, base.gammaKur),
    omega0: coerceNumber(panel.omega0, base.omega0),
    K0: coerceNumber(panel.K0, base.K0),
    epsKur: coerceNumber(panel.epsKur, base.epsKur),
    fluxX: coerceNumber(panel.fluxX, base.fluxX),
    fluxY: coerceNumber(panel.fluxY, base.fluxY),
    smallWorldWeight: coerceNumber(panel.smallWorldWeight, base.smallWorldWeight),
    p_sw: coerceNumber(panel.p_sw, base.p_sw),
    smallWorldEnabled: coerceBoolean(panel.smallWorldEnabled, base.smallWorldEnabled ?? false),
    smallWorldDegree: coerceNumber(panel.smallWorldDegree, base.smallWorldDegree ?? 12),
    smallWorldSeed: coerceNumber(panel.smallWorldSeed, base.smallWorldSeed ?? 1337),
  };
};

export const resolvePresetConfig = (
  manifest: SceneManifest,
  presetId?: string,
): ResolvedPresetConfig => {
  const preset =
    presetId && manifest.controls?.presets
      ? manifest.controls.presets.find((entry) => entry.id === presetId)
      : undefined;
  const frame = applyFrameOverrides({ ...DEFAULT_FRAME_PARAMS }, preset);
  const kernel = applyKernelOverrides(preset);
  const kuramoto = applyKuramotoOverrides(preset);
  const composer = createDefaultComposerConfig();
  const coupling = { ...DEFAULT_COUPLING };
  return { frame, kernel, composer, coupling, kuramoto };
};

export type LoadedManifest = {
  manifest: SceneManifest;
  issues: ManifestLoadResultSuccess['issues'];
};

export const loadManifest = async (path: string): Promise<LoadedManifest> => {
  const resolved = resolve(process.cwd(), path);
  const payload = await readFile(resolved, 'utf8');
  const result = await loadManifestFromJson(payload, resolved);
  if (result.kind === 'error') {
    const error = new Error(`Manifest ${resolved} invalid: ${result.message}`);
    (error as any).issues = result.issues;
    throw error;
  }
  return { manifest: result.manifest, issues: result.issues };
};

export const getDefaultSu7 = () => createDefaultSu7RuntimeParams();
