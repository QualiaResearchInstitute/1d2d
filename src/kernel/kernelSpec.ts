const clamp = (value: number, min: number, max: number) => {
  if (Number.isNaN(value)) return min;
  if (!Number.isFinite(value)) return value > 0 ? max : min;
  if (value < min) return min;
  if (value > max) return max;
  return value;
};

const KERNEL_SCALAR_KEYS = ['gain', 'k0', 'Q', 'anisotropy', 'chirality', 'transparency'] as const;
type KernelScalarKey = (typeof KERNEL_SCALAR_KEYS)[number];

export type CouplingKernelPreset = 'dmt' | '5meo';

export type CouplingKernelParams = {
  preset: CouplingKernelPreset;
  radius: number;
  nearSigma: number;
  nearGain: number;
  farSigma: number;
  farGain: number;
  baseGain: number;
  normalization: 'l1' | 'none';
};

const sanitizeFinite = (value: number, fallback: number) => {
  if (Number.isNaN(value)) return fallback;
  if (!Number.isFinite(value)) return fallback;
  return value;
};

const COUPLING_KERNEL_PRESETS_INTERNAL: Record<CouplingKernelPreset, CouplingKernelParams> = {
  dmt: {
    preset: 'dmt',
    radius: 6,
    nearSigma: 0.85,
    nearGain: 1.2,
    farSigma: 2.6,
    farGain: 0.95,
    baseGain: 0,
    normalization: 'l1',
  },
  '5meo': {
    preset: '5meo',
    radius: 6,
    nearSigma: 1,
    nearGain: 0,
    farSigma: 1,
    farGain: 0,
    baseGain: 1,
    normalization: 'l1',
  },
};

export const COUPLING_KERNEL_PRESETS: Readonly<Record<CouplingKernelPreset, CouplingKernelParams>> =
  Object.freeze(
    Object.fromEntries(
      Object.entries(COUPLING_KERNEL_PRESETS_INTERNAL).map(([key, value]) => [
        key,
        Object.freeze({ ...value }),
      ]),
    ) as Record<CouplingKernelPreset, CouplingKernelParams>,
  );

const DEFAULT_COUPLING_PRESET: CouplingKernelPreset = 'dmt';

/**
 * KernelSpec captures the dimensionless optical parameters shared across
 * the Kuramoto envelope, angular-spectrum surface solver, and hyperbolic
 * compositor. Units follow the optical notebook:
 *  - gain: scalar amplification (unitless)
 *  - k0: base spatial frequency in cycles per texture width
 *  - Q: quality factor (unitless bandwidth ratio)
 *  - anisotropy: relative gain bias along principal axes (unitless)
 *  - chirality: handedness bias, radians of phase twist scaled to unit gain
 *  - transparency: normalized transmission coefficient [0,1]
 *  - couplingPreset: oscillator coupling profile (categorical)
 */
export type KernelSpec = {
  gain: number;
  k0: number;
  Q: number;
  anisotropy: number;
  chirality: number;
  transparency: number;
  couplingPreset: CouplingKernelPreset;
};

export type KernelSpecInit = Partial<KernelSpec>;

const INTERNAL_DEFAULT_KERNEL_SPEC: KernelSpec = {
  gain: 1.0,
  k0: 0.08,
  Q: 2.2,
  anisotropy: 0.6,
  chirality: 0.4,
  transparency: 0.2,
  couplingPreset: DEFAULT_COUPLING_PRESET,
};

const KERNEL_SPEC_BOUNDS: Record<KernelScalarKey, { min: number; max: number }> = {
  gain: { min: 0, max: 6 },
  k0: { min: 0.01, max: 0.45 },
  Q: { min: 0.5, max: 10 },
  anisotropy: { min: 0, max: 2 },
  chirality: { min: 0, max: 2.5 },
  transparency: { min: 0, max: 1 },
};

const sanitizeScalar = <K extends KernelScalarKey>(
  key: K,
  value: number | undefined,
): KernelSpec[K] => {
  const bounds = KERNEL_SPEC_BOUNDS[key];
  const fallback = INTERNAL_DEFAULT_KERNEL_SPEC[key];
  if (value == null) return fallback;
  return clamp(value, bounds.min, bounds.max);
};

const sanitizeCouplingPreset = (preset: CouplingKernelPreset | undefined): CouplingKernelPreset =>
  preset && COUPLING_KERNEL_PRESETS[preset] ? preset : DEFAULT_COUPLING_PRESET;

const cloneCouplingParams = (params: CouplingKernelParams): CouplingKernelParams => {
  const reference = COUPLING_KERNEL_PRESETS_INTERNAL[params.preset];
  const normalization: CouplingKernelParams['normalization'] =
    params.normalization === 'none' ? 'none' : 'l1';
  return {
    preset: params.preset,
    radius: sanitizeFinite(params.radius, reference.radius),
    nearSigma: sanitizeFinite(params.nearSigma, reference.nearSigma),
    nearGain: sanitizeFinite(params.nearGain, reference.nearGain),
    farSigma: sanitizeFinite(params.farSigma, reference.farSigma),
    farGain: sanitizeFinite(params.farGain, reference.farGain),
    baseGain: sanitizeFinite(params.baseGain, reference.baseGain),
    normalization,
  };
};

export const createKernelSpec = (init?: KernelSpecInit): KernelSpec => ({
  gain: sanitizeScalar('gain', init?.gain),
  k0: sanitizeScalar('k0', init?.k0),
  Q: sanitizeScalar('Q', init?.Q),
  anisotropy: sanitizeScalar('anisotropy', init?.anisotropy),
  chirality: sanitizeScalar('chirality', init?.chirality),
  transparency: sanitizeScalar('transparency', init?.transparency),
  couplingPreset: sanitizeCouplingPreset(init?.couplingPreset),
});

export const clampKernelSpec = (spec: KernelSpecInit): KernelSpec => createKernelSpec(spec);

export const cloneKernelSpec = (spec: KernelSpec): KernelSpec => createKernelSpec({ ...spec });

export const kernelSpecToJSON = (spec: KernelSpec): KernelSpec => ({
  gain: spec.gain,
  k0: spec.k0,
  Q: spec.Q,
  anisotropy: spec.anisotropy,
  chirality: spec.chirality,
  transparency: spec.transparency,
  couplingPreset: spec.couplingPreset,
});

export const KERNEL_SPEC_DEFAULT: Readonly<KernelSpec> = Object.freeze(
  kernelSpecToJSON(INTERNAL_DEFAULT_KERNEL_SPEC),
);

export const DEFAULT_KERNEL_SPEC_JSON = kernelSpecToJSON(INTERNAL_DEFAULT_KERNEL_SPEC);

export const getDefaultKernelSpec = (): KernelSpec => cloneKernelSpec(INTERNAL_DEFAULT_KERNEL_SPEC);

export const getKernelSpecBounds = () => ({
  scalars: { ...KERNEL_SPEC_BOUNDS },
  couplingPresets: Object.keys(COUPLING_KERNEL_PRESETS) as CouplingKernelPreset[],
});

export const getCouplingKernelParams = (preset: CouplingKernelPreset): CouplingKernelParams =>
  cloneCouplingParams(
    COUPLING_KERNEL_PRESETS[preset] ?? COUPLING_KERNEL_PRESETS[DEFAULT_COUPLING_PRESET],
  );
