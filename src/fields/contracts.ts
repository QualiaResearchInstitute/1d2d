export type ImageDataLike = {
  width: number;
  height: number;
  data: Uint8ClampedArray;
};

const FIELD_KIND_LIST = ['rim', 'surface', 'phase', 'volume'] as const;
export type FieldKind = (typeof FIELD_KIND_LIST)[number];
export const FIELD_KINDS = FIELD_KIND_LIST;

export type FieldResolution = {
  width: number;
  height: number;
  texels: number;
};

export const makeResolution = (width: number, height: number): FieldResolution => ({
  width,
  height,
  texels: width * height,
});

export const FIELD_KIND_CODES: Record<FieldKind, number> = {
  rim: 0,
  surface: 1,
  phase: 2,
  volume: 3,
};

export const FIELD_STORAGE = {
  planar: 0,
  interleaved: 1,
} as const;
export type FieldStorage = keyof typeof FIELD_STORAGE;

export const FIELD_COMPONENT_FORMAT = {
  float32: 0,
  uint8: 1,
} as const;
export type FieldComponentFormat = keyof typeof FIELD_COMPONENT_FORMAT;

export const FIELD_LIFETIME_KIND = {
  static: 0,
  dynamic: 1,
  workerStream: 2,
} as const;
export type FieldLifetimeKind = keyof typeof FIELD_LIFETIME_KIND;

export type FieldLifetime = {
  kind: FieldLifetimeKind;
  expectedMs: number;
  staleMs: number;
};

export const CHANNEL_SEMANTICS = {
  gradientX: 0,
  gradientY: 1,
  magnitude: 2,
  baseR: 3,
  baseG: 4,
  baseB: 5,
  baseA: 6,
  phaseGradX: 7,
  phaseGradY: 8,
  phaseVorticity: 9,
  phaseCoherence: 10,
  volumePhase: 11,
  volumeDepth: 12,
  volumeIntensity: 13,
} as const;
export type ChannelSemantic = keyof typeof CHANNEL_SEMANTICS;

export const CHANNEL_DESCRIPTIONS: Record<ChannelSemantic, string> = {
  gradientX: 'Edge gradient X',
  gradientY: 'Edge gradient Y',
  magnitude: 'Edge magnitude',
  baseR: 'Surface color R',
  baseG: 'Surface color G',
  baseB: 'Surface color B',
  baseA: 'Surface alpha',
  phaseGradX: 'Kuramoto grad X',
  phaseGradY: 'Kuramoto grad Y',
  phaseVorticity: 'Kuramoto vorticity',
  phaseCoherence: 'Kuramoto coherence',
  volumePhase: 'Volume phase slice',
  volumeDepth: 'Volume depth slice',
  volumeIntensity: 'Volume intensity slice',
};

export type FieldChannelDescriptor = {
  id: string;
  semantic: ChannelSemantic;
  description: string;
  components: number;
};

export type FieldContract = {
  kind: FieldKind;
  label: string;
  storage: FieldStorage;
  format: FieldComponentFormat;
  channels: FieldChannelDescriptor[];
  lifetime: FieldLifetime;
};

export const FIELD_CONTRACTS: Record<FieldKind, FieldContract> = {
  rim: {
    kind: 'rim',
    label: 'RimField',
    storage: 'planar',
    format: 'float32',
    channels: [
      {
        id: 'gradX',
        semantic: 'gradientX',
        description: CHANNEL_DESCRIPTIONS.gradientX,
        components: 1,
      },
      {
        id: 'gradY',
        semantic: 'gradientY',
        description: CHANNEL_DESCRIPTIONS.gradientY,
        components: 1,
      },
      {
        id: 'mag',
        semantic: 'magnitude',
        description: CHANNEL_DESCRIPTIONS.magnitude,
        components: 1,
      },
    ],
    lifetime: {
      kind: 'static',
      expectedMs: Number.POSITIVE_INFINITY,
      staleMs: Number.POSITIVE_INFINITY,
    },
  },
  surface: {
    kind: 'surface',
    label: 'SurfaceField',
    storage: 'interleaved',
    format: 'uint8',
    channels: [
      { id: 'R', semantic: 'baseR', description: CHANNEL_DESCRIPTIONS.baseR, components: 1 },
      { id: 'G', semantic: 'baseG', description: CHANNEL_DESCRIPTIONS.baseG, components: 1 },
      { id: 'B', semantic: 'baseB', description: CHANNEL_DESCRIPTIONS.baseB, components: 1 },
      { id: 'A', semantic: 'baseA', description: CHANNEL_DESCRIPTIONS.baseA, components: 1 },
    ],
    lifetime: {
      kind: 'static',
      expectedMs: Number.POSITIVE_INFINITY,
      staleMs: Number.POSITIVE_INFINITY,
    },
  },
  phase: {
    kind: 'phase',
    label: 'PhaseField',
    storage: 'planar',
    format: 'float32',
    channels: [
      {
        id: 'gradX',
        semantic: 'phaseGradX',
        description: CHANNEL_DESCRIPTIONS.phaseGradX,
        components: 1,
      },
      {
        id: 'gradY',
        semantic: 'phaseGradY',
        description: CHANNEL_DESCRIPTIONS.phaseGradY,
        components: 1,
      },
      {
        id: 'vort',
        semantic: 'phaseVorticity',
        description: CHANNEL_DESCRIPTIONS.phaseVorticity,
        components: 1,
      },
      {
        id: 'coh',
        semantic: 'phaseCoherence',
        description: CHANNEL_DESCRIPTIONS.phaseCoherence,
        components: 1,
      },
    ],
    lifetime: {
      kind: 'dynamic',
      expectedMs: 16,
      staleMs: 250,
    },
  },
  volume: {
    kind: 'volume',
    label: 'VolumeField',
    storage: 'planar',
    format: 'float32',
    channels: [
      {
        id: 'phase',
        semantic: 'volumePhase',
        description: CHANNEL_DESCRIPTIONS.volumePhase,
        components: 1,
      },
      {
        id: 'depth',
        semantic: 'volumeDepth',
        description: CHANNEL_DESCRIPTIONS.volumeDepth,
        components: 1,
      },
      {
        id: 'intensity',
        semantic: 'volumeIntensity',
        description: CHANNEL_DESCRIPTIONS.volumeIntensity,
        components: 1,
      },
    ],
    lifetime: {
      kind: 'workerStream',
      expectedMs: 16,
      staleMs: 250,
    },
  },
};

const uppercaseSnake = (value: string) =>
  value
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[-\s]+/g, '_')
    .toUpperCase();

const buildChannelDefines = () =>
  Object.entries(CHANNEL_SEMANTICS)
    .map(([key, code]) => `#define FIELD_SEMANTIC_${uppercaseSnake(key)} ${code}`)
    .join('\n');

const buildKindDefines = () =>
  FIELD_KIND_LIST.map(
    (kind) => `#define FIELD_KIND_${uppercaseSnake(kind)} ${FIELD_KIND_CODES[kind]}`,
  ).join('\n');

const buildStorageDefines = () =>
  Object.entries(FIELD_STORAGE)
    .map(([key, code]) => `#define FIELD_STORAGE_${uppercaseSnake(key)} ${code}`)
    .join('\n');

const buildFormatDefines = () =>
  Object.entries(FIELD_COMPONENT_FORMAT)
    .map(([key, code]) => `#define FIELD_FORMAT_${uppercaseSnake(key)} ${code}`)
    .join('\n');

const buildLifetimeDefines = () =>
  Object.entries(FIELD_LIFETIME_KIND)
    .map(([key, code]) => `#define FIELD_LIFETIME_${uppercaseSnake(key)} ${code}`)
    .join('\n');

export const FIELD_STRUCTS_GLSL = `#define FIELD_CHANNEL_COUNT 4

${buildKindDefines()}
${buildStorageDefines()}
${buildFormatDefines()}
${buildLifetimeDefines()}
${buildChannelDefines()}

struct FieldLifetimeInfo {
  int kind;
  float expectedMs;
  float staleMs;
};

struct FieldChannelInfo {
  int semantic;
  int components;
  int format;
  int _pad;
};

struct FieldContractInfo {
  ivec2 resolution;
  int storage;
  int channelCount;
  FieldLifetimeInfo lifetime;
  FieldChannelInfo ch0;
  FieldChannelInfo ch1;
  FieldChannelInfo ch2;
  FieldChannelInfo ch3;
};
`;

export type FieldChannelUniform = {
  semantic: number;
  components: number;
  format: number;
};

export type FieldUniformSnapshot = {
  resolution: [number, number];
  storage: number;
  channelCount: number;
  lifetime: {
    kind: number;
    expectedMs: number;
    staleMs: number;
  };
  channels: [FieldChannelUniform, FieldChannelUniform, FieldChannelUniform, FieldChannelUniform];
};

export type FieldRuntimeState = {
  available: boolean;
  resolution: FieldResolution | null;
};

const padChannels = (
  channels: FieldChannelUniform[],
): [FieldChannelUniform, FieldChannelUniform, FieldChannelUniform, FieldChannelUniform] => {
  const fallback: FieldChannelUniform = {
    semantic: -1,
    components: 0,
    format: FIELD_COMPONENT_FORMAT.float32,
  };
  return [
    channels[0] ?? fallback,
    channels[1] ?? fallback,
    channels[2] ?? fallback,
    channels[3] ?? fallback,
  ];
};

export const makeFieldUniformSnapshot = (
  contract: FieldContract,
  runtime: FieldRuntimeState,
): FieldUniformSnapshot => {
  const resolution: [number, number] =
    runtime.available && runtime.resolution
      ? [runtime.resolution.width, runtime.resolution.height]
      : [0, 0];
  const channels = padChannels(
    contract.channels.map((descriptor) => ({
      semantic: CHANNEL_SEMANTICS[descriptor.semantic],
      components: descriptor.components,
      format: FIELD_COMPONENT_FORMAT[contract.format],
    })),
  );
  return {
    resolution,
    storage: FIELD_STORAGE[contract.storage],
    channelCount: contract.channels.length,
    lifetime: {
      kind: FIELD_LIFETIME_KIND[contract.lifetime.kind],
      expectedMs: contract.lifetime.expectedMs,
      staleMs: contract.lifetime.staleMs,
    },
    channels,
  };
};

type Float32Buf = Float32Array;
type Uint8Buf = Uint8ClampedArray;

export type RimField = {
  kind: 'rim';
  resolution: FieldResolution;
  gx: Float32Buf;
  gy: Float32Buf;
  mag: Float32Buf;
};

export type SurfaceField = {
  kind: 'surface';
  resolution: FieldResolution;
  rgba: Uint8Buf;
};

export type PhaseField = {
  kind: 'phase';
  resolution: FieldResolution;
  gradX: Float32Buf;
  gradY: Float32Buf;
  vort: Float32Buf;
  coh: Float32Buf;
  amp: Float32Buf;
};

export type VolumeField = {
  kind: 'volume';
  resolution: FieldResolution;
  phase: Float32Buf;
  depth: Float32Buf;
  intensity: Float32Buf;
};

const assert = (condition: boolean, message: string): void => {
  if (!condition) {
    throw new Error(message);
  }
};

const assertLength = (
  buffer: { length: number },
  expected: number,
  label: string,
  field: FieldKind,
  source: string,
) => {
  assert(
    buffer.length === expected,
    `[fields:${field}] ${label} length ${buffer.length} != expected ${expected} (${source})`,
  );
};

export const assertRimField = (field: RimField, source: string) => {
  const expected = field.resolution.texels;
  assertLength(field.gx, expected, 'gradX', 'rim', source);
  assertLength(field.gy, expected, 'gradY', 'rim', source);
  assertLength(field.mag, expected, 'mag', 'rim', source);
};

export const assertSurfaceField = (field: SurfaceField, source: string) => {
  const expected = field.resolution.texels * FIELD_CONTRACTS.surface.channels.length;
  assertLength(field.rgba, expected, 'rgba', 'surface', source);
};

export const assertPhaseField = (field: PhaseField, source: string) => {
  const expected = field.resolution.texels;
  assertLength(field.gradX, expected, 'gradX', 'phase', source);
  assertLength(field.gradY, expected, 'gradY', 'phase', source);
  assertLength(field.vort, expected, 'vort', 'phase', source);
  assertLength(field.coh, expected, 'coh', 'phase', source);
  assertLength(field.amp, expected, 'amp', 'phase', source);
};

export const assertVolumeField = (field: VolumeField, source: string) => {
  const expected = field.resolution.texels;
  assertLength(field.phase, expected, 'phase', 'volume', source);
  assertLength(field.depth, expected, 'depth', 'volume', source);
  assertLength(field.intensity, expected, 'intensity', 'volume', source);
};

export type FieldBundle = {
  rim: RimField | null;
  surface: SurfaceField | null;
  phase: PhaseField | null;
  volume: VolumeField | null;
};

export const describeImageData = (image: ImageDataLike): SurfaceField => ({
  kind: 'surface',
  resolution: makeResolution(image.width, image.height),
  rgba: image.data,
});
