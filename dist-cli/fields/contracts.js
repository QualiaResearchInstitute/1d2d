const FIELD_KIND_LIST = ['rim', 'surface', 'phase', 'volume'];
export const FIELD_KINDS = FIELD_KIND_LIST;
export const makeResolution = (width, height) => ({
  width,
  height,
  texels: width * height,
});
export const FIELD_KIND_CODES = {
  rim: 0,
  surface: 1,
  phase: 2,
  volume: 3,
};
export const FIELD_STORAGE = {
  planar: 0,
  interleaved: 1,
};
export const FIELD_COMPONENT_FORMAT = {
  float32: 0,
  uint8: 1,
};
export const FIELD_LIFETIME_KIND = {
  static: 0,
  dynamic: 1,
  workerStream: 2,
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
};
export const CHANNEL_DESCRIPTIONS = {
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
export const FIELD_CONTRACTS = {
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
const uppercaseSnake = (value) =>
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
const padChannels = (channels) => {
  const fallback = {
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
export const makeFieldUniformSnapshot = (contract, runtime) => {
  const resolution =
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
const assert = (condition, message) => {
  if (!condition) {
    throw new Error(message);
  }
};
const assertLength = (buffer, expected, label, field, source) => {
  assert(
    buffer.length === expected,
    `[fields:${field}] ${label} length ${buffer.length} != expected ${expected} (${source})`,
  );
};
export const assertRimField = (field, source) => {
  const expected = field.resolution.texels;
  assertLength(field.gx, expected, 'gradX', 'rim', source);
  assertLength(field.gy, expected, 'gradY', 'rim', source);
  assertLength(field.mag, expected, 'mag', 'rim', source);
};
export const assertSurfaceField = (field, source) => {
  const expected = field.resolution.texels * FIELD_CONTRACTS.surface.channels.length;
  assertLength(field.rgba, expected, 'rgba', 'surface', source);
};
export const assertPhaseField = (field, source) => {
  const expected = field.resolution.texels;
  assertLength(field.gradX, expected, 'gradX', 'phase', source);
  assertLength(field.gradY, expected, 'gradY', 'phase', source);
  assertLength(field.vort, expected, 'vort', 'phase', source);
  assertLength(field.coh, expected, 'coh', 'phase', source);
  assertLength(field.amp, expected, 'amp', 'phase', source);
};
export const assertVolumeField = (field, source) => {
  const expected = field.resolution.texels;
  assertLength(field.phase, expected, 'phase', 'volume', source);
  assertLength(field.depth, expected, 'depth', 'volume', source);
  assertLength(field.intensity, expected, 'intensity', 'volume', source);
};
export const describeImageData = (image) => ({
  kind: 'surface',
  resolution: makeResolution(image.width, image.height),
  rgba: image.data,
});
