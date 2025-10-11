export type Complex = {
  re: number;
  im: number;
};

export type C7Vector = [Complex, Complex, Complex, Complex, Complex, Complex, Complex];

export type Complex7x7 = [C7Vector, C7Vector, C7Vector, C7Vector, C7Vector, C7Vector, C7Vector];

export type Su7ScheduleStage = {
  gain: number;
  index?: number;
  spread?: number;
  label?: string;
};

export type Su7Schedule = readonly Su7ScheduleStage[];

export type Su7PresetId = string;

export type Su7ProjectorDescriptor = {
  id: string;
  weight?: number;
  matrix?: Complex7x7;
};

export type Su7RuntimeParams = {
  enabled: boolean;
  gain: number;
  preset: Su7PresetId;
  seed: number;
  schedule: Su7Schedule;
  projector: Su7ProjectorDescriptor;
};

export type Su7Telemetry = {
  unitaryError: number;
  determinantDrift: number;
  normDeltaMax: number;
  normDeltaMean: number;
  projectorEnergy: number;
};

function createIdentityMatrix(): Complex7x7 {
  const rows: Complex[][] = [];
  for (let i = 0; i < 7; i++) {
    const row: Complex[] = [];
    for (let j = 0; j < 7; j++) {
      row.push({ re: i === j ? 1 : 0, im: 0 });
    }
    rows.push(row);
  }
  return rows as Complex7x7;
}

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

const cloneComplex = (value: Complex): Complex => ({
  re: isFiniteNumber(value.re) ? value.re : 0,
  im: isFiniteNumber(value.im) ? value.im : 0,
});

const sanitizeComplex = (source: unknown, fallback: Complex): Complex => {
  if (!source || typeof source !== 'object') {
    return cloneComplex(fallback);
  }
  const candidate = source as { re?: unknown; im?: unknown };
  const re = isFiniteNumber(candidate.re) ? candidate.re : fallback.re;
  const im = isFiniteNumber(candidate.im) ? candidate.im : fallback.im;
  return { re, im };
};

const cloneVector = (value: C7Vector): C7Vector =>
  value.map((cell) => cloneComplex(cell)) as C7Vector;

export const cloneComplex7x7 = (value: Complex7x7): Complex7x7 =>
  value.map((row) => cloneVector(row)) as Complex7x7;

const sanitizeComplexRow = (value: unknown, fallback: C7Vector): C7Vector => {
  if (!Array.isArray(value) || value.length !== 7) {
    return cloneVector(fallback);
  }
  const sanitized = value.map((entry, idx) => sanitizeComplex(entry, fallback[idx])) as C7Vector;
  return sanitized;
};

const sanitizeComplex7x7 = (value: unknown, fallback?: Complex7x7): Complex7x7 | undefined => {
  if (!Array.isArray(value) || value.length !== 7) {
    return fallback ? cloneComplex7x7(fallback) : undefined;
  }
  if (!fallback) {
    const identity = createIdentityMatrix();
    return sanitizeComplex7x7(value, identity);
  }
  const rows = value.map((row, idx) => sanitizeComplexRow(row, fallback[idx])) as Complex7x7;
  return rows;
};

const sanitizeSchedule = (value: unknown, fallback: Su7Schedule): Su7Schedule => {
  if (!Array.isArray(value)) {
    return [...fallback];
  }
  const stages: Su7ScheduleStage[] = [];
  for (const entry of value) {
    if (!entry || typeof entry !== 'object') continue;
    const source = entry as Partial<Su7ScheduleStage>;
    if (!isFiniteNumber(source.gain)) continue;
    const stage: Su7ScheduleStage = {
      gain: source.gain,
    };
    if (isFiniteNumber(source.index)) {
      stage.index = Math.trunc(source.index);
    }
    if (isFiniteNumber(source.spread)) {
      stage.spread = source.spread;
    }
    if (typeof source.label === 'string') {
      stage.label = source.label;
    }
    stages.push(stage);
  }
  return stages.length > 0 ? stages : [...fallback];
};

const sanitizeProjector = (
  value: unknown,
  fallback: Su7ProjectorDescriptor,
): Su7ProjectorDescriptor => {
  if (!value || typeof value !== 'object') {
    return cloneProjector(fallback);
  }
  const source = value as Su7ProjectorDescriptor;
  const id = typeof source.id === 'string' && source.id.length > 0 ? source.id : fallback.id;
  const weight = isFiniteNumber(source.weight) ? source.weight : fallback.weight;
  const matrix = sanitizeComplex7x7(source.matrix, fallback.matrix);
  return {
    id,
    weight,
    matrix,
  };
};

const cloneProjector = (value: Su7ProjectorDescriptor): Su7ProjectorDescriptor => ({
  id: value.id,
  weight: isFiniteNumber(value.weight) ? value.weight : undefined,
  matrix: value.matrix ? cloneComplex7x7(value.matrix) : undefined,
});

export const createDefaultSu7RuntimeParams = (): Su7RuntimeParams => ({
  enabled: false,
  gain: 1,
  preset: 'identity',
  seed: 0,
  schedule: [],
  projector: { id: 'identity' },
});

export const cloneSu7RuntimeParams = (value: Su7RuntimeParams): Su7RuntimeParams => ({
  enabled: Boolean(value.enabled),
  gain: isFiniteNumber(value.gain) ? value.gain : 1,
  preset: value.preset,
  seed: Math.trunc(isFiniteNumber(value.seed) ? value.seed : 0),
  schedule: value.schedule.map((stage) => ({ ...stage })),
  projector: cloneProjector(value.projector),
});

export const sanitizeSu7RuntimeParams = (
  value: Partial<Su7RuntimeParams> | null | undefined,
  fallback: Su7RuntimeParams,
): Su7RuntimeParams => {
  const source = value ?? {};
  const schedule = sanitizeSchedule(source.schedule, fallback.schedule);
  const projector = sanitizeProjector(source.projector, fallback.projector);
  return {
    enabled: typeof source.enabled === 'boolean' ? source.enabled : fallback.enabled,
    gain: isFiniteNumber(source.gain) ? source.gain : fallback.gain,
    preset: typeof source.preset === 'string' ? source.preset : fallback.preset,
    seed: Math.trunc(isFiniteNumber(source.seed) ? source.seed : fallback.seed),
    schedule,
    projector,
  };
};

export const DEFAULT_SU7_TELEMETRY: Su7Telemetry = {
  unitaryError: 0,
  determinantDrift: 0,
  normDeltaMax: 0,
  normDeltaMean: 0,
  projectorEnergy: 0,
};
