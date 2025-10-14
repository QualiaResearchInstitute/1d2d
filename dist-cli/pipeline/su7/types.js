export const SU7_MAX_ACTIVE_GATES = 256;
const VECTOR_DIM = 7;
const TAU = Math.PI * 2;
function createIdentityMatrix() {
  const rows = [];
  for (let i = 0; i < VECTOR_DIM; i++) {
    const row = [];
    for (let j = 0; j < VECTOR_DIM; j++) {
      row.push({ re: i === j ? 1 : 0, im: 0 });
    }
    rows.push(row);
  }
  return rows;
}
const isFiniteNumber = (value) => typeof value === 'number' && Number.isFinite(value);
const cloneComplex = (value) => ({
  re: isFiniteNumber(value.re) ? value.re : 0,
  im: isFiniteNumber(value.im) ? value.im : 0,
});
const sanitizeComplex = (source, fallback) => {
  if (!source || typeof source !== 'object') {
    return cloneComplex(fallback);
  }
  const candidate = source;
  const re = isFiniteNumber(candidate.re) ? candidate.re : fallback.re;
  const im = isFiniteNumber(candidate.im) ? candidate.im : fallback.im;
  return { re, im };
};
const cloneVector = (value) => value.map((cell) => cloneComplex(cell));
export const cloneComplex7x7 = (value) => value.map((row) => cloneVector(row));
const sanitizeComplexRow = (value, fallback) => {
  if (!Array.isArray(value) || value.length !== 7) {
    return cloneVector(fallback);
  }
  const sanitized = value.map((entry, idx) => sanitizeComplex(entry, fallback[idx]));
  return sanitized;
};
const sanitizeComplex7x7 = (value, fallback) => {
  if (!Array.isArray(value) || value.length !== 7) {
    return fallback ? cloneComplex7x7(fallback) : undefined;
  }
  if (!fallback) {
    return sanitizeComplex7x7(value, createIdentityMatrix());
  }
  const rows = value.map((row, idx) => sanitizeComplexRow(row, fallback[idx]));
  return rows;
};
const sanitizeSchedule = (value, fallback) => {
  if (!Array.isArray(value)) {
    return [...fallback];
  }
  const stages = [];
  for (const entry of value) {
    if (!entry || typeof entry !== 'object') continue;
    const source = entry;
    if (!isFiniteNumber(source.gain)) continue;
    const stage = {
      gain: source.gain,
    };
    if (isFiniteNumber(source.index)) {
      stage.index = Math.trunc(source.index);
    }
    if (isFiniteNumber(source.spread)) {
      stage.spread = source.spread;
    }
    if (isFiniteNumber(source.phase)) {
      stage.phase = source.phase;
    }
    if (isFiniteNumber(source.time)) {
      stage.time = source.time;
    }
    if (isFiniteNumber(source.length)) {
      stage.length = Math.max(0, source.length);
    }
    if (typeof source.lane === 'string') {
      const lane = source.lane.trim();
      if (lane.length > 0) {
        stage.lane = lane;
      }
    }
    if (typeof source.macro === 'boolean') {
      stage.macro = source.macro;
    }
    if (typeof source.label === 'string') {
      stage.label = source.label;
    }
    stages.push(stage);
  }
  return stages.length > 0 ? stages : [...fallback];
};
export const cloneSu7Schedule = (schedule) => schedule.map((stage) => ({ ...stage }));
const wrapAngle = (theta) => {
  if (!Number.isFinite(theta)) return 0;
  let t = theta % TAU;
  if (t <= -Math.PI) {
    t += TAU;
  } else if (t > Math.PI) {
    t -= TAU;
  }
  return t;
};
const clampAxis = (axis) => {
  if (!Number.isFinite(axis)) return 0;
  const wrapped = Math.trunc(axis);
  return ((wrapped % VECTOR_DIM) + VECTOR_DIM) % VECTOR_DIM;
};
const sanitizePhaseVector = (value, fallback) => {
  if (!Array.isArray(value) || value.length !== VECTOR_DIM) {
    return [...fallback];
  }
  const next = [];
  for (let i = 0; i < VECTOR_DIM; i++) {
    const cell = Number(value[i]);
    next.push(Number.isFinite(cell) ? cell : fallback[i]);
  }
  return next;
};
export const cloneGate = (gate) => {
  if (gate.kind === 'phase') {
    return {
      kind: 'phase',
      phases: [...gate.phases],
      label: gate.label,
    };
  }
  return {
    kind: 'pulse',
    axis: gate.axis,
    theta: gate.theta,
    phase: gate.phase,
    label: gate.label,
  };
};
const sanitizeGate = (value, fallback) => {
  if (!value || typeof value !== 'object') {
    return fallback ? cloneGate(fallback) : null;
  }
  const source = value;
  if (source.kind === 'phase') {
    const phases = sanitizePhaseVector(
      source.phases,
      fallback?.kind === 'phase' ? fallback.phases : [0, 0, 0, 0, 0, 0, 0],
    );
    return {
      kind: 'phase',
      phases,
      label: typeof source.label === 'string' ? source.label : fallback?.label,
    };
  }
  if (source.kind === 'pulse') {
    const theta = Number(source.theta);
    const phase = Number(source.phase);
    if (!Number.isFinite(theta) || !Number.isFinite(phase)) {
      return fallback && fallback.kind === 'pulse' ? cloneGate(fallback) : null;
    }
    return {
      kind: 'pulse',
      axis: clampAxis(Number(source.axis)),
      theta,
      phase: wrapAngle(phase),
      label: typeof source.label === 'string' ? source.label : fallback?.label,
    };
  }
  return fallback ? cloneGate(fallback) : null;
};
export const sanitizeGateList = (value, fallback = []) => {
  if (!Array.isArray(value)) {
    return fallback.map((gate) => cloneGate(gate));
  }
  const sanitized = [];
  value.forEach((entry, idx) => {
    const fallbackGate = fallback[idx] ?? null;
    const gate = sanitizeGate(entry, fallbackGate ?? undefined);
    if (gate) {
      sanitized.push(gate);
    }
  });
  return sanitized;
};
const cloneGateArray = (gates) => gates.map((gate) => cloneGate(gate));
const clamp01 = (value) => (value <= 0 ? 0 : value >= 1 ? 1 : value);
const sanitizeLensAxis = (value, fallback) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback;
  }
  const clamped = Math.max(0, Math.min(6, Math.trunc(value)));
  return clamped;
};
const sanitizeHopfLensDescriptor = (value, fallback) => {
  if (!value || typeof value !== 'object') {
    return { ...fallback };
  }
  const source = value;
  const fallbackAxes = fallback.axes ?? [0, 1];
  const axisA = sanitizeLensAxis(
    Array.isArray(source.axes) ? source.axes[0] : source.axisA,
    fallbackAxes[0],
  );
  const axisB = sanitizeLensAxis(
    Array.isArray(source.axes) ? source.axes[1] : source.axisB,
    fallbackAxes[1],
  );
  const baseMix =
    typeof source.baseMix === 'number' && Number.isFinite(source.baseMix)
      ? clamp01(source.baseMix)
      : (fallback.baseMix ?? 1);
  const fiberMix =
    typeof source.fiberMix === 'number' && Number.isFinite(source.fiberMix)
      ? clamp01(source.fiberMix)
      : (fallback.fiberMix ?? 1);
  const control =
    source.controlTarget === 'base' || source.controlTarget === 'fiber'
      ? source.controlTarget
      : 'none';
  const label =
    typeof source.label === 'string' && source.label.length > 0 ? source.label : fallback.label;
  return {
    axes: [axisA, axisB],
    baseMix,
    fiberMix,
    controlTarget: control,
    label,
  };
};
const sanitizeHopfOptions = (value, fallback) => {
  if (!value || typeof value !== 'object') {
    return fallback ? { lenses: fallback.lenses.map((lens) => ({ ...lens })) } : undefined;
  }
  const source = value;
  if (!Array.isArray(source.lenses) || source.lenses.length === 0) {
    return fallback ? { lenses: fallback.lenses.map((lens) => ({ ...lens })) } : undefined;
  }
  const fallbackLenses = fallback?.lenses ?? [];
  const sanitized = source.lenses.map((lens, idx) =>
    sanitizeHopfLensDescriptor(lens, fallbackLenses[idx] ?? { axes: [idx, (idx + 1) % 7] }),
  );
  return { lenses: sanitized };
};
const sanitizeProjector = (value, fallback) => {
  if (!value || typeof value !== 'object') {
    return cloneSu7ProjectorDescriptor(fallback);
  }
  const source = value;
  const id = typeof source.id === 'string' && source.id.length > 0 ? source.id : fallback.id;
  const weight = isFiniteNumber(source.weight) ? source.weight : fallback.weight;
  const matrix = sanitizeComplex7x7(source.matrix, fallback.matrix);
  const hopf = sanitizeHopfOptions(source.hopf, fallback.hopf);
  return {
    id,
    weight,
    matrix,
    hopf,
  };
};
const cloneHopfLensDescriptor = (lens) => ({
  axes: [lens.axes[0], lens.axes[1]],
  baseMix: Number.isFinite(lens.baseMix) ? lens.baseMix : undefined,
  fiberMix: Number.isFinite(lens.fiberMix) ? lens.fiberMix : undefined,
  controlTarget: lens.controlTarget ?? 'none',
  label: lens.label,
});
const cloneHopfProjectorOptions = (options) => {
  if (!options) {
    return undefined;
  }
  return {
    lenses: options.lenses.map((lens) => cloneHopfLensDescriptor(lens)),
  };
};
export const cloneSu7ProjectorDescriptor = (value) => ({
  id: value.id,
  weight: isFiniteNumber(value.weight) ? value.weight : undefined,
  matrix: value.matrix ? cloneComplex7x7(value.matrix) : undefined,
  hopf: cloneHopfProjectorOptions(value.hopf),
});
export const createDefaultSu7RuntimeParams = () => ({
  enabled: false,
  gain: 1,
  preset: 'identity',
  seed: 0,
  schedule: [],
  projector: { id: 'identity' },
  gateAppends: [],
});
export const cloneSu7RuntimeParams = (value) => ({
  enabled: Boolean(value.enabled),
  gain: isFiniteNumber(value.gain) ? value.gain : 1,
  preset: value.preset,
  seed: Math.trunc(isFiniteNumber(value.seed) ? value.seed : 0),
  schedule: cloneSu7Schedule(value.schedule),
  projector: cloneSu7ProjectorDescriptor(value.projector),
  gateAppends: cloneGateArray(value.gateAppends ?? []),
});
export const sanitizeSu7RuntimeParams = (value, fallback) => {
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
    gateAppends: sanitizeGateList(source.gateAppends, fallback.gateAppends),
  };
};
export const DEFAULT_SU7_TELEMETRY = {
  unitaryError: 0,
  determinantDrift: 0,
  normDeltaMax: 0,
  normDeltaMean: 0,
  projectorEnergy: 0,
  geodesicFallbacks: 0,
};
export const EMPTY_SU7_GUARDRAILS = {
  events: [],
};
const normalizePhaseAngles = (angles) => {
  if (angles.length === 0) return angles;
  const sum = angles.reduce((acc, value) => acc + value, 0);
  const mean = sum / angles.length;
  for (let i = 0; i < angles.length; i++) {
    angles[i] = wrapAngle(angles[i] - mean);
  }
  return angles;
};
const toGatePhaseVector = (values) => values.map((value) => wrapAngle(value));
export const mergeGateAppends = (base, appends) => {
  const baseGates = base.gates.map((gate) => cloneGate(gate));
  const schedule = cloneSu7Schedule(base.schedule);
  const projector = cloneSu7ProjectorDescriptor(base.projector);
  if (!appends.length) {
    return {
      seed: base.seed,
      preset: base.preset,
      schedule,
      projector,
      gains: {
        baseGain: base.gains.baseGain,
        phaseAngles: [...base.gains.phaseAngles],
        pulseAngles: [...base.gains.pulseAngles],
        chiralityPhase: wrapAngle(base.gains.chiralityPhase),
      },
      gates: baseGates,
      squashedAppends: base.squashedAppends,
    };
  }
  const phases = Array.from(base.gains.phaseAngles);
  const pulses = Array.from(base.gains.pulseAngles);
  const sanitizedAppends = sanitizeGateList(appends, []);
  let weight = pulses.reduce((acc, value) => acc + Math.abs(value), 0);
  if (!Number.isFinite(weight) || weight <= 1e-6) {
    weight = 1;
  }
  let vecX = Math.cos(base.gains.chiralityPhase) * weight;
  let vecY = Math.sin(base.gains.chiralityPhase) * weight;
  sanitizedAppends.forEach((gate) => {
    if (gate.kind === 'phase') {
      for (let axis = 0; axis < VECTOR_DIM; axis++) {
        phases[axis] += gate.phases[axis];
      }
    } else if (gate.kind === 'pulse') {
      const axis = clampAxis(gate.axis);
      pulses[axis] += gate.theta;
      const magnitude = Math.abs(gate.theta);
      if (magnitude > 1e-6) {
        vecX += Math.cos(gate.phase) * magnitude;
        vecY += Math.sin(gate.phase) * magnitude;
        weight += magnitude;
      }
    }
  });
  normalizePhaseAngles(phases);
  const chiralityPhase = Math.atan2(vecY, vecX);
  const capacity = Math.max(SU7_MAX_ACTIVE_GATES - baseGates.length, 0);
  const retainedAppends =
    capacity === 0
      ? []
      : capacity >= sanitizedAppends.length
        ? sanitizedAppends
        : sanitizedAppends.slice(sanitizedAppends.length - capacity);
  const squashedAppends = sanitizedAppends.length - retainedAppends.length;
  return {
    seed: base.seed,
    preset: base.preset,
    schedule,
    projector,
    gains: {
      baseGain: base.gains.baseGain,
      phaseAngles: toGatePhaseVector(phases),
      pulseAngles: toGatePhaseVector(pulses),
      chiralityPhase: wrapAngle(chiralityPhase),
    },
    gates: [...baseGates, ...retainedAppends.map((gate) => cloneGate(gate))],
    squashedAppends,
  };
};
