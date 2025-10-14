import type { FluxSource } from './overlays.js';
import { GaugeLattice, type GaugeLinkAxis } from './lattice.js';
import {
  su3_applyToVector,
  su3_conjugateTranspose,
  su3_mul,
  type Complex3Vector,
  type Complex3x3,
} from './su3.js';

export type ProbeColorVector = Complex3Vector;

export type ProbePathStep = {
  axis: GaugeLinkAxis;
  direction: 1 | -1;
  span?: number;
};

export type ProbePath = readonly ProbePathStep[];

export type ProbeLatticeCoord = { x: number; y: number; z: number; t: number };

export type ProbeTransportPoint = {
  coord: ProbeLatticeCoord;
  vector: ProbeColorVector;
};

export type ProbeTransportSegment = {
  stepIndex: number;
  axis: GaugeLinkAxis;
  direction: 1 | -1;
  link: Complex3x3;
  entry: ProbeTransportPoint;
  exit: ProbeTransportPoint;
};

export type ProbeTransportResult = {
  start: ProbeTransportPoint;
  segments: ProbeTransportSegment[];
  end: ProbeTransportPoint;
};

export type ProbeTransportNode = {
  coord: ProbeLatticeCoord;
  rgb: [number, number, number];
  magnitude: number;
  stepIndex: number;
};

export type ProbeTransportSegmentVisual = {
  fromIndex: number;
  toIndex: number;
  rgb: [number, number, number];
  magnitude: number;
  stepIndex: number;
  axis: GaugeLinkAxis;
  direction: 1 | -1;
};

export type ProbeTransportFrameData = {
  latticeWidth: number;
  latticeHeight: number;
  nodes: ProbeTransportNode[];
  segments: ProbeTransportSegmentVisual[];
  closed: boolean;
};

const EPSILON = 1e-9;

const createIdentityMatrix = (): Complex3x3 => [
  [
    { re: 1, im: 0 },
    { re: 0, im: 0 },
    { re: 0, im: 0 },
  ],
  [
    { re: 0, im: 0 },
    { re: 1, im: 0 },
    { re: 0, im: 0 },
  ],
  [
    { re: 0, im: 0 },
    { re: 0, im: 0 },
    { re: 1, im: 0 },
  ],
];

const cloneCoord = (coord: ProbeLatticeCoord): ProbeLatticeCoord => ({
  x: coord.x,
  y: coord.y,
  z: coord.z,
  t: coord.t,
});

const cloneVector = (vector: ProbeColorVector): ProbeColorVector =>
  [
    { re: vector[0].re, im: vector[0].im },
    { re: vector[1].re, im: vector[1].im },
    { re: vector[2].re, im: vector[2].im },
  ] as ProbeColorVector;

const clampSpan = (span: number | undefined): number => {
  if (!(typeof span === 'number' && Number.isFinite(span))) {
    return 1;
  }
  const magnitude = Math.abs(Math.floor(span));
  return magnitude > 0 ? magnitude : 1;
};

export const createProbeBasisVector = (color: 0 | 1 | 2 = 0): ProbeColorVector => {
  const basis: ProbeColorVector = [
    { re: 0, im: 0 },
    { re: 0, im: 0 },
    { re: 0, im: 0 },
  ];
  basis[color] = { re: 1, im: 0 };
  return basis;
};

const resolveStep = (
  lattice: GaugeLattice,
  coord: ProbeLatticeCoord,
  axis: GaugeLinkAxis,
  direction: 1 | -1,
): { link: Complex3x3; nextCoord: ProbeLatticeCoord } => {
  if (direction === 1) {
    const link = lattice.getLinkMatrix(coord.x, coord.y, axis, coord.z, coord.t);
    const nextCoord = lattice.shiftCoordinate(coord, axis, 1);
    return { link, nextCoord };
  }
  const prevCoord = lattice.shiftCoordinate(coord, axis, -1);
  const rawLink = lattice.getLinkMatrix(prevCoord.x, prevCoord.y, axis, prevCoord.z, prevCoord.t);
  const link = su3_conjugateTranspose(rawLink);
  return { link, nextCoord: prevCoord };
};

export const transportProbe = (params: {
  lattice: GaugeLattice;
  origin: ProbeLatticeCoord;
  path: ProbePath;
  vector?: ProbeColorVector;
}): ProbeTransportResult => {
  const { lattice, origin, path } = params;
  const baseVector = params.vector ? cloneVector(params.vector) : createProbeBasisVector(0);
  let coord = cloneCoord(origin);
  let transportMatrix = createIdentityMatrix();
  const startVector = su3_applyToVector(transportMatrix, baseVector);
  const start: ProbeTransportPoint = {
    coord: cloneCoord(coord),
    vector: cloneVector(startVector),
  };
  const segments: ProbeTransportSegment[] = [];
  let stepIndex = 0;

  for (const step of path) {
    const span = clampSpan(step.span);
    for (let iter = 0; iter < span; iter++) {
      const entryVector = su3_applyToVector(transportMatrix, baseVector);
      const entry: ProbeTransportPoint = {
        coord: cloneCoord(coord),
        vector: cloneVector(entryVector),
      };
      const { link, nextCoord } = resolveStep(lattice, coord, step.axis, step.direction);
      transportMatrix = su3_mul(transportMatrix, link);
      const exitVector = su3_applyToVector(transportMatrix, baseVector);
      const exit: ProbeTransportPoint = {
        coord: cloneCoord(nextCoord),
        vector: cloneVector(exitVector),
      };
      segments.push({
        stepIndex,
        axis: step.axis,
        direction: step.direction,
        link,
        entry,
        exit,
      });
      coord = nextCoord;
      stepIndex += 1;
    }
  }

  const endVector = su3_applyToVector(transportMatrix, baseVector);
  const end: ProbeTransportPoint = {
    coord: cloneCoord(coord),
    vector: cloneVector(endVector),
  };
  return { start, segments, end };
};

const componentMagnitudeSquared = (value: { re: number; im: number }): number =>
  value.re * value.re + value.im * value.im;

const vectorMagnitudeSquared = (vector: ProbeColorVector): number =>
  componentMagnitudeSquared(vector[0]) +
  componentMagnitudeSquared(vector[1]) +
  componentMagnitudeSquared(vector[2]);

const clamp01 = (value: number): number => (value <= 0 ? 0 : value >= 1 ? 1 : value);

const vectorToRgb = (vector: ProbeColorVector): [number, number, number] => {
  const components = [
    componentMagnitudeSquared(vector[0]),
    componentMagnitudeSquared(vector[1]),
    componentMagnitudeSquared(vector[2]),
  ];
  const total = components[0] + components[1] + components[2];
  if (total <= EPSILON) {
    return [0, 0, 0];
  }
  return [
    clamp01(components[0] / total),
    clamp01(components[1] / total),
    clamp01(components[2] / total),
  ];
};

const toNode = (point: ProbeTransportPoint, stepIndex: number): ProbeTransportNode => ({
  coord: cloneCoord(point.coord),
  rgb: vectorToRgb(point.vector),
  magnitude: Math.sqrt(Math.max(vectorMagnitudeSquared(point.vector), 0)),
  stepIndex,
});

export const buildProbeTransportFrameData = (
  lattice: GaugeLattice,
  result: ProbeTransportResult,
): ProbeTransportFrameData => {
  const nodes: ProbeTransportNode[] = [];
  const segments: ProbeTransportSegmentVisual[] = [];
  nodes.push(toNode(result.start, 0));
  let prevIndex = 0;
  result.segments.forEach((segment) => {
    const nextNode = toNode(segment.exit, segment.stepIndex + 1);
    const nextIndex = nodes.length;
    nodes.push(nextNode);
    segments.push({
      fromIndex: prevIndex,
      toIndex: nextIndex,
      rgb: vectorToRgb(segment.exit.vector),
      magnitude: nextNode.magnitude,
      stepIndex: segment.stepIndex,
      axis: segment.axis,
      direction: segment.direction,
    });
    prevIndex = nextIndex;
  });
  const closed =
    nodes.length > 1 &&
    result.start.coord.x === result.end.coord.x &&
    result.start.coord.y === result.end.coord.y &&
    result.start.coord.z === result.end.coord.z &&
    result.start.coord.t === result.end.coord.t;
  return {
    latticeWidth: lattice.width,
    latticeHeight: lattice.height,
    nodes,
    segments,
    closed,
  };
};

const wrapDelta = (delta: number, size: number): number => {
  if (size <= 0) return 0;
  let norm = delta % size;
  if (norm > size / 2) {
    norm -= size;
  } else if (norm < -size / 2) {
    norm += size;
  }
  return norm;
};

const clampIndex = (value: number, size: number): number => {
  if (!Number.isFinite(value) || size <= 0) {
    return 0;
  }
  let index = Math.floor(value);
  if (index < 0) {
    index = ((index % size) + size) % size;
  }
  if (index >= size) {
    index %= size;
  }
  return index;
};

export const createPlaquettePath = (axes: readonly GaugeLinkAxis[]): ProbePath => {
  if (axes.length === 0) {
    return [];
  }
  if (axes.length === 1) {
    return [
      { axis: axes[0], direction: 1 },
      { axis: axes[0], direction: -1 },
    ];
  }
  const a = axes[0]!;
  const b = axes[1]!;
  return [
    { axis: a, direction: 1 },
    { axis: b, direction: 1 },
    { axis: a, direction: -1 },
    { axis: b, direction: -1 },
  ];
};

const buildManhattanPath = (
  lattice: GaugeLattice,
  origin: ProbeLatticeCoord,
  target: ProbeLatticeCoord,
): ProbePath => {
  const dx = wrapDelta(target.x - origin.x, lattice.width);
  const dy = wrapDelta(target.y - origin.y, lattice.height);
  const steps: ProbePathStep[] = [];
  if (dx !== 0) {
    steps.push({ axis: 'x', direction: dx > 0 ? 1 : -1, span: Math.abs(dx) });
  }
  if (dy !== 0) {
    steps.push({ axis: 'y', direction: dy > 0 ? 1 : -1, span: Math.abs(dy) });
  }
  if (steps.length === 0) {
    return createPlaquettePath(lattice.axes);
  }
  const reverse: ProbePathStep[] = [];
  for (let idx = steps.length - 1; idx >= 0; idx--) {
    const step = steps[idx]!;
    reverse.push({ axis: step.axis, direction: step.direction === 1 ? -1 : 1, span: step.span });
  }
  return [...steps, ...reverse];
};

export const deriveProbePathFromSources = (
  lattice: GaugeLattice,
  sources: readonly FluxSource[],
): { origin: ProbeLatticeCoord; path: ProbePath } => {
  const origin: ProbeLatticeCoord = {
    x: Math.floor(lattice.width / 2),
    y: Math.floor(lattice.height / 2),
    z: 0,
    t: 0,
  };

  if (sources.length === 0) {
    const path = createPlaquettePath(lattice.axes);
    return { origin, path };
  }

  const primary = sources[0]!;
  origin.x = clampIndex(Math.round(primary.x), lattice.width);
  origin.y = clampIndex(Math.round(primary.y), lattice.height);

  if (sources.length === 1) {
    const path = createPlaquettePath(lattice.axes);
    return { origin, path };
  }

  const secondary = sources[1]!;
  const target: ProbeLatticeCoord = {
    x: clampIndex(Math.round(secondary.x), lattice.width),
    y: clampIndex(Math.round(secondary.y), lattice.height),
    z: 0,
    t: 0,
  };
  const path = buildManhattanPath(lattice, origin, target);
  return { origin, path };
};

export const buildProbeTransportVisualization = (
  lattice: GaugeLattice,
  sources: readonly FluxSource[],
  vector?: ProbeColorVector,
): ProbeTransportFrameData | null => {
  if (lattice.width <= 0 || lattice.height <= 0) {
    return null;
  }
  const { origin, path } = deriveProbePathFromSources(lattice, sources);
  if (path.length === 0) {
    return null;
  }
  const transport = transportProbe({
    lattice,
    origin,
    path,
    vector,
  });
  return buildProbeTransportFrameData(lattice, transport);
};
