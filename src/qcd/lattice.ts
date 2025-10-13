import type { FieldResolution } from '../fields/contracts.js';
import type { Complex } from '../pipeline/su7/types.js';
import type { Complex3x3 } from './su3.js';

export type GaugeLinkAxis = 'x' | 'y' | 'z' | 't';

export const GAUGE_LATTICE_AXES: readonly GaugeLinkAxis[] = ['x', 'y', 'z', 't'] as const;

const MATRIX_DIM = 3;
const COMPLEX_COMPONENTS = 2;
const COMPLEX_PER_MATRIX = MATRIX_DIM * MATRIX_DIM;
export const FLOATS_PER_MATRIX = COMPLEX_PER_MATRIX * COMPLEX_COMPONENTS;
const ROW_STRIDE = MATRIX_DIM * COMPLEX_COMPONENTS;

export type GaugeLatticeDimensions = {
  width: number;
  height: number;
  depth?: number;
  temporalExtent?: number;
};

export const GAUGE_LATTICE_SCHEMA_VERSION = 2 as const;

export type GaugeLatticeSnapshot = {
  schemaVersion: number;
  width: number;
  height: number;
  depth: number;
  temporalExtent: number;
  activeAxes: GaugeLinkAxis[];
  siteStride: number;
  linkStride: number;
  rowStride: number;
  complexStride: number;
  values: number[];
};

const identityEntry = (): Complex => ({ re: 1, im: 0 });
const zeroEntry = (): Complex => ({ re: 0, im: 0 });

const createIdentityMatrix = (): Complex3x3 => [
  [identityEntry(), zeroEntry(), zeroEntry()],
  [zeroEntry(), identityEntry(), zeroEntry()],
  [zeroEntry(), zeroEntry(), identityEntry()],
];
const IDENTITY_MATRIX = createIdentityMatrix();

const coerceDimension = (label: string, value: number): number => {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(
      `Gauge lattice ${label} must be a positive finite integer (received ${value})`,
    );
  }
  const intValue = Math.floor(value);
  if (intValue !== value) {
    throw new RangeError(`Gauge lattice ${label} must be integral (received ${value})`);
  }
  return intValue;
};

const normalizeDimensions = (input: GaugeLatticeDimensions | FieldResolution) => {
  const baseWidth = coerceDimension('width', input.width);
  const baseHeight = coerceDimension('height', input.height);
  const depth = 'depth' in input && input.depth != null ? coerceDimension('depth', input.depth) : 1;
  const temporalExtent =
    'temporalExtent' in input && input.temporalExtent != null
      ? coerceDimension('temporalExtent', input.temporalExtent)
      : 1;
  return {
    width: baseWidth,
    height: baseHeight,
    depth,
    temporalExtent,
  };
};

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

const clampAxisIndex = (value: number, limit: number) => {
  let next = value;
  while (next < 0) {
    next += limit;
  }
  return next % limit;
};

/**
 * GaugeLattice stores per-site SU(3) link matrices for an arbitrary subset
 * of the four lattice axes (x, y, z, t). Axes with extent 1 are omitted.
 *
 * Memory layout (Float32):
 *   siteIndex = (((t * depth) + z) * height + y) * width + x
 *   axisOffset = FLOATS_PER_MATRIX * axisIndex
 *   matrixOffset = row * MATRIX_DIM + column
 *   componentOffset = 0 (real), 1 (imag)
 *
 *   Float index = siteIndex * siteStride
 *               + axisOffset
 *               + matrixOffset * complexStride
 *               + componentOffset
 */
export class GaugeLattice {
  readonly width: number;
  readonly height: number;
  readonly depth: number;
  readonly temporalExtent: number;
  readonly axes: GaugeLinkAxis[];
  readonly data: Float32Array;
  readonly siteStride: number;
  readonly linkStride = FLOATS_PER_MATRIX;
  readonly rowStride = ROW_STRIDE;
  readonly complexStride = COMPLEX_COMPONENTS;

  private readonly axisOffsets: Record<GaugeLinkAxis, number>;

  constructor(dimensions: GaugeLatticeDimensions | FieldResolution, buffer?: Float32Array) {
    const { width, height, depth, temporalExtent } = normalizeDimensions(dimensions);
    this.width = width;
    this.height = height;
    this.depth = depth;
    this.temporalExtent = temporalExtent;
    this.axes = GAUGE_LATTICE_AXES.filter((axis) => {
      if (axis === 'z') {
        return depth > 1;
      }
      if (axis === 't') {
        return temporalExtent > 1;
      }
      return true;
    });
    if (this.axes.length === 0) {
      throw new RangeError('Gauge lattice requires at least one active axis');
    }
    this.axisOffsets = this.axes.reduce<Record<GaugeLinkAxis, number>>(
      (acc, axis, index) => {
        acc[axis] = index * FLOATS_PER_MATRIX;
        return acc;
      },
      { x: 0, y: 0, z: 0, t: 0 },
    );
    this.siteStride = FLOATS_PER_MATRIX * this.axes.length;
    const siteCount = this.siteCount;
    const expectedFloats = siteCount * this.siteStride;
    if (buffer) {
      if (!(buffer instanceof Float32Array)) {
        throw new TypeError('Gauge lattice buffer must be a Float32Array');
      }
      if (buffer.length !== expectedFloats) {
        throw new RangeError(
          `Gauge lattice buffer length mismatch (expected ${expectedFloats}, received ${buffer.length})`,
        );
      }
      this.data = buffer;
    } else {
      this.data = new Float32Array(expectedFloats);
    }
  }

  get siteCount(): number {
    return this.width * this.height * this.depth * this.temporalExtent;
  }

  private validateAxis(axis: GaugeLinkAxis) {
    if (!this.axes.includes(axis)) {
      throw new RangeError(`Gauge lattice axis ${axis} is inactive for the current dimensions`);
    }
  }

  private indexFor(x: number, y: number, axis: GaugeLinkAxis, z = 0, t = 0): number {
    if (!Number.isInteger(x) || x < 0 || x >= this.width) {
      throw new RangeError(`Gauge lattice X index out of bounds (received ${x})`);
    }
    if (!Number.isInteger(y) || y < 0 || y >= this.height) {
      throw new RangeError(`Gauge lattice Y index out of bounds (received ${y})`);
    }
    if (!Number.isInteger(z) || z < 0 || z >= this.depth) {
      throw new RangeError(`Gauge lattice Z index out of bounds (received ${z})`);
    }
    if (!Number.isInteger(t) || t < 0 || t >= this.temporalExtent) {
      throw new RangeError(`Gauge lattice temporal index out of bounds (received ${t})`);
    }
    this.validateAxis(axis);
    const axisOffset = this.axisOffsets[axis];
    const siteIndex = (((t * this.depth + z) * this.height + y) * this.width + x) * this.siteStride;
    return siteIndex + axisOffset;
  }

  getLinkSlice(x: number, y: number, axis: GaugeLinkAxis, z = 0, t = 0): Float32Array {
    const start = this.indexFor(x, y, axis, z, t);
    return this.data.subarray(start, start + FLOATS_PER_MATRIX);
  }

  setLinkSlice(
    x: number,
    y: number,
    axis: GaugeLinkAxis,
    source: ArrayLike<number>,
    z = 0,
    t = 0,
  ): void {
    if (source.length !== FLOATS_PER_MATRIX) {
      throw new RangeError(
        `Gauge lattice link slice expects ${FLOATS_PER_MATRIX} floats per matrix (received ${source.length})`,
      );
    }
    const start = this.indexFor(x, y, axis, z, t);
    if (Array.isArray(source)) {
      for (let i = 0; i < FLOATS_PER_MATRIX; i++) {
        const value = source[i];
        this.data[start + i] = isFiniteNumber(value) ? value : 0;
      }
      return;
    }
    this.data.set(source, start);
  }

  setLinkMatrix(x: number, y: number, axis: GaugeLinkAxis, matrix: Complex3x3, z = 0, t = 0): void {
    const start = this.indexFor(x, y, axis, z, t);
    let cursor = start;
    for (let row = 0; row < MATRIX_DIM; row++) {
      const rowData = matrix[row];
      if (!Array.isArray(rowData) || rowData.length !== MATRIX_DIM) {
        throw new TypeError('Gauge lattice link requires a 3x3 complex matrix');
      }
      for (let col = 0; col < MATRIX_DIM; col++) {
        const entry = rowData[col] as Complex | undefined;
        const re = entry && isFiniteNumber(entry.re) ? entry.re : 0;
        const im = entry && isFiniteNumber(entry.im) ? entry.im : 0;
        this.data[cursor++] = re;
        this.data[cursor++] = im;
      }
    }
  }

  getLinkMatrix(x: number, y: number, axis: GaugeLinkAxis, z = 0, t = 0): Complex3x3 {
    const start = this.indexFor(x, y, axis, z, t);
    let cursor = start;
    const rows: Complex[][] = [];
    for (let row = 0; row < MATRIX_DIM; row++) {
      const cols: Complex[] = [];
      for (let col = 0; col < MATRIX_DIM; col++) {
        const re = this.data[cursor++];
        const im = this.data[cursor++];
        cols.push({ re, im });
      }
      rows.push(cols);
    }
    return rows as Complex3x3;
  }

  fillIdentity(): void {
    for (let t = 0; t < this.temporalExtent; t++) {
      for (let z = 0; z < this.depth; z++) {
        for (let y = 0; y < this.height; y++) {
          for (let x = 0; x < this.width; x++) {
            for (const axis of this.axes) {
              this.setLinkMatrix(x, y, axis, IDENTITY_MATRIX, z, t);
            }
          }
        }
      }
    }
  }

  shiftCoordinate(
    coord: { x: number; y: number; z: number; t: number },
    axis: GaugeLinkAxis,
    delta: -1 | 1,
  ): { x: number; y: number; z: number; t: number } {
    if (axis === 'x') {
      return {
        ...coord,
        x: clampAxisIndex(coord.x + delta, this.width),
      };
    }
    if (axis === 'y') {
      return {
        ...coord,
        y: clampAxisIndex(coord.y + delta, this.height),
      };
    }
    if (axis === 'z') {
      return {
        ...coord,
        z: clampAxisIndex(coord.z + delta, this.depth),
      };
    }
    return {
      ...coord,
      t: clampAxisIndex(coord.t + delta, this.temporalExtent),
    };
  }

  snapshot(): GaugeLatticeSnapshot {
    return {
      schemaVersion: GAUGE_LATTICE_SCHEMA_VERSION,
      width: this.width,
      height: this.height,
      depth: this.depth,
      temporalExtent: this.temporalExtent,
      activeAxes: [...this.axes],
      siteStride: this.siteStride,
      linkStride: this.linkStride,
      rowStride: this.rowStride,
      complexStride: this.complexStride,
      values: Array.from(this.data),
    };
  }

  static restore(snapshot: GaugeLatticeSnapshot): GaugeLattice {
    if (snapshot.schemaVersion !== GAUGE_LATTICE_SCHEMA_VERSION) {
      throw new RangeError(
        `Unsupported gauge lattice snapshot version (received ${snapshot.schemaVersion})`,
      );
    }
    const lattice = new GaugeLattice({
      width: snapshot.width,
      height: snapshot.height,
      depth: snapshot.depth,
      temporalExtent: snapshot.temporalExtent,
    });
    if (
      snapshot.siteStride !== lattice.siteStride ||
      snapshot.linkStride !== lattice.linkStride ||
      snapshot.rowStride !== lattice.rowStride ||
      snapshot.complexStride !== lattice.complexStride
    ) {
      throw new RangeError('Gauge lattice snapshot stride metadata does not match current layout');
    }
    if (snapshot.values.length !== lattice.data.length) {
      throw new RangeError(
        `Gauge lattice snapshot value length mismatch (expected ${lattice.data.length}, received ${snapshot.values.length})`,
      );
    }
    lattice.data.set(snapshot.values);
    return lattice;
  }
}

export const snapshotGaugeLattice = (lattice: GaugeLattice): GaugeLatticeSnapshot =>
  lattice.snapshot();

export const restoreGaugeLattice = (snapshot: GaugeLatticeSnapshot): GaugeLattice =>
  GaugeLattice.restore(snapshot);
