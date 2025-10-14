import type { PolarizationMatrix } from '../kuramotoCore.js';
import type { C7Vector, Complex7x7, Complex } from './su7/types.js';

type ComplexSource = Complex | PolarizationMatrix['m00'];

const sanitizeComplex = (source: ComplexSource | undefined): { re: number; im: number } => {
  const re = source && Number.isFinite(source.re) ? source.re : 0;
  const im = source && Number.isFinite(source.im) ? source.im : 0;
  return { re, im };
};

const normalizeMatrix = (matrix: PolarizationMatrix, gain = 1): PolarizationMatrix => {
  const sum =
    matrix.m00.re * matrix.m00.re +
    matrix.m00.im * matrix.m00.im +
    matrix.m01.re * matrix.m01.re +
    matrix.m01.im * matrix.m01.im +
    matrix.m10.re * matrix.m10.re +
    matrix.m10.im * matrix.m10.im +
    matrix.m11.re * matrix.m11.re +
    matrix.m11.im * matrix.m11.im;
  const norm = Math.sqrt(sum);
  const scale = norm > 1e-12 ? gain / norm : gain;
  if (scale === 1) {
    return matrix;
  }
  return {
    m00: { re: matrix.m00.re * scale, im: matrix.m00.im * scale },
    m01: { re: matrix.m01.re * scale, im: matrix.m01.im * scale },
    m10: { re: matrix.m10.re * scale, im: matrix.m10.im * scale },
    m11: { re: matrix.m11.re * scale, im: matrix.m11.im * scale },
  };
};

const vectorToMatrix = (
  vector: readonly Complex[],
  indices: readonly [number, number, number, number],
  gain = 1,
): PolarizationMatrix => {
  const [i00, i01, i10, i11] = indices;
  const base: PolarizationMatrix = {
    m00: sanitizeComplex(vector[i00]),
    m01: sanitizeComplex(vector[i01]),
    m10: sanitizeComplex(vector[i10]),
    m11: sanitizeComplex(vector[i11]),
  };
  return normalizeMatrix(base, gain);
};

export const su7VectorToPolarizationMatrix = (
  vector: C7Vector,
  options?: {
    indices?: [number, number, number, number];
    gain?: number;
  },
): PolarizationMatrix => {
  const indices = options?.indices ?? [0, 1, 2, 3];
  return vectorToMatrix(vector, indices, options?.gain ?? 1);
};

export const su7UnitaryColumnToPolarizationMatrix = (
  unitary: Complex7x7,
  options?: {
    column?: number;
    gain?: number;
  },
): PolarizationMatrix => {
  const column = options?.column ?? 0;
  const vector: Complex[] = unitary.map((row) => row[column]) as Complex[];
  return vectorToMatrix(vector, [0, 1, 2, 3], options?.gain ?? 1);
};
