import type { C7Vector, Complex7x7 } from './types.js';

export type Su7VectorBuffers = {
  tex0: Float32Array;
  tex1: Float32Array;
  tex2: Float32Array;
  tex3: Float32Array;
};

export const SU7_TILE_SIZE = 32;
export const SU7_TILE_TEXTURE_WIDTH = 8;
export const SU7_TILE_TEXTURE_ROWS_PER_TILE = 4;

const VECTOR_COMPONENTS_PER_TEXEL = 4;
const COMPLEX_PER_TEXEL = 2;
const COMPLEX_PER_VECTOR = 7;
const FLOATS_PER_COMPLEX = 2;
export const SU7_VECTOR_FLOATS = COMPLEX_PER_VECTOR * FLOATS_PER_COMPLEX;
const TOTAL_COMPLEX = 7 * 7;
const TILE_TEXELS_PER_TILE = SU7_TILE_TEXTURE_WIDTH * SU7_TILE_TEXTURE_ROWS_PER_TILE;
const TILE_TEXELS_USED = Math.ceil(TOTAL_COMPLEX / COMPLEX_PER_TEXEL);

export const ensureSu7VectorBuffers = (
  buffers: Su7VectorBuffers | null,
  texelCount: number,
): Su7VectorBuffers => {
  const required = texelCount * VECTOR_COMPONENTS_PER_TEXEL;
  if (!buffers) {
    return {
      tex0: new Float32Array(required),
      tex1: new Float32Array(required),
      tex2: new Float32Array(required),
      tex3: new Float32Array(required),
    };
  }
  if (
    buffers.tex0.length !== required ||
    buffers.tex1.length !== required ||
    buffers.tex2.length !== required ||
    buffers.tex3.length !== required
  ) {
    return {
      tex0: new Float32Array(required),
      tex1: new Float32Array(required),
      tex2: new Float32Array(required),
      tex3: new Float32Array(required),
    };
  }
  return buffers;
};

export const fillSu7VectorBuffers = (
  vectors: C7Vector[],
  norms: Float32Array,
  texelCount: number,
  target: Su7VectorBuffers,
) => {
  const { tex0, tex1, tex2, tex3 } = target;
  for (let i = 0; i < texelCount; i++) {
    const base = i * VECTOR_COMPONENTS_PER_TEXEL;
    const vector = vectors[i];
    const norm = norms[i] ?? 0;
    if (vector) {
      tex0[base + 0] = vector[0]?.re ?? 0;
      tex0[base + 1] = vector[0]?.im ?? 0;
      tex0[base + 2] = vector[1]?.re ?? 0;
      tex0[base + 3] = vector[1]?.im ?? 0;

      tex1[base + 0] = vector[2]?.re ?? 0;
      tex1[base + 1] = vector[2]?.im ?? 0;
      tex1[base + 2] = vector[3]?.re ?? 0;
      tex1[base + 3] = vector[3]?.im ?? 0;

      tex2[base + 0] = vector[4]?.re ?? 0;
      tex2[base + 1] = vector[4]?.im ?? 0;
      tex2[base + 2] = vector[5]?.re ?? 0;
      tex2[base + 3] = vector[5]?.im ?? 0;

      tex3[base + 0] = vector[6]?.re ?? 0;
      tex3[base + 1] = vector[6]?.im ?? 0;
    } else {
      tex0[base + 0] = 0;
      tex0[base + 1] = 0;
      tex0[base + 2] = 0;
      tex0[base + 3] = 0;
      tex1[base + 0] = 0;
      tex1[base + 1] = 0;
      tex1[base + 2] = 0;
      tex1[base + 3] = 0;
      tex2[base + 0] = 0;
      tex2[base + 1] = 0;
      tex2[base + 2] = 0;
      tex2[base + 3] = 0;
      tex3[base + 0] = 0;
      tex3[base + 1] = 0;
    }
    tex3[base + 2] = norm;
    tex3[base + 3] = 0;
  }
};

export const fillSu7VectorBuffersFromPacked = (
  packed: Float32Array,
  norms: Float32Array,
  texelCount: number,
  target: Su7VectorBuffers,
  strideFloats: number = SU7_VECTOR_FLOATS,
) => {
  const required = texelCount * strideFloats;
  if (packed.length < required) {
    throw new Error(
      `[su7-gpu] packed vector buffer length ${packed.length} expected >= ${required}`,
    );
  }
  const { tex0, tex1, tex2, tex3 } = target;
  for (let i = 0; i < texelCount; i++) {
    const baseTex = i * VECTOR_COMPONENTS_PER_TEXEL;
    const basePacked = i * strideFloats;
    const norm = norms[i] ?? 0;

    tex0[baseTex + 0] = packed[basePacked + 0];
    tex0[baseTex + 1] = packed[basePacked + 1];
    tex0[baseTex + 2] = packed[basePacked + 2];
    tex0[baseTex + 3] = packed[basePacked + 3];

    tex1[baseTex + 0] = packed[basePacked + 4];
    tex1[baseTex + 1] = packed[basePacked + 5];
    tex1[baseTex + 2] = packed[basePacked + 6];
    tex1[baseTex + 3] = packed[basePacked + 7];

    tex2[baseTex + 0] = packed[basePacked + 8];
    tex2[baseTex + 1] = packed[basePacked + 9];
    tex2[baseTex + 2] = packed[basePacked + 10];
    tex2[baseTex + 3] = packed[basePacked + 11];

    tex3[baseTex + 0] = packed[basePacked + 12];
    tex3[baseTex + 1] = packed[basePacked + 13];
    tex3[baseTex + 2] = norm;
    tex3[baseTex + 3] = 0;
  }
};

export const ensureSu7TileBuffer = (
  buffer: Float32Array | null,
  tileCols: number,
  tileRows: number,
): Float32Array => {
  const width = SU7_TILE_TEXTURE_WIDTH;
  const height = tileRows * SU7_TILE_TEXTURE_ROWS_PER_TILE;
  const required = width * height * VECTOR_COMPONENTS_PER_TEXEL;
  if (!buffer || buffer.length !== required) {
    return new Float32Array(required);
  }
  return buffer;
};

export const fillSu7TileBuffer = (
  unitary: Complex7x7,
  tileCols: number,
  tileRows: number,
  target: Float32Array,
) => {
  target.fill(0);
  const width = SU7_TILE_TEXTURE_WIDTH;
  const rowsPerTile = SU7_TILE_TEXTURE_ROWS_PER_TILE;
  const tileCount = tileCols * tileRows;

  for (let tileIndex = 0; tileIndex < tileCount; tileIndex++) {
    const tileRow = Math.floor(tileIndex / tileCols);
    const tileCol = tileIndex % tileCols;
    const tileBaseRow = (tileRow * tileCols + tileCol) * rowsPerTile;
    for (let row = 0; row < 7; row++) {
      for (let col = 0; col < 7; col++) {
        const complexIndex = row * 7 + col;
        const texelIndex = Math.floor(complexIndex / COMPLEX_PER_TEXEL);
        const lane = complexIndex % COMPLEX_PER_TEXEL;
        const texY = tileBaseRow + Math.floor(texelIndex / width);
        const texX = texelIndex % width;
        const offset = (texY * width + texX) * VECTOR_COMPONENTS_PER_TEXEL;
        const entry = unitary[row][col];
        if (lane === 0) {
          target[offset + 0] = entry.re;
          target[offset + 1] = entry.im;
        } else {
          target[offset + 2] = entry.re;
          target[offset + 3] = entry.im;
        }
      }
    }
  }
};

export const computeSu7TileTextureHeight = (tileRows: number): number =>
  tileRows * SU7_TILE_TEXTURE_ROWS_PER_TILE;
