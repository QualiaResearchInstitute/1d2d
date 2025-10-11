import { lmsToRgb, linearLuma, linearToSrgb, srgbToLinear } from '../colorSpaces.js';
import type { C7Vector, Su7ProjectorDescriptor } from './types.js';

const EPSILON = 1e-9;

const clamp = (value: number, min: number, max: number): number =>
  Math.min(Math.max(value, min), max);

const clamp01 = (value: number): number => clamp(value, 0, 1);

const safeNumber = (value: number, fallback = 0): number =>
  Number.isFinite(value) ? (value as number) : fallback;

const cellMagnitude = (cell: { re: number; im: number }): number => Math.hypot(cell.re, cell.im);

export type Su7ComposerWeights = {
  rim?: number;
  surface?: number;
  kur?: number;
  volume?: number;
};

export type Su7ProjectionOverlay = {
  rgb: [number, number, number];
  mix: number;
};

export type Su7ProjectionResult = {
  rgb: [number, number, number];
  mix: number;
  energy: number;
  composerWeights?: Su7ComposerWeights;
  overlays?: Su7ProjectionOverlay[];
};

export type ProjectSu7VectorParams = {
  vector: C7Vector;
  norm: number;
  projector: Su7ProjectorDescriptor;
  gain: number;
  frameGain: number;
  baseColor: [number, number, number];
};

type EnergyShares = {
  rim: number;
  surface: number;
  kur: number;
  volume: number;
  total: number;
};

const computeShares = (vector: C7Vector): EnergyShares => {
  const rim = (cellMagnitude(vector[0]) + cellMagnitude(vector[1]) + cellMagnitude(vector[2])) / 3;
  const surface = cellMagnitude(vector[3]);
  const kur = (cellMagnitude(vector[4]) + cellMagnitude(vector[5])) * 0.5;
  const volume = cellMagnitude(vector[6]);
  const total = rim + surface + kur + volume;
  if (total <= EPSILON) {
    return { rim: 0, surface: 0, kur: 0, volume: 0, total: 0 };
  }
  return {
    rim: rim / total,
    surface: surface / total,
    kur: kur / total,
    volume: volume / total,
    total,
  };
};

const sharesToWeights = (shares: EnergyShares): Su7ComposerWeights => {
  if (shares.total <= EPSILON) {
    return {};
  }
  const baseShare = 0.25;
  const scale = 2;
  const toWeight = (value: number) => clamp(1 + (value - baseShare) * scale, 0.2, 2.5);
  return {
    rim: toWeight(shares.rim),
    surface: toWeight(shares.surface),
    kur: toWeight(shares.kur),
    volume: toWeight(shares.volume),
  };
};

const tintColor = (
  share: number,
  tint: readonly [number, number, number],
): [number, number, number] => {
  const scale = Math.sqrt(clamp01(share));
  return [clamp01(scale * tint[0]), clamp01(scale * tint[1]), clamp01(scale * tint[2])];
};

const sharesToOverlays = (
  shares: EnergyShares,
  projectedRgb: [number, number, number],
  mix: number,
): Su7ProjectionOverlay[] => {
  if (shares.total <= EPSILON || mix <= EPSILON) {
    return [];
  }
  const overlays: Su7ProjectionOverlay[] = [
    {
      rgb: projectedRgb,
      mix: clamp01(shares.rim * mix),
    },
    {
      rgb: tintColor(shares.surface, [0.95, 0.72, 0.33]),
      mix: clamp01(shares.surface * mix),
    },
    {
      rgb: tintColor(shares.kur, [0.28, 0.78, 1.0]),
      mix: clamp01(shares.kur * mix),
    },
    {
      rgb: tintColor(shares.volume, [0.42, 0.9, 0.56]),
      mix: clamp01(shares.volume * mix),
    },
  ];
  return overlays.filter((overlay) => overlay.mix > EPSILON);
};

export const projectSu7Vector = (params: ProjectSu7VectorParams): Su7ProjectionResult | null => {
  const { vector, norm, projector, gain, frameGain, baseColor } = params;
  if (!vector) {
    return null;
  }

  const projectorId = typeof projector.id === 'string' ? projector.id.toLowerCase() : 'identity';
  const projectorWeight = clamp01(Math.abs(safeNumber(projector.weight, 1)));
  const effectiveGain = Math.abs(safeNumber(gain, 1));
  const normValue = Math.max(0, safeNumber(norm, 0));
  if (effectiveGain <= EPSILON || normValue <= EPSILON) {
    return null;
  }

  const scale = normValue * effectiveGain;
  const toneScale = scale * (projectorWeight > 0 ? projectorWeight : 1);
  const magnitude0 = cellMagnitude(vector[0]);
  const magnitude1 = cellMagnitude(vector[1]);
  const magnitude2 = cellMagnitude(vector[2]);

  let linearRgb: [number, number, number];
  if (projectorId === 'directrgb') {
    linearRgb = [magnitude0 * toneScale, magnitude1 * toneScale, magnitude2 * toneScale];
  } else {
    const L = magnitude0 * toneScale;
    const M = magnitude1 * toneScale;
    const S = magnitude2 * toneScale;
    linearRgb = lmsToRgb(L, M, S);
  }

  linearRgb = linearRgb.map((channel) => Math.max(0, channel * frameGain)) as [
    number,
    number,
    number,
  ];

  const baseLinear = [
    srgbToLinear(clamp01(baseColor[0])),
    srgbToLinear(clamp01(baseColor[1])),
    srgbToLinear(clamp01(baseColor[2])),
  ] as [number, number, number];
  const baseLuma = linearLuma(baseLinear[0], baseLinear[1], baseLinear[2]);
  let su7Luma = linearLuma(linearRgb[0], linearRgb[1], linearRgb[2]);

  if (su7Luma > EPSILON && baseLuma > EPSILON) {
    const lumaScale = clamp(baseLuma / su7Luma, 0.25, 4);
    linearRgb = linearRgb.map((channel) => channel * lumaScale) as [number, number, number];
    su7Luma = linearLuma(linearRgb[0], linearRgb[1], linearRgb[2]);
  }

  const srgb = linearRgb.map((channel) => clamp01(linearToSrgb(channel))) as [
    number,
    number,
    number,
  ];

  const mix = clamp01(projectorWeight * clamp01(effectiveGain));
  if (mix <= EPSILON) {
    return {
      rgb: srgb,
      mix,
      energy: su7Luma,
    };
  }

  const shares = computeShares(vector);
  const result: Su7ProjectionResult = {
    rgb: srgb,
    mix,
    energy: su7Luma,
  };

  if (projectorId === 'composerweights' || projectorId === 'overlaysplit') {
    result.composerWeights = sharesToWeights(shares);
  }
  if (projectorId === 'overlaysplit') {
    result.overlays = sharesToOverlays(shares, srgb, mix);
  }

  return result;
};
