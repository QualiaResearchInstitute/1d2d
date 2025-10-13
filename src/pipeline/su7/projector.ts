import { lmsToRgb, linearLuma, linearToSrgb, srgbToLinear } from '../colorSpaces.js';
import { buildFluxSu3Block, fluxEnergyToOverlayColor } from '../../qcd/overlays.js';
import { computeHopfCoordinates, hopfBaseToRgb, hopfFiberToRgb } from './geodesic.js';
import { su3_embed } from './math.js';
import type { Complex3x3 } from '../../qcd/su3.js';
import type {
  C7Vector,
  HopfLensDescriptor,
  Su7GuardrailEvent,
  Su7ProjectorDescriptor,
} from './types.js';

const EPSILON = 1e-9;
const FLUX_OVERLAY_EPSILON = 1e-6;

const clamp = (value: number, min: number, max: number): number =>
  Math.min(Math.max(value, min), max);

const clamp01 = (value: number): number => clamp(value, 0, 1);

const safeNumber = (value: number | null | undefined, fallback = 0): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

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
  guardrailEvent?: Extract<Su7GuardrailEvent, { kind: 'autoGain' }>;
  hopf?: HopfLensProjection[];
};

export type FluxOverlaySample = {
  energy: number;
  energyScale: number;
  dirX: number;
  dirY: number;
};

export type ProjectSu7VectorParams = {
  vector: C7Vector;
  norm: number;
  projector: Su7ProjectorDescriptor;
  gain: number;
  frameGain: number;
  baseColor: [number, number, number];
  fluxOverlay?: FluxOverlaySample;
};

export type HopfLensProjection = {
  index: number;
  axes: [number, number];
  base: [number, number, number];
  fiber: number;
  magnitude: number;
  share: number;
  baseMix: number;
  fiberMix: number;
  label: string | undefined;
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

export const MAX_HOPF_LENSES = 3;
const MAX_HOPF_OVERLAYS = MAX_HOPF_LENSES * 2;

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

const applyFluxBlockHead = (
  block: Complex3x3,
  vector: C7Vector,
): [{ re: number; im: number }, { re: number; im: number }, { re: number; im: number }] => {
  const embedded = su3_embed(block);
  const out: [{ re: number; im: number }, { re: number; im: number }, { re: number; im: number }] =
    [
      { re: 0, im: 0 },
      { re: 0, im: 0 },
      { re: 0, im: 0 },
    ];
  for (let row = 0; row < 3; row++) {
    let sumRe = 0;
    let sumIm = 0;
    for (let col = 0; col < 3; col++) {
      const coeff = embedded[row][col];
      if (coeff.re === 0 && coeff.im === 0) continue;
      const cell = vector[col];
      sumRe += coeff.re * cell.re - coeff.im * cell.im;
      sumIm += coeff.re * cell.im + coeff.im * cell.re;
    }
    out[row] = { re: sumRe, im: sumIm };
  }
  return out;
};

export const DEFAULT_HOPF_LENSES: readonly HopfLensDescriptor[] = Object.freeze([
  { axes: [0, 1], baseMix: 1, fiberMix: 1, controlTarget: 'none', label: 'Lens 1' },
  { axes: [2, 3], baseMix: 1, fiberMix: 1, controlTarget: 'none', label: 'Lens 2' },
  { axes: [4, 5], baseMix: 1, fiberMix: 1, controlTarget: 'none', label: 'Lens 3' },
]);

const sanitizeLensAxes = (axes: [number, number]): [number, number] => {
  const a = Math.max(0, Math.min(6, Math.trunc(axes[0])));
  const b = Math.max(0, Math.min(6, Math.trunc(axes[1])));
  if (a === b) {
    const fallback = (a + 1) % 7;
    return [a, fallback];
  }
  return [a, b];
};

export const resolveHopfLenses = (descriptor: Su7ProjectorDescriptor): HopfLensDescriptor[] => {
  const options = descriptor.hopf;
  if (!options || !Array.isArray(options.lenses) || options.lenses.length === 0) {
    return DEFAULT_HOPF_LENSES.map((lens) => ({ ...lens }));
  }
  const sanitized = options.lenses.map((lens, idx) => {
    const tuple =
      Array.isArray(lens.axes) && lens.axes.length === 2 ? lens.axes : [idx % 7, (idx + 1) % 7];
    const axes = sanitizeLensAxes(tuple as [number, number]);
    const baseMix =
      typeof lens.baseMix === 'number' && Number.isFinite(lens.baseMix) ? clamp01(lens.baseMix) : 1;
    const fiberMix =
      typeof lens.fiberMix === 'number' && Number.isFinite(lens.fiberMix)
        ? clamp01(lens.fiberMix)
        : 1;
    return {
      axes,
      baseMix,
      fiberMix,
      controlTarget: lens.controlTarget ?? 'none',
      label: lens.label,
    };
  });
  return sanitized.slice(0, MAX_HOPF_LENSES);
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

  const targetEnergy = normValue * effectiveGain * frameGain;
  const su7LumaBeforeAutoGain = su7Luma;
  let autoGainEvent: Extract<Su7GuardrailEvent, { kind: 'autoGain' }> | null = null;
  if (targetEnergy > EPSILON) {
    const tolerance = Math.max(targetEnergy * 0.05, 1e-6);
    if (Math.abs(su7Luma - targetEnergy) > tolerance) {
      const correction = clamp(targetEnergy / Math.max(su7Luma, 1e-6), 0.25, 4);
      linearRgb = linearRgb.map((channel) => channel * correction) as [number, number, number];
      su7Luma = Math.max(0, su7Luma * correction);
      autoGainEvent = {
        kind: 'autoGain',
        before: su7LumaBeforeAutoGain,
        after: su7Luma,
        target: targetEnergy,
        sampleCount: 1,
      };
    }
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
      guardrailEvent: autoGainEvent ?? undefined,
    };
  }

  const shares = computeShares(vector);
  const result: Su7ProjectionResult = {
    rgb: srgb,
    mix,
    energy: su7Luma,
    guardrailEvent: autoGainEvent ?? undefined,
  };

  if (projectorId === 'composerweights' || projectorId === 'overlaysplit') {
    result.composerWeights = sharesToWeights(shares);
  }
  if (projectorId === 'overlaysplit') {
    result.overlays = sharesToOverlays(shares, srgb, mix);
  } else if (projectorId === 'hopflens') {
    const lenses = resolveHopfLenses(projector);
    const hopf: HopfLensProjection[] = [];
    const overlays: Su7ProjectionOverlay[] = [];
    const epsilon = 1e-6;
    if (lenses.length > 0 && mix > epsilon) {
      const invNorm = norm > epsilon ? 1 / norm : 0;
      lenses.forEach((lens, index) => {
        const [axisA, axisB] = lens.axes;
        const compA = vector[axisA];
        const compB = vector[axisB];
        const coords = computeHopfCoordinates(compA, compB);
        if (coords.magnitude <= epsilon) {
          return;
        }
        const share = clamp01(coords.magnitude * invNorm);
        const baseMix = clamp01(share * mix * (lens.baseMix ?? 1));
        const fiberMix = clamp01(share * mix * (lens.fiberMix ?? 1));
        const projection: HopfLensProjection = {
          index,
          axes: [axisA, axisB],
          base: coords.base,
          fiber: coords.fiber,
          magnitude: coords.magnitude,
          share,
          baseMix,
          fiberMix,
          label: lens.label,
        };
        hopf.push(projection);
        if (baseMix > epsilon && overlays.length < MAX_HOPF_OVERLAYS) {
          overlays.push({
            rgb: hopfBaseToRgb(coords.base),
            mix: baseMix,
          });
        }
        if (fiberMix > epsilon && overlays.length < MAX_HOPF_OVERLAYS) {
          overlays.push({
            rgb: hopfFiberToRgb(coords.fiber),
            mix: fiberMix,
          });
        }
      });
    }
    if (hopf.length > 0) {
      result.hopf = hopf;
    }
    if (overlays.length > 0) {
      result.overlays = overlays;
    }
  }

  const fluxSample = params.fluxOverlay;
  if (fluxSample && fluxSample.energyScale > FLUX_OVERLAY_EPSILON && mix > FLUX_OVERLAY_EPSILON) {
    const normalizedEnergy = clamp01(fluxSample.energy * fluxSample.energyScale);
    const dirMagnitude = Math.hypot(fluxSample.dirX, fluxSample.dirY);
    if (normalizedEnergy > FLUX_OVERLAY_EPSILON && dirMagnitude > FLUX_OVERLAY_EPSILON) {
      const angle = Math.atan2(fluxSample.dirY, fluxSample.dirX);
      const block = buildFluxSu3Block(normalizedEnergy, angle);
      const transformedHead = applyFluxBlockHead(block, vector);
      const headMagnitudes = transformedHead.map((cell) => clamp01(cellMagnitude(cell))) as [
        number,
        number,
        number,
      ];
      const headMean = (headMagnitudes[0] + headMagnitudes[1] + headMagnitudes[2]) / 3;
      const modulation = clamp(headMean * 0.85 + 0.15, 0.1, 1.35);
      const overlayMix = clamp01(normalizedEnergy * modulation * mix);
      if (overlayMix > FLUX_OVERLAY_EPSILON) {
        const baseOverlay = fluxEnergyToOverlayColor(normalizedEnergy, angle);
        const tintedOverlay = baseOverlay.map((channel, idx) =>
          clamp01(channel * 0.6 + headMagnitudes[Math.min(idx, 2)] * 0.4),
        ) as [number, number, number];
        const overlay: Su7ProjectionOverlay = {
          rgb: tintedOverlay,
          mix: overlayMix,
        };
        if (result.overlays) {
          result.overlays.push(overlay);
        } else {
          result.overlays = [overlay];
        }
      }
    }
  }

  return result;
};
