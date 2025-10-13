import { normalizeTimeline } from './normalize.js';
import type {
  Timeline,
  TimelineEvaluation,
  TimelineKeyframe,
  TimelineLane,
  TimelineSeedKeyframe,
  TimelineSeedTrack,
  TimelineValue,
} from './types.js';

const findSurroundingKeyframes = <T extends TimelineValue>(
  keyframes: TimelineKeyframe<T>[],
  frame: number,
): { prev: TimelineKeyframe<T> | null; next: TimelineKeyframe<T> | null } => {
  if (keyframes.length === 0) {
    return { prev: null, next: null };
  }
  if (frame <= keyframes[0].frame) {
    return { prev: keyframes[0], next: keyframes[0] };
  }
  const last = keyframes[keyframes.length - 1];
  if (frame >= last.frame) {
    return { prev: last, next: last };
  }
  let prev: TimelineKeyframe<T> | null = keyframes[0];
  for (let i = 1; i < keyframes.length; i++) {
    const current = keyframes[i];
    if (current.frame === frame) {
      return { prev: current, next: current };
    }
    if (current.frame > frame) {
      return { prev, next: current };
    }
    prev = current;
  }
  return { prev: last, next: last };
};

const interpolateLinear = (a: number, b: number, t: number) => a + (b - a) * t;

const findSurroundingSeedKeyframes = (
  keyframes: TimelineSeedKeyframe[],
  frame: number,
): { prev: TimelineSeedKeyframe | null; next: TimelineSeedKeyframe | null } => {
  if (keyframes.length === 0) {
    return { prev: null, next: null };
  }
  if (frame <= keyframes[0].frame) {
    return { prev: keyframes[0], next: keyframes[0] };
  }
  const last = keyframes[keyframes.length - 1];
  if (frame >= last.frame) {
    return { prev: last, next: last };
  }
  let prev: TimelineSeedKeyframe | null = keyframes[0];
  for (let i = 1; i < keyframes.length; i++) {
    const current = keyframes[i];
    if (current.frame === frame) {
      return { prev: current, next: current };
    }
    if (current.frame > frame) {
      return { prev, next: current };
    }
    prev = current;
  }
  return { prev: last, next: last };
};

const evaluateLane = <T extends TimelineValue>(
  lane: TimelineLane<T>,
  frame: number,
): T | undefined => {
  if (lane.keyframes.length === 0) {
    return undefined;
  }
  const { prev, next } = findSurroundingKeyframes(lane.keyframes, frame);
  if (!prev) {
    return next ? next.value : undefined;
  }
  if (!next || prev === next) {
    return prev.value;
  }
  const interpolation = lane.interpolation ?? 'step';
  if (interpolation !== 'linear') {
    return prev.value;
  }
  // Only interpolate numbers; fall back to step for other types.
  if (typeof prev.value !== 'number' || typeof next.value !== 'number') {
    return prev.value;
  }
  const span = next.frame - prev.frame;
  if (span <= 0) {
    return prev.value;
  }
  const t = (frame - prev.frame) / span;
  return interpolateLinear(prev.value, next.value, t) as T;
};

const evaluateSeedTrack = (track: TimelineSeedTrack, frame: number): number | undefined => {
  if (track.keyframes.length === 0) {
    return undefined;
  }
  const { prev, next } = findSurroundingSeedKeyframes(track.keyframes, frame);
  if (!prev) {
    return next?.seed;
  }
  if (!next || prev === next) {
    return prev.seed;
  }
  // Seeds always use step interpolation.
  return prev.seed;
};

export type EvaluateTimelineOptions = {
  clampFrame?: boolean;
};

export const evaluateTimeline = (
  source: Timeline,
  frame: number,
  options: EvaluateTimelineOptions = {},
): TimelineEvaluation => {
  const timeline = normalizeTimeline(source);
  const clampedFrame =
    options.clampFrame === false
      ? Math.floor(frame)
      : Math.max(0, Math.min(timeline.durationFrames, Math.floor(frame)));
  const values: Record<string, TimelineValue> = {};
  for (const lane of timeline.lanes) {
    const value = evaluateLane(lane, clampedFrame);
    if (value !== undefined) {
      values[lane.id] = value;
    }
  }
  const seeds: Record<string, number> = {};
  for (const track of timeline.seeds) {
    const seed = evaluateSeedTrack(track, clampedFrame);
    if (seed !== undefined) {
      seeds[track.scope] = seed;
    }
  }
  return {
    frame: clampedFrame,
    values,
    seeds,
  };
};
