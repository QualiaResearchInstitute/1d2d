import type {
  Timeline,
  TimelineKeyframe,
  TimelineLane,
  TimelineSeedKeyframe,
  TimelineSeedTrack,
  TimelineValue,
} from './types.js';

const clampFrame = (frame: number): number => {
  if (!Number.isFinite(frame) || frame < 0) {
    return 0;
  }
  return Math.floor(frame);
};

const normalizeKeyframes = <T extends TimelineValue>(
  keyframes: TimelineKeyframe<T>[],
): TimelineKeyframe<T>[] => {
  const normalized = keyframes
    .map((entry) => ({
      frame: clampFrame(entry.frame),
      value: entry.value,
    }))
    .sort((a, b) => a.frame - b.frame);
  const deduped: TimelineKeyframe<T>[] = [];
  let lastFrame = -1;
  for (const entry of normalized) {
    if (entry.frame === lastFrame) {
      deduped[deduped.length - 1] = entry;
    } else {
      deduped.push(entry);
      lastFrame = entry.frame;
    }
  }
  return deduped;
};

const normalizeSeedKeyframes = (keyframes: TimelineSeedKeyframe[]): TimelineSeedKeyframe[] => {
  const normalized = keyframes
    .map((entry) => ({
      frame: clampFrame(entry.frame),
      seed: Math.floor(entry.seed >>> 0),
    }))
    .sort((a, b) => a.frame - b.frame);
  const deduped: TimelineSeedKeyframe[] = [];
  let lastFrame = -1;
  for (const entry of normalized) {
    if (entry.frame === lastFrame) {
      deduped[deduped.length - 1] = entry;
    } else {
      deduped.push(entry);
      lastFrame = entry.frame;
    }
  }
  return deduped;
};

const normalizeLane = <T extends TimelineValue>(lane: TimelineLane<T>): TimelineLane<T> => ({
  id: lane.id,
  label: lane.label,
  kind: lane.kind,
  interpolation: lane.interpolation ?? 'step',
  keyframes: normalizeKeyframes(lane.keyframes),
});

const normalizeSeedTrack = (track: TimelineSeedTrack): TimelineSeedTrack => ({
  scope: track.scope,
  keyframes: normalizeSeedKeyframes(track.keyframes),
});

export const normalizeTimeline = (timeline: Timeline): Timeline => {
  const duration =
    Number.isFinite(timeline.durationFrames) && timeline.durationFrames >= 0
      ? Math.floor(timeline.durationFrames)
      : 0;
  const lanes = [...timeline.lanes]
    .map((lane) => normalizeLane(lane))
    .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const seeds = [...timeline.seeds]
    .map((track) => normalizeSeedTrack(track))
    .sort((a, b) => (a.scope < b.scope ? -1 : a.scope > b.scope ? 1 : 0));

  return {
    version: 1,
    fps: Number.isFinite(timeline.fps) && timeline.fps > 0 ? timeline.fps : 60,
    durationFrames: duration,
    lanes,
    seeds,
    metadata: timeline.metadata
      ? {
          ...timeline.metadata,
        }
      : undefined,
  };
};
