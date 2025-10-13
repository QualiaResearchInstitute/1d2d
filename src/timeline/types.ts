export type TimelineValue = number | boolean | string;

export type TimelineInterpolation = 'step' | 'linear';

export type TimelineKeyframe<T extends TimelineValue = TimelineValue> = {
  frame: number;
  value: T;
};

export type TimelineLane<T extends TimelineValue = TimelineValue> = {
  id: string;
  label?: string;
  kind?: 'scalar' | 'boolean' | 'enum';
  interpolation?: TimelineInterpolation;
  keyframes: TimelineKeyframe<T>[];
};

export type TimelineSeedKeyframe = {
  frame: number;
  seed: number;
};

export type TimelineSeedTrack = {
  scope: string;
  keyframes: TimelineSeedKeyframe[];
};

export type TimelineMetadata = {
  name?: string;
  description?: string;
  createdAt?: string;
  updatedAt?: string;
  versionTag?: string;
};

export type Timeline = {
  version: 1;
  fps: number;
  durationFrames: number;
  lanes: TimelineLane[];
  seeds: TimelineSeedTrack[];
  metadata?: TimelineMetadata;
};

export type TimelineEvaluation = {
  frame: number;
  values: Record<string, TimelineValue>;
  seeds: Record<string, number>;
};
