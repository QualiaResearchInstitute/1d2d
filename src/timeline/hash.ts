import { hashCanonicalJson, writeCanonicalJson } from '../serialization/canonicalJson.js';
import { normalizeTimeline } from './normalize.js';
import type { Timeline } from './types.js';

export type TimelineSerializeOptions = {
  indent?: number;
};

export const canonicalizeTimeline = (timeline: Timeline): Timeline => normalizeTimeline(timeline);

export const serializeTimeline = (
  timeline: Timeline,
  options: TimelineSerializeOptions = {},
): { json: string; hash: string } => {
  const normalized = canonicalizeTimeline(timeline);
  return hashCanonicalJson(normalized, options);
};

export const timelineToJson = (timeline: Timeline, options: TimelineSerializeOptions = {}) => {
  const normalized = canonicalizeTimeline(timeline);
  return writeCanonicalJson(normalized, options);
};
