import { createHash } from 'blake3';

import { evaluateTimeline } from './evaluate.js';
import { canonicalizeTimeline, serializeTimeline } from './hash.js';
import type { Timeline, TimelineEvaluation, TimelineSeedTrack, TimelineValue } from './types.js';

const textEncoder = new TextEncoder();

export const hexToBytes = (hex: string): Uint8Array => {
  const clean = hex.length % 2 === 0 ? hex : `0${hex}`;
  const result = new Uint8Array(clean.length / 2);
  for (let i = 0; i < clean.length; i += 2) {
    result[i / 2] = parseInt(clean.slice(i, i + 2), 16);
  }
  return result;
};

export const deriveSeedFromHash = (timelineHash: string, scope: string, frame: number): number => {
  const hasher = createHash();
  hasher.update(hexToBytes(timelineHash));
  hasher.update(textEncoder.encode(scope));
  const frameBytes = new Uint8Array(4);
  new DataView(frameBytes.buffer).setUint32(0, frame >>> 0, true);
  hasher.update(frameBytes);
  const digest = hasher.digest();
  const view = new DataView(digest.buffer, digest.byteOffset, digest.byteLength);
  return view.getUint32(0, true);
};

export type TimelineFrameEvaluation = TimelineEvaluation & {
  frameIndex: number;
};

export class TimelinePlayer {
  readonly timeline: Timeline;
  readonly hash: string;
  readonly json: string;
  readonly fps: number;
  readonly durationFrames: number;
  private readonly seedScopes: Set<string>;

  constructor(source: Timeline) {
    this.timeline = canonicalizeTimeline(source);
    const { hash, json } = serializeTimeline(this.timeline);
    this.hash = hash;
    this.json = json;
    this.fps = this.timeline.fps;
    this.durationFrames = this.timeline.durationFrames;
    this.seedScopes = new Set(this.timeline.seeds.map((track) => track.scope));
  }

  getSeedTrack(scope: string): TimelineSeedTrack | undefined {
    return this.timeline.seeds.find((track) => track.scope === scope);
  }

  getFrameForTime(timeSeconds: number): number {
    if (!Number.isFinite(timeSeconds) || timeSeconds <= 0) {
      return 0;
    }
    const raw = Math.round(timeSeconds * this.fps);
    if (raw <= 0) return 0;
    if (raw >= this.durationFrames) {
      return this.durationFrames;
    }
    return raw;
  }

  evaluateAtFrame(frame: number): TimelineFrameEvaluation {
    const clampedFrame = Math.max(0, Math.min(this.durationFrames, Math.floor(frame)));
    const evaluation = evaluateTimeline(this.timeline, clampedFrame, { clampFrame: true });
    return {
      frame: evaluation.frame,
      values: evaluation.values,
      seeds: evaluation.seeds,
      frameIndex: clampedFrame,
    };
  }

  evaluateAtTimeSeconds(timeSeconds: number): TimelineFrameEvaluation {
    const frameIndex = this.getFrameForTime(timeSeconds);
    return this.evaluateAtFrame(frameIndex);
  }

  getValueAtTime(id: string, timeSeconds: number): TimelineValue | undefined {
    const evaluation = this.evaluateAtTimeSeconds(timeSeconds);
    return evaluation.values[id];
  }

  getSeedAtFrame(scope: string, frame: number): number {
    const evaluation = this.evaluateAtFrame(frame);
    if (Object.prototype.hasOwnProperty.call(evaluation.seeds, scope)) {
      return evaluation.seeds[scope];
    }
    return deriveSeedFromHash(this.hash, scope, evaluation.frameIndex);
  }

  getSeedAtTime(scope: string, timeSeconds: number): number {
    const frameIndex = this.getFrameForTime(timeSeconds);
    return this.getSeedAtFrame(scope, frameIndex);
  }

  hasScopedSeed(scope: string): boolean {
    return this.seedScopes.has(scope);
  }
}
