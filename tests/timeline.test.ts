import assert from 'node:assert/strict';
import test from 'node:test';

import {
  canonicalizeTimeline,
  evaluateTimeline,
  serializeTimeline,
  type Timeline,
  type TimelineLane,
  TimelinePlayer,
} from '../src/timeline/index.js';

const createTimeline = (): Timeline => ({
  version: 1,
  fps: 24,
  durationFrames: 48,
  lanes: [
    {
      id: 'mode',
      keyframes: [
        { frame: 12, value: 'warp' },
        { frame: 0, value: 'rim' },
      ],
    },
    {
      id: 'dmt',
      interpolation: 'linear',
      keyframes: [
        { frame: 16, value: 0.6 },
        { frame: 0, value: 0.2 },
        { frame: 8, value: 0.4 },
      ],
    },
  ],
  seeds: [
    {
      scope: 'kuramoto',
      keyframes: [
        { frame: 20, seed: 2024 },
        { frame: 0, seed: 1234 },
      ],
    },
  ],
  metadata: {
    name: 'demo',
    description: 'Phase 8 timeline sample',
  },
});

test('canonicalizeTimeline sorts lanes and keyframes', () => {
  const normalized = canonicalizeTimeline(createTimeline());
  const laneIds = normalized.lanes.map((lane) => lane.id);
  assert.deepEqual(laneIds, ['dmt', 'mode']);
  const dmtLane = normalized.lanes.find((lane) => lane.id === 'dmt') as TimelineLane<number>;
  assert.ok(dmtLane);
  const frameOrder = dmtLane.keyframes.map((kf) => kf.frame);
  assert.deepEqual(frameOrder, [0, 8, 16]);
});

test('evaluateTimeline handles step and linear interpolation', () => {
  const timeline = createTimeline();
  const mid = evaluateTimeline(timeline, 12);
  assert.equal(mid.frame, 12);
  assert.equal(mid.values.mode, 'warp');
  assert.ok(Math.abs((mid.values.dmt as number) - 0.5) < 1e-9);
  assert.equal(mid.seeds.kuramoto, 1234);

  const before = evaluateTimeline(timeline, 4);
  assert.equal(before.values.mode, 'rim');
  assert.ok(Math.abs((before.values.dmt as number) - 0.3) < 1e-9);
  assert.equal(before.seeds.kuramoto, 1234);

  const later = evaluateTimeline(timeline, 24);
  assert.equal(later.seeds.kuramoto, 2024);
});

test('serializeTimeline produces stable hash for equivalent data', () => {
  const timelineA = createTimeline();
  const timelineB = createTimeline();
  // reorder lanes to confirm canonicalization drives identical hash
  timelineB.lanes.reverse();
  const { hash: hashA, json: jsonA } = serializeTimeline(timelineA, { indent: 2 });
  const { hash: hashB, json: jsonB } = serializeTimeline(timelineB, { indent: 2 });
  assert.equal(hashA, hashB);
  assert.equal(jsonA, jsonB);

  timelineB.lanes[0].keyframes[0].value = 0.25;
  const { hash: hashC } = serializeTimeline(timelineB);
  assert.notEqual(hashA, hashC);
});

test('TimelinePlayer derives scoped seeds deterministically', () => {
  const timeline = createTimeline();
  const player = new TimelinePlayer(timeline);
  const explicitSeed = player.getSeedAtFrame('kuramoto', 0);
  assert.equal(explicitSeed, 1234);

  const derivedA = player.getSeedAtFrame('volume', 5);
  const derivedB = player.getSeedAtFrame('volume', 5);
  assert.equal(derivedA, derivedB);
  const derivedC = player.getSeedAtFrame('volume', 12);
  assert.notEqual(derivedA, derivedC);

  const timeSeed = player.getSeedAtTime('volume', 0.5);
  assert.equal(timeSeed, player.getSeedAtFrame('volume', player.getFrameForTime(0.5)));
});
