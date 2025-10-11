import test from 'node:test';
import assert from 'node:assert/strict';

import { createSmallWorldRewiring } from '../src/kuramotoCore.js';

test('small-world rewiring is deterministic for a fixed seed', () => {
  const width = 8;
  const height = 8;
  const degree = 12;
  const seed = 12345;

  const first = createSmallWorldRewiring(width, height, degree, seed);
  const second = createSmallWorldRewiring(width, height, degree, seed);

  assert.equal(first.degree, degree, 'first rewiring degree should match input');
  assert.equal(second.degree, degree, 'second rewiring degree should match input');
  assert.equal(first.targets.length, width * height * degree);
  assert.equal(second.targets.length, width * height * degree);
  assert.deepEqual(Array.from(first.targets), Array.from(second.targets));
  for (let idx = 0; idx < width * height; idx++) {
    const offset = idx * degree;
    for (let edge = 0; edge < degree; edge++) {
      assert.notEqual(
        first.targets[offset + edge],
        idx,
        `rewiring should avoid self loops (node ${idx}, edge ${edge})`,
      );
    }
  }
});

test('small-world rewiring changes when the seed changes', () => {
  const width = 8;
  const height = 8;
  const degree = 12;

  const base = createSmallWorldRewiring(width, height, degree, 111);
  const variant = createSmallWorldRewiring(width, height, degree, 222);

  assert.equal(base.degree, degree);
  assert.equal(variant.degree, degree);
  assert.equal(base.targets.length, variant.targets.length);
  assert.notDeepEqual(Array.from(base.targets), Array.from(variant.targets));
});
