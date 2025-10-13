import assert from 'node:assert/strict';
import test from 'node:test';

import { parseRationalFps } from '../scripts/videoUtils.js';

test('parseRationalFps parses rational values', () => {
  assert.equal(parseRationalFps('60000/1001'), 60000 / 1001);
  assert.equal(parseRationalFps('24/1'), 24);
});

test('parseRationalFps parses numeric values', () => {
  assert.equal(parseRationalFps('29.97'), 29.97);
});

test('parseRationalFps rejects invalid input', () => {
  assert.throws(() => parseRationalFps('0/0'));
  assert.throws(() => parseRationalFps('not-a-number'));
  assert.throws(() => parseRationalFps(''));
});
