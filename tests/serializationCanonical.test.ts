import test from 'node:test';
import assert from 'node:assert/strict';
import process from 'node:process';

import {
  writeCanonicalJson,
  readCanonicalJson,
  hashCanonicalJson,
  hashCanonicalJsonString,
} from '../src/serialization/canonicalJson.js';

test('canonical JSON round-trip preserves ordering and normalizes -0', () => {
  const input = {
    beta: 2,
    alpha: {
      gamma: -0,
      delta: [1, undefined, -0],
    },
  };

  const json = writeCanonicalJson(input);
  assert.equal(json, '{"alpha":{"delta":[1,null,0],"gamma":0},"beta":2}');

  const parsed = readCanonicalJson<typeof input>(json);
  const rerendered = writeCanonicalJson(parsed);
  assert.equal(rerendered, json);
});

test('canonical JSON hashing remains stable across locales', () => {
  const payload = {
    value: 1.2345,
    nested: {
      zeta: -0,
      eta: [0.1, 0.2, 0.3],
    },
  };

  const nodeProcess = process as unknown as NodeJS.Process;
  const originalLocale = nodeProcess.env.LC_ALL;
  const baseline = hashCanonicalJson(payload);
  try {
    nodeProcess.env.LC_ALL = 'fr_FR';
    const variant = hashCanonicalJson(payload);
    assert.equal(variant.hash, baseline.hash);
    assert.equal(variant.json, baseline.json);
    const hashFromString = hashCanonicalJsonString(baseline.json);
    assert.equal(hashFromString, baseline.hash);
  } finally {
    if (originalLocale === undefined) {
      delete nodeProcess.env.LC_ALL;
    } else {
      nodeProcess.env.LC_ALL = originalLocale;
    }
  }
});

test('canonical JSON rejects non-finite numbers', () => {
  assert.throws(
    () => writeCanonicalJson({ invalid: Number.POSITIVE_INFINITY }),
    /non-finite numbers/i,
  );
});

test('canonical JSON hashing emits 256-bit hex digests', () => {
  const payload = { foo: 'bar', baz: [1, 2, 3] };
  const { hash } = hashCanonicalJson(payload);
  assert.equal(hash.length, 64);
  assert.ok(/^[0-9a-f]+$/i.test(hash), 'hash should be hexadecimal');

  const json = writeCanonicalJson(payload);
  const hashFromString = hashCanonicalJsonString(json);
  assert.equal(hashFromString.length, 64);
  assert.equal(hashFromString, hash);
});
