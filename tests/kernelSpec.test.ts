import test from "node:test";
import assert from "node:assert/strict";

import {
  clampKernelSpec,
  createKernelSpec,
  getDefaultKernelSpec,
  getKernelSpecBounds,
  kernelSpecToJSON,
  type KernelSpec
} from "../src/kernel/kernelSpec.js";

test("getDefaultKernelSpec returns sanitized clone", () => {
  const a = getDefaultKernelSpec();
  const b = getDefaultKernelSpec();
  assert.notStrictEqual(a, b);
  assert.deepEqual(a, b);
});

test("createKernelSpec clamps to allowed bounds", () => {
  const bounds = getKernelSpecBounds();
  const extreme: Partial<KernelSpec> = {
    gain: bounds.gain.max * 4,
    k0: bounds.k0.min / 10,
    Q: bounds.Q.max * 3,
    anisotropy: -1,
    chirality: Number.POSITIVE_INFINITY,
    transparency: Number.NaN
  };
  const sanitized = createKernelSpec(extreme);
  assert.equal(sanitized.gain, bounds.gain.max);
  assert.equal(sanitized.k0, bounds.k0.min);
  assert.equal(sanitized.Q, bounds.Q.max);
  assert.equal(sanitized.anisotropy, bounds.anisotropy.min);
  assert.equal(sanitized.chirality, bounds.chirality.max);
  assert.equal(sanitized.transparency, bounds.transparency.min);
});

test("clampKernelSpec preserves provided values within bounds", () => {
  const sample = clampKernelSpec({
    gain: 2.2,
    k0: 0.24,
    Q: 5.1,
    anisotropy: 0.8,
    chirality: 1.2,
    transparency: 0.6
  });
  assert.equal(sample.gain, 2.2);
  assert.equal(sample.k0, 0.24);
  assert.equal(sample.Q, 5.1);
  assert.equal(sample.anisotropy, 0.8);
  assert.equal(sample.chirality, 1.2);
  assert.equal(sample.transparency, 0.6);
});

test("kernelSpecToJSON survives round-tripping", () => {
  const spec = clampKernelSpec({
    gain: 1.75,
    k0: 0.19,
    Q: 3.3,
    anisotropy: 0.55,
    chirality: 0.9,
    transparency: 0.4
  });
  const serialized = kernelSpecToJSON(spec);
  const roundTripped = createKernelSpec(JSON.parse(JSON.stringify(serialized)));
  assert.deepEqual(roundTripped, spec);
});
