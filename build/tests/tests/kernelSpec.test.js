import test from 'node:test';
import assert from 'node:assert/strict';
import { clampKernelSpec, createKernelSpec, getCouplingKernelParams, getDefaultKernelSpec, getKernelSpecBounds, kernelSpecToJSON, } from '../src/kernel/kernelSpec.js';
test('getDefaultKernelSpec returns sanitized clone', () => {
    const a = getDefaultKernelSpec();
    const b = getDefaultKernelSpec();
    assert.notStrictEqual(a, b);
    assert.deepEqual(a, b);
});
test('createKernelSpec clamps to allowed bounds', () => {
    const bounds = getKernelSpecBounds();
    const scalars = bounds.scalars;
    const extreme = {
        gain: scalars.gain.max * 4,
        k0: scalars.k0.min / 10,
        Q: scalars.Q.max * 3,
        anisotropy: -1,
        chirality: Number.POSITIVE_INFINITY,
        transparency: Number.NaN,
        couplingPreset: 'invalid',
    };
    const sanitized = createKernelSpec(extreme);
    assert.equal(sanitized.gain, scalars.gain.max);
    assert.equal(sanitized.k0, scalars.k0.min);
    assert.equal(sanitized.Q, scalars.Q.max);
    assert.equal(sanitized.anisotropy, scalars.anisotropy.min);
    assert.equal(sanitized.chirality, scalars.chirality.max);
    assert.equal(sanitized.transparency, scalars.transparency.min);
    assert.equal(sanitized.couplingPreset, 'dmt');
});
test('clampKernelSpec preserves provided values within bounds', () => {
    const sample = clampKernelSpec({
        gain: 2.2,
        k0: 0.24,
        Q: 5.1,
        anisotropy: 0.8,
        chirality: 1.2,
        transparency: 0.6,
        couplingPreset: '5meo',
    });
    assert.equal(sample.gain, 2.2);
    assert.equal(sample.k0, 0.24);
    assert.equal(sample.Q, 5.1);
    assert.equal(sample.anisotropy, 0.8);
    assert.equal(sample.chirality, 1.2);
    assert.equal(sample.transparency, 0.6);
    assert.equal(sample.couplingPreset, '5meo');
});
test('kernelSpecToJSON survives round-tripping', () => {
    const spec = clampKernelSpec({
        gain: 1.75,
        k0: 0.19,
        Q: 3.3,
        anisotropy: 0.55,
        chirality: 0.9,
        transparency: 0.4,
        couplingPreset: '5meo',
    });
    const serialized = kernelSpecToJSON(spec);
    const roundTripped = createKernelSpec(JSON.parse(JSON.stringify(serialized)));
    assert.deepEqual(roundTripped, spec);
});
test('getCouplingKernelParams returns immutable presets', () => {
    const original = getCouplingKernelParams('dmt');
    assert.equal(original.preset, 'dmt');
    assert.ok(original.radius > 0);
    const mutated = getCouplingKernelParams('dmt');
    mutated.radius += 1;
    const next = getCouplingKernelParams('dmt');
    assert.equal(next.radius, original.radius);
});
