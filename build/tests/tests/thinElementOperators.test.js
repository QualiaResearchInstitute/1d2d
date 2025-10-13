import test from 'node:test';
import assert from 'node:assert/strict';
import { createKuramotoState, createDerivedViews, createNormalGenerator, derivedBufferSize, deriveKuramotoFields, getThinElementOperatorGains, initKuramotoState, stepKuramotoState, } from '../src/kuramotoCore.js';
import { clampKernelSpec, KERNEL_SPEC_DEFAULT } from '../src/kernel/kernelSpec.js';
const defaultParams = {
    alphaKur: 0.2,
    gammaKur: 0.15,
    omega0: 0,
    K0: 0.6,
    epsKur: 0.001,
    fluxX: 0,
    fluxY: 0,
    smallWorldWeight: 0,
    p_sw: 0,
};
test('thin-element gains respond deterministically to DMT/arousal controls', () => {
    const kernel = clampKernelSpec({
        gain: 2.4,
        k0: 0.18,
        Q: 3.5,
        anisotropy: 1.1,
        chirality: 1.2,
        transparency: 0.42,
    });
    const baseGains = getThinElementOperatorGains(kernel);
    const dmtGains = getThinElementOperatorGains(kernel, { dmt: 0.75 });
    const arousalGains = getThinElementOperatorGains(kernel, { arousal: 0.8 });
    const combinedGains = getThinElementOperatorGains(kernel, { dmt: 0.5, arousal: 0.5 });
    assert.ok(dmtGains.amplitude > baseGains.amplitude, 'DMT should raise amplitude gain');
    assert.ok(arousalGains.phase > baseGains.phase, 'arousal should raise phase gain');
    assert.deepEqual(combinedGains, getThinElementOperatorGains(kernel, { dmt: 0.5, arousal: 0.5 }), 'gain mapping must be deterministic');
});
const computeMaxDiff = (a, b) => {
    let max = 0;
    for (let i = 0; i < a.length; i++) {
        const diff = Math.abs(a[i] - b[i]);
        if (diff > max)
            max = diff;
    }
    return max;
};
const computeRelativeAmpDelta = (baseline, variant) => {
    let maxRelative = 0;
    for (let i = 0; i < baseline.length; i++) {
        const ref = Math.abs(baseline[i]) + 1e-6;
        const delta = Math.abs(variant[i] - baseline[i]) / ref;
        if (delta > maxRelative)
            maxRelative = delta;
    }
    return maxRelative;
};
test('beam-split schedule reproduces baseline thin-element chaining', () => {
    const width = 12;
    const height = 10;
    const state = createKuramotoState(width, height);
    const bufferBaseline = new ArrayBuffer(derivedBufferSize(width, height));
    const baselineDerived = createDerivedViews(bufferBaseline, width, height);
    initKuramotoState(state, 1, baselineDerived);
    const randn = createNormalGenerator(42);
    const dt = 0.016;
    const steps = 5;
    for (let i = 0; i < steps; i++) {
        stepKuramotoState(state, defaultParams, dt, randn, dt * (i + 1));
    }
    deriveKuramotoFields(state, baselineDerived);
    const bufferScheduled = new ArrayBuffer(derivedBufferSize(width, height));
    const scheduledDerived = createDerivedViews(bufferScheduled, width, height);
    const schedule = [
        {
            kind: 'beamSplit',
            recombine: 'average',
            branches: [
                { weight: 1, steps: [] },
                { weight: 1, steps: [] },
            ],
        },
        { kind: 'operator', operator: 'amplitude' },
        { kind: 'operator', operator: 'phase' },
    ];
    deriveKuramotoFields(state, scheduledDerived, { schedule });
    const metrics = {
        gradX: computeMaxDiff(baselineDerived.gradX, scheduledDerived.gradX),
        gradY: computeMaxDiff(baselineDerived.gradY, scheduledDerived.gradY),
        vort: computeMaxDiff(baselineDerived.vort, scheduledDerived.vort),
        coh: computeMaxDiff(baselineDerived.coh, scheduledDerived.coh),
        amp: computeMaxDiff(baselineDerived.amp, scheduledDerived.amp),
    };
    Object.entries(metrics).forEach(([key, value]) => {
        assert.ok(value < 1e-6, `${key} diff ${value} exceeds tolerance`);
    });
    const relativeAmp = computeRelativeAmpDelta(baselineDerived.amp, scheduledDerived.amp);
    assert.ok(relativeAmp <= 0.02, `amplitude should match within 2%, got ${relativeAmp}`);
});
test('DMT/arousal controls reshape derived amplitude deterministically', () => {
    const width = 16;
    const height = 12;
    const state = createKuramotoState(width, height);
    const buffer = new ArrayBuffer(derivedBufferSize(width, height));
    const derivedBase = createDerivedViews(buffer, width, height);
    initKuramotoState(state, 2, derivedBase);
    const randn = () => 0;
    const dt = 0.02;
    for (let i = 0; i < 4; i++) {
        stepKuramotoState(state, defaultParams, dt, randn, dt * (i + 1));
    }
    deriveKuramotoFields(state, derivedBase, { kernel: KERNEL_SPEC_DEFAULT });
    const bufferMod = new ArrayBuffer(derivedBufferSize(width, height));
    const derivedMod = createDerivedViews(bufferMod, width, height);
    deriveKuramotoFields(state, derivedMod, {
        kernel: KERNEL_SPEC_DEFAULT,
        controls: { dmt: 0.6, arousal: 0.4 },
    });
    let sumBase = 0;
    let sumMod = 0;
    for (let i = 0; i < derivedBase.amp.length; i++) {
        sumBase += derivedBase.amp[i];
        sumMod += derivedMod.amp[i];
    }
    assert.ok(sumMod > sumBase, 'combined controls should raise average amplitude');
    assert.deepEqual(getThinElementOperatorGains(KERNEL_SPEC_DEFAULT, { dmt: 0.6, arousal: 0.4 }), getThinElementOperatorGains(KERNEL_SPEC_DEFAULT, { dmt: 0.6, arousal: 0.4 }), 'operator gains must stay deterministic across repeated queries');
});
