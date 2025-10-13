import test from 'node:test';
import assert from 'node:assert/strict';
import { runCrossTierValidation, summarizeAlerts, } from '../src/validation/crossTierValidation.js';
const TIERS = ['rim1p5D', 'surface2D', 'volume2p5D'];
test('cross-tier validation estimates tracer half-life close to configured targets', () => {
    const report = runCrossTierValidation({
        dt: 0.05,
        steps: 28,
        halfLives: {
            rim1p5D: 0.4,
            surface2D: 0.55,
            volume2p5D: 0.7,
        },
        kernelDelta: { gain: 2.8 },
        dmt: 0.3,
    });
    for (const tier of TIERS) {
        const diff = Math.abs(report.baseline[tier].measuredHalfLife - report.baseline[tier].expectedHalfLife);
        assert.ok(diff < 0.12, `${tier} baseline half-life diff ${diff.toFixed(3)} exceeds tolerance`);
        const variantDiff = Math.abs(report.variant[tier].measuredHalfLife - report.variant[tier].expectedHalfLife);
        assert.ok(variantDiff < 0.12, `${tier} variant half-life diff ${variantDiff.toFixed(3)} exceeds tolerance`);
    }
});
test('cross-tier validation records kernel-induced deltas across tiers', () => {
    const report = runCrossTierValidation({
        kernelDelta: { gain: 3.2, k0: 0.08 },
        steps: 24,
    });
    for (const tier of TIERS) {
        const delta = report.kernelDelta.perTier[tier].delta;
        assert.ok(Math.abs(delta) > 1e-4, `${tier} expected measurable kernel delta magnitude, received ${delta.toFixed(4)}`);
    }
});
test('cross-tier validation raises coherence alerts when noise exceeds tolerance', () => {
    const report = runCrossTierValidation({
        variantNoise: { surface2D: 0.95, volume2p5D: 0.7 },
        coherenceTolerance: 0.9,
        steps: 20,
    });
    const coherenceAlerts = report.alerts.filter((alert) => alert.kind === 'coherence');
    assert.ok(coherenceAlerts.length > 0, 'expected at least one coherence alert');
    const summaries = summarizeAlerts(coherenceAlerts);
    assert.ok(summaries.some((line) => line.includes('coherence drop')), 'summaries should include coherence drop message');
});
