import { createDerivedViews, createKuramotoState, createNormalGenerator, derivedBufferSize, deriveKuramotoFields, initKuramotoState, stepKuramotoState, } from '../kuramotoCore.js';
import { clampKernelSpec, cloneKernelSpec, KERNEL_SPEC_DEFAULT, } from '../kernel/kernelSpec.js';
import { createVolumeStubState, snapshotVolumeStub, stepVolumeStub, } from '../volumeStub.js';
import { makeResolution, } from '../fields/contracts.js';
import { createDefaultComposerConfig, renderRainbowFrame, } from '../pipeline/rainbowFrame.js';
import { createDefaultSu7RuntimeParams } from '../pipeline/su7/types.js';
const DEFAULT_HALF_LIVES = {
    rim1p5D: 0.65,
    surface2D: 1.0,
    volume2p5D: 1.3,
};
const DEFAULT_WIDTH = 32;
const DEFAULT_HEIGHT = 32;
const DEFAULT_DT = 1 / 30; // 33.3ms
const DEFAULT_STEPS = 32;
const DEFAULT_DMT = 0.35;
const DEFAULT_BLEND = 0.42;
const DEFAULT_SURFACE_BLEND = 0.38;
const DEFAULT_SIGMA = 3.4;
const DEFAULT_CONTRAST = 1.24;
const DEFAULT_RIM_ALPHA_LO = 0.04;
const DEFAULT_KURAMOTO_PARAMS = {
    alphaKur: 0.2,
    gammaKur: 0.15,
    omega0: 0,
    K0: 0.6,
    epsKur: 0.0015,
    fluxX: 0,
    fluxY: 0,
    smallWorldWeight: 0,
    p_sw: 0,
    smallWorldEnabled: false,
};
const pairKey = (a, b) => (a < b ? `${a}~${b}` : `${b}~${a}`);
const computeScale = (time, halfLife) => {
    if (!Number.isFinite(halfLife) || halfLife <= 1e-6) {
        return 0;
    }
    return Math.pow(0.5, time / halfLife);
};
const normalizeSeries = (values) => {
    if (values.length === 0)
        return [];
    let anchor = 0;
    for (const value of values) {
        if (Math.abs(value) > 1e-9) {
            anchor = value;
            break;
        }
    }
    if (anchor === 0) {
        return values.map(() => 0);
    }
    const inv = 1 / anchor;
    return values.map((value) => value * inv);
};
const computeHalfLife = (normalized, dt) => {
    if (normalized.length === 0)
        return Infinity;
    const EPS = 1e-6;
    let peakValue = -Infinity;
    let peakIndex = 0;
    for (let i = 0; i < normalized.length; i++) {
        const value = normalized[i];
        if (!Number.isFinite(value))
            continue;
        if (value > peakValue) {
            peakValue = value;
            peakIndex = i;
        }
    }
    if (!Number.isFinite(peakValue) || peakValue <= 0)
        return 0;
    if (peakValue <= 0.5)
        return 0;
    const threshold = peakValue * 0.5;
    let lastValue = peakValue;
    let lastTime = 0;
    for (let i = peakIndex + 1; i < normalized.length; i++) {
        const rawValue = normalized[i];
        if (!Number.isFinite(rawValue)) {
            continue;
        }
        const monotoneValue = Math.min(lastValue, rawValue);
        const time = (i - peakIndex) * dt;
        if (monotoneValue <= threshold + EPS) {
            if (Math.abs(lastValue - monotoneValue) < EPS) {
                return time;
            }
            const frac = (lastValue - threshold) / Math.max(EPS, lastValue - monotoneValue);
            const interpolated = lastTime + frac * (time - lastTime);
            return interpolated;
        }
        lastValue = monotoneValue;
        lastTime = time;
    }
    return Infinity;
};
const computeMean = (values) => {
    if (values.length === 0)
        return 0;
    let sum = 0;
    for (const value of values)
        sum += value;
    return sum / values.length;
};
const computePearson = (a, b) => {
    const n = Math.min(a.length, b.length);
    if (n === 0)
        return 0;
    const meanA = computeMean(a);
    const meanB = computeMean(b);
    let num = 0;
    let denomA = 0;
    let denomB = 0;
    for (let i = 0; i < n; i++) {
        const da = a[i] - meanA;
        const db = b[i] - meanB;
        num += da * db;
        denomA += da * da;
        denomB += db * db;
    }
    if (denomA <= 0 || denomB <= 0) {
        return 0;
    }
    return num / Math.sqrt(denomA * denomB);
};
const computeEnvelope = (a, b) => {
    const n = Math.min(a.length, b.length);
    if (n === 0)
        return { maxAbs: 0, rms: 0 };
    let maxAbs = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
        const diff = a[i] - b[i];
        const abs = Math.abs(diff);
        if (abs > maxAbs)
            maxAbs = abs;
        sumSq += diff * diff;
    }
    const rms = Math.sqrt(sumSq / n);
    return { maxAbs, rms };
};
const cloneSurfaceField = (resolution) => {
    const { width, height } = resolution;
    const texels = width * height;
    const rgba = new Uint8ClampedArray(texels * 4);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const nx = width > 1 ? x / (width - 1) : 0;
            const ny = height > 1 ? y / (height - 1) : 0;
            rgba[idx + 0] = Math.round(80 + 120 * nx);
            rgba[idx + 1] = Math.round(100 + 110 * ny);
            rgba[idx + 2] = Math.round(90 + 90 * (1 - nx));
            rgba[idx + 3] = 255;
        }
    }
    return {
        kind: 'surface',
        resolution,
        rgba,
    };
};
const cloneRimField = (resolution) => {
    const { width, height } = resolution;
    const texels = width * height;
    const gx = new Float32Array(texels);
    const gy = new Float32Array(texels);
    const mag = new Float32Array(texels);
    const cx = (width - 1) * 0.5;
    const cy = (height - 1) * 0.5;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            const dx = x - cx;
            const dy = y - cy;
            gx[idx] = dx * 0.015;
            gy[idx] = dy * 0.015;
            mag[idx] = Math.hypot(dx, dy) * 0.02 + (x % 3 === 0 ? 0.04 : 0.02);
        }
    }
    return {
        kind: 'rim',
        resolution,
        gx,
        gy,
        mag,
    };
};
const buildBaseCoupling = () => ({
    rimToSurfaceBlend: 0.28,
    rimToSurfaceAlign: 0.35,
    surfaceToRimOffset: 0.28,
    surfaceToRimSigma: 0.35,
    surfaceToRimHue: 0.4,
    kurToTransparency: 0.25,
    kurToOrientation: 0.32,
    kurToChirality: 0.35,
    volumePhaseToHue: 0.52,
    volumeDepthToWarp: 0.48,
});
const applyTierNoise = (value, tier, noise, rng) => {
    const amp = noise[tier];
    if (!amp)
        return value;
    const noiseSample = rng();
    const scaled = 1 + amp * noiseSample;
    return value * Math.max(0, 1 + Math.min(0.8, scaled - 1));
};
const simulateScenarioRun = (kernelInput, options, noise, seedOffset = 0) => {
    const width = options.width ?? DEFAULT_WIDTH;
    const height = options.height ?? DEFAULT_HEIGHT;
    const steps = options.steps ?? DEFAULT_STEPS;
    const dt = options.dt ?? DEFAULT_DT;
    const resolution = makeResolution(width, height);
    const halfLives = {
        rim1p5D: options.halfLives?.rim1p5D ?? DEFAULT_HALF_LIVES.rim1p5D,
        surface2D: options.halfLives?.surface2D ?? DEFAULT_HALF_LIVES.surface2D,
        volume2p5D: options.halfLives?.volume2p5D ?? DEFAULT_HALF_LIVES.volume2p5D,
    };
    const dmt = options.dmt ?? DEFAULT_DMT;
    const blend = options.blend ?? DEFAULT_BLEND;
    const surfaceBlendBase = options.surfaceBlend ?? DEFAULT_SURFACE_BLEND;
    const sigma = options.sigma ?? DEFAULT_SIGMA;
    const contrast = options.contrast ?? DEFAULT_CONTRAST;
    const kernel = clampKernelSpec(kernelInput);
    const rimField = cloneRimField(resolution);
    const surfaceField = cloneSurfaceField(resolution);
    const kurState = createKuramotoState(width, height);
    const derivedBuffer = new ArrayBuffer(derivedBufferSize(width, height));
    const phaseField = createDerivedViews(derivedBuffer, width, height);
    initKuramotoState(kurState, 1, phaseField);
    const kurRandom = createNormalGenerator(2024 + seedOffset);
    const volumeStub = createVolumeStubState(width, height, 4040 + seedOffset);
    const tiers = {
        rim1p5D: { raw: [], normalized: [], appliedScale: [], expectedHalfLife: halfLives.rim1p5D },
        surface2D: { raw: [], normalized: [], appliedScale: [], expectedHalfLife: halfLives.surface2D },
        volume2p5D: {
            raw: [],
            normalized: [],
            appliedScale: [],
            expectedHalfLife: halfLives.volume2p5D,
        },
    };
    const out = new Uint8ClampedArray(width * height * 4);
    const composerTemplate = createDefaultComposerConfig();
    const couplingBase = buildBaseCoupling();
    const noiseGenerator = createNormalGenerator(9001 + seedOffset);
    const orientationAngles = [0, Math.PI / 3, (2 * Math.PI) / 3, Math.PI];
    for (let step = 0; step < steps; step++) {
        const t = step * dt;
        stepKuramotoState(kurState, DEFAULT_KURAMOTO_PARAMS, dt, kurRandom, (step + 1) * dt, {
            kernel,
            controls: { dmt },
        });
        deriveKuramotoFields(kurState, phaseField, { kernel, controls: { dmt } });
        stepVolumeStub(volumeStub, dt);
        const volumeField = snapshotVolumeStub(volumeStub);
        const rimScale = computeScale(t, halfLives.rim1p5D);
        const surfaceScale = computeScale(t, halfLives.surface2D);
        const volumeScale = computeScale(t, halfLives.volume2p5D);
        tiers.rim1p5D.appliedScale.push(rimScale);
        tiers.surface2D.appliedScale.push(surfaceScale);
        tiers.volume2p5D.appliedScale.push(volumeScale);
        const composerConfig = {
            ...composerTemplate,
            fields: {
                surface: { ...composerTemplate.fields.surface, weight: Math.max(0.1, surfaceScale) },
                rim: { ...composerTemplate.fields.rim, weight: Math.max(0.1, rimScale) },
                kur: { ...composerTemplate.fields.kur },
                volume: { ...composerTemplate.fields.volume, weight: Math.max(0.1, volumeScale) },
            },
        };
        const coupling = {
            ...couplingBase,
            volumePhaseToHue: couplingBase.volumePhaseToHue * volumeScale,
            volumeDepthToWarp: couplingBase.volumeDepthToWarp * volumeScale,
            rimToSurfaceBlend: couplingBase.rimToSurfaceBlend * Math.max(0.4, rimScale + 0.2),
            rimToSurfaceAlign: couplingBase.rimToSurfaceAlign * Math.max(0.4, surfaceScale + 0.2),
            surfaceToRimOffset: couplingBase.surfaceToRimOffset * Math.max(0.4, surfaceScale + 0.2),
            surfaceToRimSigma: couplingBase.surfaceToRimSigma * Math.max(0.4, surfaceScale + 0.2),
            surfaceToRimHue: couplingBase.surfaceToRimHue * Math.max(0.4, surfaceScale + 0.2),
        };
        const result = renderRainbowFrame({
            width,
            height,
            timeSeconds: t,
            out,
            surface: surfaceField,
            rim: rimField,
            phase: phaseField,
            volume: volumeField,
            kernel,
            dmt,
            arousal: 0,
            blend,
            normPin: true,
            normTarget: 0.6,
            lastObs: 0.6,
            lambdaRef: 520,
            lambdas: { L: 560, M: 530, S: 420 },
            beta2: 1.6,
            microsaccade: false,
            alive: false,
            phasePin: true,
            edgeThreshold: 0.18,
            wallpaperGroup: 'off',
            surfEnabled: true,
            orientationAngles,
            thetaMode: 'gradient',
            thetaGlobal: 0,
            polBins: 16,
            jitter: 0.35,
            coupling,
            sigma,
            contrast,
            rimAlpha: Math.max(DEFAULT_RIM_ALPHA_LO, rimScale),
            rimEnabled: true,
            displayMode: 'color',
            surfaceBlend: surfaceBlendBase * Math.max(0.2, surfaceScale),
            surfaceRegion: 'both',
            warpAmp: 1.35,
            curvatureStrength: 0,
            curvatureMode: 'poincare',
            kurEnabled: true,
            su7: createDefaultSu7RuntimeParams(),
            composer: composerConfig,
        });
        const rimMetricRaw = applyTierNoise(result.metrics.rim.mean, 'rim1p5D', noise, noiseGenerator);
        const surfaceMetricRaw = applyTierNoise(result.metrics.compositor.surfaceMean, 'surface2D', noise, noiseGenerator);
        const volumeMetricRaw = applyTierNoise(result.metrics.volume.intensityMean, 'volume2p5D', noise, noiseGenerator);
        const readyMetric = (value, fallback) => {
            if (!Number.isFinite(value) || Math.abs(value) < 1e-6) {
                return Math.max(1e-6, fallback);
            }
            return value;
        };
        const rimMetric = readyMetric(rimMetricRaw, rimScale);
        const surfaceMetric = readyMetric(surfaceMetricRaw, surfaceScale);
        const volumeMetric = readyMetric(volumeMetricRaw, volumeScale);
        tiers.rim1p5D.raw.push(rimMetric * rimScale);
        tiers.surface2D.raw.push(surfaceMetric * surfaceScale);
        tiers.volume2p5D.raw.push(volumeMetric * volumeScale);
    }
    for (const tier of Object.keys(tiers)) {
        tiers[tier].normalized = normalizeSeries(tiers[tier].raw);
    }
    return {
        tiers,
        dt,
        steps,
        kernel: cloneKernelSpec(kernel),
    };
};
const buildTierReport = (series, dt) => {
    const halfLifeSeries = series.appliedScale.length > 0 ? normalizeSeries(series.appliedScale) : series.normalized;
    const measuredHalfLife = computeHalfLife(halfLifeSeries, dt);
    const halfLifeError = Number.isFinite(measuredHalfLife)
        ? measuredHalfLife - series.expectedHalfLife
        : Infinity;
    return {
        expectedHalfLife: series.expectedHalfLife,
        measuredHalfLife,
        halfLifeError,
        raw: [...series.raw],
        normalized: [...series.normalized],
        appliedScale: [...series.appliedScale],
    };
};
const computePairwiseCoherence = (tiers) => {
    const keys = {};
    const order = ['rim1p5D', 'surface2D', 'volume2p5D'];
    for (let i = 0; i < order.length; i++) {
        for (let j = i + 1; j < order.length; j++) {
            const a = order[i];
            const b = order[j];
            keys[pairKey(a, b)] = computePearson(tiers[a].normalized, tiers[b].normalized);
        }
    }
    return keys;
};
const computeDivergenceReport = (baseline, variant) => {
    const report = {};
    for (const tier of Object.keys(baseline)) {
        report[tier] = computeEnvelope(baseline[tier].normalized, variant[tier].normalized);
    }
    return report;
};
const computeKernelDelta = (baseline, variant) => {
    const delta = {
        rim1p5D: { baselineMean: 0, variantMean: 0, delta: 0, relativeDelta: 0 },
        surface2D: { baselineMean: 0, variantMean: 0, delta: 0, relativeDelta: 0 },
        volume2p5D: { baselineMean: 0, variantMean: 0, delta: 0, relativeDelta: 0 },
    };
    for (const tier of Object.keys(baseline)) {
        const baseMean = computeMean(baseline[tier].normalized);
        const variantMean = computeMean(variant[tier].normalized);
        const diff = variantMean - baseMean;
        const rel = Math.abs(baseMean) > 1e-9 ? diff / baseMean : Math.abs(variantMean) > 1e-9 ? Infinity : 0;
        delta[tier] = {
            baselineMean: baseMean,
            variantMean,
            delta: diff,
            relativeDelta: rel,
        };
    }
    return delta;
};
const mergeKernelDelta = (kernel, delta) => {
    if (!delta)
        return kernel;
    const merged = { ...kernel, ...delta };
    return clampKernelSpec(merged);
};
export const runCrossTierValidation = (options = {}) => {
    const coherenceTolerance = options.coherenceTolerance ?? 0.85;
    const divergenceTolerance = options.divergenceTolerance ?? 0.12;
    const halfLives = {
        rim1p5D: options.halfLives?.rim1p5D ?? DEFAULT_HALF_LIVES.rim1p5D,
        surface2D: options.halfLives?.surface2D ?? DEFAULT_HALF_LIVES.surface2D,
        volume2p5D: options.halfLives?.volume2p5D ?? DEFAULT_HALF_LIVES.volume2p5D,
    };
    const baseKernel = clampKernelSpec(options.kernel ?? KERNEL_SPEC_DEFAULT);
    const variantKernel = clampKernelSpec(options.enforceVariantKernel ??
        mergeKernelDelta(baseKernel, options.kernelDelta ?? { gain: baseKernel.gain * 1.08 }));
    const baselineRun = simulateScenarioRun(baseKernel, options, options.baselineNoise ?? {});
    const variantRun = simulateScenarioRun(variantKernel, options, options.variantNoise ?? {}, 10000);
    const baselineReport = {
        rim1p5D: buildTierReport(baselineRun.tiers.rim1p5D, baselineRun.dt),
        surface2D: buildTierReport(baselineRun.tiers.surface2D, baselineRun.dt),
        volume2p5D: buildTierReport(baselineRun.tiers.volume2p5D, baselineRun.dt),
    };
    const variantReport = {
        rim1p5D: buildTierReport(variantRun.tiers.rim1p5D, variantRun.dt),
        surface2D: buildTierReport(variantRun.tiers.surface2D, variantRun.dt),
        volume2p5D: buildTierReport(variantRun.tiers.volume2p5D, variantRun.dt),
    };
    const coherenceBaseline = computePairwiseCoherence(baselineRun.tiers);
    const coherenceVariant = computePairwiseCoherence(variantRun.tiers);
    const divergence = computeDivergenceReport(baselineRun.tiers, variantRun.tiers);
    const kernelDelta = computeKernelDelta(baselineRun.tiers, variantRun.tiers);
    const alerts = [];
    for (const [pair, value] of Object.entries(coherenceBaseline)) {
        if (value < coherenceTolerance) {
            alerts.push({ kind: 'coherence', pair, value, threshold: coherenceTolerance });
        }
    }
    for (const [pair, value] of Object.entries(coherenceVariant)) {
        if (value < coherenceTolerance) {
            alerts.push({ kind: 'coherence', pair, value, threshold: coherenceTolerance });
        }
    }
    for (const tier of Object.keys(divergence)) {
        if (divergence[tier].maxAbs > divergenceTolerance) {
            alerts.push({
                kind: 'divergence',
                tier,
                value: divergence[tier].maxAbs,
                threshold: divergenceTolerance,
            });
        }
    }
    return {
        scenario: {
            width: options.width ?? DEFAULT_WIDTH,
            height: options.height ?? DEFAULT_HEIGHT,
            steps: options.steps ?? DEFAULT_STEPS,
            dt: options.dt ?? DEFAULT_DT,
            halfLives,
            coherenceTolerance,
            divergenceTolerance,
        },
        baseline: baselineReport,
        variant: variantReport,
        coherence: {
            baseline: coherenceBaseline,
            variant: coherenceVariant,
        },
        divergence,
        kernelDelta: {
            baseline: cloneKernelSpec(baseKernel),
            variant: cloneKernelSpec(variantKernel),
            perTier: kernelDelta,
        },
        alerts,
    };
};
export const summarizeAlerts = (alerts) => alerts.map((alert) => {
    if (alert.kind === 'coherence') {
        return `coherence drop for ${alert.pair}: ${alert.value.toFixed(3)} < ${alert.threshold.toFixed(3)}`;
    }
    return `divergence envelope breach for ${alert.tier}: ${alert.value.toFixed(3)} > ${alert.threshold.toFixed(3)}`;
});
