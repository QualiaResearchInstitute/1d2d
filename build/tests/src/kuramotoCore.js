import { makeResolution, } from './fields/contracts.js';
import { OpticalFieldManager, OpticalFieldFrame, } from './fields/opticalField.js';
import { KERNEL_SPEC_DEFAULT, COUPLING_KERNEL_PRESETS, cloneKernelSpec, } from './kernel/kernelSpec.js';
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const clamp01 = (v) => clamp(v, 0, 1);
const wrapAngle = (a) => {
    let ang = a;
    while (ang > Math.PI)
        ang -= 2 * Math.PI;
    while (ang <= -Math.PI)
        ang += 2 * Math.PI;
    return ang;
};
const mulberry32 = (seed) => {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let r = Math.imul(t ^ (t >>> 15), 1 | t);
        r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
};
const SMALL_WORLD_DEFAULT_DEGREE = 12;
const SMALL_WORLD_MAX_DEGREE = 64;
const SMALL_WORLD_CACHE = new Map();
const clampSmallWorldDegree = (value) => {
    if (!Number.isFinite(value))
        return 0;
    if (value <= 0)
        return 0;
    const degree = Math.floor(value);
    return Math.max(1, Math.min(SMALL_WORLD_MAX_DEGREE, degree));
};
export const createSmallWorldRewiring = (width, height, degree, seed) => {
    const clampedDegree = clampSmallWorldDegree(degree);
    if (clampedDegree === 0) {
        return { degree: 0, targets: new Int32Array(0) };
    }
    const total = width * height;
    const targets = new Int32Array(total * clampedDegree);
    if (total <= 1) {
        targets.fill(0);
        return { degree: clampedDegree, targets };
    }
    const rng = mulberry32(seed);
    for (let idx = 0; idx < total; idx++) {
        const offset = idx * clampedDegree;
        for (let edge = 0; edge < clampedDegree; edge++) {
            let candidate = Math.floor(rng() * total);
            if (candidate === idx) {
                candidate = (candidate + 1) % total;
            }
            targets[offset + edge] = candidate;
        }
    }
    return { degree: clampedDegree, targets };
};
const getSmallWorldRewiring = (width, height, degree, seed) => {
    const clampedDegree = clampSmallWorldDegree(degree);
    if (clampedDegree === 0)
        return null;
    const key = `${width}x${height}:${clampedDegree}:${seed}`;
    const cached = SMALL_WORLD_CACHE.get(key);
    if (cached)
        return cached;
    const rewiring = createSmallWorldRewiring(width, height, clampedDegree, seed);
    SMALL_WORLD_CACHE.set(key, rewiring);
    return rewiring;
};
export const IRRADIANCE_FRAME_SCHEMA_VERSION = 1;
/**
 * Canonical ordering for thin-element operators applied during phase/flux derivation.
 * Consumers should respect this order when constructing custom schedules to ensure that
 * amplitude updates run before phase gradients and that any flux phase masks execute first.
 */
export const THIN_ELEMENT_OPERATOR_ORDER = ['flux', 'amplitude', 'phase'];
const COUPLING_KERNEL_CACHE = new Map();
const clampRadius = (value) => Math.max(0, Math.floor(value));
export const computeCouplingWeight = (distance, params) => {
    if (!Number.isFinite(distance) || distance > params.radius)
        return 0;
    const gaussian = (gain, sigma) => {
        if (gain === 0)
            return 0;
        if (sigma <= 0)
            return 0;
        const scaled = distance / sigma;
        return gain * Math.exp(-0.5 * scaled * scaled);
    };
    const far = gaussian(params.farGain, params.farSigma);
    const near = gaussian(params.nearGain, params.nearSigma);
    return params.baseGain + far - near;
};
export const computeCouplingWeights = (distances, params, out) => {
    const target = out ?? new Float32Array(distances.length);
    for (let i = 0; i < distances.length; i++) {
        target[i] = computeCouplingWeight(distances[i], params);
    }
    return target;
};
const buildCouplingKernelTable = (params) => {
    const radius = clampRadius(params.radius);
    const offsetsX = [];
    const offsetsY = [];
    const weights = [];
    const orientations = [];
    let centerWeight = computeCouplingWeight(0, params);
    let l1 = Math.abs(centerWeight);
    for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
            if (dx === 0 && dy === 0)
                continue;
            const distance = Math.hypot(dx, dy);
            if (distance > radius + 1e-6)
                continue;
            const weight = computeCouplingWeight(distance, params);
            if (Math.abs(weight) < 1e-5)
                continue;
            offsetsX.push(dx);
            offsetsY.push(dy);
            weights.push(weight);
            l1 += Math.abs(weight);
            const denom = dx * dx + dy * dy;
            orientations.push(denom === 0 ? 0 : (dx * dx - dy * dy) / denom);
        }
    }
    if (params.normalization === 'l1' && l1 > 0) {
        const inv = 1 / l1;
        centerWeight *= inv;
        for (let i = 0; i < weights.length; i++) {
            weights[i] *= inv;
        }
    }
    return {
        key: `${params.preset}:${params.radius}:${params.nearSigma}:${params.nearGain}:${params.farSigma}:${params.farGain}:${params.baseGain}:${params.normalization}`,
        params,
        radius,
        selfWeight: centerWeight,
        offsetsX: Int16Array.from(offsetsX),
        offsetsY: Int16Array.from(offsetsY),
        weights: Float32Array.from(weights),
        orientations: Float32Array.from(orientations),
    };
};
const getCouplingKernelTable = (params) => {
    const key = `${params.preset}:${params.radius}:${params.nearSigma}:${params.nearGain}:${params.farSigma}:${params.farGain}:${params.baseGain}:${params.normalization}`;
    const cached = COUPLING_KERNEL_CACHE.get(key);
    if (cached)
        return cached;
    const table = buildCouplingKernelTable(params);
    COUPLING_KERNEL_CACHE.set(key, table);
    return table;
};
const wrapIndex = (x, y, width, height) => {
    const xx = ((x % width) + width) % width;
    const yy = ((y % height) + height) % height;
    return yy * width + xx;
};
const controlValue = (value) => clamp01(value ?? 0);
const cloneOpticalMeta = (meta) => ({
    ...meta,
    phaseOrigin: meta.phaseOrigin ? { ...meta.phaseOrigin } : undefined,
    userTags: meta.userTags ? { ...meta.userTags } : undefined,
});
export const createIrradianceFrameBuffer = (width, height, initialMeta) => {
    const resolution = makeResolution(width, height);
    const texels = resolution.texels;
    const buffer = new Float32Array(texels * 3);
    return {
        resolution,
        exposureSeconds: 0,
        foveaPx: [Math.floor(width / 2), Math.floor(height / 2)],
        buffer,
        L: buffer.subarray(0, texels),
        M: buffer.subarray(texels, texels * 2),
        S: buffer.subarray(texels * 2, texels * 3),
        opticalMeta: cloneOpticalMeta(initialMeta),
        kernel: cloneKernelSpec(KERNEL_SPEC_DEFAULT),
        kernelVersion: 0,
    };
};
export const createTelemetrySnapshot = () => ({
    frameId: -1,
    timestamp: 0,
    dt: 0,
    kernelVersion: 0,
    kernel: cloneKernelSpec(KERNEL_SPEC_DEFAULT),
    orderParameter: {
        magnitude: 0,
        phase: 0,
        real: 0,
        imag: 0,
        sampleCount: 0,
    },
    interference: {
        mean: 0,
        variance: 0,
        max: 0,
    },
});
const computeOperatorGains = (kernel, controls) => {
    const d = controlValue(controls?.dmt);
    const a = controlValue(controls?.arousal);
    const anisBaseline = KERNEL_SPEC_DEFAULT.anisotropy;
    const chiralityBaseline = KERNEL_SPEC_DEFAULT.chirality;
    const transparencyBaseline = KERNEL_SPEC_DEFAULT.transparency;
    const anisBias = clamp(kernel.anisotropy - anisBaseline, -1, 1);
    const fluxBase = 1 + 0.45 * anisBias;
    const flux = Math.max(0, fluxBase + 0.25 * d + 0.2 * a);
    const amplitude = Math.max(0, kernel.gain * (1 + 0.35 * d + 0.25 * a));
    const phaseBase = 1 + 0.4 * (kernel.chirality - chiralityBaseline);
    const phase = Math.max(0, phaseBase + 0.2 * d + 0.15 * a);
    const transparency = Math.max(0, 1 + 0.55 * (kernel.transparency - transparencyBaseline) + 0.25 * d);
    return { flux, amplitude, phase, transparency };
};
export const getThinElementOperatorGains = (kernel, controls) => computeOperatorGains(kernel, controls);
const ensureThetaScratch = (scratch, size) => {
    if (!scratch.theta || scratch.theta.length !== size) {
        scratch.theta = new Float32Array(size);
    }
    return scratch.theta;
};
const computeGradientScale = (gains, kernel) => {
    const k0Baseline = KERNEL_SPEC_DEFAULT.k0;
    const relative = k0Baseline > 0 ? (kernel.k0 - k0Baseline) / k0Baseline : 0;
    const limited = clamp(relative, -0.95, 3);
    return gains.phase * (1 + 0.5 * limited);
};
const computeVorticityScale = (gains, kernel) => {
    const delta = clamp(kernel.chirality - KERNEL_SPEC_DEFAULT.chirality, -2.5, 2.5);
    return gains.phase * (1 + 0.35 * delta);
};
const applyFluxPhaseMask = (field, params, gains) => {
    const { fluxX = 0, fluxY = 0 } = params;
    if (fluxX === 0 && fluxY === 0)
        return;
    const { width, height } = field.resolution;
    const real = field.real;
    const imag = field.imag;
    const normX = width > 1 ? 1 / (width - 1) : 0;
    const normY = height > 1 ? 1 / (height - 1) : 0;
    const fluxScale = gains.flux;
    for (let y = 0; y < height; y++) {
        const ny = height > 1 ? y * normY * 2 - 1 : 0;
        for (let x = 0; x < width; x++) {
            const nx = width > 1 ? x * normX * 2 - 1 : 0;
            const idx = y * width + x;
            const phaseShift = fluxScale * (fluxX * nx + fluxY * ny);
            if (phaseShift === 0)
                continue;
            const cos = Math.cos(phaseShift);
            const sin = Math.sin(phaseShift);
            const zr = real[idx];
            const zi = imag[idx];
            real[idx] = zr * cos - zi * sin;
            imag[idx] = zr * sin + zi * cos;
        }
    }
};
const createFluxOperator = (field, params, kernel, gains) => {
    const { width, height } = field.resolution;
    const real = field.real;
    const imag = field.imag;
    const { fluxX = 0, fluxY = 0 } = params;
    const hasFluxX = fluxX !== 0;
    const hasFluxY = fluxY !== 0;
    const fluxScale = 0.2 * gains.flux;
    const smallWorldEnabled = params.smallWorldEnabled ?? true;
    const rawSmallWorldWeight = clamp(params.smallWorldWeight, 0, 4);
    const baseSmallWorldWeight = smallWorldEnabled ? rawSmallWorldWeight : 0;
    const pSwValue = clamp(params.p_sw, -0.5, 0.5);
    const pSw = baseSmallWorldWeight !== 0 ? pSwValue : 0;
    const smallWorldDegree = params.smallWorldDegree ?? SMALL_WORLD_DEFAULT_DEGREE;
    const smallWorldSeed = params.smallWorldSeed ?? 0;
    const smallWorldRewiring = baseSmallWorldWeight !== 0 && pSw !== 0
        ? getSmallWorldRewiring(width, height, smallWorldDegree, smallWorldSeed)
        : null;
    const smallWorldScale = smallWorldRewiring && smallWorldRewiring.degree > 0
        ? clamp(baseSmallWorldWeight * pSw, -4, 4)
        : 0;
    const smallWorldFactor = smallWorldRewiring && smallWorldRewiring.degree > 0
        ? smallWorldScale / smallWorldRewiring.degree
        : 0;
    const couplingPreset = kernel.couplingPreset;
    const couplingParams = COUPLING_KERNEL_PRESETS[couplingPreset] ?? COUPLING_KERNEL_PRESETS.dmt;
    const table = getCouplingKernelTable(couplingParams);
    const offsetsX = table.offsetsX;
    const offsetsY = table.offsetsY;
    const weights = table.weights;
    const orientations = table.orientations;
    const selfWeight = table.selfWeight;
    const anisBaseline = KERNEL_SPEC_DEFAULT.anisotropy;
    const anisBias = clamp(kernel.anisotropy - anisBaseline, -1, 1);
    const anisScale = 0.6;
    return {
        coupling(x, y, idx) {
            const selfR = real[idx];
            const selfI = imag[idx];
            let sumR = selfWeight * selfR;
            let sumI = selfWeight * selfI;
            for (let i = 0; i < weights.length; i++) {
                const baseWeight = weights[i];
                const orientation = orientations[i];
                const weight = anisBias === 0 ? baseWeight : baseWeight * (1 + anisScale * anisBias * orientation);
                let nx = x + offsetsX[i];
                let ny = y + offsetsY[i];
                let phaseShift = 0;
                let wraps = 0;
                while (nx < 0) {
                    nx += width;
                    wraps -= 1;
                }
                while (nx >= width) {
                    nx -= width;
                    wraps += 1;
                }
                if (wraps !== 0 && hasFluxX) {
                    phaseShift += wraps * fluxX;
                }
                wraps = 0;
                while (ny < 0) {
                    ny += height;
                    wraps -= 1;
                }
                while (ny >= height) {
                    ny -= height;
                    wraps += 1;
                }
                if (wraps !== 0 && hasFluxY) {
                    phaseShift += wraps * fluxY;
                }
                const neighborIdx = ny * width + nx;
                let nr = real[neighborIdx];
                let ni = imag[neighborIdx];
                if (phaseShift !== 0) {
                    const cos = Math.cos(phaseShift);
                    const sin = Math.sin(phaseShift);
                    const rotR = nr * cos - ni * sin;
                    const rotI = nr * sin + ni * cos;
                    nr = rotR;
                    ni = rotI;
                }
                sumR += weight * nr;
                sumI += weight * ni;
            }
            if (smallWorldRewiring && smallWorldFactor !== 0) {
                // This executes only when small-world coupling is active. We accumulate the mean
                // long-range delta to gently bias the oscillator toward (or away from) distant peers.
                const offset = idx * smallWorldRewiring.degree;
                let deltaR = 0;
                let deltaI = 0;
                for (let edge = 0; edge < smallWorldRewiring.degree; edge++) {
                    const targetIdx = smallWorldRewiring.targets[offset + edge];
                    const nr = real[targetIdx];
                    const ni = imag[targetIdx];
                    deltaR += nr - selfR;
                    deltaI += ni - selfI;
                }
                sumR += smallWorldFactor * deltaR;
                sumI += smallWorldFactor * deltaI;
            }
            return {
                Hr: fluxScale * sumR,
                Hi: fluxScale * sumI,
            };
        },
    };
};
const applyAmplitudeOperator = (field, phase, gains) => {
    const { amp, coh } = phase;
    const { real, imag } = field;
    const transparencyScale = gains.transparency;
    for (let i = 0; i < real.length; i++) {
        const magnitude = Math.hypot(real[i], imag[i]) * gains.amplitude;
        amp[i] = magnitude;
        coh[i] = clamp01(magnitude * transparencyScale);
    }
};
const applyPhaseOperator = (field, phase, gains, kernel, scratch) => {
    const { gradX, gradY, vort } = phase;
    const { width, height } = field.resolution;
    const real = field.real;
    const imag = field.imag;
    const total = real.length;
    const theta = ensureThetaScratch(scratch, total);
    for (let i = 0; i < total; i++) {
        theta[i] = Math.atan2(imag[i], real[i]);
    }
    const gradScale = computeGradientScale(gains, kernel);
    const vortScale = computeVorticityScale(gains, kernel);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            const left = wrapIndex(x - 1, y, width, height);
            const right = wrapIndex(x + 1, y, width, height);
            const up = wrapIndex(x, y - 1, width, height);
            const down = wrapIndex(x, y + 1, width, height);
            gradX[idx] = 0.5 * gradScale * wrapAngle(theta[right] - theta[left]);
            gradY[idx] = 0.5 * gradScale * wrapAngle(theta[down] - theta[up]);
        }
    }
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i00 = y * width + x;
            const i10 = wrapIndex(x + 1, y, width, height);
            const i11 = wrapIndex(x + 1, y + 1, width, height);
            const i01 = wrapIndex(x, y + 1, width, height);
            const a = wrapAngle(theta[i10] - theta[i00]);
            const b = wrapAngle(theta[i11] - theta[i10]);
            const c = wrapAngle(theta[i01] - theta[i11]);
            const d = wrapAngle(theta[i00] - theta[i01]);
            vort[i00] = (vortScale * (a + b + c + d)) / (2 * Math.PI);
        }
    }
};
const applyOperatorKind = (kind, ctx) => {
    switch (kind) {
        case 'flux':
            if (ctx.params)
                applyFluxPhaseMask(ctx.field, ctx.params, ctx.gains);
            break;
        case 'amplitude':
            if (ctx.derived)
                applyAmplitudeOperator(ctx.field, ctx.derived, ctx.gains);
            break;
        case 'phase':
            if (ctx.derived)
                applyPhaseOperator(ctx.field, ctx.derived, ctx.gains, ctx.kernel, ctx.scratch);
            break;
        default:
            // no-op
            break;
    }
};
const executeBeamSplitStep = (step, ctx) => {
    const { field } = ctx;
    const { branches, recombine = 'sum' } = step;
    if (!branches.length)
        return;
    const resolution = field.resolution;
    const texels = resolution.texels;
    const accumReal = new Float32Array(texels);
    const accumImag = new Float32Array(texels);
    let weightSum = 0;
    let weightSqSum = 0;
    for (const branch of branches) {
        const weight = branch.weight ?? 1;
        weightSum += weight;
        weightSqSum += weight * weight;
        const branchFrame = new OpticalFieldFrame(resolution);
        branchFrame.real.set(field.real);
        branchFrame.imag.set(field.imag);
        const branchCtx = {
            field: branchFrame,
            derived: undefined,
            params: ctx.params,
            kernel: ctx.kernel,
            gains: ctx.gains,
            scratch: {},
        };
        executeThinElementSchedule(branch.steps, branchCtx);
        const branchReal = branchFrame.real;
        const branchImag = branchFrame.imag;
        for (let i = 0; i < texels; i++) {
            accumReal[i] += branchReal[i] * weight;
            accumImag[i] += branchImag[i] * weight;
        }
    }
    let norm = 1;
    switch (recombine) {
        case 'average':
            if (weightSum !== 0)
                norm = 1 / weightSum;
            break;
        case 'energy':
            if (weightSqSum !== 0)
                norm = 1 / Math.sqrt(weightSqSum);
            break;
        case 'sum':
        default:
            norm = 1;
            break;
    }
    const real = field.real;
    const imag = field.imag;
    for (let i = 0; i < texels; i++) {
        real[i] = accumReal[i] * norm;
        imag[i] = accumImag[i] * norm;
    }
};
const executeThinElementSchedule = (schedule, ctx) => {
    for (const step of schedule) {
        if (step.kind === 'operator') {
            applyOperatorKind(step.operator, ctx);
        }
        else if (step.kind === 'beamSplit') {
            executeBeamSplitStep(step, ctx);
        }
    }
};
const DEFAULT_PHASE_SCHEDULE = [
    { kind: 'operator', operator: 'amplitude' },
    { kind: 'operator', operator: 'phase' },
];
export const createKuramotoState = (width, height, manager) => {
    const resolution = makeResolution(width, height);
    const fieldManager = manager ??
        new OpticalFieldManager({
            solver: 'kuramoto',
            resolution,
            initialFrameId: 0,
        });
    const frame = fieldManager.acquireFrame();
    const telemetry = createTelemetrySnapshot();
    const initialMeta = frame.getMeta();
    const irradiance = createIrradianceFrameBuffer(width, height, initialMeta);
    return {
        width,
        height,
        manager: fieldManager,
        field: frame,
        Zr: frame.real,
        Zi: frame.imag,
        telemetry,
        irradiance,
    };
};
export const derivedFieldCount = 5;
export const derivedBufferSize = (width, height) => width * height * derivedFieldCount * Float32Array.BYTES_PER_ELEMENT;
export const createDerivedViews = (buffer, width, height) => {
    const total = width * height;
    const view = new Float32Array(buffer);
    const gradX = view.subarray(0, total);
    const gradY = view.subarray(total, total * 2);
    const vort = view.subarray(total * 2, total * 3);
    const coh = view.subarray(total * 3, total * 4);
    const amp = view.subarray(total * 4, total * 5);
    return {
        kind: 'phase',
        resolution: makeResolution(width, height),
        gradX,
        gradY,
        vort,
        coh,
        amp,
    };
};
export const initKuramotoState = (state, q, phase) => {
    const { width, height, Zr, Zi } = state;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const theta = (2 * Math.PI * q * x) / width;
            const idx = y * width + x;
            Zr[idx] = Math.cos(theta);
            Zi[idx] = Math.sin(theta);
        }
    }
    const meta = state.field.getMeta();
    const telemetry = state.telemetry;
    telemetry.frameId = -1;
    telemetry.timestamp = 0;
    telemetry.dt = 0;
    telemetry.kernelVersion = 0;
    telemetry.kernel = cloneKernelSpec(KERNEL_SPEC_DEFAULT);
    telemetry.orderParameter.magnitude = 0;
    telemetry.orderParameter.phase = 0;
    telemetry.orderParameter.real = 0;
    telemetry.orderParameter.imag = 0;
    telemetry.orderParameter.sampleCount = 0;
    telemetry.interference.mean = 0;
    telemetry.interference.variance = 0;
    telemetry.interference.max = 0;
    const irradiance = state.irradiance;
    irradiance.exposureSeconds = 0;
    irradiance.foveaPx = [Math.floor(width / 2), Math.floor(height / 2)];
    irradiance.L.fill(0);
    irradiance.M.fill(0);
    irradiance.S.fill(0);
    irradiance.kernelVersion = 0;
    irradiance.kernel = cloneKernelSpec(KERNEL_SPEC_DEFAULT);
    irradiance.opticalMeta = cloneOpticalMeta(meta);
    if (phase) {
        phase.gradX.fill(0);
        phase.gradY.fill(0);
        phase.vort.fill(0);
        phase.coh.fill(0.5);
        phase.amp.fill(0);
    }
};
export const stepKuramotoState = (state, params, dt, randn, timestamp, options) => {
    const { width, height, Zr, Zi } = state;
    const { alphaKur, gammaKur, omega0, K0, epsKur } = params;
    const kernel = options?.kernel ?? KERNEL_SPEC_DEFAULT;
    const gains = computeOperatorGains(kernel, options?.controls);
    const telemetryRequest = options?.telemetry;
    const captureIrradiance = telemetryRequest?.captureIrradiance ?? true;
    if (options?.schedule) {
        executeThinElementSchedule(options.schedule, {
            field: state.field,
            derived: undefined,
            params,
            kernel,
            gains,
            scratch: {},
        });
    }
    const fluxOperator = createFluxOperator(state.field, params, kernel, gains);
    const ca = Math.cos(alphaKur);
    const sa = Math.sin(alphaKur);
    const couplingGain = 0.5 * K0 * gains.phase;
    const noiseScale = Math.sqrt(Math.max(dt * epsKur, 0));
    const irradiance = state.irradiance;
    const texels = width * height;
    const irrL = irradiance.L;
    const irrM = irradiance.M;
    const irrS = irradiance.S;
    if (telemetryRequest?.foveaPx) {
        irradiance.foveaPx = [telemetryRequest.foveaPx[0], telemetryRequest.foveaPx[1]];
    }
    let orderSumR = 0;
    let orderSumI = 0;
    let orderSamples = 0;
    let energySum = 0;
    let energySumSq = 0;
    let energyMax = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            const { Hr, Hi } = fluxOperator.coupling(x, y, idx);
            const Zre = Zr[idx];
            const Zim = Zi[idx];
            const Z2r = Zre * Zre - Zim * Zim;
            const Z2i = 2 * Zre * Zim;
            const H1r = ca * Hr + sa * Hi;
            const H1i = -sa * Hr + ca * Hi;
            const HrConj = Hr;
            const HiConj = -Hi;
            const Tr = Z2r * HrConj - Z2i * HiConj;
            const Ti = Z2r * HiConj + Z2i * HrConj;
            const H2r = ca * Tr - sa * Ti;
            const H2i = sa * Tr + ca * Ti;
            const dZr = -gammaKur * Zre - omega0 * Zim + couplingGain * (H1r - H2r);
            const dZi = -gammaKur * Zim + omega0 * Zre + couplingGain * (H1i - H2i);
            const nextR = Zre + dt * dZr + noiseScale * randn();
            const nextI = Zim + dt * dZi + noiseScale * randn();
            Zr[idx] = nextR;
            Zi[idx] = nextI;
            const ampSq = nextR * nextR + nextI * nextI;
            energySum += ampSq;
            energySumSq += ampSq * ampSq;
            if (ampSq > energyMax)
                energyMax = ampSq;
            const amp = Math.sqrt(ampSq);
            if (amp > 1e-12) {
                const invAmp = 1 / amp;
                orderSumR += nextR * invAmp;
                orderSumI += nextI * invAmp;
                orderSamples += 1;
            }
            if (captureIrradiance) {
                irrL[idx] = ampSq;
                irrM[idx] = ampSq;
                irrS[idx] = ampSq;
            }
        }
    }
    const meta = state.manager.stampFrame(state.field, { dt, timestamp });
    const telemetry = state.telemetry;
    const invSamples = orderSamples > 0 ? 1 / orderSamples : 0;
    const avgReal = orderSumR * invSamples;
    const avgImag = orderSumI * invSamples;
    telemetry.frameId = meta.frameId;
    telemetry.timestamp = meta.timestamp;
    telemetry.dt = dt;
    const kernelVersion = telemetryRequest?.kernelVersion ?? telemetry.kernelVersion;
    telemetry.kernelVersion = kernelVersion;
    telemetry.kernel = cloneKernelSpec(kernel);
    telemetry.orderParameter.real = avgReal;
    telemetry.orderParameter.imag = avgImag;
    telemetry.orderParameter.magnitude = Math.hypot(avgReal, avgImag);
    telemetry.orderParameter.phase = Math.atan2(avgImag, avgReal);
    telemetry.orderParameter.sampleCount = orderSamples;
    const denom = texels > 0 ? texels : 1;
    const meanEnergy = energySum / denom;
    const varianceEnergy = Math.max(0, energySumSq / denom - meanEnergy * meanEnergy);
    telemetry.interference.mean = meanEnergy;
    telemetry.interference.variance = varianceEnergy;
    telemetry.interference.max = energyMax;
    irradiance.exposureSeconds = captureIrradiance ? dt : 0;
    irradiance.kernelVersion = kernelVersion;
    irradiance.kernel = cloneKernelSpec(kernel);
    irradiance.opticalMeta = cloneOpticalMeta(meta);
    return {
        telemetry,
        irradiance,
    };
};
export const createKuramotoInstrumentationSnapshot = (state) => ({
    telemetry: {
        frameId: state.telemetry.frameId,
        timestamp: state.telemetry.timestamp,
        dt: state.telemetry.dt,
        kernelVersion: state.telemetry.kernelVersion,
        kernel: cloneKernelSpec(state.telemetry.kernel),
        orderParameter: {
            magnitude: state.telemetry.orderParameter.magnitude,
            phase: state.telemetry.orderParameter.phase,
            real: state.telemetry.orderParameter.real,
            imag: state.telemetry.orderParameter.imag,
            sampleCount: state.telemetry.orderParameter.sampleCount,
        },
        interference: {
            mean: state.telemetry.interference.mean,
            variance: state.telemetry.interference.variance,
            max: state.telemetry.interference.max,
        },
    },
    irradiance: {
        exposureSeconds: state.irradiance.exposureSeconds,
        foveaPx: [state.irradiance.foveaPx[0], state.irradiance.foveaPx[1]],
        kernelVersion: state.irradiance.kernelVersion,
        kernel: cloneKernelSpec(state.irradiance.kernel),
        opticalMeta: cloneOpticalMeta(state.irradiance.opticalMeta),
    },
});
export const deriveKuramotoFields = (state, phase, options) => {
    const kernel = options?.kernel ?? KERNEL_SPEC_DEFAULT;
    const gains = computeOperatorGains(kernel, options?.controls);
    const schedule = options?.schedule ?? DEFAULT_PHASE_SCHEDULE;
    const context = {
        field: state.field,
        derived: phase,
        params: options?.params,
        kernel,
        gains,
        scratch: {},
    };
    executeThinElementSchedule(schedule, context);
};
export const createNormalGenerator = (seed) => {
    const rng = seed == null ? Math.random : mulberry32(seed);
    let spare = null;
    return () => {
        if (spare != null) {
            const value = spare;
            spare = null;
            return value;
        }
        let u = 0;
        let v = 0;
        while (u === 0)
            u = rng();
        while (v === 0)
            v = rng();
        const mag = Math.sqrt(-2.0 * Math.log(u));
        const z0 = mag * Math.cos(2 * Math.PI * v);
        const z1 = mag * Math.sin(2 * Math.PI * v);
        spare = z1;
        return z0;
    };
};
export const snapshotVolumeField = (state) => {
    const { width, height, Zr, Zi } = state;
    const total = width * height;
    const phase = new Float32Array(total);
    const depth = new Float32Array(total);
    const intensity = new Float32Array(total);
    for (let y = 0; y < height; y++) {
        const ny = height > 1 ? (y / (height - 1)) * 2 - 1 : 0;
        for (let x = 0; x < width; x++) {
            const nx = width > 1 ? (x / (width - 1)) * 2 - 1 : 0;
            const idx = y * width + x;
            const zr = Zr[idx];
            const zi = Zi[idx];
            const amp = Math.hypot(zr, zi);
            const phi = Math.atan2(zi, zr);
            phase[idx] = phi;
            intensity[idx] = amp;
            const radial = Math.hypot(nx, ny);
            // Depth proxy: standing-wave shell gated by amplitude and radius
            depth[idx] = clamp(0.5 + 0.5 * Math.sin(phi + radial * 3.0) * Math.min(1, amp), 0, 1);
        }
    }
    return {
        kind: 'volume',
        resolution: makeResolution(width, height),
        phase,
        depth,
        intensity,
    };
};
