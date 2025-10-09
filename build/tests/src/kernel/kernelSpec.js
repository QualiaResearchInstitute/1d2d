const clamp = (value, min, max) => {
    if (Number.isNaN(value))
        return min;
    if (!Number.isFinite(value))
        return value > 0 ? max : min;
    if (value < min)
        return min;
    if (value > max)
        return max;
    return value;
};
const INTERNAL_DEFAULT_KERNEL_SPEC = {
    gain: 1.0,
    k0: 0.08,
    Q: 2.2,
    anisotropy: 0.6,
    chirality: 0.4,
    transparency: 0.2
};
const KERNEL_SPEC_BOUNDS = {
    gain: { min: 0, max: 6 },
    k0: { min: 0.01, max: 0.45 },
    Q: { min: 0.5, max: 10 },
    anisotropy: { min: 0, max: 2 },
    chirality: { min: 0, max: 2.5 },
    transparency: { min: 0, max: 1 }
};
const sanitizeValue = (key, value) => {
    const bounds = KERNEL_SPEC_BOUNDS[key];
    const fallback = INTERNAL_DEFAULT_KERNEL_SPEC[key];
    if (value == null)
        return fallback;
    return clamp(value, bounds.min, bounds.max);
};
export const createKernelSpec = (init) => ({
    gain: sanitizeValue("gain", init?.gain),
    k0: sanitizeValue("k0", init?.k0),
    Q: sanitizeValue("Q", init?.Q),
    anisotropy: sanitizeValue("anisotropy", init?.anisotropy),
    chirality: sanitizeValue("chirality", init?.chirality),
    transparency: sanitizeValue("transparency", init?.transparency)
});
export const clampKernelSpec = (spec) => createKernelSpec(spec);
export const cloneKernelSpec = (spec) => createKernelSpec({ ...spec });
export const kernelSpecToJSON = (spec) => ({
    gain: spec.gain,
    k0: spec.k0,
    Q: spec.Q,
    anisotropy: spec.anisotropy,
    chirality: spec.chirality,
    transparency: spec.transparency
});
export const KERNEL_SPEC_DEFAULT = Object.freeze(kernelSpecToJSON(INTERNAL_DEFAULT_KERNEL_SPEC));
export const DEFAULT_KERNEL_SPEC_JSON = kernelSpecToJSON(INTERNAL_DEFAULT_KERNEL_SPEC);
export const getDefaultKernelSpec = () => cloneKernelSpec(INTERNAL_DEFAULT_KERNEL_SPEC);
export const getKernelSpecBounds = () => ({ ...KERNEL_SPEC_BOUNDS });
