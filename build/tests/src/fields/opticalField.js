import { makeResolution } from "./contracts.js";
export const OPTICAL_FIELD_SCHEMA_VERSION = 2;
const TAU = Math.PI * 2;
const wrapAngle = (theta) => {
    let t = theta;
    while (t > Math.PI)
        t -= TAU;
    while (t <= -Math.PI)
        t += TAU;
    return t;
};
let nextBufferId = 0;
export class OpticalFieldFrame {
    constructor(resolution, buffer) {
        Object.defineProperty(this, "id", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "resolution", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "buffer", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "view", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "real", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "imag", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "metadata", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "inUse", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: false
        });
        this.id = nextBufferId++;
        this.resolution = makeResolution(resolution.width, resolution.height);
        const texels = resolution.texels;
        const bytes = texels * 2 * Float32Array.BYTES_PER_ELEMENT;
        this.buffer = buffer ?? new ArrayBuffer(bytes);
        this.view = new Float32Array(this.buffer);
        this.real = this.view.subarray(0, texels);
        this.imag = this.view.subarray(texels, texels * 2);
        this.metadata = {
            schemaVersion: OPTICAL_FIELD_SCHEMA_VERSION,
            solver: "dispatcher",
            solverInstanceId: "unassigned",
            frameId: -1,
            timestamp: 0,
            dt: 0,
            wavelengthNm: 550,
            pixelPitchMeters: 1e-6,
            space: "screen",
            phaseReference: "wrapped"
        };
    }
    markAcquired(meta) {
        this.metadata = meta;
        this.inUse = true;
    }
    markReleased() {
        this.inUse = false;
    }
    getMeta() {
        return this.metadata;
    }
    updateMeta(update) {
        this.metadata = {
            ...this.metadata,
            ...update,
            phaseOrigin: update.phaseOrigin ?? this.metadata.phaseOrigin,
            userTags: update.userTags ?? this.metadata.userTags
        };
    }
    isInUse() {
        return this.inUse;
    }
    getPhase(index) {
        return Math.atan2(this.imag[index], this.real[index]);
    }
    applyPhaseRotation(delta) {
        const cos = Math.cos(delta);
        const sin = Math.sin(delta);
        for (let i = 0; i < this.real.length; i++) {
            const r = this.real[i];
            const im = this.imag[i];
            this.real[i] = r * cos - im * sin;
            this.imag[i] = r * sin + im * cos;
        }
    }
}
const defaultTimestamp = () => {
    if (typeof performance !== "undefined" && typeof performance.now === "function") {
        return performance.now();
    }
    return Date.now();
};
export class OpticalFieldManager {
    constructor(options) {
        Object.defineProperty(this, "solver", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "solverInstanceId", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "resolution", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "capacity", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "pool", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: []
        });
        Object.defineProperty(this, "live", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: new Set()
        });
        Object.defineProperty(this, "nextFrameId", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "hooks", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: new Set()
        });
        Object.defineProperty(this, "defaults", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        this.solver = options.solver;
        this.solverInstanceId = options.solverInstanceId ?? `${options.solver}-default`;
        this.resolution = makeResolution(options.resolution.width, options.resolution.height);
        this.capacity = Math.max(1, options.capacity ?? 4);
        this.nextFrameId = options.initialFrameId ?? 0;
        this.defaults = {
            dt: options.defaultDt ?? 0,
            wavelengthNm: options.defaultWavelengthNm ?? 550,
            pixelPitchMeters: options.defaultPixelPitchMeters ?? 1e-6,
            space: options.defaultSpace ?? "screen"
        };
    }
    acquireFrame(options) {
        let frame = this.pool.pop();
        if (!frame) {
            frame = new OpticalFieldFrame(this.resolution);
        }
        const meta = this.makeMetadata(options);
        frame.markAcquired(meta);
        this.live.add(frame);
        return frame;
    }
    releaseFrame(frame) {
        if (!this.live.has(frame)) {
            throw new Error("[opticalField] attempt to release frame not owned by manager");
        }
        this.live.delete(frame);
        frame.markReleased();
        if (this.pool.length < this.capacity) {
            this.pool.push(frame);
        }
    }
    alignPhase(frame, request) {
        if (!this.live.has(frame)) {
            throw new Error("[opticalField] cannot align phase for unmanaged frame");
        }
        const { anchorIndex, referencePhase } = request;
        if (anchorIndex < 0 || anchorIndex >= frame.real.length) {
            throw new Error(`[opticalField] anchor index ${anchorIndex} out of bounds for ${frame.real.length} samples`);
        }
        const tolerance = request.tolerance ?? 1e-6;
        const current = frame.getPhase(anchorIndex);
        if (!Number.isFinite(current)) {
            return 0;
        }
        const delta = wrapAngle(referencePhase - current);
        if (Math.abs(delta) > tolerance) {
            frame.applyPhaseRotation(delta);
        }
        frame.updateMeta({
            phaseReference: "aligned",
            phaseOrigin: {
                anchorIndex,
                referencePhase,
                appliedDelta: delta
            }
        });
        for (const hook of this.hooks) {
            hook({
                field: frame,
                request,
                phaseDelta: delta
            });
        }
        return delta;
    }
    stampFrame(frame, options) {
        if (!this.live.has(frame)) {
            throw new Error("[opticalField] cannot stamp metadata for unmanaged frame");
        }
        const prev = frame.getMeta();
        const frameId = this.allocateFrameId(options?.frameId);
        const meta = {
            ...this.makeMetadata(options, prev),
            frameId,
            timestamp: options?.timestamp ?? defaultTimestamp(),
            dt: options?.dt ?? prev.dt ?? this.defaults.dt,
            wavelengthNm: options?.wavelengthNm ?? prev.wavelengthNm ?? this.defaults.wavelengthNm,
            pixelPitchMeters: options?.pixelPitchMeters ?? prev.pixelPitchMeters ?? this.defaults.pixelPitchMeters,
            space: options?.space ?? prev.space ?? this.defaults.space,
            phaseReference: options?.phaseReference ?? "wrapped",
            phaseOrigin: undefined,
            notes: options?.notes ?? prev.notes,
            userTags: options?.userTags
                ? { ...options.userTags }
                : prev.userTags
                    ? { ...prev.userTags }
                    : undefined
        };
        frame.markAcquired(meta);
        return meta;
    }
    registerPhaseHook(hook) {
        this.hooks.add(hook);
        return () => {
            this.hooks.delete(hook);
        };
    }
    getResolution() {
        return this.resolution;
    }
    getLatestFrameId() {
        return this.nextFrameId - 1;
    }
    allocateFrameId(requested) {
        if (requested != null) {
            if (requested >= this.nextFrameId) {
                this.nextFrameId = requested + 1;
            }
            return requested;
        }
        const frameId = this.nextFrameId;
        this.nextFrameId += 1;
        return frameId;
    }
    makeMetadata(options, prev) {
        const userTags = options?.userTags ?? prev?.userTags;
        return {
            schemaVersion: OPTICAL_FIELD_SCHEMA_VERSION,
            solver: this.solver,
            solverInstanceId: this.solverInstanceId,
            frameId: options?.frameId ?? prev?.frameId ?? -1,
            timestamp: options?.timestamp ?? prev?.timestamp ?? 0,
            dt: options?.dt ?? prev?.dt ?? this.defaults.dt,
            wavelengthNm: options?.wavelengthNm ?? prev?.wavelengthNm ?? this.defaults.wavelengthNm,
            pixelPitchMeters: options?.pixelPitchMeters ?? prev?.pixelPitchMeters ?? this.defaults.pixelPitchMeters,
            space: options?.space ?? prev?.space ?? this.defaults.space,
            phaseReference: options?.phaseReference ?? prev?.phaseReference ?? "wrapped",
            phaseOrigin: prev?.phaseOrigin,
            notes: options?.notes ?? prev?.notes,
            userTags: userTags ? { ...userTags } : undefined
        };
    }
}
