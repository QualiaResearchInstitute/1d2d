import { clampKernelSpec, cloneKernelSpec, getDefaultKernelSpec } from "./kernelSpec.js";
const now = () => {
    if (typeof performance !== "undefined" && typeof performance.now === "function") {
        return performance.now();
    }
    return Date.now();
};
const EPS = 1e-6;
const computeChangedKeys = (prev, next) => {
    const keys = [];
    Object.keys(prev).forEach((key) => {
        const prevValue = prev[key];
        const nextValue = next[key];
        if (typeof prevValue === "number" && typeof nextValue === "number") {
            if (Math.abs(prevValue - nextValue) > EPS) {
                keys.push(key);
            }
            return;
        }
        if (prevValue !== nextValue) {
            keys.push(key);
        }
    });
    return keys;
};
const cloneEvent = (event) => ({
    spec: cloneKernelSpec(event.spec),
    version: event.version,
    timestamp: event.timestamp,
    source: event.source,
    changed: [...event.changed]
});
export class KernelSpecHub {
    constructor(initial, source = "init") {
        Object.defineProperty(this, "snapshot", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "subscribers", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "pendingEvent", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "scheduled", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: false
        });
        Object.defineProperty(this, "lastDispatchLatency", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        });
        const base = clampKernelSpec(initial ?? getDefaultKernelSpec());
        this.snapshot = {
            spec: base,
            version: 0,
            timestamp: now(),
            source,
            changed: ["gain", "k0", "Q", "anisotropy", "chirality", "transparency", "couplingPreset"]
        };
        this.subscribers = new Set();
        this.pendingEvent = null;
    }
    getSnapshot() {
        return cloneEvent(this.snapshot);
    }
    subscribe(listener, options) {
        this.subscribers.add(listener);
        if (options?.immediate ?? true) {
            listener(this.getSnapshot());
        }
        return () => {
            this.subscribers.delete(listener);
        };
    }
    update(update, options) {
        const merged = clampKernelSpec({ ...this.snapshot.spec, ...update });
        const changed = computeChangedKeys(this.snapshot.spec, merged);
        if (!options?.force && changed.length === 0) {
            return null;
        }
        const event = {
            spec: merged,
            version: this.snapshot.version + 1,
            timestamp: now(),
            source: options?.source ?? "unspecified",
            changed
        };
        this.snapshot = event;
        this.enqueueBroadcast(event);
        return cloneEvent(event);
    }
    replace(spec, options) {
        return this.update(spec, options);
    }
    enqueueBroadcast(event) {
        this.pendingEvent = event;
        if (this.scheduled)
            return;
        this.scheduled = true;
        queueMicrotask(() => this.flush());
    }
    flush() {
        this.scheduled = false;
        const event = this.pendingEvent;
        if (!event)
            return;
        this.pendingEvent = null;
        this.lastDispatchLatency = Math.max(0, now() - event.timestamp);
        const payload = cloneEvent(event);
        for (const subscriber of this.subscribers) {
            subscriber(cloneEvent(payload));
        }
    }
    getDiagnostics() {
        return {
            subscriberCount: this.subscribers.size,
            lastVersion: this.snapshot.version,
            lastSource: this.snapshot.source,
            lastDispatchLatency: this.lastDispatchLatency
        };
    }
    getSubscriberCount() {
        return this.subscribers.size;
    }
}
const sharedHub = new KernelSpecHub();
export const getKernelSpecHub = () => sharedHub;
