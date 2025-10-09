import {
  clampKernelSpec,
  cloneKernelSpec,
  getDefaultKernelSpec,
  type KernelSpec,
  type KernelSpecInit
} from "./kernelSpec.js";

const now = () => {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
};

type KernelSpecChange = readonly (keyof KernelSpec)[];

export type KernelSpecEvent = {
  spec: KernelSpec;
  version: number;
  timestamp: number;
  source: string;
  changed: KernelSpecChange;
};

export type KernelSpecSubscriber = (event: KernelSpecEvent) => void;

const EPS = 1e-6;

const computeChangedKeys = (prev: KernelSpec, next: KernelSpec): (keyof KernelSpec)[] => {
  const keys: (keyof KernelSpec)[] = [];
  (Object.keys(prev) as (keyof KernelSpec)[]).forEach((key) => {
    if (Math.abs(prev[key] - next[key]) > EPS) {
      keys.push(key);
    }
  });
  return keys;
};

const cloneEvent = (event: KernelSpecEvent): KernelSpecEvent => ({
  spec: cloneKernelSpec(event.spec),
  version: event.version,
  timestamp: event.timestamp,
  source: event.source,
  changed: [...event.changed]
});

export class KernelSpecHub {
  private snapshot: KernelSpecEvent;
  private subscribers: Set<KernelSpecSubscriber>;
  private pendingEvent: KernelSpecEvent | null;
  private scheduled = false;
  private lastDispatchLatency = 0;

  constructor(initial?: KernelSpecInit, source = "init") {
    const base = clampKernelSpec(initial ?? getDefaultKernelSpec());
    this.snapshot = {
      spec: base,
      version: 0,
      timestamp: now(),
      source,
      changed: ["gain", "k0", "Q", "anisotropy", "chirality", "transparency"]
    };
    this.subscribers = new Set();
    this.pendingEvent = null;
  }

  getSnapshot(): KernelSpecEvent {
    return cloneEvent(this.snapshot);
  }

  subscribe(listener: KernelSpecSubscriber, options?: { immediate?: boolean }): () => void {
    this.subscribers.add(listener);
    if (options?.immediate ?? true) {
      listener(this.getSnapshot());
    }
    return () => {
      this.subscribers.delete(listener);
    };
  }

  update(update: KernelSpecInit | KernelSpec, options?: { source?: string; force?: boolean }) {
    const merged = clampKernelSpec({ ...this.snapshot.spec, ...update });
    const changed = computeChangedKeys(this.snapshot.spec, merged);
    if (!options?.force && changed.length === 0) {
      return null;
    }
    const event: KernelSpecEvent = {
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

  replace(spec: KernelSpec, options?: { source?: string; force?: boolean }) {
    return this.update(spec, options);
  }

  private enqueueBroadcast(event: KernelSpecEvent) {
    this.pendingEvent = event;
    if (this.scheduled) return;
    this.scheduled = true;
    queueMicrotask(() => this.flush());
  }

  private flush() {
    this.scheduled = false;
    const event = this.pendingEvent;
    if (!event) return;
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
