import { makeResolution, type FieldResolution } from './contracts.js';

export const OPTICAL_FIELD_SCHEMA_VERSION = 3 as const;

export type OpticalSpace = 'screen' | 'pupil' | 'volumeSlice';

export type OpticalSolverId = 'kuramoto' | 'angularSpectrum' | 'volumeStub' | 'dispatcher' | string;

export type PhaseReferenceKind = 'wrapped' | 'aligned';

export type OpticalFieldMetadata = {
  schemaVersion: number;
  solver: OpticalSolverId;
  solverInstanceId: string;
  frameId: number;
  componentCount: number;
  timestamp: number;
  dt: number;
  wavelengthNm: number;
  pixelPitchMeters: number;
  space: OpticalSpace;
  phaseReference: PhaseReferenceKind;
  phaseOrigin?: {
    anchorIndex: number;
    referencePhase: number;
    appliedDelta: number;
  };
  notes?: string;
  userTags?: Record<string, unknown>;
};

export type OpticalFieldComponentView = {
  readonly index: number;
  readonly real: Float32Array;
  readonly imag: Float32Array;
};

export type OpticalFieldAcquireOptions = {
  dt?: number;
  timestamp?: number;
  wavelengthNm?: number;
  pixelPitchMeters?: number;
  space?: OpticalSpace;
  phaseReference?: PhaseReferenceKind;
  frameId?: number;
  notes?: string;
  userTags?: Record<string, unknown>;
  componentCount?: number;
};

export type PhaseAlignmentRequest = {
  anchorIndex: number;
  referencePhase: number;
  reason?: string;
  tolerance?: number;
};

export type PhaseAlignmentHook = (payload: {
  field: OpticalFieldFrame;
  request: PhaseAlignmentRequest;
  phaseDelta: number;
}) => void;

const TAU = Math.PI * 2;

const wrapAngle = (theta: number) => {
  let t = theta;
  while (t > Math.PI) t -= TAU;
  while (t <= -Math.PI) t += TAU;
  return t;
};

let nextBufferId = 0;

const ensureFiniteComponentIndex = (index: number, componentCount: number) => {
  if (!Number.isInteger(index) || index < 0 || index >= componentCount) {
    throw new Error(
      `[opticalField] component index ${index} out of range 0..${componentCount - 1}`,
    );
  }
  return index;
};

export class OpticalFieldFrame {
  readonly id: number;
  readonly resolution: FieldResolution;
  readonly buffer: ArrayBuffer;
  readonly view: Float32Array;
  readonly real: Float32Array;
  readonly imag: Float32Array;
  readonly componentCount: number;
  readonly components: readonly OpticalFieldComponentView[];
  private metadata: OpticalFieldMetadata;
  private inUse = false;

  constructor(
    resolution: FieldResolution,
    options?: { buffer?: ArrayBuffer; componentCount?: number },
  ) {
    this.id = nextBufferId++;
    this.resolution = makeResolution(resolution.width, resolution.height);
    const texels = resolution.texels;
    this.componentCount = Math.max(1, options?.componentCount ?? 1);
    const expectedLength = texels * this.componentCount * 2;
    const bytes = expectedLength * Float32Array.BYTES_PER_ELEMENT;
    this.buffer = options?.buffer ?? new ArrayBuffer(bytes);
    this.view = new Float32Array(this.buffer);
    if (this.view.length !== expectedLength) {
      throw new Error(
        `[opticalField] buffer length ${this.view.length} does not match expected ${expectedLength} for ${this.componentCount} components`,
      );
    }
    const componentViews: OpticalFieldComponentView[] = [];
    for (let componentIndex = 0; componentIndex < this.componentCount; componentIndex++) {
      const base = componentIndex * texels * 2;
      const real = this.view.subarray(base, base + texels);
      const imag = this.view.subarray(base + texels, base + texels * 2);
      componentViews.push({ index: componentIndex, real, imag });
    }
    this.components = componentViews;
    this.real = componentViews[0]!.real;
    this.imag = componentViews[0]!.imag;
    this.metadata = {
      schemaVersion: OPTICAL_FIELD_SCHEMA_VERSION,
      solver: 'dispatcher',
      solverInstanceId: 'unassigned',
      frameId: -1,
      componentCount: this.componentCount,
      timestamp: 0,
      dt: 0,
      wavelengthNm: 550,
      pixelPitchMeters: 1e-6,
      space: 'screen',
      phaseReference: 'wrapped',
    };
  }

  markAcquired(meta: OpticalFieldMetadata) {
    this.metadata = meta;
    this.inUse = true;
  }

  markReleased() {
    this.inUse = false;
  }

  getMeta(): OpticalFieldMetadata {
    return this.metadata;
  }

  updateMeta(update: Partial<OpticalFieldMetadata>) {
    this.metadata = {
      ...this.metadata,
      componentCount: update.componentCount ?? this.metadata.componentCount,
      ...update,
      phaseOrigin: update.phaseOrigin ?? this.metadata.phaseOrigin,
      userTags: update.userTags ?? this.metadata.userTags,
    };
  }

  isInUse() {
    return this.inUse;
  }

  getPhase(index: number, component = 0): number {
    const target =
      component === 0
        ? this
        : this.getComponentView(ensureFiniteComponentIndex(component, this.componentCount));
    return Math.atan2(target.imag[index], target.real[index]);
  }

  applyPhaseRotation(delta: number, component?: number) {
    if (delta === 0) return;
    if (component != null) {
      const target = this.getComponentView(
        ensureFiniteComponentIndex(component, this.componentCount),
      );
      rotateComponent(target.real, target.imag, delta);
      return;
    }
    for (const { real, imag } of this.components) {
      rotateComponent(real, imag, delta);
    }
  }

  getComponentView(index: number): OpticalFieldComponentView {
    const safeIndex = ensureFiniteComponentIndex(index, this.componentCount);
    return this.components[safeIndex]!;
  }
}

const rotateComponent = (real: Float32Array, imag: Float32Array, delta: number) => {
  const cos = Math.cos(delta);
  const sin = Math.sin(delta);
  for (let i = 0; i < real.length; i++) {
    const r = real[i];
    const im = imag[i];
    real[i] = r * cos - im * sin;
    imag[i] = r * sin + im * cos;
  }
};

const defaultTimestamp = () => {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
};

export type OpticalFieldManagerOptions = {
  solver: OpticalSolverId;
  solverInstanceId?: string;
  resolution: FieldResolution | { width: number; height: number };
  capacity?: number;
  initialFrameId?: number;
  defaultDt?: number;
  defaultWavelengthNm?: number;
  defaultPixelPitchMeters?: number;
  defaultSpace?: OpticalSpace;
  componentCount?: number;
};

export class OpticalFieldManager {
  private readonly solver: OpticalSolverId;
  private readonly solverInstanceId: string;
  private readonly resolution: FieldResolution;
  private readonly capacity: number;
  private readonly pool: OpticalFieldFrame[] = [];
  private readonly live = new Set<OpticalFieldFrame>();
  private nextFrameId: number;
  private readonly hooks = new Set<PhaseAlignmentHook>();
  private readonly componentCount: number;
  private readonly defaults: {
    dt: number;
    wavelengthNm: number;
    pixelPitchMeters: number;
    space: OpticalSpace;
  };

  constructor(options: OpticalFieldManagerOptions) {
    this.solver = options.solver;
    this.solverInstanceId = options.solverInstanceId ?? `${options.solver}-default`;
    this.resolution = makeResolution(options.resolution.width, options.resolution.height);
    this.capacity = Math.max(1, options.capacity ?? 4);
    this.nextFrameId = options.initialFrameId ?? 0;
    this.componentCount = Math.max(1, options.componentCount ?? 1);
    this.defaults = {
      dt: options.defaultDt ?? 0,
      wavelengthNm: options.defaultWavelengthNm ?? 550,
      pixelPitchMeters: options.defaultPixelPitchMeters ?? 1e-6,
      space: options.defaultSpace ?? 'screen',
    };
  }

  acquireFrame(options?: OpticalFieldAcquireOptions): OpticalFieldFrame {
    const requestedComponents = options?.componentCount ?? this.componentCount;
    if (requestedComponents !== this.componentCount) {
      throw new Error(
        `[opticalField] manager configured for ${this.componentCount} components; requested ${requestedComponents}`,
      );
    }
    let frame: OpticalFieldFrame | undefined = this.pool.pop();
    if (!frame) {
      frame = new OpticalFieldFrame(this.resolution, { componentCount: this.componentCount });
    } else if (frame.componentCount !== this.componentCount) {
      throw new Error(
        `[opticalField] pooled frame component count ${frame.componentCount} does not match manager component count ${this.componentCount}`,
      );
    }
    const meta = this.makeMetadata(options);
    frame.markAcquired(meta);
    this.live.add(frame);
    return frame;
  }

  releaseFrame(frame: OpticalFieldFrame) {
    if (!this.live.has(frame)) {
      throw new Error('[opticalField] attempt to release frame not owned by manager');
    }
    this.live.delete(frame);
    frame.markReleased();
    if (this.pool.length < this.capacity) {
      this.pool.push(frame);
    }
  }

  alignPhase(frame: OpticalFieldFrame, request: PhaseAlignmentRequest): number {
    if (!this.live.has(frame)) {
      throw new Error('[opticalField] cannot align phase for unmanaged frame');
    }
    const { anchorIndex, referencePhase } = request;
    if (anchorIndex < 0 || anchorIndex >= frame.real.length) {
      throw new Error(
        `[opticalField] anchor index ${anchorIndex} out of bounds for ${frame.real.length} samples`,
      );
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
      phaseReference: 'aligned',
      phaseOrigin: {
        anchorIndex,
        referencePhase,
        appliedDelta: delta,
      },
    });
    for (const hook of this.hooks) {
      hook({
        field: frame,
        request,
        phaseDelta: delta,
      });
    }
    return delta;
  }

  stampFrame(frame: OpticalFieldFrame, options?: OpticalFieldAcquireOptions): OpticalFieldMetadata {
    if (!this.live.has(frame)) {
      throw new Error('[opticalField] cannot stamp metadata for unmanaged frame');
    }
    const prev = frame.getMeta();
    const frameId = this.allocateFrameId(options?.frameId);
    const meta = {
      ...this.makeMetadata(options, prev),
      frameId,
      componentCount: frame.componentCount,
      timestamp: options?.timestamp ?? defaultTimestamp(),
      dt: options?.dt ?? prev.dt ?? this.defaults.dt,
      wavelengthNm: options?.wavelengthNm ?? prev.wavelengthNm ?? this.defaults.wavelengthNm,
      pixelPitchMeters:
        options?.pixelPitchMeters ?? prev.pixelPitchMeters ?? this.defaults.pixelPitchMeters,
      space: options?.space ?? prev.space ?? this.defaults.space,
      phaseReference: options?.phaseReference ?? 'wrapped',
      phaseOrigin: undefined,
      notes: options?.notes ?? prev.notes,
      userTags: options?.userTags
        ? { ...options.userTags }
        : prev.userTags
          ? { ...prev.userTags }
          : undefined,
    };
    frame.markAcquired(meta);
    return meta;
  }

  registerPhaseHook(hook: PhaseAlignmentHook): () => void {
    this.hooks.add(hook);
    return () => {
      this.hooks.delete(hook);
    };
  }

  getResolution(): FieldResolution {
    return this.resolution;
  }

  getComponentCount(): number {
    return this.componentCount;
  }

  getLatestFrameId(): number {
    return this.nextFrameId - 1;
  }

  private allocateFrameId(requested?: number): number {
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

  private makeMetadata(
    options?: OpticalFieldAcquireOptions,
    prev?: OpticalFieldMetadata,
  ): OpticalFieldMetadata {
    const userTags = options?.userTags ?? prev?.userTags;
    return {
      schemaVersion: OPTICAL_FIELD_SCHEMA_VERSION,
      solver: this.solver,
      solverInstanceId: this.solverInstanceId,
      frameId: options?.frameId ?? prev?.frameId ?? -1,
      componentCount: options?.componentCount ?? prev?.componentCount ?? this.componentCount,
      timestamp: options?.timestamp ?? prev?.timestamp ?? 0,
      dt: options?.dt ?? prev?.dt ?? this.defaults.dt,
      wavelengthNm: options?.wavelengthNm ?? prev?.wavelengthNm ?? this.defaults.wavelengthNm,
      pixelPitchMeters:
        options?.pixelPitchMeters ?? prev?.pixelPitchMeters ?? this.defaults.pixelPitchMeters,
      space: options?.space ?? prev?.space ?? this.defaults.space,
      phaseReference: options?.phaseReference ?? prev?.phaseReference ?? 'wrapped',
      phaseOrigin: prev?.phaseOrigin,
      notes: options?.notes ?? prev?.notes,
      userTags: userTags ? { ...userTags } : undefined,
    };
  }
}
