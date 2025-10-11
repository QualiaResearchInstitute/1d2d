import { makeResolution, type FieldResolution } from './contracts.js';

export const OPTICAL_FIELD_SCHEMA_VERSION = 2 as const;

export type OpticalSpace = 'screen' | 'pupil' | 'volumeSlice';

export type OpticalSolverId = 'kuramoto' | 'angularSpectrum' | 'volumeStub' | 'dispatcher' | string;

export type PhaseReferenceKind = 'wrapped' | 'aligned';

export type OpticalFieldMetadata = {
  schemaVersion: number;
  solver: OpticalSolverId;
  solverInstanceId: string;
  frameId: number;
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

export class OpticalFieldFrame {
  readonly id: number;
  readonly resolution: FieldResolution;
  readonly buffer: ArrayBuffer;
  readonly view: Float32Array;
  readonly real: Float32Array;
  readonly imag: Float32Array;
  private metadata: OpticalFieldMetadata;
  private inUse = false;

  constructor(resolution: FieldResolution, buffer?: ArrayBuffer) {
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
      solver: 'dispatcher',
      solverInstanceId: 'unassigned',
      frameId: -1,
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
      ...update,
      phaseOrigin: update.phaseOrigin ?? this.metadata.phaseOrigin,
      userTags: update.userTags ?? this.metadata.userTags,
    };
  }

  isInUse() {
    return this.inUse;
  }

  getPhase(index: number): number {
    return Math.atan2(this.imag[index], this.real[index]);
  }

  applyPhaseRotation(delta: number) {
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
    this.defaults = {
      dt: options.defaultDt ?? 0,
      wavelengthNm: options.defaultWavelengthNm ?? 550,
      pixelPitchMeters: options.defaultPixelPitchMeters ?? 1e-6,
      space: options.defaultSpace ?? 'screen',
    };
  }

  acquireFrame(options?: OpticalFieldAcquireOptions): OpticalFieldFrame {
    let frame: OpticalFieldFrame | undefined = this.pool.pop();
    if (!frame) {
      frame = new OpticalFieldFrame(this.resolution);
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
