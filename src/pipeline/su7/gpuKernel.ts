import type { C7Vector, Complex7x7 } from './types.js';

const SU7_DIM = 7;
const COMPLEX_PARTS = 2;
const DEFAULT_STRIDE_COMPLEX = SU7_DIM;
const DEFAULT_VECTOR_STRIDE = SU7_DIM * COMPLEX_PARTS;
const MATRIX_FLOATS = SU7_DIM * SU7_DIM * COMPLEX_PARTS;
const WORKGROUP_SIZE = 64;
const PROFILE_RING_CAPACITY = 240;
const PROFILE_BASELINE_SAMPLES = 90;
const DRIFT_THRESHOLD = 0.1;

type GpuFlagTable = Readonly<Record<string, number>>;

const FALLBACK_GPU_BUFFER_USAGE: GpuFlagTable = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
} as const;

const FALLBACK_GPU_MAP_MODE: GpuFlagTable = {
  READ: 0x0001,
  WRITE: 0x0002,
} as const;

type MaybeGpuEnvironment = {
  GPUBufferUsage?: GpuFlagTable;
  GPUMapMode?: GpuFlagTable;
};

const gpuEnvironment: MaybeGpuEnvironment =
  typeof globalThis === 'undefined' ? {} : (globalThis as MaybeGpuEnvironment);

const GPU_BUFFER_USAGE: GpuFlagTable = gpuEnvironment.GPUBufferUsage ?? FALLBACK_GPU_BUFFER_USAGE;

const GPU_MAP_MODE: GpuFlagTable = gpuEnvironment.GPUMapMode ?? FALLBACK_GPU_MAP_MODE;

export type Su7GpuKernelSources = {
  wgsl: string;
  glsl: string;
};

let cachedKernelSources: Su7GpuKernelSources | null = null;

const getNodeProcess = ():
  | {
      versions?: Record<string, unknown>;
    }
  | undefined =>
  typeof globalThis === 'undefined'
    ? undefined
    : ((globalThis as { process?: unknown }).process as {
        versions?: Record<string, unknown>;
      });

const isNodeEnvironment = (): boolean => {
  const nodeProcess = getNodeProcess();
  const versions = nodeProcess?.versions;
  return typeof versions === 'object' && versions != null && 'node' in versions;
};

const loadSu7GpuKernelSources = async (): Promise<Su7GpuKernelSources> => {
  if (cachedKernelSources) {
    return cachedKernelSources;
  }
  if (isNodeEnvironment()) {
    const [{ readFile }, { fileURLToPath }, { resolve, dirname }] = await Promise.all([
      import('node:fs/promises'),
      import('node:url'),
      import('node:path'),
    ]);
    const baseDir = dirname(fileURLToPath(import.meta.url));
    const [wgsl, glsl] = await Promise.all([
      readFile(resolve(baseDir, 'gpuKernel.wgsl'), 'utf8'),
      readFile(resolve(baseDir, 'gpuKernel.glsl'), 'utf8'),
    ]);
    cachedKernelSources = { wgsl, glsl };
    return cachedKernelSources;
  }
  const [wgslModule, glslModule] = await Promise.all([
    import('./gpuKernel.wgsl?raw'),
    import('./gpuKernel.glsl?raw'),
  ]);
  cachedKernelSources = {
    wgsl: (wgslModule as { default: string }).default,
    glsl: (glslModule as { default: string }).default,
  };
  return cachedKernelSources;
};

export const getSu7GpuKernelSources = async (): Promise<Su7GpuKernelSources> =>
  loadSu7GpuKernelSources();

type BackendKind = 'gpu' | 'cpu';

export type Su7GpuKernelDispatchParams = {
  unitary: Float32Array;
  input: Float32Array;
  vectorCount: number;
  stride?: number;
  output?: Float32Array | null;
};

export type Su7GpuKernelProfile = {
  backend: BackendKind;
  timeMs: number;
  vectorCount: number;
};

export type Su7GpuKernelStats = {
  backend: BackendKind;
  sampleCount: number;
  medianMs: number;
  meanMs: number;
  baselineMs: number | null;
  drift: number | null;
  warning: boolean;
};

export type Su7GpuKernelWarningEvent = {
  drift: number;
  medianMs: number;
  baselineMs: number;
};

export type Su7GpuKernelInitOptions = {
  backend?: 'auto' | 'gpu-first' | 'cpu-only';
  device?: unknown;
  now?: () => number;
  onWarning?: (event: Su7GpuKernelWarningEvent) => void;
  profileCapacity?: number;
  label?: string;
};

export const SU7_GPU_KERNEL_DIM = SU7_DIM;
export const SU7_GPU_KERNEL_VECTOR_STRIDE = DEFAULT_VECTOR_STRIDE;
export const SU7_GPU_KERNEL_MATRIX_FLOATS = MATRIX_FLOATS;
export const SU7_GPU_KERNEL_WORKGROUP_SIZE = WORKGROUP_SIZE;

const ensureStrideComplex = (stride?: number): number => {
  if (stride == null || !Number.isFinite(stride) || stride <= 0) {
    return DEFAULT_STRIDE_COMPLEX;
  }
  return Math.max(1, Math.trunc(stride));
};

const assertPackedDimensions = (
  unitary: Float32Array,
  input: Float32Array,
  vectorCount: number,
  strideComplex: number,
) => {
  if (unitary.length !== MATRIX_FLOATS) {
    throw new Error(
      `[su7-gpu-kernel] expected unitary buffer length ${MATRIX_FLOATS}, received ${unitary.length}`,
    );
  }
  const strideFloats = strideComplex * COMPLEX_PARTS;
  const required = vectorCount * strideFloats;
  if (input.length < required) {
    throw new Error(
      `[su7-gpu-kernel] input length ${input.length} too small for ${vectorCount} vectors stride ${strideFloats}`,
    );
  }
};

export const packSu7Unitary = (matrix: Complex7x7, target?: Float32Array | null): Float32Array => {
  const output =
    target && target.length === MATRIX_FLOATS ? target : new Float32Array(MATRIX_FLOATS);
  let ptr = 0;
  for (let row = 0; row < SU7_DIM; row++) {
    for (let col = 0; col < SU7_DIM; col++) {
      const cell = matrix[row]?.[col];
      const re = cell?.re ?? 0;
      const im = cell?.im ?? 0;
      output[ptr++] = re;
      output[ptr++] = im;
    }
  }
  return output;
};

export const packSu7Vectors = (
  vectors: readonly C7Vector[],
  target?: Float32Array | null,
): Float32Array => {
  const count = vectors.length;
  const strideFloats = DEFAULT_VECTOR_STRIDE;
  const output =
    target && target.length === count * strideFloats
      ? target
      : new Float32Array(count * strideFloats);
  for (let index = 0; index < count; index++) {
    const vector = vectors[index];
    const base = index * strideFloats;
    for (let dim = 0; dim < SU7_DIM; dim++) {
      const entry = vector?.[dim];
      output[base + dim * COMPLEX_PARTS + 0] = entry?.re ?? 0;
      output[base + dim * COMPLEX_PARTS + 1] = entry?.im ?? 0;
    }
  }
  return output;
};

export const multiplySu7CpuPacked = (
  unitary: Float32Array,
  input: Float32Array,
  vectorCount: number,
  strideComplex = DEFAULT_STRIDE_COMPLEX,
  target?: Float32Array | null,
): Float32Array => {
  assertPackedDimensions(unitary, input, vectorCount, strideComplex);
  const strideFloats = strideComplex * COMPLEX_PARTS;
  const output =
    target && target.length >= vectorCount * strideFloats
      ? target
      : new Float32Array(vectorCount * strideFloats);

  for (let vectorIndex = 0; vectorIndex < vectorCount; vectorIndex++) {
    const vectorBase = vectorIndex * strideFloats;
    for (let row = 0; row < SU7_DIM; row++) {
      let sumRe = 0;
      let sumIm = 0;
      for (let col = 0; col < SU7_DIM; col++) {
        const matrixOffset = (row * SU7_DIM + col) * COMPLEX_PARTS;
        const coeffRe = unitary[matrixOffset + 0];
        const coeffIm = unitary[matrixOffset + 1];
        const vectorOffset = vectorBase + col * COMPLEX_PARTS;
        const valueRe = input[vectorOffset + 0];
        const valueIm = input[vectorOffset + 1];
        sumRe += coeffRe * valueRe - coeffIm * valueIm;
        sumIm += coeffRe * valueIm + coeffIm * valueRe;
      }
      const outputOffset = vectorBase + row * COMPLEX_PARTS;
      output[outputOffset + 0] = sumRe;
      output[outputOffset + 1] = sumIm;
    }
  }
  return output;
};

const computeMedian = (values: readonly number[]): number => {
  if (!values.length) return 0;
  const copy = [...values].sort((a, b) => a - b);
  const mid = Math.floor(copy.length / 2);
  if (copy.length % 2 === 0) {
    return (copy[mid - 1] + copy[mid]) * 0.5;
  }
  return copy[mid];
};

const computeMean = (values: readonly number[]): number => {
  if (!values.length) return 0;
  let sum = 0;
  for (const value of values) {
    sum += value;
  }
  return sum / values.length;
};

const defaultNow = () => {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
};

type GpuState = {
  device: any;
  pipeline: any;
  unitaryBuffer: any;
  inputBuffer: any;
  outputBuffer: any;
  readbackBuffer: any;
  uniformBuffer: any;
  bindGroupLayout: any;
  bindGroup: any;
  bufferCapacity: number;
};

export class Su7GpuKernel {
  private readonly backend: BackendKind;
  private readonly now: () => number;
  private readonly samples: number[] = [];
  private readonly capacity: number;
  private readonly onWarning?: (event: Su7GpuKernelWarningEvent) => void;
  private readonly label: string;
  private baselineMs: number | null = null;
  private warningActive = false;
  private gpu: GpuState | null;
  private lastProfile: Su7GpuKernelProfile | null = null;

  private constructor(
    backend: BackendKind,
    nowFn: () => number,
    gpu: GpuState | null,
    options: Su7GpuKernelInitOptions,
  ) {
    this.backend = backend;
    this.now = nowFn;
    this.gpu = gpu;
    this.capacity = Math.max(32, Math.trunc(options.profileCapacity ?? PROFILE_RING_CAPACITY));
    this.onWarning = options.onWarning;
    this.label = options.label ?? 'su7-gpu-kernel';
  }

  static async create(options: Su7GpuKernelInitOptions = {}): Promise<Su7GpuKernel> {
    const backendPref = options.backend ?? 'auto';
    const nowFn = options.now ?? defaultNow;
    if (backendPref === 'cpu-only') {
      return new Su7GpuKernel('cpu', nowFn, null, options);
    }

    const device =
      (options.device as any) ??
      (typeof navigator !== 'undefined' && (navigator as any).gpu
        ? await (navigator as any).gpu
            .requestAdapter()
            .then((adapter: any) => adapter?.requestDevice?.())
            .catch(() => null)
        : null);

    if (!device) {
      if (backendPref === 'gpu-first') {
        throw new Error('[su7-gpu-kernel] WebGPU device unavailable');
      }
      return new Su7GpuKernel('cpu', nowFn, null, options);
    }

    try {
      const { wgsl: kernelWgslSource } = await getSu7GpuKernelSources();
      const module = device.createShaderModule({
        label: options.label ?? 'su7-gpu-kernel',
        code: kernelWgslSource,
      });
      const pipeline =
        device.createComputePipelineAsync != null
          ? await device.createComputePipelineAsync({
              layout: 'auto',
              label: options.label ?? 'su7-gpu-kernel',
              compute: {
                module,
                entryPoint: 'main',
              },
            })
          : device.createComputePipeline({
              layout: 'auto',
              label: options.label ?? 'su7-gpu-kernel',
              compute: {
                module,
                entryPoint: 'main',
              },
            });
      const bindGroupLayout = pipeline.getBindGroupLayout(0);
      const unitaryBuffer = device.createBuffer({
        label: 'su7-unitary',
        size: MATRIX_FLOATS * Float32Array.BYTES_PER_ELEMENT,
        usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
      });
      const uniformBuffer = device.createBuffer({
        label: 'su7-uniforms',
        size: 16,
        usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
      });
      const gpu: GpuState = {
        device,
        pipeline,
        unitaryBuffer,
        inputBuffer: null,
        outputBuffer: null,
        readbackBuffer: null,
        uniformBuffer,
        bindGroupLayout,
        bindGroup: null,
        bufferCapacity: 0,
      };
      return new Su7GpuKernel('gpu', nowFn, gpu, options);
    } catch (error) {
      console.warn('[su7-gpu-kernel] falling back to CPU backend', error);
      if (backendPref === 'gpu-first') {
        throw new Error('[su7-gpu-kernel] failed to initialize GPU backend');
      }
      return new Su7GpuKernel('cpu', nowFn, null, options);
    }
  }

  getBackend(): BackendKind {
    return this.backend;
  }

  getLastProfile(): Su7GpuKernelProfile | null {
    return this.lastProfile;
  }

  getStats(): Su7GpuKernelStats | null {
    if (!this.samples.length) {
      return null;
    }
    const median = computeMedian(this.samples);
    const mean = computeMean(this.samples);
    if (this.baselineMs == null && this.samples.length >= PROFILE_BASELINE_SAMPLES) {
      this.baselineMs = median;
    }
    const baseline = this.baselineMs;
    const medianDrift = baseline ? (median - baseline) / baseline : null;
    const latestSample = this.samples[this.samples.length - 1] ?? null;
    const latestDrift =
      baseline != null && latestSample != null ? (latestSample - baseline) / baseline : null;
    const driftCandidates = [medianDrift, latestDrift].filter(
      (value): value is number => value != null,
    );
    const drift = driftCandidates.length ? Math.max(...driftCandidates) : null;
    const warning = drift != null && drift > DRIFT_THRESHOLD;
    if (warning && !this.warningActive && baseline != null) {
      this.onWarning?.({
        drift,
        medianMs: median,
        baselineMs: baseline,
      });
    }
    this.warningActive = warning;
    return {
      backend: this.backend,
      sampleCount: this.samples.length,
      medianMs: median,
      meanMs: mean,
      baselineMs: baseline ?? null,
      drift: drift ?? null,
      warning,
    };
  }

  async dispatch(params: Su7GpuKernelDispatchParams): Promise<Float32Array> {
    const strideComplex = ensureStrideComplex(params.stride);
    const strideFloats = strideComplex * COMPLEX_PARTS;
    const nowStart = this.now();
    let output: Float32Array;
    if (this.backend === 'gpu' && this.gpu) {
      output = await this.dispatchGpu(params, strideComplex, strideFloats);
    } else {
      output = multiplySu7CpuPacked(
        params.unitary,
        params.input,
        params.vectorCount,
        strideComplex,
        params.output ?? undefined,
      );
    }
    const timeMs = this.now() - nowStart;
    this.samples.push(timeMs);
    if (this.samples.length > this.capacity) {
      this.samples.shift();
    }
    this.lastProfile = {
      backend: this.backend,
      timeMs,
      vectorCount: params.vectorCount,
    };
    this.getStats();
    return output;
  }

  dispose(): void {
    if (!this.gpu) return;
    const { unitaryBuffer, inputBuffer, outputBuffer, readbackBuffer, uniformBuffer } = this.gpu;
    unitaryBuffer?.destroy?.();
    inputBuffer?.destroy?.();
    outputBuffer?.destroy?.();
    readbackBuffer?.destroy?.();
    uniformBuffer?.destroy?.();
    this.gpu = null;
  }

  private ensureGpuBuffers(capacity: number, strideFloats: number) {
    if (!this.gpu) return;
    const byteLength = capacity * strideFloats * Float32Array.BYTES_PER_ELEMENT;
    if (
      this.gpu.bufferCapacity >= capacity &&
      this.gpu.inputBuffer &&
      this.gpu.outputBuffer &&
      this.gpu.readbackBuffer
    ) {
      return;
    }
    const device = this.gpu.device;
    this.gpu.inputBuffer?.destroy?.();
    this.gpu.outputBuffer?.destroy?.();
    this.gpu.readbackBuffer?.destroy?.();
    const inputBuffer = device.createBuffer({
      label: `${this.label}-input`,
      size: byteLength,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
    });
    const outputBuffer = device.createBuffer({
      label: `${this.label}-output`,
      size: byteLength,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_SRC,
    });
    const readbackBuffer = device.createBuffer({
      label: `${this.label}-readback`,
      size: byteLength,
      usage: GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.MAP_READ,
    });
    this.gpu.inputBuffer = inputBuffer;
    this.gpu.outputBuffer = outputBuffer;
    this.gpu.readbackBuffer = readbackBuffer;
    this.gpu.bufferCapacity = capacity;
    this.gpu.bindGroup = null;
  }

  private createBindGroup() {
    if (!this.gpu) return;
    if (this.gpu.bindGroup && this.gpu.bindGroupLayout) {
      return;
    }
    const device = this.gpu.device;
    this.gpu.bindGroup = device.createBindGroup({
      layout: this.gpu.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.gpu.unitaryBuffer } },
        { binding: 1, resource: { buffer: this.gpu.inputBuffer } },
        { binding: 2, resource: { buffer: this.gpu.outputBuffer } },
        { binding: 3, resource: { buffer: this.gpu.uniformBuffer } },
      ],
    });
  }

  private async dispatchGpu(
    params: Su7GpuKernelDispatchParams,
    strideComplex: number,
    strideFloats: number,
  ): Promise<Float32Array> {
    if (!this.gpu) {
      throw new Error('[su7-gpu-kernel] GPU backend not available');
    }
    assertPackedDimensions(params.unitary, params.input, params.vectorCount, strideComplex);
    const requiredCapacity = Math.max(1, params.vectorCount);
    this.ensureGpuBuffers(requiredCapacity, strideFloats);
    this.createBindGroup();
    const device = this.gpu.device;
    const queue = device.queue;
    queue.writeBuffer(this.gpu.unitaryBuffer, 0, params.unitary);
    queue.writeBuffer(this.gpu.inputBuffer, 0, params.input);
    const uniforms = new Uint32Array([params.vectorCount >>> 0, strideComplex >>> 0, 0, 0]);
    queue.writeBuffer(this.gpu.uniformBuffer, 0, uniforms);
    const encoder = device.createCommandEncoder({
      label: `${this.label}-commands`,
    });
    const pass = encoder.beginComputePass({
      label: `${this.label}-pass`,
    });
    pass.setPipeline(this.gpu.pipeline);
    pass.setBindGroup(0, this.gpu.bindGroup);
    const workgroups = Math.max(1, Math.ceil(params.vectorCount / WORKGROUP_SIZE));
    pass.dispatchWorkgroups(workgroups);
    pass.end();
    const byteLength = params.vectorCount * strideFloats * Float32Array.BYTES_PER_ELEMENT;
    encoder.copyBufferToBuffer(this.gpu.outputBuffer, 0, this.gpu.readbackBuffer, 0, byteLength);
    const commandBuffer = encoder.finish();
    queue.submit([commandBuffer]);
    await (queue.onSubmittedWorkDone?.() ?? Promise.resolve());

    await this.gpu.readbackBuffer.mapAsync(GPU_MAP_MODE.READ);
    const arrayBuffer = this.gpu.readbackBuffer.getMappedRange(0, byteLength);
    const output =
      params.output && params.output.length >= params.vectorCount * strideFloats
        ? params.output
        : new Float32Array(params.vectorCount * strideFloats);
    output.set(new Float32Array(arrayBuffer));
    this.gpu.readbackBuffer.unmap();
    return output;
  }
}
