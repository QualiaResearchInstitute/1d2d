/// <reference lib="webworker" />

import {
  createKuramotoState,
  createDerivedViews,
  createNormalGenerator,
  derivedBufferSize,
  deriveKuramotoFields,
  initKuramotoState,
  stepKuramotoState,
  createKuramotoInstrumentationSnapshot,
  type KuramotoParams,
  type KuramotoState,
  type KuramotoInstrumentationSnapshot,
  type ThinElementSchedule,
} from './kuramotoCore';
import { assertPhaseField } from './fields/contracts.js';
import { clampKernelSpec, KERNEL_SPEC_DEFAULT, type KernelSpec } from './kernel/kernelSpec';
import type { OpticalFieldMetadata } from './fields/opticalField.js';

type InitMessage = {
  kind: 'init';
  width: number;
  height: number;
  params: KuramotoParams;
  qInit: number;
  buffers: ArrayBuffer[];
  seed?: number;
  componentCount?: number;
};

type TickMessage = {
  kind: 'tick';
  dt: number;
  timestamp: number;
  frameId: number;
  seed?: number;
  schedule?: ThinElementSchedule | null;
  componentCount?: number;
};

type UpdateParamsMessage = {
  kind: 'updateParams';
  params: KuramotoParams;
};

type KernelSpecMessage = {
  kind: 'kernelSpec';
  spec: KernelSpec;
  version: number;
};

type ResetMessage = {
  kind: 'reset';
  qInit: number;
  seed?: number;
};

type ReturnBufferMessage = {
  kind: 'returnBuffer';
  buffer: ArrayBuffer;
};

type SimulateMessage = {
  kind: 'simulate';
  frameCount: number;
  dt: number;
  params: KuramotoParams;
  width: number;
  height: number;
  qInit: number;
  seed?: number;
  schedule?: ThinElementSchedule | null;
  componentCount?: number;
};

type IncomingMessage =
  | InitMessage
  | TickMessage
  | UpdateParamsMessage
  | KernelSpecMessage
  | ResetMessage
  | ReturnBufferMessage
  | SimulateMessage;

type FrameMessage = {
  kind: 'frame';
  buffer: ArrayBuffer;
  timestamp: number;
  frameId: number;
  queueDepth: number;
  meta: OpticalFieldMetadata;
  kernelVersion: number;
  instrumentation: KuramotoInstrumentationSnapshot;
};

type ReadyMessage = { kind: 'ready'; width: number; height: number };
type LogMessage = { kind: 'log'; message: string };
type SimulateResultMessage = {
  kind: 'simulateResult';
  buffers: ArrayBuffer[];
  width: number;
  height: number;
  frameCount: number;
};

const ctx: DedicatedWorkerGlobalScope = self as any;

let state: KuramotoState | null = null;
let params: KuramotoParams | null = null;
let randn = createNormalGenerator();
let bufferPool: ArrayBuffer[] = [];
let width = 0;
let height = 0;
let componentCount = 1;
let kernelSpec: KernelSpec | null = null;
let kernelSpecVersion = 0;

const post = (
  message: FrameMessage | ReadyMessage | LogMessage | SimulateResultMessage,
  transfer?: Transferable[],
) => {
  if (transfer) {
    ctx.postMessage(message, transfer);
  } else {
    ctx.postMessage(message);
  }
};

const ensureState = (w: number, h: number, requestedComponents: number) => {
  if (
    !state ||
    state.width !== w ||
    state.height !== h ||
    state.componentCount !== requestedComponents
  ) {
    state = createKuramotoState(w, h, undefined, { componentCount: requestedComponents });
    componentCount = requestedComponents;
  }
};

const ensureRand = (seed?: number) => {
  randn = createNormalGenerator(seed);
};

const acquireBuffer = () => bufferPool.shift() ?? null;

const releaseBuffer = (buffer: ArrayBuffer) => {
  bufferPool.push(buffer);
};

const handleTick = (msg: TickMessage) => {
  if (!state || !params) return;
  const requestedComponents = msg.componentCount ?? componentCount;
  ensureState(width, height, requestedComponents);
  componentCount = requestedComponents;
  const buffer = acquireBuffer();
  if (!buffer) {
    post({
      kind: 'log',
      message: '[kur-worker] dropping frame: buffer pool empty',
    });
    return;
  }
  if (msg.seed != null) {
    ensureRand(msg.seed);
  }
  const activeKernel = kernelSpec ?? KERNEL_SPEC_DEFAULT;
  stepKuramotoState(state, params, msg.dt, randn, msg.timestamp, {
    kernel: activeKernel,
    telemetry: { kernelVersion: kernelSpecVersion },
    schedule: msg.schedule ?? undefined,
  });
  const meta = state.field.getMeta();
  const derived = createDerivedViews(buffer, state.width, state.height);
  assertPhaseField(derived, 'worker:tick');
  deriveKuramotoFields(state, derived, {
    kernel: activeKernel,
    schedule: msg.schedule ?? undefined,
  });
  const instrumentation = createKuramotoInstrumentationSnapshot(state);
  post(
    {
      kind: 'frame',
      buffer,
      timestamp: msg.timestamp,
      frameId: meta.frameId,
      queueDepth: bufferPool.length,
      kernelVersion: kernelSpecVersion,
      meta,
      instrumentation,
    },
    [buffer],
  );
};

const handleInit = (msg: InitMessage) => {
  width = msg.width;
  height = msg.height;
  params = { ...msg.params };
  componentCount = msg.componentCount ?? componentCount;
  ensureState(width, height, componentCount);
  ensureRand(msg.seed);
  bufferPool = [...msg.buffers];
  if (!state) return;
  initKuramotoState(state, msg.qInit);
  post({ kind: 'ready', width: state.width, height: state.height });
};

const handleReset = (msg: ResetMessage) => {
  if (!state) return;
  ensureRand(msg.seed);
  initKuramotoState(state, msg.qInit);
};

const handleSimulate = (msg: SimulateMessage) => {
  const simState = createKuramotoState(msg.width, msg.height, undefined, {
    componentCount: msg.componentCount ?? 1,
  });
  const simParams = { ...msg.params };
  const simRand = createNormalGenerator(msg.seed);
  initKuramotoState(simState, msg.qInit);
  const size = derivedBufferSize(msg.width, msg.height);
  const buffers: ArrayBuffer[] = [];
  const activeKernel = kernelSpec ?? KERNEL_SPEC_DEFAULT;
  const schedule = msg.schedule ?? undefined;
  for (let frame = 0; frame < msg.frameCount; frame++) {
    stepKuramotoState(simState, simParams, msg.dt, simRand, msg.dt * (frame + 1), {
      kernel: activeKernel,
      schedule,
    });
    const out = new ArrayBuffer(size);
    const derived = createDerivedViews(out, msg.width, msg.height);
    assertPhaseField(derived, 'worker:simulate');
    deriveKuramotoFields(simState, derived, {
      kernel: activeKernel,
      schedule,
    });
    buffers.push(out);
  }
  post(
    {
      kind: 'simulateResult',
      buffers,
      width: msg.width,
      height: msg.height,
      frameCount: msg.frameCount,
    },
    buffers,
  );
};

ctx.onmessage = (event: MessageEvent<IncomingMessage>) => {
  const msg = event.data;
  switch (msg.kind) {
    case 'init':
      handleInit(msg);
      break;
    case 'tick':
      handleTick(msg);
      break;
    case 'updateParams':
      params = { ...msg.params };
      break;
    case 'kernelSpec':
      kernelSpec = clampKernelSpec(msg.spec);
      kernelSpecVersion = msg.version;
      break;
    case 'reset':
      handleReset(msg);
      break;
    case 'returnBuffer':
      releaseBuffer(msg.buffer);
      break;
    case 'simulate':
      handleSimulate(msg);
      break;
    default:
      post({
        kind: 'log',
        message: `[kur-worker] unhandled message ${(msg as any)?.kind}`,
      });
      break;
  }
};
