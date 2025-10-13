import type { RainbowFrameResult } from '../../src/pipeline/rainbowFrame.js';
import {
  PerformanceWatchdog,
  type PerformanceBudget,
  type PerformanceSample,
  type PerformanceSnapshot,
  type PerformanceWatchdogOptions,
} from './performanceWatchdog.js';

export type OfflineRendererConfig = {
  width: number;
  height: number;
  budgets: PerformanceBudget;
  tileHeight?: number;
  watchdog?: PerformanceWatchdogOptions;
};

export type OfflineFrameContext = {
  frameIndex: number;
};

export type OfflineRenderCallback = (
  outBuffer: Uint8ClampedArray,
) => Promise<RainbowFrameResult> | RainbowFrameResult;

export type OfflineFrameResult = {
  frameIndex: number;
  rainbow: RainbowFrameResult;
  performance: PerformanceSample;
  /** View over the renderer's internal output buffer. Consumers should finish reading before invoking renderFrame again. */
  output10Bit: Uint16Array;
};

const createLut8to10 = () => {
  const lut = new Uint16Array(256);
  for (let i = 0; i < 256; i++) {
    lut[i] = Math.round((i / 255) * 1023);
  }
  return lut;
};

const clampTileHeight = (value: number, totalRows: number) => {
  if (!Number.isFinite(value) || value <= 0) {
    return Math.min(256, totalRows);
  }
  return Math.min(Math.max(1, Math.floor(value)), totalRows);
};

export class OfflineRenderer {
  private readonly width: number;
  private readonly height: number;
  private readonly tileHeight: number;
  private readonly watchdog: PerformanceWatchdog;
  private readonly outBuffer: Uint8ClampedArray;
  private readonly out10Bit: Uint16Array;
  private readonly lut8to10: Uint16Array;

  constructor(config: OfflineRendererConfig) {
    this.width = config.width;
    this.height = config.height;
    this.tileHeight = clampTileHeight(config.tileHeight ?? 256, this.height);
    this.watchdog = new PerformanceWatchdog(config.budgets, {
      label: 'offline-renderer',
      historySize: 120,
      ...config.watchdog,
    });
    this.outBuffer = new Uint8ClampedArray(this.width * this.height * 4);
    this.out10Bit = new Uint16Array(this.outBuffer.length);
    this.lut8to10 = createLut8to10();
  }

  getPerformanceSnapshot(): PerformanceSnapshot {
    return this.watchdog.snapshot();
  }

  getScratchBuffer(): Uint8ClampedArray {
    return this.outBuffer;
  }

  async renderFrame(
    context: OfflineFrameContext,
    callback: OfflineRenderCallback,
  ): Promise<OfflineFrameResult> {
    this.watchdog.beginFrame(context.frameIndex);
    const rainbow = await callback(this.outBuffer);
    this.convertTo10Bit();
    const performance = this.watchdog.endFrame();
    return {
      frameIndex: context.frameIndex,
      rainbow,
      performance,
      output10Bit: this.out10Bit,
    };
  }

  private convertTo10Bit() {
    const rowStride = this.width * 4;
    const chunkStride = this.tileHeight * rowStride;
    const total = this.outBuffer.length;
    for (let offset = 0; offset < total; offset += chunkStride) {
      const end = Math.min(total, offset + chunkStride);
      for (let i = offset; i < end; i++) {
        this.out10Bit[i] = this.lut8to10[this.outBuffer[i] ?? 0];
      }
    }
  }
}
