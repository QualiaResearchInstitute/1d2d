export type PerformanceBudget = {
  frameMs?: number;
  rssMb?: number;
  heapMb?: number;
  cpuPercent?: number;
};

export type PerformanceSample = {
  frameMs: number;
  rssMb: number;
  heapMb: number;
  cpuPercent: number;
  frameIndex: number;
};

export type WatchdogViolationType = keyof PerformanceBudget;

export type WatchdogViolation = {
  type: WatchdogViolationType;
  value: number;
  limit: number;
  frameIndex: number;
};

export type PerformanceSnapshot = {
  frames: number;
  frameMsAvg: number;
  frameMsMax: number;
  rssMbMax: number;
  heapMbMax: number;
  cpuPercentAvg: number;
  cpuPercentMax: number;
  lastSample: PerformanceSample | null;
  history: PerformanceSample[];
  violations: WatchdogViolation[];
};

type MeasurementProvider = {
  now: () => bigint;
  cpu: () => NodeJS.CpuUsage;
  memory: () => NodeJS.MemoryUsage;
};

export type PerformanceWatchdogOptions = {
  label?: string;
  tolerance?: number;
  historySize?: number;
};

const convertBytesToMb = (bytes: number) => bytes / (1024 * 1024);

const DEFAULT_PROVIDER: MeasurementProvider = {
  now: () => process.hrtime.bigint(),
  cpu: () => process.cpuUsage(),
  memory: () => process.memoryUsage(),
};

export class PerformanceWatchdog {
  private readonly budget: PerformanceBudget;
  private readonly tolerance: number;
  private readonly historySize: number;
  private readonly provider: MeasurementProvider;
  private readonly label: string;
  private readonly history: PerformanceSample[] = [];
  private readonly violations: WatchdogViolation[] = [];

  private frameCount = 0;
  private frameMsTotal = 0;
  private frameMsMax = 0;
  private cpuPercentTotal = 0;
  private cpuPercentMax = 0;
  private rssMbMax = 0;
  private heapMbMax = 0;
  private lastSample: PerformanceSample | null = null;

  private frameStart: {
    time: bigint;
    cpu: NodeJS.CpuUsage;
    memory: NodeJS.MemoryUsage;
    frameIndex: number;
  } | null = null;

  constructor(
    budget: PerformanceBudget,
    options: PerformanceWatchdogOptions = {},
    provider: MeasurementProvider = DEFAULT_PROVIDER,
  ) {
    this.budget = { ...budget };
    this.tolerance = Math.max(0, options.tolerance ?? 0.1);
    this.historySize = Math.max(0, Math.floor(options.historySize ?? 32));
    this.provider = provider;
    this.label = options.label ?? 'performance';
  }

  beginFrame(frameIndex: number) {
    if (this.frameStart) {
      throw new Error(`[${this.label}] beginFrame called twice without endFrame.`);
    }
    this.frameStart = {
      time: this.provider.now(),
      cpu: this.provider.cpu(),
      memory: this.provider.memory(),
      frameIndex,
    };
  }

  endFrame(): PerformanceSample {
    if (!this.frameStart) {
      throw new Error(`[${this.label}] endFrame called without beginFrame.`);
    }
    const endTime = this.provider.now();
    const endCpu = this.provider.cpu();
    const endMemory = this.provider.memory();
    const start = this.frameStart;
    this.frameStart = null;

    const elapsedNs = endTime - start.time;
    const elapsedMs = Number(elapsedNs) / 1_000_000;
    const cpuDeltaUser = endCpu.user - start.cpu.user;
    const cpuDeltaSystem = endCpu.system - start.cpu.system;
    const cpuTotalUs = cpuDeltaUser + cpuDeltaSystem;
    const cpuMs = cpuTotalUs / 1000;
    const cpuPercent = elapsedMs > 0 ? (cpuMs / elapsedMs) * 100 : 0;

    const rssMb = convertBytesToMb(endMemory.rss);
    const heapMb = convertBytesToMb(endMemory.heapUsed);

    const sample: PerformanceSample = {
      frameMs: elapsedMs,
      rssMb,
      heapMb,
      cpuPercent,
      frameIndex: start.frameIndex,
    };

    this.frameCount += 1;
    this.frameMsTotal += elapsedMs;
    this.cpuPercentTotal += cpuPercent;
    this.frameMsMax = Math.max(this.frameMsMax, elapsedMs);
    this.cpuPercentMax = Math.max(this.cpuPercentMax, cpuPercent);
    this.rssMbMax = Math.max(this.rssMbMax, rssMb);
    this.heapMbMax = Math.max(this.heapMbMax, heapMb);
    this.lastSample = sample;

    if (this.historySize > 0) {
      this.history.push(sample);
      if (this.history.length > this.historySize) {
        this.history.shift();
      }
    }

    this.checkBudgets(sample);
    return sample;
  }

  private checkBudgets(sample: PerformanceSample) {
    const slack = 1 + this.tolerance;
    const checks: Array<[WatchdogViolationType, number | undefined, number]> = [
      ['frameMs', this.budget.frameMs, sample.frameMs],
      ['rssMb', this.budget.rssMb, sample.rssMb],
      ['heapMb', this.budget.heapMb, sample.heapMb],
      ['cpuPercent', this.budget.cpuPercent, sample.cpuPercent],
    ];
    for (const [type, limit, value] of checks) {
      if (typeof limit === 'number' && Number.isFinite(limit) && value > limit * slack) {
        this.violations.push({
          type,
          value,
          limit,
          frameIndex: sample.frameIndex,
        });
      }
    }
  }

  snapshot(): PerformanceSnapshot {
    const frames = this.frameCount;
    const frameMsAvg = frames > 0 ? this.frameMsTotal / frames : 0;
    const cpuPercentAvg = frames > 0 ? this.cpuPercentTotal / frames : 0;
    return {
      frames,
      frameMsAvg,
      frameMsMax: this.frameMsMax,
      rssMbMax: this.rssMbMax,
      heapMbMax: this.heapMbMax,
      cpuPercentAvg,
      cpuPercentMax: this.cpuPercentMax,
      lastSample: this.lastSample,
      history: [...this.history],
      violations: [...this.violations],
    };
  }

  reset(): void {
    this.frameCount = 0;
    this.frameMsTotal = 0;
    this.frameMsMax = 0;
    this.cpuPercentTotal = 0;
    this.cpuPercentMax = 0;
    this.rssMbMax = 0;
    this.heapMbMax = 0;
    this.lastSample = null;
    this.history.length = 0;
    this.violations.length = 0;
    this.frameStart = null;
  }
}
