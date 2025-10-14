import { OpticalFieldManager, type OpticalFieldFrame } from '../fields/opticalField.js';

export type AngularSpectrumConfig = {
  width: number;
  height: number;
  wavelengthNm: number;
  pixelPitchMeters: number;
  dzMeters: number;
  componentCount?: number;
};

export class AngularSpectrumSolver {
  private manager: OpticalFieldManager;
  private readonly config: AngularSpectrumConfig;
  private componentCount: number;

  constructor(config: AngularSpectrumConfig) {
    this.config = config;
    this.componentCount = Math.max(1, config.componentCount ?? 1);
    this.manager = this.createManager(this.componentCount);
  }

  propagate(
    input: OpticalFieldFrame,
    options?: { dzMeters?: number; timestamp?: number },
  ): OpticalFieldFrame {
    if (input.componentCount !== this.componentCount) {
      this.componentCount = input.componentCount;
      this.manager = this.createManager(this.componentCount);
    }
    const frame = this.manager.acquireFrame({
      dt: options?.dzMeters ?? this.config.dzMeters,
      timestamp: options?.timestamp,
      componentCount: this.componentCount,
    });
    for (let componentIndex = 0; componentIndex < this.componentCount; componentIndex++) {
      const source = input.components[componentIndex] ?? input.components[0]!;
      const target = frame.components[componentIndex]!;
      target.real.set(source.real);
      target.imag.set(source.imag);
    }
    const shift = (options?.dzMeters ?? this.config.dzMeters) * 0.1;
    if (Math.abs(shift) > 1e-12) {
      const cos = Math.cos(shift);
      const sin = Math.sin(shift);
      for (const { real, imag } of frame.components) {
        for (let i = 0; i < real.length; i++) {
          const r = real[i];
          const im = imag[i];
          real[i] = r * cos - im * sin;
          imag[i] = r * sin + im * cos;
        }
      }
    }
    this.manager.stampFrame(frame, {
      dt: options?.dzMeters ?? this.config.dzMeters,
      timestamp: options?.timestamp,
    });
    return frame;
  }

  alignPhase(frame: OpticalFieldFrame, anchorIndex: number, referencePhase: number) {
    this.manager.alignPhase(frame, { anchorIndex, referencePhase });
  }

  getManager() {
    return this.manager;
  }

  private createManager(componentCount: number) {
    return new OpticalFieldManager({
      solver: 'angularSpectrum',
      solverInstanceId: 'angularSpectrum-main',
      resolution: { width: this.config.width, height: this.config.height },
      defaultWavelengthNm: this.config.wavelengthNm,
      defaultPixelPitchMeters: this.config.pixelPitchMeters,
      defaultDt: this.config.dzMeters,
      initialFrameId: 0,
      componentCount,
    });
  }
}
