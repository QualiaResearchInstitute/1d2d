import { OpticalFieldManager, type OpticalFieldFrame } from "../fields/opticalField.js";

export type AngularSpectrumConfig = {
  width: number;
  height: number;
  wavelengthNm: number;
  pixelPitchMeters: number;
  dzMeters: number;
};

export class AngularSpectrumSolver {
  private readonly manager: OpticalFieldManager;
  private readonly config: AngularSpectrumConfig;

  constructor(config: AngularSpectrumConfig) {
    this.config = config;
    this.manager = new OpticalFieldManager({
      solver: "angularSpectrum",
      solverInstanceId: "angularSpectrum-main",
      resolution: { width: config.width, height: config.height },
      defaultWavelengthNm: config.wavelengthNm,
      defaultPixelPitchMeters: config.pixelPitchMeters,
      defaultDt: config.dzMeters,
      initialFrameId: 0
    });
  }

  propagate(input: OpticalFieldFrame, options?: { dzMeters?: number; timestamp?: number }): OpticalFieldFrame {
    const frame = this.manager.acquireFrame({
      dt: options?.dzMeters ?? this.config.dzMeters,
      timestamp: options?.timestamp
    });
    frame.real.set(input.real);
    frame.imag.set(input.imag);
    const shift = (options?.dzMeters ?? this.config.dzMeters) * 0.1;
    if (Math.abs(shift) > 1e-12) {
      const cos = Math.cos(shift);
      const sin = Math.sin(shift);
      for (let i = 0; i < frame.real.length; i++) {
        const r = frame.real[i];
        const im = frame.imag[i];
        frame.real[i] = r * cos - im * sin;
        frame.imag[i] = r * sin + im * cos;
      }
    }
    this.manager.stampFrame(frame, {
      dt: options?.dzMeters ?? this.config.dzMeters,
      timestamp: options?.timestamp
    });
    return frame;
  }

  alignPhase(frame: OpticalFieldFrame, anchorIndex: number, referencePhase: number) {
    this.manager.alignPhase(frame, { anchorIndex, referencePhase });
  }

  getManager() {
    return this.manager;
  }
}
