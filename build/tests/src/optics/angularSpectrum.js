import { OpticalFieldManager } from "../fields/opticalField.js";
export class AngularSpectrumSolver {
    constructor(config) {
        Object.defineProperty(this, "manager", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "config", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
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
    propagate(input, options) {
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
    alignPhase(frame, anchorIndex, referencePhase) {
        this.manager.alignPhase(frame, { anchorIndex, referencePhase });
    }
    getManager() {
        return this.manager;
    }
}
