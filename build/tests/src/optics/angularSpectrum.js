import { OpticalFieldManager } from '../fields/opticalField.js';
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
        Object.defineProperty(this, "componentCount", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        this.config = config;
        this.componentCount = Math.max(1, config.componentCount ?? 1);
        this.manager = this.createManager(this.componentCount);
    }
    propagate(input, options) {
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
            const source = input.components[componentIndex] ?? input.components[0];
            const target = frame.components[componentIndex];
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
    alignPhase(frame, anchorIndex, referencePhase) {
        this.manager.alignPhase(frame, { anchorIndex, referencePhase });
    }
    getManager() {
        return this.manager;
    }
    createManager(componentCount) {
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
