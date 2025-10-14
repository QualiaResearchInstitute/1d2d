import { useEffect, useRef } from 'react';
import {
  BeamSplitterRenderer,
  type BeamSplitterRendererConfig,
} from '../optics/beamSplitterRenderer';
import { useGpuState } from '../state/AppState';

const CLEAR_COLOR = { r: 0.02, g: 0.06, b: 0.12, a: 1 };
const TARGET_REFRESH_HZ = 120;

const ensureCanvasSize = (canvas: HTMLCanvasElement) => {
  const dpr = window.devicePixelRatio ?? 1;
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
};

export function useViewportRenderer(
  canvas: HTMLCanvasElement | null,
  config: BeamSplitterRendererConfig | null,
) {
  const { setInitializing, setReady, setError, setTargetFrameRate, updateBeamSplitterDiagnostics } =
    useGpuState();
  const engineRef = useRef<BeamSplitterRenderer | null>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    if (!canvas) {
      return;
    }

    setInitializing();
    setTargetFrameRate(TARGET_REFRESH_HZ);

    const gl = canvas.getContext('webgl2', {
      antialias: true,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    if (!gl) {
      setError('WebGL2 context is unavailable');
      return;
    }

    gl.clearColor(CLEAR_COLOR.r, CLEAR_COLOR.g, CLEAR_COLOR.b, CLEAR_COLOR.a);
    ensureCanvasSize(canvas);

    const engine = new BeamSplitterRenderer(gl, (entries) =>
      updateBeamSplitterDiagnostics(entries),
    );
    engineRef.current = engine;
    setReady('webgl2');

    const resizeObserver = new ResizeObserver(() => ensureCanvasSize(canvas));
    resizeObserver.observe(canvas);
    resizeObserverRef.current = resizeObserver;

    const renderLoop = () => {
      ensureCanvasSize(canvas);
      if (engineRef.current) {
        engineRef.current.render(canvas.width, canvas.height);
      } else {
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clear(gl.COLOR_BUFFER_BIT);
      }
      rafRef.current = window.requestAnimationFrame(renderLoop);
    };

    renderLoop();

    return () => {
      window.cancelAnimationFrame(rafRef.current);
      resizeObserver.disconnect();
      engine.dispose();
      engineRef.current = null;
    };
  }, [
    canvas,
    setError,
    setInitializing,
    setReady,
    setTargetFrameRate,
    updateBeamSplitterDiagnostics,
  ]);

  useEffect(() => {
    if (!canvas || !engineRef.current) {
      return;
    }
    ensureCanvasSize(canvas);
    const engine = engineRef.current;
    const finalConfig: BeamSplitterRendererConfig = {
      canvasWidth: canvas.width,
      canvasHeight: canvas.height,
      nodeId: config?.nodeId,
      asset: config?.asset,
      pipeline: config?.pipeline,
      branches: config?.branches ?? [],
      recombine: config?.recombine ?? 'sum',
    };
    void engine.configure(finalConfig).catch((error: unknown) => {
      console.error('[gpu] Failed to configure beam splitter renderer', error);
    });
  }, [canvas, config]);
}
