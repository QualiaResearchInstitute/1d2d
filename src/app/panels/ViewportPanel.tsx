import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { PanelFrame } from '../layout/PanelFrame';
import { useViewportRenderer } from '../../gpu/useViewportRenderer';
import { useMediaController } from '../../media/useMediaController';
import { useSceneGraph } from '../../state/AppState';
import { parseBranches } from '../../state/beamSplitter';
import type { BeamSplitterRendererConfig } from '../../optics/beamSplitterRenderer';
import type { PhaseField, RimField } from '../../fields/contracts';
import { useI18n } from '../../i18n/LocalizationProvider';

type OverlayMode = 'source' | 'edges' | 'phase';

const createCanvas = (width: number, height: number): HTMLCanvasElement | null => {
  if (typeof document === 'undefined') {
    return null;
  }
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
};

const createEdgeOverlay = (rim: RimField): string | null => {
  const { width, height } = rim.resolution;
  const canvas = createCanvas(width, height);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  const imageData = ctx.createImageData(width, height);
  const { data } = imageData;
  const mag = rim.mag;
  for (let i = 0; i < mag.length; i++) {
    const value = Math.max(0, Math.min(255, Math.round(mag[i] * 255)));
    const idx = i * 4;
    data[idx] = value;
    data[idx + 1] = value;
    data[idx + 2] = value;
    data[idx + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
};

const createPhaseOverlay = (phase: PhaseField): string | null => {
  const { width, height } = phase.resolution;
  const canvas = createCanvas(width, height);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  const imageData = ctx.createImageData(width, height);
  const { data } = imageData;
  const amp = phase.amp;
  const vort = phase.vort;
  for (let i = 0; i < amp.length; i++) {
    const a = Math.max(0, Math.min(1, amp[i]));
    const v = Math.max(-1, Math.min(1, vort[i]));
    const idx = i * 4;
    const r = Math.round(a * 255);
    const g = Math.round((1 - Math.abs(v)) * 255);
    const b = Math.round((1 - a) * 255);
    data[idx] = r;
    data[idx + 1] = g;
    data[idx + 2] = b;
    data[idx + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
};

export interface ViewportPanelProps {
  readonly onCanvasReady?: (canvas: HTMLCanvasElement) => void;
}

export function ViewportPanel({ onCanvasReady }: ViewportPanelProps) {
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('source');
  const [edgeUrl, setEdgeUrl] = useState<string | null>(null);
  const [phaseUrl, setPhaseUrl] = useState<string | null>(null);
  const { media, getPipelineResult, selectAsset } = useMediaController();
  const { scene } = useSceneGraph();
  const { t } = useI18n();

  useEffect(() => {
    if (canvas && onCanvasReady) {
      onCanvasReady(canvas);
    }
  }, [canvas, onCanvasReady]);

  const handleCanvasRef = useCallback((element: HTMLCanvasElement | null) => {
    setCanvas(element);
  }, []);

  const selectedAsset = useMemo(
    () => media.assets.find((asset) => asset.id === media.selectedAssetId),
    [media.assets, media.selectedAssetId],
  );

  const beamSplitterNode = useMemo(
    () => scene.nodes.find((node) => node.type === 'BeamSplitter'),
    [scene.nodes],
  );

  const branchConfig = useMemo((): BeamSplitterRendererConfig['branches'] => {
    if (!beamSplitterNode) {
      return [];
    }
    const branches = parseBranches(beamSplitterNode.metadata ?? {});
    return branches.map((branch) => ({
      id: branch.id,
      label: branch.label,
      weight: branch.weight ?? 1,
      priority: branch.priority ?? 0,
      source: branch.source ?? 'source',
      transformStack: branch.transformStack.map((step) => ({ ...step })),
    }));
  }, [beamSplitterNode]);

  const recombineMode = useMemo(() => {
    const parameter = beamSplitterNode?.parameters.find((entry) => entry.id === 'recombine');
    return typeof parameter?.value === 'string' ? parameter.value : 'sum';
  }, [beamSplitterNode]);

  const pipelineResult = useMemo(
    () => (selectedAsset ? getPipelineResult(selectedAsset.id) : undefined),
    [getPipelineResult, selectedAsset],
  );

  const rendererConfig = useMemo<BeamSplitterRendererConfig | null>(() => {
    if (!selectedAsset) {
      return null;
    }
    const previewUrl = selectedAsset.previewUrl ?? selectedAsset.sourceUrl;
    if (!previewUrl) {
      return null;
    }
    return {
      canvasWidth: 0,
      canvasHeight: 0,
      nodeId: beamSplitterNode?.id,
      asset: {
        id: selectedAsset.id,
        previewUrl,
        width: selectedAsset.width ?? pipelineResult?.rim.resolution.width ?? 1,
        height: selectedAsset.height ?? pipelineResult?.rim.resolution.height ?? 1,
      },
      pipeline: pipelineResult,
      branches: branchConfig,
      recombine: recombineMode,
    };
  }, [selectedAsset, beamSplitterNode?.id, pipelineResult, branchConfig, recombineMode]);

  useViewportRenderer(canvas, rendererConfig);

  useEffect(() => {
    if (!selectedAsset || !pipelineResult) {
      setEdgeUrl(null);
      setPhaseUrl(null);
      return;
    }
    const edges = createEdgeOverlay(pipelineResult.rim);
    const phase = createPhaseOverlay(pipelineResult.phase.field);
    setEdgeUrl(edges);
    setPhaseUrl(phase);
  }, [pipelineResult, selectedAsset]);

  useEffect(() => {
    if (overlayMode === 'edges' && !edgeUrl) {
      setOverlayMode('source');
    } else if (overlayMode === 'phase' && !phaseUrl) {
      setOverlayMode('source');
    }
  }, [edgeUrl, phaseUrl, overlayMode]);

  useEffect(() => {
    setOverlayMode('source');
  }, [selectedAsset?.id]);

  const overlaySource =
    overlayMode === 'source'
      ? (selectedAsset?.previewUrl ?? selectedAsset?.sourceUrl ?? null)
      : overlayMode === 'edges'
        ? edgeUrl
        : phaseUrl;

  const overlayLabel =
    overlayMode === 'source'
      ? t('viewport.overlay.source')
      : overlayMode === 'edges'
        ? t('viewport.overlay.edges')
        : t('viewport.overlay.phase');

  return (
    <PanelFrame
      title={t('viewport.title')}
      footer={
        <div className="panel-footnote">
          <span>{t('viewport.footer')}</span>
        </div>
      }
    >
      <div className="viewport">
        <canvas
          ref={handleCanvasRef}
          className="viewport__canvas"
          role="img"
          aria-label={t('viewport.canvas.label')}
        />
        {selectedAsset && overlaySource ? (
          <div className="viewport__media">
            <img src={overlaySource} alt={`${selectedAsset.name} ${overlayLabel}`} />
          </div>
        ) : null}
        {selectedAsset && (
          <div className="viewport__controls">
            <span>{selectedAsset.name}</span>
            <div className="viewport__control-group">
              <button
                type="button"
                className={
                  overlayMode === 'source'
                    ? 'viewport__toggle viewport__toggle--active'
                    : 'viewport__toggle'
                }
                onClick={() => setOverlayMode('source')}
                aria-pressed={overlayMode === 'source'}
              >
                {t('viewport.overlay.source')}
              </button>
              <button
                type="button"
                className={
                  overlayMode === 'edges' && edgeUrl
                    ? 'viewport__toggle viewport__toggle--active'
                    : 'viewport__toggle'
                }
                onClick={() => edgeUrl && setOverlayMode('edges')}
                disabled={!edgeUrl}
                aria-pressed={overlayMode === 'edges'}
              >
                {t('viewport.overlay.edges')}
              </button>
              <button
                type="button"
                className={
                  overlayMode === 'phase' && phaseUrl
                    ? 'viewport__toggle viewport__toggle--active'
                    : 'viewport__toggle'
                }
                onClick={() => phaseUrl && setOverlayMode('phase')}
                disabled={!phaseUrl}
                aria-pressed={overlayMode === 'phase'}
              >
                {t('viewport.overlay.phase')}
              </button>
            </div>
            <button
              type="button"
              className="viewport__toggle viewport__toggle--link"
              onClick={() => selectedAsset && selectAsset(selectedAsset.id)}
            >
              {t('viewport.overlay.focus')}
            </button>
          </div>
        )}
        {!selectedAsset ? (
          <div className="viewport__placeholder">
            <p>{t('viewport.placeholder.idle')}</p>
            <p>{t('viewport.placeholder.load')}</p>
          </div>
        ) : null}
      </div>
    </PanelFrame>
  );
}
