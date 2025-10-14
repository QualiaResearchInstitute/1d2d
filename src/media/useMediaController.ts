import { useCallback, useMemo } from 'react';
import { useMediaLibrary, useSceneGraph } from '../state/AppState';
import type {
  MediaAsset,
  MediaAssetKind,
  MediaProcessingStatus,
  MediaTelemetryEntry,
} from '../state/types';
import {
  runMediaPipeline,
  type MediaPipelineOptions,
  type MediaPipelineResult,
} from './mediaPipeline.js';
import { storeMediaResult, removeMediaResult, getMediaResult } from './cache.js';

const MAX_MEDIA_PIXELS = 3840 * 2160;
const THUMBNAIL_SIZE = 256;

const now = () =>
  typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();

const detectMediaKind = (file: File): MediaAssetKind => {
  if (file.type.startsWith('image/')) {
    return 'image';
  }
  if (file.type.startsWith('video/')) {
    return 'video';
  }
  throw new Error(`Unsupported media type: ${file.type || 'unknown'}`);
};

const createAssetId = () => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `media-${Math.random().toString(36).slice(2, 10)}`;
};

const clampPixels = (width: number, height: number) => {
  const pixels = width * height;
  if (pixels <= MAX_MEDIA_PIXELS) {
    return { width, height, scale: 1 };
  }
  const scale = Math.sqrt(MAX_MEDIA_PIXELS / pixels);
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
    scale,
  };
};

const loadImageElement = (url: string): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Failed to decode image'));
    image.src = url;
  });

const loadVideoElement = (url: string): Promise<HTMLVideoElement> =>
  new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'auto';
    video.crossOrigin = 'anonymous';
    video.muted = true;
    const handleLoaded = () => resolve(video);
    const handleError = () => reject(new Error('Failed to decode video'));
    video.addEventListener('loadeddata', handleLoaded, { once: true });
    video.addEventListener('error', handleError, { once: true });
    video.src = url;
  });

const createCanvas = (width: number, height: number): HTMLCanvasElement => {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
};

const renderToImageData = (
  element: CanvasImageSource,
  width: number,
  height: number,
): ImageData => {
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('Canvas 2D context unavailable');
  }
  ctx.drawImage(element, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
};

const createThumbnail = (element: CanvasImageSource, width: number, height: number): string => {
  const aspect = width / height;
  let thumbWidth = THUMBNAIL_SIZE;
  let thumbHeight = THUMBNAIL_SIZE;
  if (aspect >= 1) {
    thumbHeight = Math.max(1, Math.round(THUMBNAIL_SIZE / aspect));
  } else {
    thumbWidth = Math.max(1, Math.round(THUMBNAIL_SIZE * aspect));
  }
  const canvas = createCanvas(thumbWidth, thumbHeight);
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Canvas 2D context unavailable');
  }
  ctx.drawImage(element, 0, 0, thumbWidth, thumbHeight);
  return canvas.toDataURL('image/png', 0.85);
};

type DecodedMedia =
  | {
      kind: 'image';
      buffer: { data: Uint8ClampedArray; width: number; height: number };
      originalWidth: number;
      originalHeight: number;
      previewUrl: string;
      downscale: number;
    }
  | {
      kind: 'video';
      buffer: { data: Uint8ClampedArray; width: number; height: number };
      originalWidth: number;
      originalHeight: number;
      previewUrl: string;
      downscale: number;
      duration: number;
    };

const decodeMedia = async (
  file: File,
  sourceUrl: string,
  kind: MediaAssetKind,
): Promise<DecodedMedia> => {
  if (kind === 'image') {
    const image = await loadImageElement(sourceUrl);
    const {
      width: scaledWidth,
      height: scaledHeight,
      scale,
    } = clampPixels(image.naturalWidth, image.naturalHeight);
    const imageData = renderToImageData(image, scaledWidth, scaledHeight);
    const previewUrl = createThumbnail(image, image.naturalWidth, image.naturalHeight);
    return {
      kind: 'image',
      buffer: { data: imageData.data, width: imageData.width, height: imageData.height },
      originalWidth: image.naturalWidth,
      originalHeight: image.naturalHeight,
      previewUrl,
      downscale: scale,
    };
  }

  const video = await loadVideoElement(sourceUrl);
  await new Promise<void>((resolve, reject) => {
    const handleSeeked = () => resolve();
    const handleError = () => reject(new Error('Video seek failed'));
    video.addEventListener('seeked', handleSeeked, { once: true });
    video.addEventListener('error', handleError, { once: true });
    try {
      video.currentTime = 0;
    } catch (error) {
      reject(error instanceof Error ? error : new Error('Video seek threw an unknown error'));
    }
  });
  const {
    width: scaledWidth,
    height: scaledHeight,
    scale,
  } = clampPixels(video.videoWidth, video.videoHeight);
  const frameData = renderToImageData(video, scaledWidth, scaledHeight);
  const previewUrl = createThumbnail(video, video.videoWidth, video.videoHeight);
  return {
    kind: 'video',
    buffer: { data: frameData.data, width: frameData.width, height: frameData.height },
    originalWidth: video.videoWidth,
    originalHeight: video.videoHeight,
    previewUrl,
    downscale: scale,
    duration: Number.isFinite(video.duration) ? video.duration : 0,
  };
};

const initialProcessingStatus = (assetId: string): MediaProcessingStatus => ({
  assetId,
  stage: 'loading',
  progress: 0.05,
  message: 'Decoding media…',
  startedAt: Date.now(),
});

const updateStatus = (
  setStatus: (status: MediaProcessingStatus) => void,
  status: MediaProcessingStatus,
  patch: Partial<MediaProcessingStatus>,
) => {
  setStatus({
    ...status,
    ...patch,
    startedAt: status.startedAt ?? Date.now(),
  });
};

const toTelemetryEntry = (
  asset: MediaAsset,
  pipeline: MediaPipelineResult,
  loadMs: number,
): MediaTelemetryEntry => {
  const durations = {
    loadMs,
    ...pipeline.telemetry.durations,
  };
  return {
    assetId: asset.id,
    timestamp: Date.now(),
    durations,
    metrics: pipeline.telemetry.metrics,
    notes:
      pipeline.kuramoto &&
      pipeline.kuramoto.deterministic &&
      !pipeline.kuramoto.deterministic.verified
        ? 'Kuramoto determinism check exceeded tolerance'
        : undefined,
  };
};

export function useMediaController() {
  const {
    media,
    registerAsset,
    updateAsset,
    removeAsset,
    selectAsset,
    setProcessingStatus,
    recordTelemetry,
  } = useMediaLibrary();
  const { scene, addNode, removeNode } = useSceneGraph();

  const importMedia = useCallback(
    async (files: readonly File[], pipelineOptions?: MediaPipelineOptions) => {
      for (const file of files) {
        const kind = detectMediaKind(file);
        const assetId = createAssetId();
        const sourceUrl = URL.createObjectURL(file);
        const importedAt = Date.now();
        const asset: MediaAsset = {
          id: assetId,
          name: file.name || `Media ${importedAt}`,
          kind,
          sourceUrl,
          importedAt,
          status: 'pending',
          metadata: {
            fileSizeBytes: file.size,
            mimeType: file.type,
          },
        };
        registerAsset(asset);
        selectAsset(assetId);

        let status = initialProcessingStatus(assetId);
        setProcessingStatus(status);
        const decodeStart = now();
        try {
          const decoded = await decodeMedia(file, sourceUrl, kind);
          const decodeEnd = now();
          const loadDuration = decodeEnd - decodeStart;
          status = {
            ...status,
            stage: 'edge',
            progress: 0.35,
            message: 'Computing edge map…',
          };
          setProcessingStatus(status);
          const pipelineResult = runMediaPipeline(decoded.buffer, {
            ...pipelineOptions,
            onStage: (stage) => {
              if (stage === 'phase') {
                status = {
                  ...status,
                  stage: 'phase',
                  progress: 0.6,
                  message: 'Deriving phase field…',
                };
                setProcessingStatus(status);
              } else if (stage === 'kuramoto') {
                status = {
                  ...status,
                  stage: 'kuramoto',
                  progress: 0.85,
                  message: 'Integrating Kuramoto lattice…',
                };
                setProcessingStatus(status);
              }
            },
          });
          storeMediaResult(assetId, pipelineResult);

          const telemetryEntry = toTelemetryEntry(asset, pipelineResult, loadDuration);
          recordTelemetry(telemetryEntry);

          updateAsset(assetId, {
            status: 'ready',
            width: decoded.originalWidth,
            height: decoded.originalHeight,
            previewUrl: decoded.previewUrl,
            metadata: {
              ...asset.metadata,
              scaledWidth: decoded.buffer.width,
              scaledHeight: decoded.buffer.height,
              downscale: decoded.downscale,
              durationSeconds: decoded.kind === 'video' ? decoded.duration : undefined,
              phaseMetrics: pipelineResult.phase.metrics,
              kuramotoDeterminism: pipelineResult.kuramoto?.deterministic,
            },
          });

          const nodePosition = {
            x: 120 + scene.nodes.length * 40,
            y: 120 + (scene.nodes.length % 4) * 60,
          };
          addNode({
            id: `media-${assetId}`,
            type: kind === 'image' ? 'ImageSource' : 'VideoSource',
            label: asset.name,
            position: nodePosition,
            parameters: [
              {
                id: 'resolution',
                label: 'Resolution',
                value: {
                  kind: 'vector',
                  size: 2,
                  values: [decoded.buffer.width, decoded.buffer.height],
                },
              },
              {
                id: 'edgePixels',
                label: 'Edge Pixels',
                value: pipelineResult.telemetry.metrics.edgePixelCount ?? 0,
              },
              {
                id: 'phaseVariance',
                label: 'Phase Variance',
                value: pipelineResult.telemetry.metrics.phaseVariance ?? 0,
              },
              ...(decoded.kind === 'video'
                ? [
                    {
                      id: 'duration',
                      label: 'Duration (s)',
                      value: Number(decoded.duration.toFixed(2)),
                    },
                  ]
                : []),
            ],
          });

          status = {
            ...status,
            stage: 'complete',
            progress: 1,
            message: 'Media pipeline ready',
            finishedAt: Date.now(),
          };
          setProcessingStatus(status);
          if (typeof window !== 'undefined') {
            window.setTimeout(() => {
              setProcessingStatus({
                stage: 'idle',
                progress: 0,
              });
            }, 1800);
          }
        } catch (error) {
          console.error('[media] failed to ingest media', error);
          setProcessingStatus({
            assetId,
            stage: 'error',
            progress: 1,
            message: error instanceof Error ? error.message : 'Unknown media processing error',
            startedAt: status.startedAt ?? Date.now(),
            finishedAt: Date.now(),
            error: error instanceof Error ? error.message : String(error),
          });
          updateAsset(assetId, {
            status: 'error',
            errorMessage: error instanceof Error ? error.message : String(error),
          });
          URL.revokeObjectURL(sourceUrl);
          removeMediaResult(assetId);
        }
      }
    },
    [
      addNode,
      recordTelemetry,
      registerAsset,
      scene.nodes.length,
      selectAsset,
      setProcessingStatus,
      updateAsset,
    ],
  );

  const deleteMedia = useCallback(
    (assetId: string) => {
      const asset = media.assets.find((entry) => entry.id === assetId);
      if (asset?.sourceUrl && asset.sourceUrl.startsWith('blob:')) {
        URL.revokeObjectURL(asset.sourceUrl);
      }
      removeMediaResult(assetId);
      removeAsset(assetId);
      removeNode(`media-${assetId}`);
    },
    [media.assets, removeAsset, removeNode],
  );

  return useMemo(
    () => ({
      media,
      importMedia,
      deleteMedia,
      selectAsset,
      getPipelineResult: getMediaResult,
    }),
    [deleteMedia, importMedia, media, selectAsset],
  );
}
