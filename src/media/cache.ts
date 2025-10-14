import type { MediaPipelineResult } from './mediaPipeline';

const pipelineCache = new Map<string, MediaPipelineResult>();

export const storeMediaResult = (assetId: string, result: MediaPipelineResult): void => {
  pipelineCache.set(assetId, result);
};

export const getMediaResult = (assetId: string): MediaPipelineResult | undefined => {
  return pipelineCache.get(assetId);
};

export const removeMediaResult = (assetId: string): void => {
  pipelineCache.delete(assetId);
};

export const clearMediaCache = (): void => {
  pipelineCache.clear();
};
