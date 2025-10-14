import { useCallback } from 'react';
import { useAppState } from '../../state/AppState';
import { loadManifestFromFile, loadManifestFromJson } from '../../manifest/loader';
import type { ManifestLoadResult } from '../../manifest/loader';

export function useManifestLoader() {
  const { dispatch } = useAppState();

  const applyResult = useCallback(
    async (promise: Promise<ManifestLoadResult>, sourceName?: string) => {
      const result = await promise;
      if (result.kind === 'success') {
        dispatch({
          type: 'manifest/loadSuccess',
          scene: result.scene,
          timeline: result.timeline,
          path: sourceName ?? result.sourceName,
          metadata: result.manifest.metadata,
          issues: result.issues,
          controls: result.controls,
        });
      } else {
        dispatch({
          type: 'manifest/loadError',
          message: result.message,
          path: sourceName ?? result.sourceName,
          issues: result.issues,
        });
      }
    },
    [dispatch],
  );

  const loadFromFile = useCallback(
    async (file: File) => {
      dispatch({ type: 'manifest/loadStart', path: file.name });
      await applyResult(loadManifestFromFile(file), file.name);
    },
    [applyResult, dispatch],
  );

  const loadFromText = useCallback(
    async (json: string, sourceName?: string) => {
      dispatch({ type: 'manifest/loadStart', path: sourceName });
      await applyResult(loadManifestFromJson(json, sourceName), sourceName);
    },
    [applyResult, dispatch],
  );

  const loadFromUrl = useCallback(
    async (url: string) => {
      dispatch({ type: 'manifest/loadStart', path: url });
      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to fetch manifest: ${response.status} ${response.statusText}`);
        }
        const text = await response.text();
        await applyResult(loadManifestFromJson(text, url), url);
      } catch (error) {
        dispatch({
          type: 'manifest/loadError',
          message: error instanceof Error ? error.message : String(error),
          path: url,
        });
      }
    },
    [applyResult, dispatch],
  );

  return {
    loadFromFile,
    loadFromText,
    loadFromUrl,
  };
}
