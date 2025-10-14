import { createRuntimeBundle, type ManifestRuntimeBundle } from './runtime.js';
import { ManifestValidationError, validateManifest } from './schema.js';
import type { ManifestValidationResult, SceneManifest } from './types.js';

export interface ManifestLoadSuccess extends ManifestRuntimeBundle {
  readonly manifest: SceneManifest;
  readonly issues: ManifestValidationResult['issues'];
  readonly sourceName?: string;
}

export type ManifestLoadResult =
  | ({ readonly kind: 'success' } & ManifestLoadSuccess)
  | {
      readonly kind: 'error';
      readonly message: string;
      readonly issues: ManifestValidationError['issues'] | undefined;
      readonly sourceName?: string;
    };

const readFile = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener('error', () => {
      reject(reader.error ?? new Error('Unknown file read error'));
    });
    reader.addEventListener('load', () => {
      resolve(typeof reader.result === 'string' ? reader.result : '');
    });
    reader.readAsText(file);
  });

export async function loadManifestFromJson(
  json: string,
  sourceName?: string,
): Promise<ManifestLoadResult> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch (error) {
    return {
      kind: 'error',
      message: error instanceof Error ? error.message : 'Failed to parse JSON manifest',
      issues: undefined,
      sourceName,
    };
  }

  try {
    const { manifest, issues } = validateManifest(parsed);
    const bundle = createRuntimeBundle(manifest);
    return {
      kind: 'success',
      manifest,
      issues,
      sourceName,
      ...bundle,
    };
  } catch (error) {
    if (error instanceof ManifestValidationError) {
      return {
        kind: 'error',
        message: error.message,
        issues: error.issues,
        sourceName,
      };
    }
    return {
      kind: 'error',
      message: error instanceof Error ? error.message : 'Unknown manifest validation error',
      issues: undefined,
      sourceName,
    };
  }
}

export async function loadManifestFromFile(file: File): Promise<ManifestLoadResult> {
  const json = await readFile(file);
  return loadManifestFromJson(json, file.name);
}
