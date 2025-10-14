import { createRuntimeBundle } from './runtime.js';
import { ManifestValidationError, validateManifest } from './schema.js';
const readFile = (file) =>
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
export async function loadManifestFromJson(json, sourceName) {
  let parsed;
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
export async function loadManifestFromFile(file) {
  const json = await readFile(file);
  return loadManifestFromJson(json, file.name);
}
