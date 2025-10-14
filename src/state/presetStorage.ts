import type { ManifestMetadata } from '../manifest/types.js';
import type { PresetDefinition } from './types.js';

const STORAGE_PREFIX = 'indra-presets';

const hasStorage = (): boolean => typeof window !== 'undefined' && !!window.localStorage;

const toStorageKey = (metadata: ManifestMetadata | undefined): string => {
  const name = metadata?.name?.trim() ?? 'manifest';
  const version = metadata?.version?.trim() ?? '0';
  return `${STORAGE_PREFIX}::${name}::${version}`;
};

type StoredPreset = {
  readonly id: string;
  readonly label: string;
  readonly description?: string;
  readonly panels: Record<string, Record<string, unknown>>;
  readonly createdAt: number;
};

type StoredPayload = {
  readonly presets: StoredPreset[];
};

export const loadUserPresets = (metadata: ManifestMetadata | undefined): PresetDefinition[] => {
  if (!hasStorage()) {
    return [];
  }
  try {
    const key = toStorageKey(metadata);
    const raw = window.localStorage.getItem(key);
    if (!raw) {
      return [];
    }
    const stored = JSON.parse(raw) as StoredPayload;
    if (!stored || !Array.isArray(stored.presets)) {
      return [];
    }
    return stored.presets.map((preset) => ({
      id: preset.id,
      label: preset.label,
      description: preset.description,
      panels: preset.panels,
      createdAt: preset.createdAt,
      thumbnail: undefined,
      kind: 'user',
    }));
  } catch (error) {
    console.warn('[presets] failed to load user presets from storage', error);
    return [];
  }
};

export const saveUserPresets = (
  metadata: ManifestMetadata | undefined,
  presets: PresetDefinition[],
) => {
  if (!hasStorage()) {
    return;
  }
  try {
    const key = toStorageKey(metadata);
    const payload: StoredPayload = {
      presets: presets
        .filter((preset) => preset.kind === 'user')
        .map((preset) => ({
          id: preset.id,
          label: preset.label,
          description: preset.description,
          panels: preset.panels ?? {},
          createdAt: preset.createdAt,
        })),
    };
    window.localStorage.setItem(key, JSON.stringify(payload));
  } catch (error) {
    console.warn('[presets] failed to persist user presets', error);
  }
};
