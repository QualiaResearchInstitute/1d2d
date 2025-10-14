import React, { useCallback, useId, useMemo, useRef, useState } from 'react';
import { PanelFrame } from '../layout/PanelFrame';
import type { ManifestValidationIssue } from '../../manifest/types';
import { useMediaController } from '../../media/useMediaController';
import {
  useControlPanels,
  useManifestStatus,
  usePresetLibrary,
  useSceneGraph,
  useTimelineState,
} from '../../state/AppState';
import { serializeSceneManifest } from '../../manifest/serializer';
import { useI18n } from '../../i18n/LocalizationProvider';

export interface SamplePresetDefinition {
  readonly id: string;
  readonly label: string;
  readonly description?: string;
}

export interface PresetsPanelProps {
  readonly manifestName?: string;
  readonly manifestStatus: 'idle' | 'loading' | 'loaded' | 'error';
  readonly onManifestSelected: (file: File) => void;
  readonly onSampleSelected?: (presetId: string) => void;
  readonly samplePresets?: readonly SamplePresetDefinition[];
  readonly lastError?: string;
  readonly validationIssues?: readonly ManifestValidationIssue[];
}

export function PresetsPanel({
  manifestName,
  manifestStatus,
  onManifestSelected,
  onSampleSelected,
  samplePresets = [
    { id: 'phase-one-demo', label: 'Phase One Demo' },
    { id: 'hyperbolic-prototype', label: 'Hyperbolic Prototype' },
    { id: 'optics-blank', label: 'Optics Blank Slate' },
  ],
  lastError,
  validationIssues,
}: PresetsPanelProps) {
  const inputId = useId();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragActive, setDragActive] = useState(false);
  const { t } = useI18n();
  const { media, importMedia, deleteMedia, selectAsset } = useMediaController();
  const { presets, activePresetId, applyPreset, savePreset, deletePreset } = usePresetLibrary();
  const { controlsState } = useControlPanels();
  const manifestState = useManifestStatus();
  const { scene } = useSceneGraph();
  const { timeline } = useTimelineState();

  const builtinPresets = useMemo(
    () => presets.filter((preset) => preset.kind === 'builtin'),
    [presets],
  );
  const userPresets = useMemo(() => presets.filter((preset) => preset.kind === 'user'), [presets]);

  const handlePresetApply = useCallback(
    (presetId: string) => {
      applyPreset(presetId);
    },
    [applyPreset],
  );

  const handlePresetSave = useCallback(() => {
    if (manifestStatus !== 'loaded') {
      return;
    }
    const label =
      typeof window !== 'undefined'
        ? window.prompt(t('presetsPanel.prompt.name'), t('presetsPanel.prompt.name.placeholder'))
        : null;
    if (!label) {
      return;
    }
    const trimmed = label.trim();
    if (!trimmed) {
      return;
    }
    const description =
      typeof window !== 'undefined'
        ? (window.prompt(t('presetsPanel.prompt.description'), '') ?? undefined)
        : undefined;
    savePreset(trimmed, description?.trim() || undefined);
  }, [manifestStatus, savePreset, t]);

  const handlePresetDelete = useCallback(
    (presetId: string) => {
      deletePreset(presetId);
    },
    [deletePreset],
  );

  const handleExportManifest = useCallback(() => {
    if (manifestStatus !== 'loaded' || !manifestState.metadata) {
      return;
    }
    const manifest = serializeSceneManifest(
      scene,
      timeline,
      manifestState.metadata,
      controlsState,
      presets,
    );
    const json = JSON.stringify(manifest, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${manifestState.metadata.name?.replace(/\s+/g, '-') ?? 'scene'}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }, [controlsState, manifestState.metadata, manifestStatus, presets, scene, timeline]);

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const [file] = event.target.files ?? [];
      if (file) {
        onManifestSelected(file);
        event.target.value = '';
      }
    },
    [onManifestSelected],
  );

  const handleLoadClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleMediaInput = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = event.target.files ? Array.from(event.target.files).filter(Boolean) : [];
      if (files.length > 0) {
        void importMedia(files);
      }
      event.target.value = '';
    },
    [importMedia],
  );

  const handleMediaBrowse = useCallback(() => {
    mediaInputRef.current?.click();
  }, []);

  const handleDragOver = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      if (!isDragActive) {
        setDragActive(true);
      }
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = 'copy';
      }
    },
    [isDragActive],
  );

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      const items = Array.from(event.dataTransfer?.files ?? []);
      const files = items.filter(
        (file) => file.type.startsWith('image/') || file.type.startsWith('video/'),
      );
      if (files.length > 0) {
        void importMedia(files);
      }
      setDragActive(false);
    },
    [importMedia],
  );

  const warnings = useMemo(
    () => validationIssues?.filter((issue) => issue.severity === 'warning') ?? [],
    [validationIssues],
  );

  const handleDropZoneKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        handleMediaBrowse();
      }
    },
    [handleMediaBrowse],
  );

  const processing = media.processing.stage !== 'idle' ? media.processing : undefined;
  const processingPercent =
    processing && Number.isFinite(processing.progress) ? Math.round(processing.progress * 100) : 0;

  return (
    <PanelFrame
      title={t('presetsPanel.title')}
      footer={
        <div className="panel-footnote" aria-live="polite">
          {manifestStatus === 'loading' && <span>{t('presetsPanel.loading')}</span>}
          {manifestStatus === 'loaded' && manifestName && (
            <span>{t('presetsPanel.loaded', { values: { name: manifestName } })}</span>
          )}
          {manifestStatus === 'error' && lastError && (
            <span className="panel-footnote--error">
              {t('presetsPanel.error', { values: { message: lastError } })}
            </span>
          )}
          {manifestStatus === 'idle' && <span>{t('presetsPanel.idle')}</span>}
          {processing ? (
            <span>
              {t('presetsPanel.media.processing', {
                values: {
                  message: processing.message ?? t('presetsPanel.media.processing.default'),
                  progress: processingPercent,
                },
              })}
            </span>
          ) : (
            <span>
              {t(
                media.assets.length === 1
                  ? 'presetsPanel.media.count'
                  : 'presetsPanel.media.count.plural',
                {
                  values: { count: media.assets.length },
                },
              )}
            </span>
          )}
        </div>
      }
    >
      <div className="stack gap-sm">
        <button
          className="primary-button"
          type="button"
          onClick={handleLoadClick}
          disabled={manifestStatus === 'loading'}
        >
          {t('presetsPanel.loadManifest')}
        </button>
        <input
          ref={fileInputRef}
          id={inputId}
          className="hidden-input"
          type="file"
          accept="application/json"
          onChange={handleFileChange}
        />
        <div className="panel-section">
          <h3>{t('presetsPanel.mediaInput.title')}</h3>
          <div
            className={isDragActive ? 'media-dropzone media-dropzone--active' : 'media-dropzone'}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onKeyDown={handleDropZoneKeyDown}
            role="button"
            tabIndex={0}
            aria-label={t('presetsPanel.mediaInput.drop.label')}
          >
            <p>{t('presetsPanel.mediaInput.drop')}</p>
            <button className="ghost-button" type="button" onClick={handleMediaBrowse}>
              {t('presetsPanel.mediaInput.browse')}
            </button>
            <input
              ref={mediaInputRef}
              className="hidden-input"
              type="file"
              accept="image/*,video/*"
              multiple
              onChange={handleMediaInput}
            />
          </div>
          {media.assets.length === 0 ? (
            <p className="panel-muted">{t('presetsPanel.mediaInput.empty')}</p>
          ) : (
            <ul className="media-gallery">
              {media.assets.map((asset) => {
                const isSelected = media.selectedAssetId === asset.id;
                const inFlight =
                  media.processing.assetId === asset.id && media.processing.stage !== 'complete';
                return (
                  <li
                    key={asset.id}
                    className={isSelected ? 'media-card media-card--active' : 'media-card'}
                  >
                    <button
                      type="button"
                      className="media-card__body"
                      onClick={() => selectAsset(asset.id)}
                      title={asset.name}
                    >
                      <div className="media-card__thumbnail">
                        {asset.previewUrl || asset.sourceUrl ? (
                          <img src={asset.previewUrl ?? asset.sourceUrl} alt={asset.name} />
                        ) : (
                          <span className="media-card__placeholder">
                            {t('presetsPanel.media.noPreview')}
                          </span>
                        )}
                        {inFlight ? (
                          <span className="media-card__badge">
                            {t('presetsPanel.media.processing.badge')}
                          </span>
                        ) : null}
                        {asset.status === 'error' ? (
                          <span className="media-card__badge media-card__badge--error">
                            {t('presetsPanel.media.error')}
                          </span>
                        ) : null}
                      </div>
                      <span className="media-card__label">{asset.name}</span>
                      {asset.width && asset.height ? (
                        <span className="media-card__meta">
                          {asset.width}×{asset.height}
                        </span>
                      ) : null}
                    </button>
                    <button
                      className="media-card__remove"
                      type="button"
                      onClick={() => deleteMedia(asset.id)}
                    >
                      {t('presetsPanel.media.remove')}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
        <div className="panel-section">
          <div className="panel-section__header">
            <h3>{t('presetsPanel.gallery.title')}</h3>
            <div className="panel-section__actions">
              <button
                className="ghost-button"
                type="button"
                onClick={handlePresetSave}
                disabled={manifestStatus !== 'loaded'}
              >
                {t('presetsPanel.gallery.save')}
              </button>
              <button
                className="ghost-button"
                type="button"
                onClick={handleExportManifest}
                disabled={manifestStatus !== 'loaded'}
              >
                {t('presetsPanel.gallery.export')}
              </button>
            </div>
          </div>
          {presets.length === 0 ? (
            <p className="panel-muted">{t('presetsPanel.gallery.empty')}</p>
          ) : (
            <div className="preset-gallery">
              {builtinPresets.length > 0 ? (
                <div className="preset-group">
                  <h4>{t('presetsPanel.gallery.builtin')}</h4>
                  <div className="preset-grid">
                    {builtinPresets.map((preset) => {
                      const isActive = preset.id === activePresetId;
                      return (
                        <div
                          key={preset.id}
                          className={isActive ? 'preset-card preset-card--active' : 'preset-card'}
                        >
                          <button
                            type="button"
                            onClick={() => handlePresetApply(preset.id)}
                            disabled={manifestStatus !== 'loaded'}
                          >
                            <span className="preset-card__label">{preset.label}</span>
                            {preset.description ? (
                              <span className="preset-card__description">{preset.description}</span>
                            ) : null}
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : null}
              {userPresets.length > 0 ? (
                <div className="preset-group">
                  <h4>{t('presetsPanel.gallery.saved')}</h4>
                  <div className="preset-grid">
                    {userPresets.map((preset) => {
                      const isActive = preset.id === activePresetId;
                      return (
                        <div
                          key={preset.id}
                          className={isActive ? 'preset-card preset-card--active' : 'preset-card'}
                        >
                          <button
                            type="button"
                            onClick={() => handlePresetApply(preset.id)}
                            disabled={manifestStatus !== 'loaded'}
                          >
                            <span className="preset-card__label">{preset.label}</span>
                            {preset.description ? (
                              <span className="preset-card__description">{preset.description}</span>
                            ) : null}
                          </button>
                          <button
                            type="button"
                            className="preset-card__remove"
                            onClick={() => handlePresetDelete(preset.id)}
                            disabled={manifestStatus !== 'loaded'}
                          >
                            {t('presetsPanel.media.remove')}
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : null}
            </div>
          )}
        </div>
        <div className="panel-section">
          <h3>{t('presetsPanel.examples.title')}</h3>
          <ul>
            {samplePresets.map((preset) => (
              <li key={preset.id}>
                <button
                  className="ghost-button"
                  type="button"
                  onClick={() => onSampleSelected?.(preset.id)}
                  disabled={!onSampleSelected}
                  title={preset.description}
                >
                  {preset.label}
                </button>
              </li>
            ))}
          </ul>
        </div>
        {warnings.length > 0 ? (
          <div className="panel-section">
            <h3>{t('presetsPanel.validation.title')}</h3>
            <ul>
              {warnings.slice(0, 3).map((issue) => (
                <li key={`${issue.code}-${issue.path.join('.')}`} className="panel-muted">
                  {issue.message}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </PanelFrame>
  );
}
