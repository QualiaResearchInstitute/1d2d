import React, { useCallback, useEffect, useMemo } from 'react';
import {
  useGpuState,
  useManifestStatus,
  usePanelLayout,
  useSceneGraph,
  useHistoryController,
} from '../../state/AppState';
import { PresetsPanel } from '../panels/PresetsPanel';
import { InspectorPanel } from '../panels/InspectorPanel';
import { TimelinePanel } from '../panels/TimelinePanel';
import { ViewportPanel } from '../panels/ViewportPanel';
import { useResizeHandle } from './useResizeHandle';
import { useManifestLoader } from '../hooks/useManifestLoader';
import { useI18n, type Locale } from '../../i18n/LocalizationProvider';
import { useAccessibilityPreferences } from '../providers/AccessibilityPreferences';

const formatFileName = (path?: string) => {
  if (!path) {
    return undefined;
  }
  const segments = path.split(/[\\/]/);
  return segments[segments.length - 1] || path;
};

export function AppLayout() {
  const { t, locale, setLocale, locales } = useI18n();
  const { highContrast, setHighContrast, largeText, setLargeText } = useAccessibilityPreferences();
  const presetsPanel = usePanelLayout('presets');
  const inspectorPanel = usePanelLayout('inspector');
  const timelinePanel = usePanelLayout('timeline');
  const manifest = useManifestStatus();
  const { scene } = useSceneGraph();
  const { loadFromFile, loadFromUrl } = useManifestLoader();
  const { gpu } = useGpuState();
  const { canUndo, canRedo, undo, redo } = useHistoryController();

  const samplePresets = useMemo(
    () => [
      {
        id: 'phase-one-demo',
        label: t('presetsPanel.preset.sample.phaseOne'),
        description: t('presetsPanel.preset.sample.phaseOne.description'),
      },
    ],
    [t],
  );

  const samplePresetSources = useMemo(
    () => new Map<string, string>([['phase-one-demo', '/sample-manifest.json']]),
    [],
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      const isMeta = event.metaKey || event.ctrlKey;
      if (!isMeta) {
        return;
      }
      if (key === 'z' && !event.shiftKey) {
        event.preventDefault();
        if (canUndo) {
          undo();
        }
      } else if ((key === 'z' && event.shiftKey) || key === 'y') {
        event.preventDefault();
        if (canRedo) {
          redo();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown, { passive: false });
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [canRedo, canUndo, redo, undo]);

  const handleManifestSelected = useCallback(
    (file: File) => {
      void loadFromFile(file);
    },
    [loadFromFile],
  );

  const handleSampleSelected = useCallback(
    (presetId: string) => {
      const source = samplePresetSources.get(presetId);
      if (!source) {
        return;
      }
      void loadFromUrl(source);
    },
    [loadFromUrl, samplePresetSources],
  );

  const presetsResizeHandle = useResizeHandle({
    orientation: 'vertical',
    direction: 'positive',
    size: presetsPanel.layout.size,
    minSize: presetsPanel.layout.minSize,
    maxSize: presetsPanel.layout.maxSize,
    onSizeChange: presetsPanel.setSize,
  });

  const inspectorResizeHandle = useResizeHandle({
    orientation: 'vertical',
    direction: 'negative',
    size: inspectorPanel.layout.size,
    minSize: inspectorPanel.layout.minSize,
    maxSize: inspectorPanel.layout.maxSize,
    onSizeChange: inspectorPanel.setSize,
  });

  const timelineResizeHandle = useResizeHandle({
    orientation: 'horizontal',
    direction: 'negative',
    size: timelinePanel.layout.size,
    minSize: timelinePanel.layout.minSize,
    maxSize: timelinePanel.layout.maxSize,
    onSizeChange: timelinePanel.setSize,
  });

  return (
    <div className="app-shell">
      <header className="app-toolbar">
        <div className="app-toolbar__group">
          <h1>{t('app.title')}</h1>
          <span className="app-toolbar__tag">{t('app.phaseTag')}</span>
        </div>
        <nav className="app-toolbar__toggles" aria-label={t('toolbar.navigation')}>
          <button
            type="button"
            className={
              presetsPanel.layout.visible
                ? 'toolbar-toggle toolbar-toggle--active'
                : 'toolbar-toggle'
            }
            onClick={presetsPanel.toggle}
            aria-label={t('toolbar.toggle.presets.aria')}
            aria-pressed={presetsPanel.layout.visible}
          >
            {t('toolbar.toggle.presets')}
          </button>
          <button
            type="button"
            className={
              inspectorPanel.layout.visible
                ? 'toolbar-toggle toolbar-toggle--active'
                : 'toolbar-toggle'
            }
            onClick={inspectorPanel.toggle}
            aria-label={t('toolbar.toggle.inspector.aria')}
            aria-pressed={inspectorPanel.layout.visible}
          >
            {t('toolbar.toggle.inspector')}
          </button>
          <button
            type="button"
            className={
              timelinePanel.layout.visible
                ? 'toolbar-toggle toolbar-toggle--active'
                : 'toolbar-toggle'
            }
            onClick={timelinePanel.toggle}
            aria-label={t('toolbar.toggle.timeline.aria')}
            aria-pressed={timelinePanel.layout.visible}
          >
            {t('toolbar.toggle.timeline')}
          </button>
        </nav>
        <div className="app-toolbar__language">
          <label className="app-toolbar__language-label" htmlFor="language-select">
            {t('app.language.label')}
          </label>
          <select
            id="language-select"
            value={locale}
            onChange={(event) => {
              const next = event.target.value as Locale;
              setLocale(next);
            }}
          >
            {locales.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div className="app-toolbar__accessibility">
          <span className="app-toolbar__language-label">{t('accessibility.label')}</span>
          <button
            type="button"
            className={highContrast ? 'toolbar-toggle toolbar-toggle--active' : 'toolbar-toggle'}
            onClick={() => setHighContrast(!highContrast)}
            aria-pressed={highContrast}
            aria-label={t('accessibility.highContrast.aria')}
          >
            {t('accessibility.highContrast')}
          </button>
          <button
            type="button"
            className={largeText ? 'toolbar-toggle toolbar-toggle--active' : 'toolbar-toggle'}
            onClick={() => setLargeText(!largeText)}
            aria-pressed={largeText}
            aria-label={t('accessibility.largeText.aria')}
          >
            {t('accessibility.largeText')}
          </button>
        </div>
        <div className="app-toolbar__history">
          <button type="button" className="toolbar-action" onClick={undo} disabled={!canUndo}>
            {t('toolbar.action.undo')}
          </button>
          <button type="button" className="toolbar-action" onClick={redo} disabled={!canRedo}>
            {t('toolbar.action.redo')}
          </button>
        </div>
      </header>
      <div className="app-main">
        <div className="app-panels">
          {presetsPanel.layout.visible ? (
            <aside
              id="presets-panel"
              className="docked-panel docked-panel--left"
              style={{ width: `${presetsPanel.layout.size}px` }}
            >
              <PresetsPanel
                manifestName={manifest.metadata?.name ?? formatFileName(manifest.lastLoadedPath)}
                manifestStatus={manifest.status}
                lastError={manifest.errorMessage}
                validationIssues={manifest.issues}
                samplePresets={samplePresets}
                onSampleSelected={handleSampleSelected}
                onManifestSelected={handleManifestSelected}
              />
              <div
                className={
                  presetsResizeHandle.isDragging
                    ? 'resize-handle resize-handle--vertical resize-handle--active'
                    : 'resize-handle resize-handle--vertical'
                }
                onPointerDown={presetsResizeHandle.onPointerDown}
                onKeyDown={presetsResizeHandle.onKeyDown}
                role="separator"
                aria-orientation="vertical"
                aria-label={t('layout.resize.presets')}
                aria-valuemin={presetsPanel.layout.minSize}
                aria-valuemax={presetsPanel.layout.maxSize ?? undefined}
                aria-valuenow={Math.round(presetsPanel.layout.size)}
                aria-controls="presets-panel"
                tabIndex={0}
              />
            </aside>
          ) : null}
          <main className="app-viewport">
            <ViewportPanel />
          </main>
          {inspectorPanel.layout.visible ? (
            <aside
              id="inspector-panel"
              className="docked-panel docked-panel--right"
              style={{ width: `${inspectorPanel.layout.size}px` }}
            >
              <div
                className={
                  inspectorResizeHandle.isDragging
                    ? 'resize-handle resize-handle--vertical resize-handle--active'
                    : 'resize-handle resize-handle--vertical'
                }
                onPointerDown={inspectorResizeHandle.onPointerDown}
                onKeyDown={inspectorResizeHandle.onKeyDown}
                role="separator"
                aria-orientation="vertical"
                aria-label={t('layout.resize.inspector')}
                aria-valuemin={inspectorPanel.layout.minSize}
                aria-valuemax={inspectorPanel.layout.maxSize ?? undefined}
                aria-valuenow={Math.round(inspectorPanel.layout.size)}
                aria-controls="inspector-panel"
                tabIndex={0}
              />
              <InspectorPanel />
            </aside>
          ) : null}
        </div>
        {timelinePanel.layout.visible ? (
          <section
            id="timeline-panel"
            className="timeline-dock"
            style={{ height: `${timelinePanel.layout.size}px` }}
          >
            <div
              className={
                timelineResizeHandle.isDragging
                  ? 'resize-handle resize-handle--horizontal resize-handle--active'
                  : 'resize-handle resize-handle--horizontal'
              }
              onPointerDown={timelineResizeHandle.onPointerDown}
              onKeyDown={timelineResizeHandle.onKeyDown}
              role="separator"
              aria-orientation="horizontal"
              aria-label={t('layout.resize.timeline')}
              aria-valuemin={timelinePanel.layout.minSize}
              aria-valuemax={timelinePanel.layout.maxSize ?? undefined}
              aria-valuenow={Math.round(timelinePanel.layout.size)}
              aria-controls="timeline-panel"
              tabIndex={0}
            />
            <TimelinePanel />
          </section>
        ) : null}
      </div>
      <footer className="app-status-bar" role="status" aria-live="polite">
        <span>{t('status.nodes', { values: { count: scene.nodes.length } })}</span>
        <span>{t('status.connections', { values: { count: scene.links.length } })}</span>
        <span>
          {manifest.status === 'loaded'
            ? t('status.manifest.loaded', {
                values: {
                  name:
                    manifest.metadata?.name ??
                    formatFileName(manifest.lastLoadedPath) ??
                    t('status.manifest.ready'),
                },
              })
            : manifest.status === 'loading'
              ? t('status.manifest.loading')
              : t('status.manifest.idle')}
        </span>
        <span>
          {gpu.status === 'ready'
            ? t('status.gpu.ready', {
                values: {
                  backend: gpu.backend.toUpperCase(),
                  hz: gpu.targetFrameRate,
                  adapter: gpu.adapterName
                    ? t('status.gpu.adapter', { values: { name: gpu.adapterName } })
                    : '',
                },
              })
            : gpu.status === 'initializing'
              ? t('status.gpu.initializing')
              : gpu.status === 'error'
                ? t('status.gpu.error', {
                    values: { message: gpu.errorMessage ?? t('status.gpu.unavailable') },
                  })
                : t('status.gpu.idle')}
        </span>
      </footer>
    </div>
  );
}
