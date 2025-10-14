import React from 'react';
import { useAutotune } from '../hooks/useAutotune';
import { useI18n } from '../../i18n/LocalizationProvider';

const formatNumber = (value: number) => {
  if (Math.abs(value) >= 10) {
    return value.toFixed(2);
  }
  if (Math.abs(value) >= 1) {
    return value.toFixed(3);
  }
  return value.toFixed(4);
};

export function AutotunePanel() {
  const { state, run, apply, reset } = useAutotune();
  const { t } = useI18n();

  const isRunning = state.status === 'running';
  const hasResult = state.status === 'completed' && state.result;
  const errorKey = state.status === 'error' ? state.error : undefined;
  const errorMessage = state.status === 'error' ? state.errorMessage : undefined;

  const handleRunClick = () => {
    if (!isRunning) {
      void run();
    }
  };

  const handleApply = () => {
    if (hasResult) {
      apply();
    }
  };

  const handleDismiss = () => {
    reset();
  };

  return (
    <section className="autotune-panel" aria-live="polite">
      <header className="autotune-panel__header">
        <h3>{t('autotune.title')}</h3>
        <p>{t('autotune.description')}</p>
      </header>
      {state.status === 'idle' ? (
        <button type="button" className="primary-button" onClick={handleRunClick}>
          {t('autotune.run')}
        </button>
      ) : null}
      {isRunning ? (
        <div className="autotune-panel__progress">
          <div className="autotune-panel__progress-bar">
            <div
              className="autotune-panel__progress-fill"
              style={{ width: `${Math.round(state.progress * 100)}%` }}
            />
          </div>
          <span>
            {t('autotune.progress', {
              values: { current: state.currentTrial, total: state.totalTrials },
            })}
          </span>
        </div>
      ) : null}
      {hasResult && state.result ? (
        <div className="autotune-panel__result">
          <p>
            {t('autotune.result.summary', {
              values: {
                score: formatNumber(state.result.score),
                delta: formatNumber(state.result.score - state.result.baselineScore),
                seconds: (state.result.durationMs / 1000).toFixed(1),
              },
            })}
          </p>
          {state.result.suggestions.length > 0 ? (
            <ul className="autotune-panel__suggestions">
              {state.result.suggestions.map((suggestion) => (
                <li key={`${suggestion.panelId}:${suggestion.pointer}`}>
                  <span className="autotune-panel__suggestion-label">
                    {suggestion.panelLabel}: {suggestion.label}
                  </span>
                  <span className="autotune-panel__suggestion-diff">
                    {formatNumber(suggestion.from)} → {formatNumber(suggestion.to)}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="panel-muted">{t('autotune.result.noChanges')}</p>
          )}
          <div className="autotune-panel__actions">
            <button type="button" className="primary-button" onClick={handleApply}>
              {t('autotune.result.apply')}
            </button>
            <button type="button" className="ghost-button" onClick={handleDismiss}>
              {t('autotune.result.dismiss')}
            </button>
          </div>
        </div>
      ) : null}
      {state.status === 'error' && errorKey ? (
        <p className="panel-footnote--error">
          {errorKey === 'failed'
            ? t('autotune.error.failed', { values: { message: errorMessage ?? '' } })
            : t(`autotune.error.${errorKey}`)}
        </p>
      ) : null}
      {state.status === 'idle' && !isRunning ? (
        <p className="autotune-panel__hint">{t('autotune.status.idle')}</p>
      ) : null}
    </section>
  );
}
