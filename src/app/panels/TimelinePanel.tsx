import React, { useCallback } from 'react';
import { PanelFrame } from '../layout/PanelFrame';
import { useTimelineState } from '../../state/AppState';
import { useI18n } from '../../i18n/LocalizationProvider';

export function TimelinePanel() {
  const {
    timeline: { duration, currentTime, fps, clips, playing },
    setTime,
    setPlaying,
  } = useTimelineState();
  const { t } = useI18n();

  const handleTogglePlayback = useCallback(() => {
    setPlaying(!playing);
  }, [playing, setPlaying]);

  const handleTimeChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const value = Number.parseFloat(event.target.value);
      if (!Number.isNaN(value)) {
        setTime(value);
      }
    },
    [setTime],
  );

  return (
    <PanelFrame
      title={t('timeline.title')}
      footer={
        <span>{t('timeline.footer', { values: { fps, duration: duration.toFixed(1) } })}</span>
      }
    >
      <div className="timeline-controls">
        <button className="primary-button" type="button" onClick={handleTogglePlayback}>
          {playing ? t('timeline.pause') : t('timeline.play')}
        </button>
        <label className="timeline-slider">
          <span>{t('timeline.timeLabel')}</span>
          <input
            type="range"
            min="0"
            max={duration}
            step="0.01"
            value={currentTime}
            onChange={handleTimeChange}
          />
          <span>{currentTime.toFixed(2)}s</span>
        </label>
      </div>
      <div className="timeline-clips">
        {clips.length === 0 ? (
          <p className="panel-muted">{t('timeline.empty')}</p>
        ) : (
          <ul>
            {clips.map((clip) => (
              <li key={clip.id}>
                <span className="timeline-clip__name">{clip.parameterId}</span>
                <span className="timeline-clip__node">{clip.nodeId}</span>
                <span className="timeline-clip__keyframes">
                  {t('timeline.clip.keyframes', { values: { count: clip.keyframes.length } })}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </PanelFrame>
  );
}
