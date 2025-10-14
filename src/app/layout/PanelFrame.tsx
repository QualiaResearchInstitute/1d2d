import React from 'react';

export interface PanelFrameProps {
  readonly title: string;
  readonly children: React.ReactNode;
  readonly footer?: React.ReactNode;
}

export function PanelFrame({ title, children, footer }: PanelFrameProps) {
  return (
    <section className="panel-frame">
      <header className="panel-frame__header">
        <h2>{title}</h2>
      </header>
      <div className="panel-frame__body">{children}</div>
      {footer ? <footer className="panel-frame__footer">{footer}</footer> : null}
    </section>
  );
}
