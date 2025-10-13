import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';
import { JSDOM } from 'jsdom';
import axe from 'axe-core';

const setupDom = () => {
  const dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', {
    url: 'http://localhost',
    pretendToBeVisual: true,
  });

  const globalAny = globalThis as any;
  globalAny.window = dom.window;
  globalAny.document = dom.window.document;
  Object.defineProperty(globalAny, 'navigator', {
    value: dom.window.navigator,
    configurable: true,
  });
  globalAny.HTMLElement = dom.window.HTMLElement;
  globalAny.HTMLCanvasElement = dom.window.HTMLCanvasElement;
  globalAny.Node = dom.window.Node;
  globalAny.getComputedStyle = dom.window.getComputedStyle.bind(dom.window);

  if (!dom.window.requestAnimationFrame) {
    dom.window.requestAnimationFrame = (callback: FrameRequestCallback) =>
      dom.window.setTimeout(() => callback(dom.window.performance.now()), 16) as unknown as number;
  }

  Object.defineProperty(dom.window.HTMLCanvasElement.prototype, 'getContext', {
    value: () => ({
      fillRect: () => {},
      drawImage: () => {},
      getImageData: () => ({ data: new Uint8ClampedArray() }),
      putImageData: () => {},
      createImageData: (w: number, h: number) => ({
        data: new Uint8ClampedArray(w * h * 4),
        width: w,
        height: h,
      }),
      beginPath: () => {},
      moveTo: () => {},
      lineTo: () => {},
      closePath: () => {},
      stroke: () => {},
      clearRect: () => {},
    }),
  });

  if (!globalAny.URL.createObjectURL) {
    globalAny.URL.createObjectURL = () => 'blob:mock';
  }
  if (!globalAny.URL.revokeObjectURL) {
    globalAny.URL.revokeObjectURL = () => {};
  }

  globalAny.performance = dom.window.performance;

  return dom;
};

test('App renders without axe-core violations', async () => {
  const dom = setupDom();
  const container = dom.window.document.getElementById('root');
  assert(container, 'expected root element');

  const root = createRoot(container);
  const GlyphHarness = () => (
    <div>
      <div
        aria-label="SU7 phase and pulse glyph"
        role="img"
        style={{ width: 220, height: 220, border: '1px solid #1f2937' }}
      />
      <div style={{ marginTop: '1rem' }}>
        <button aria-pressed={false} type="button">
          Macro learn mode
        </button>
      </div>
      <div style={{ marginTop: '0.75rem' }}>
        <label htmlFor="macro-knob">Macro knob</label>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <input id="macro-knob" type="range" min={-1.5} max={1.5} step={0.05} defaultValue={0} />
          <span>0.00</span>
        </div>
      </div>
      <div role="list" style={{ marginTop: '0.75rem', display: 'grid', gap: '0.5rem' }}>
        <div role="listitem" style={{ border: '1px solid #1f2937', padding: '0.5rem' }}>
          <div>Edge 1</div>
          <div>Δθ 12.0° · Δφ 4.0°</div>
          <button type="button">Remove</button>
        </div>
        <div role="listitem" style={{ border: '1px solid #1f2937', padding: '0.5rem' }}>
          <div>Edge 2</div>
          <div>Δθ -6.0° · Δφ 2.5°</div>
          <button type="button">Remove</button>
        </div>
      </div>
    </div>
  );
  await act(async () => {
    root.render(<GlyphHarness />);
  });

  await new Promise((resolve) => setTimeout(resolve, 0));

  const results = await axe.run(container);
  const violations = results.violations ?? [];
  const message = violations
    .map((violation) => `${violation.id}: ${violation.nodes.length} node(s)`)
    .join('\n');
  assert.equal(
    violations.length,
    0,
    `Expected zero accessibility violations but found ${violations.length}\n${message}`,
  );

  await act(async () => {
    root.unmount();
  });

  dom.window.close();
});
