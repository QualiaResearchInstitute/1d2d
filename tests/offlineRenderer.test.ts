import assert from 'node:assert/strict';
import test from 'node:test';

import { OfflineRenderer } from '../scripts/lib/offlineRenderer.js';
import type { RainbowFrameResult } from '../src/pipeline/rainbowFrame.js';

const createRainbowStub = (): RainbowFrameResult => ({
  metrics: {} as any,
  obsAverage: null,
});

test('OfflineRenderer converts RGBA8 to 10-bit with tile stride', async () => {
  const renderer = new OfflineRenderer({
    width: 2,
    height: 1,
    tileHeight: 1,
    budgets: {
      frameMs: 500,
      rssMb: 1024,
      heapMb: 512,
      cpuPercent: 500,
    },
  });

  const result = await renderer.renderFrame({ frameIndex: 0 }, (out) => {
    out.set([0, 128, 255, 64, 32, 64, 96, 255]);
    return createRainbowStub();
  });

  assert.equal(result.frameIndex, 0);
  assert.ok(result.performance.frameMs >= 0);
  assert.deepEqual(Array.from(result.output10Bit), [0, 514, 1023, 257, 128, 257, 385, 1023]);

  const snapshot = renderer.getPerformanceSnapshot();
  assert.equal(snapshot.frames, 1);
  assert.equal(snapshot.lastSample?.frameIndex, 0);
});
