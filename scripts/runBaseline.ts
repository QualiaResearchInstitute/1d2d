import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  BASELINE_DIM,
  baselinePaths,
  CANONICAL_KUR_PARAMS,
  CANONICAL_PARAMS,
  encodePpm,
  generateBaselineFrame,
} from './lib/baselineHarness.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const ensureDir = (path: string) => mkdirSync(path, { recursive: true });

const main = async () => {
  const baselineRoot = join(__dirname, '..', '..', 'baseline');
  const paths = baselinePaths(baselineRoot);
  ensureDir(join(baselineRoot, 'renders'));
  ensureDir(join(baselineRoot, 'metrics'));

  const { pixels, metrics, obsAverage } = generateBaselineFrame(0);
  const ppmBytes = encodePpm(pixels, BASELINE_DIM.width, BASELINE_DIM.height);
  const payload = {
    width: BASELINE_DIM.width,
    height: BASELINE_DIM.height,
    preset: 'Rainbow Rims + DMT Kernel Effects',
    timestamp: new Date().toISOString(),
    metrics,
    obsAverage,
    kurParams: CANONICAL_KUR_PARAMS,
    canonicalParams: CANONICAL_PARAMS,
  };

  writeFileSync(paths.render, ppmBytes);
  writeFileSync(paths.metrics, JSON.stringify(payload, null, 2));

  console.log('Baseline render saved to', paths.render);
  console.log('Metrics saved to', paths.metrics);
};

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
