import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'blake3';

import {
  BASELINE_DIM,
  baselinePaths,
  CANONICAL_KUR_PARAMS,
  CANONICAL_PARAMS,
  encodePpm,
  generateBaselineFrame,
  downsamplePixels,
} from './lib/baselineHarness.js';
import { hashCanonicalJson } from '../src/serialization/canonicalJson.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const THUMB_DIM = { width: 64, height: 64 } as const;

const toHex = (bytes: Uint8Array): string => {
  let result = '';
  for (let i = 0; i < bytes.length; i++) {
    result += bytes[i].toString(16).padStart(2, '0');
  }
  return result;
};

const hashBytesHex = (data: Uint8Array | Uint8ClampedArray): string => {
  const view =
    data instanceof Uint8Array
      ? data
      : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return toHex(createHash().update(view).digest());
};

const ensureDir = (path: string) => mkdirSync(path, { recursive: true });

const main = async () => {
  const baselineRoot = join(__dirname, '..', '..', 'baseline');
  const paths = baselinePaths(baselineRoot);
  ensureDir(join(baselineRoot, 'renders'));
  ensureDir(join(baselineRoot, 'metrics'));

  const { pixels, metrics, obsAverage, gateSnapshot } = generateBaselineFrame(0);
  const ppmBytes = encodePpm(pixels, BASELINE_DIM.width, BASELINE_DIM.height);
  const renderHash = hashBytesHex(ppmBytes);
  const thumbnailPixels = downsamplePixels(
    pixels,
    BASELINE_DIM.width,
    BASELINE_DIM.height,
    THUMB_DIM.width,
    THUMB_DIM.height,
  );
  const thumbnailPpm = encodePpm(thumbnailPixels, THUMB_DIM.width, THUMB_DIM.height);
  const thumbnailHash = hashBytesHex(thumbnailPpm);
  const presetCanonical = hashCanonicalJson(gateSnapshot);

  const payload = {
    width: BASELINE_DIM.width,
    height: BASELINE_DIM.height,
    preset: 'Rainbow Rims + DMT Kernel Effects',
    timestamp: new Date().toISOString(),
    metrics,
    obsAverage,
    kurParams: CANONICAL_KUR_PARAMS,
    canonicalParams: CANONICAL_PARAMS,
    su7Gate: {
      snapshot: gateSnapshot,
      hash: presetCanonical.hash,
    },
    goldens: {
      presetHash: presetCanonical.hash,
      render: {
        width: BASELINE_DIM.width,
        height: BASELINE_DIM.height,
        hash: renderHash,
      },
      thumbnail64: {
        width: THUMB_DIM.width,
        height: THUMB_DIM.height,
        hash: thumbnailHash,
      },
    },
  };

  writeFileSync(paths.render, ppmBytes);
  writeFileSync(paths.renderHash, `${renderHash}\n`);
  writeFileSync(paths.thumbnail, thumbnailPpm);
  writeFileSync(paths.thumbnailHash, `${thumbnailHash}\n`);
  writeFileSync(paths.preset, presetCanonical.json);
  writeFileSync(paths.presetHash, `${presetCanonical.hash}\n`);
  const { json: canonicalJson, hash: metricsHash } = hashCanonicalJson(payload, { indent: 2 });
  writeFileSync(paths.metrics, canonicalJson);

  console.log('Baseline render saved to', paths.render);
  console.log('Baseline render hash saved to', paths.renderHash);
  console.log('Thumbnail render saved to', paths.thumbnail);
  console.log('Thumbnail hash saved to', paths.thumbnailHash);
  console.log('SU7 preset saved to', paths.preset);
  console.log('SU7 preset hash saved to', paths.presetHash);
  console.log('Metrics saved to', paths.metrics);
  console.log('Canonical metrics hash (BLAKE3-256):', metricsHash);
  console.log('SU7 gate snapshot hash (BLAKE3-256):', presetCanonical.hash);
  console.log('Canonical render hash (BLAKE3-256):', renderHash);
  console.log('Thumbnail render hash (BLAKE3-256):', thumbnailHash);
};

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
