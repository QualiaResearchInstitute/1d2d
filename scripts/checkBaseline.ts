import { readFileSync } from 'node:fs';
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
import { hashCanonicalJson, readCanonicalJson } from '../src/serialization/canonicalJson.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

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

const readHashFile = (path: string): string => readFileSync(path, 'utf8').trim();

type MetricComparison = {
  label: string;
  baseline: number;
  current: number;
  tolerance: number;
  kind: 'relative' | 'absolute';
};

type StoredBaseline = {
  metrics: Record<string, any>;
  obsAverage: number | null;
  su7Gate?: {
    snapshot: unknown;
    hash?: string;
  };
  goldens?: {
    presetHash?: string;
    render?: { width: number; height: number; hash: string };
    thumbnail64?: { width: number; height: number; hash: string };
  };
};

const loadBaseline = (root: string): StoredBaseline => {
  const paths = baselinePaths(root);
  const text = readFileSync(paths.metrics, 'utf8');
  return readCanonicalJson<StoredBaseline>(text);
};

const relativeDiff = (baseline: number, current: number) => {
  if (!Number.isFinite(baseline) || Math.abs(baseline) < 1e-9) {
    return Math.abs(current - baseline);
  }
  return Math.abs((current - baseline) / baseline);
};

const compareMetrics = (comparisons: MetricComparison[]) => {
  const failures: string[] = [];
  for (const item of comparisons) {
    const delta =
      item.kind === 'relative'
        ? relativeDiff(item.baseline, item.current)
        : Math.abs(item.current - item.baseline);
    if (delta > item.tolerance) {
      failures.push(
        `${item.label}: baseline=${item.baseline} current=${item.current} diff=${delta}`,
      );
    }
  }
  return failures;
};

const main = async () => {
  const baselineRoot = join(__dirname, '..', '..', 'baseline');
  const paths = baselinePaths(baselineRoot);
  const stored = loadBaseline(baselineRoot);
  const current = generateBaselineFrame(0);

  const canonicalPpm = encodePpm(current.pixels, BASELINE_DIM.width, BASELINE_DIM.height);
  const renderHash = hashBytesHex(canonicalPpm);
  const thumbnailPixels = downsamplePixels(
    current.pixels,
    BASELINE_DIM.width,
    BASELINE_DIM.height,
    64,
    64,
  );
  const thumbnailPpm = encodePpm(thumbnailPixels, 64, 64);
  const thumbnailHash = hashBytesHex(thumbnailPpm);
  const presetCanonical = hashCanonicalJson(current.gateSnapshot);
  const currentGateHash = presetCanonical.hash;

  const { hash: storedHash } = hashCanonicalJson(stored);
  const storedGateHash =
    stored.su7Gate != null ? hashCanonicalJson(stored.su7Gate.snapshot).hash : null;

  const storedPresetJson = readFileSync(paths.preset, 'utf8');
  const storedPresetSnapshot = readCanonicalJson<unknown>(storedPresetJson);
  const storedPresetHash = readHashFile(paths.presetHash);
  const storedPresetHashComputed = hashCanonicalJson(storedPresetSnapshot).hash;
  const storedRenderHash = readHashFile(paths.renderHash);
  const storedThumbnailHash = readHashFile(paths.thumbnailHash);

  const b = stored.metrics;
  const c = current.metrics as any;

  const comparisons: MetricComparison[] = [
    {
      label: 'rim.mean',
      baseline: b.rim.mean,
      current: c.rim.mean,
      tolerance: 0.01,
      kind: 'relative',
    },
    {
      label: 'rim.std',
      baseline: b.rim.std,
      current: c.rim.std,
      tolerance: 0.01,
      kind: 'relative',
    },
    {
      label: 'rim.max',
      baseline: b.rim.max,
      current: c.rim.max,
      tolerance: 0.02,
      kind: 'relative',
    },
    {
      label: 'warp.mean',
      baseline: b.warp.mean,
      current: c.warp.mean,
      tolerance: 0.01,
      kind: 'relative',
    },
    {
      label: 'warp.std',
      baseline: b.warp.std,
      current: c.warp.std,
      tolerance: 0.01,
      kind: 'relative',
    },
    {
      label: 'warp.angle',
      baseline: b.warp.dominantAngle,
      current: c.warp.dominantAngle,
      tolerance: 0.01,
      kind: 'absolute',
    },
    {
      label: 'grad.mean',
      baseline: b.gradient.gradMean,
      current: c.gradient.gradMean,
      tolerance: 0.01,
      kind: 'relative',
    },
    {
      label: 'grad.std',
      baseline: b.gradient.gradStd,
      current: c.gradient.gradStd,
      tolerance: 0.01,
      kind: 'relative',
    },
    {
      label: 'coh.mean',
      baseline: b.gradient.cohMean,
      current: c.gradient.cohMean,
      tolerance: 0.01,
      kind: 'relative',
    },
    {
      label: 'coh.std',
      baseline: b.gradient.cohStd,
      current: c.gradient.cohStd,
      tolerance: 0.01,
      kind: 'relative',
    },
    {
      label: 'compositor.effectiveBlend',
      baseline: b.compositor.effectiveBlend,
      current: c.compositor.effectiveBlend,
      tolerance: 0.005,
      kind: 'relative',
    },
    {
      label: 'compositor.surfaceMean',
      baseline: b.compositor.surfaceMean,
      current: c.compositor.surfaceMean,
      tolerance: 0.01,
      kind: 'absolute',
    },
    {
      label: 'obsAverage',
      baseline: stored.obsAverage ?? 0,
      current: current.obsAverage ?? 0,
      tolerance: 0.01,
      kind: 'relative',
    },
  ];

  const failures = compareMetrics(comparisons);
  if (storedPresetHash !== storedPresetHashComputed) {
    failures.push(
      `Stored preset hash mismatch: file=${storedPresetHash} computed=${storedPresetHashComputed}`,
    );
  }
  if (storedGateHash && storedGateHash !== storedPresetHash) {
    failures.push(
      `Stored gate hash mismatch: metrics=${storedGateHash} presetFile=${storedPresetHash}`,
    );
  }
  if (stored.goldens?.presetHash && stored.goldens.presetHash !== storedPresetHash) {
    failures.push(
      `Stored golden preset hash mismatch: metrics=${stored.goldens.presetHash} file=${storedPresetHash}`,
    );
  }
  if (stored.goldens?.render && stored.goldens.render.hash !== storedRenderHash) {
    failures.push(
      `Stored golden render hash mismatch: metrics=${stored.goldens.render.hash} file=${storedRenderHash}`,
    );
  }
  if (stored.goldens?.thumbnail64 && stored.goldens.thumbnail64.hash !== storedThumbnailHash) {
    failures.push(
      `Stored golden thumbnail hash mismatch: metrics=${stored.goldens.thumbnail64.hash} file=${storedThumbnailHash}`,
    );
  }
  if (storedPresetHash !== currentGateHash) {
    failures.push(
      `SU7 preset hash mismatch: baseline=${storedPresetHash} current=${currentGateHash}`,
    );
  }
  if (renderHash !== storedRenderHash) {
    failures.push(
      `Canonical render hash mismatch: baseline=${storedRenderHash} current=${renderHash}`,
    );
  }
  if (thumbnailHash !== storedThumbnailHash) {
    failures.push(
      `Thumbnail render hash mismatch: baseline=${storedThumbnailHash} current=${thumbnailHash}`,
    );
  }

  if (failures.length > 0) {
    console.error('\nBaseline regression check failed:');
    for (const line of failures) {
      console.error('  ', line);
    }
    process.exitCode = 1;
    return;
  }

  console.log('Baseline metrics within tolerance (<=1%)');
  console.log('Stored canonical baseline hash (BLAKE3-256):', storedHash);
  if (storedGateHash) {
    console.log('Stored SU7 gate hash (BLAKE3-256):', storedGateHash);
  }
  console.log('Stored SU7 preset hash (BLAKE3-256):', storedPresetHash);
  console.log('Current SU7 gate hash (BLAKE3-256):', currentGateHash);
  console.log('Canonical render hash (baseline/current):', storedRenderHash, renderHash);
  console.log('Thumbnail render hash (baseline/current):', storedThumbnailHash, thumbnailHash);
};

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
