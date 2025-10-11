import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  runCrossTierValidation,
  summarizeAlerts,
  type CrossTierValidationReport,
  type CrossTierAlert,
  type PairwiseKey,
  type TierId,
} from '../src/validation/crossTierValidation.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

type ToleranceConfig = {
  halfLife: number;
  coherence: number;
  kernelDelta: number;
  divergence: number;
};

type StoredBaseline = {
  report: CrossTierValidationReport;
  tolerance?: Partial<ToleranceConfig>;
};

const DEFAULT_TOLERANCE: ToleranceConfig = {
  halfLife: 0.08,
  coherence: 0.05,
  kernelDelta: 0.12,
  divergence: 0.08,
};

const baselinePath = join(__dirname, '..', '..', 'baseline', 'metrics', 'cross-tier.json');
const outputPath = join(__dirname, '..', '..', 'dist', 'cross-tier', 'latest.json');

const ensureDir = (filePath: string) => {
  const dir = dirname(filePath);
  mkdirSync(dir, { recursive: true });
};

const computeDiff = (
  current: CrossTierValidationReport,
  baseline: CrossTierValidationReport,
  tolerance: ToleranceConfig,
) => {
  const failures: string[] = [];

  const tiers: TierId[] = ['rim1p5D', 'surface2D', 'volume2p5D'];
  const pairs: PairwiseKey[] = ['rim1p5D~surface2D', 'rim1p5D~volume2p5D', 'surface2D~volume2p5D'];

  for (const tier of tiers) {
    const diffBaseline = Math.abs(
      current.baseline[tier].measuredHalfLife - baseline.baseline[tier].measuredHalfLife,
    );
    if (diffBaseline > tolerance.halfLife) {
      failures.push(
        `${tier} baseline half-life diff ${diffBaseline.toFixed(3)} exceeds tolerance ${tolerance.halfLife}`,
      );
    }
    const diffVariant = Math.abs(
      current.variant[tier].measuredHalfLife - baseline.variant[tier].measuredHalfLife,
    );
    if (diffVariant > tolerance.halfLife) {
      failures.push(
        `${tier} variant half-life diff ${diffVariant.toFixed(3)} exceeds tolerance ${tolerance.halfLife}`,
      );
    }

    const divergence = current.divergence[tier].maxAbs;
    if (divergence > tolerance.divergence) {
      failures.push(
        `${tier} divergence max ${divergence.toFixed(3)} exceeds tolerance ${tolerance.divergence}`,
      );
    }

    const deltaDiff = Math.abs(
      current.kernelDelta.perTier[tier].delta - baseline.kernelDelta.perTier[tier].delta,
    );
    if (deltaDiff > tolerance.kernelDelta) {
      failures.push(
        `${tier} kernel delta change ${deltaDiff.toFixed(3)} exceeds tolerance ${tolerance.kernelDelta}`,
      );
    }
  }

  for (const key of pairs) {
    const baseDiff = Math.abs(current.coherence.baseline[key] - baseline.coherence.baseline[key]);
    if (baseDiff > tolerance.coherence) {
      failures.push(
        `${key} baseline coherence diff ${baseDiff.toFixed(3)} exceeds tolerance ${tolerance.coherence}`,
      );
    }
    const variantDiff = Math.abs(current.coherence.variant[key] - baseline.coherence.variant[key]);
    if (variantDiff > tolerance.coherence) {
      failures.push(
        `${key} variant coherence diff ${variantDiff.toFixed(3)} exceeds tolerance ${tolerance.coherence}`,
      );
    }
  }

  return failures;
};

const main = () => {
  const report = runCrossTierValidation();

  ensureDir(outputPath);
  writeFileSync(outputPath, JSON.stringify(report, null, 2));

  const baselineExists = existsSync(baselinePath);
  let tolerance = DEFAULT_TOLERANCE;
  if (baselineExists) {
    const raw = readFileSync(baselinePath, 'utf8');
    const stored = JSON.parse(raw) as StoredBaseline;
    tolerance = {
      ...DEFAULT_TOLERANCE,
      ...(stored.tolerance ?? {}),
    };
    const failures = computeDiff(report, stored.report, tolerance);
    if (failures.length > 0) {
      console.error('\nCross-tier regression detected:');
      for (const failure of failures) {
        console.error('  -', failure);
      }
      process.exitCode = 1;
    } else {
      console.log('Cross-tier metrics within tolerance.');
    }
  } else {
    console.log('No cross-tier baseline found; skipping regression diff.');
  }

  if (report.alerts.length > 0) {
    console.error('\nCross-tier alerts:');
    for (const message of summarizeAlerts(report.alerts)) {
      console.error('  -', message);
    }
    process.exitCode = 1;
  }

  if (!baselineExists) {
    console.log('Tip: capture baseline via `npm run cross-tier:update`.');
  }
};

main();
