import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  BASELINE_DIM,
  baselinePaths,
  CANONICAL_KUR_PARAMS,
  CANONICAL_PARAMS,
  generateBaselineFrame
} from "./lib/baselineHarness.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

type MetricComparison = {
  label: string;
  baseline: number;
  current: number;
  tolerance: number;
  kind: "relative" | "absolute";
};

const loadBaseline = (root: string) => {
  const paths = baselinePaths(root);
  const text = readFileSync(paths.metrics, "utf8");
  return JSON.parse(text) as {
    metrics: Record<string, any>;
    obsAverage: number | null;
  };
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
      item.kind === "relative"
        ? relativeDiff(item.baseline, item.current)
        : Math.abs(item.current - item.baseline);
    if (delta > item.tolerance) {
      failures.push(
        `${item.label}: baseline=${item.baseline} current=${item.current} diff=${delta}`
      );
    }
  }
  return failures;
};

const main = async () => {
  const baselineRoot = join(__dirname, "..", "..", "baseline");
  const stored = loadBaseline(baselineRoot);
  const current = generateBaselineFrame(0);

  const b = stored.metrics;
  const c = current.metrics as any;

  const comparisons: MetricComparison[] = [
    { label: "rim.mean", baseline: b.rim.mean, current: c.rim.mean, tolerance: 0.01, kind: "relative" },
    { label: "rim.std", baseline: b.rim.std, current: c.rim.std, tolerance: 0.01, kind: "relative" },
    { label: "rim.max", baseline: b.rim.max, current: c.rim.max, tolerance: 0.02, kind: "relative" },
    { label: "warp.mean", baseline: b.warp.mean, current: c.warp.mean, tolerance: 0.01, kind: "relative" },
    { label: "warp.std", baseline: b.warp.std, current: c.warp.std, tolerance: 0.01, kind: "relative" },
    { label: "warp.angle", baseline: b.warp.dominantAngle, current: c.warp.dominantAngle, tolerance: 0.01, kind: "absolute" },
    { label: "grad.mean", baseline: b.gradient.gradMean, current: c.gradient.gradMean, tolerance: 0.01, kind: "relative" },
    { label: "grad.std", baseline: b.gradient.gradStd, current: c.gradient.gradStd, tolerance: 0.01, kind: "relative" },
    { label: "coh.mean", baseline: b.gradient.cohMean, current: c.gradient.cohMean, tolerance: 0.01, kind: "relative" },
    { label: "coh.std", baseline: b.gradient.cohStd, current: c.gradient.cohStd, tolerance: 0.01, kind: "relative" },
    { label: "compositor.effectiveBlend", baseline: b.compositor.effectiveBlend, current: c.compositor.effectiveBlend, tolerance: 0.005, kind: "relative" },
    { label: "compositor.surfaceMean", baseline: b.compositor.surfaceMean, current: c.compositor.surfaceMean, tolerance: 0.01, kind: "absolute" },
    { label: "obsAverage", baseline: stored.obsAverage ?? 0, current: current.obsAverage ?? 0, tolerance: 0.01, kind: "relative" }
  ];

  const failures = compareMetrics(comparisons);
  if (failures.length > 0) {
    console.error("\nBaseline regression check failed:");
    for (const line of failures) {
      console.error("  ", line);
    }
    process.exitCode = 1;
    return;
  }

  console.log("Baseline metrics within tolerance (<=1%)");
};

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
