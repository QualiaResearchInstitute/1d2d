import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { runCrossTierValidation, type CrossTierValidationReport } from "../src/validation/crossTierValidation.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

const baselinePath = join(__dirname, "..", "..", "baseline", "metrics", "cross-tier.json");

const DEFAULT_TOLERANCE = {
  halfLife: 0.08,
  coherence: 0.05,
  kernelDelta: 0.12,
  divergence: 0.08
} as const;

const ensureDir = (filePath: string) => {
  const dir = dirname(filePath);
  mkdirSync(dir, { recursive: true });
};

const writeBaseline = (report: CrossTierValidationReport) => {
  ensureDir(baselinePath);
  const payload = {
    report,
    tolerance: { ...DEFAULT_TOLERANCE }
  };
  writeFileSync(baselinePath, JSON.stringify(payload, null, 2));
};

const main = () => {
  const report = runCrossTierValidation();
  writeBaseline(report);
  console.log(`Cross-tier baseline saved to ${baselinePath}`);
};

main();
