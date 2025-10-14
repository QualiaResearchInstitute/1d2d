/**
 * Example usage of the TypeScript SDK.
 *
 * Build the SDK first:
 *   cd sdk/typescript && npm install && npm run build && npm link
 *   npm link @indra/sdk
 * Then execute this script with ts-node or compile with tsc.
 */

import { IndraClient } from '@indra/sdk';

async function main() {
  const client = new IndraClient();
  const response = await client.render({
    input: './examples/assets/input.png',
    output: './examples/artifacts/render-ts.png',
    manifest: './public/sample-manifest.json',
    preset: 'balanced-optics',
  });

  console.log(`Indra index: ${response.metrics.indraIndex.toFixed(3)}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
