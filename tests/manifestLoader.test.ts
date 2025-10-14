import { readFileSync } from 'node:fs';
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { resolve } from 'node:path';
import { validateManifest } from '../src/manifest/schema.js';
import { createRuntimeBundle } from '../src/manifest/runtime.js';

const fixturePath = resolve(process.cwd(), 'public/sample-manifest.json');

const loadFixture = () => {
  const buffer = readFileSync(fixturePath, 'utf8');
  return JSON.parse(buffer);
};

test('sample manifest validates and maps to runtime bundle', () => {
  const manifestJson = loadFixture();
  const result = validateManifest(manifestJson);
  const errorIssues = result.issues.filter((issue) => issue.severity === 'error');
  assert.equal(
    errorIssues.length,
    0,
    `Expected no errors, received ${JSON.stringify(errorIssues)}`,
  );

  const bundle = createRuntimeBundle(result.manifest);
  assert.equal(bundle.scene.nodes.length, 7);
  assert.equal(bundle.scene.links.length, 6);
  assert.equal(bundle.timeline.duration, 18);
  assert.equal(bundle.timeline.clips.length, 2);
  assert.equal(bundle.scene.nodes[0]?.parameters.length, 5);
  assert.equal(bundle.controls?.panels.length ?? 0, 7);
  assert.equal(bundle.controls?.presets?.length ?? 0, 2);
});

test('invalid manifest surfaces validation errors', () => {
  const manifest = {
    schemaVersion: '1.0.0',
    metadata: { name: 'Invalid manifest' },
    nodes: [
      {
        id: 'nodeA',
        type: 'TestNode',
        label: 'Test node',
      },
    ],
    links: [
      {
        id: 'invalid-link',
        from: { nodeId: 'missing-node', port: 'out' },
        to: { nodeId: 'nodeA', port: 'in' },
      },
    ],
  };

  try {
    validateManifest(manifest);
    assert.fail('Expected manifest validation to throw');
  } catch (error) {
    assert.equal(error instanceof Error, true);
  }
});
