# TypeScript SDK

The SDK lives in `sdk/typescript`. Build it with:

```bash
cd sdk/typescript
npm install
npm run build
```

Consume the generated package via a workspace reference or by linking (`npm link`).

```ts
import { IndraClient } from '@indra/sdk';

const client = new IndraClient({ baseUrl: 'http://127.0.0.1:8787' });

const render = await client.render({
  input: './input.png',
  output: './output.png',
  manifest: './sample-manifest.json',
  preset: 'balanced-optics',
});

console.log(render.metrics.indraIndex);
```

## API

### Constructor

`new IndraClient(options?)`

- `baseUrl` – REST endpoint (default `http://127.0.0.1:8787`)
- `fetchImpl` – custom `fetch` implementation (optional)

### `render(request)`

Processes a single image. Request mirrors the REST `/render` body. Returns metrics, telemetry, and the resolved output path.

### `simulate(request)`

Runs the headless simulation pipeline. Request parameters match `/simulate`. Returns aggregated metrics.

### `capture(request)`

Batch-processes a video. Request parameters match `/capture`. Returns frame stats and output path.

### `validateManifest(request)`

Validates inline JSON (`manifest`) or a file path (`manifestPath`). Returns manifest stats and warnings.

### `health()`

Returns `true` when the server responds with `status: ok`.
