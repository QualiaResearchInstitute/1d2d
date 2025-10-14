# Indra Studio User Guide

This guide walks through the end-to-end workflow for operating the Indra studio in headless or scripted environments. It complements the interactive UI by documenting the new Phase� 10 interfaces (CLI, REST API, and SDKs) and providing reproducible recipes.

## Prerequisites

- Node.js 18+
- Python 3.9+ (for notebook and scripting workflows)
- ffmpeg/ffprobe available on the `PATH`

Clone the repository, install dependencies, and build:

```bash
npm install
npm run build
```

## Running the desktop studio

```bash
npm run dev
```

Open the printed local URL to access the UI. Use the **Presets** panel to load manifests and the **Output** panel to drive capture.

## Using the CLI

Build the CLI bundle and inspect the available commands:

```bash
npm run cli:build
node dist-cli/cli/indraCli.js --help
```

Typical workflows:

- **Validate a manifest** – `indra-cli manifest validate docs/assets/sample-manifest.json`
- **Apply a preset to an image** – `indra-cli apply --input input.png --output rendered.png --manifest sample.json --preset balanced-optics`
- **Offline simulation** – `indra-cli simulate --input input.png --frames 240 --output metrics.json`
- **Capture video** – `indra-cli capture --input footage.mov --output indra-output.mp4 --manifest sample.json`
- **Telemetry sink** – `indra-cli telemetry --output run.jsonl`

See `docs/reference/cli.md` for exhaustive flag documentation.

## REST service

Start the local API server:

```bash
npm run cli:build
node dist-cli/server/index.js -- (optional)
```

The server listens on `http://127.0.0.1:8787` by default. Important endpoints:

- `GET /health` – readiness probe
- `POST /render` – process an image
- `POST /simulate` – headless simulation
- `POST /capture` – video pipeline
- `POST /manifest/validate` – inline or path-based validation

Each endpoint returns JSON. See `docs/reference/rest.md` for detailed schemas.

## SDKs

- **TypeScript** – located at `sdk/typescript`. Build with `npm run build` inside that folder and consume via `import { IndraClient } from '@indra/sdk';`.
- **Python** – located at `sdk/python`. Install in editable mode (`pip install -e sdk/python`) and use `from indra_sdk import IndraClient`.

Example scripts are available under `docs/examples/` and in the `examples/` directory.

## Tutorials

1. **Kaleidoscopic Oscillator Pattern**
   - Use the CLI to apply the `balanced-optics` preset to `examples/assets/kaleidoscope.png`.
   - Enable Kuramoto (`--preset balanced-optics` already includes tuned values).
   - Capture the output video: `indra-cli capture --input kaleidoscope.mov --output kaleidoscope-indra.mp4`.

2. **Python parameter sweep**
   - Start the REST server.
   - Run `python examples/python/parameter_sweep.py` to generate a CSV of Indra Index vs. rim weight.
   - Plot the results (the script includes a matplotlib snippet).

3. **Telemetry logging**
   - Launch `indra-cli telemetry --output session.jsonl`.
   - In the UI, enable telemetry streaming to `ws://localhost:8090/telemetry`.
   - Inspect the JSONL output for frame-by-frame metrics.

## Next steps

- Review the parameter reference (`docs/manual/parameter-reference.md`) for detailed descriptions of every control and measurement.
- Use the developer guide (`docs/manual/developer-guide.md`) to understand the project layout, extension points, and release process.
- Consult the glossary (`docs/manual/glossary.md`) when encountering specialised terminology.
