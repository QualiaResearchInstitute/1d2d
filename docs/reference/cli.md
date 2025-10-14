# CLI Reference

The `indra-cli` binary ships with the project (see `package.json` bin entry). Build it via `npm run cli:build` and invoke through `node dist-cli/cli/indraCli.js` or via `npx indra-cli` after linking.

## Global usage

```bash
indra-cli <command> [options]
```

### Commands

| Command             | Summary                                        |
| ------------------- | ---------------------------------------------- |
| `manifest validate` | Validate a manifest JSON file.                 |
| `manifest diff`     | Show structural diffs between two manifests.   |
| `apply`             | Process a single image with a preset/manifest. |
| `simulate`          | Run an offline simulation and emit metrics.    |
| `capture`           | Batch process a video into a rendered output.  |
| `telemetry`         | Start a WebSocket sink for telemetry frames.   |

### manifest validate

```
indra-cli manifest validate <manifest.json> [--json] [--verbose]
```

- `--json` – Emit a machine-readable response
- `--verbose` – Print validation warnings

### manifest diff

```
indra-cli manifest diff <manifest-a.json> <manifest-b.json> [--json]
```

Outputs structural differences (added/removed/changed paths).

### apply

```
indra-cli apply --input <image> --output <image>
                [--manifest <path>] [--preset <id>]
                [--ffmpeg <path>] [--ffprobe <path>]
                [--bit-depth 8|10|16] [--json]
```

Produces a processed frame using the same physics pipeline as the UI.

### simulate

```
indra-cli simulate --input <image>
                   [--manifest <path>] [--preset <id>]
                   [--frames <count>] [--dt <seconds>] [--seed <number>]
                   [--ffmpeg <path>] [--ffprobe <path>]
                   [--output metrics.json] [--json]
```

Aggregates key metrics across `frames` iterations.

### capture

```
indra-cli capture --input <video> --output <video>
                  [--manifest <path>] [--preset <id>]
                  [--frames <count>] [--keep-temp]
                  [--ffmpeg <path>] [--ffprobe <path>] [--json]
```

Processes each decoded frame, re-encodes with ffmpeg, and prints a summary.

### telemetry

```
indra-cli telemetry [--port <number>] [--output metrics.jsonl]
```

Starts a WebSocket server on `ws://localhost:<port>/telemetry` and optionally appends frames to a JSONL file.
