# REST API Reference

Start the server via `node dist-cli/server/index.js` (after `npm run cli:build`). All endpoints accept and return JSON. Default base URL: `http://127.0.0.1:8787`.

## `GET /health`

Returns `{ "status": "ok" }` when the service is ready.

## `POST /manifest/validate`

Validates a manifest supplied inline or via path.

### Request body

```
{
  "manifest": { ... }  // optional
  "manifestPath": "path/to/manifest.json" // optional
}
```

At least one of `manifest` or `manifestPath` must be provided.

### Response (success)

```
{
  "status": "ok",
  "manifest": {
    "name": "Phase Three Interactive Demo",
    "schemaVersion": "1.0.0",
    "nodes": 42,
    "links": 17
  },
  "warnings": [ { "code": "...", "message": "...", "severity": "warning" } ]
}
```

### Response (error)

```
{
  "status": "error",
  "message": "Manifest invalid: ...",
  "issues": [ { "code": "...", "path": [...], "severity": "error" } ]
}
```

## `POST /render`

Processes a single frame.

### Request

```
{
  "input": "path/to/image.png",
  "output": "path/to/output.png",
  "manifest": "path/to/manifest.json",   // optional
  "preset": "balanced-optics",          // optional
  "ffmpeg": "/usr/bin/ffmpeg",          // optional
  "ffprobe": "/usr/bin/ffprobe",        // optional
  "bitDepth": 8                           // optional (8|10|16)
}
```

### Response

```
{
  "status": "ok",
  "output": "<absolute path>",
  "width": 1920,
  "height": 1080,
  "manifest": "path/to/manifest.json",
  "preset": "balanced-optics",
  "telemetry": { ... },
  "metrics": {
    "rimMean": 0.42,
    "warpMean": 0.31,
    "coherenceMean": 0.78,
    "indraIndex": 0.68
  }
}
```

## `POST /simulate`

Runs a headless simulation and aggregates metrics.

### Request

```
{
  "input": "path/to/image.png",
  "manifest": "path/to/manifest.json", // optional
  "preset": "balanced-optics",         // optional
  "frames": 240,                        // optional (default 120)
  "dt": 0.016,                          // optional (default 1/60)
  "seed": 1337,                         // optional (default 1337)
  "ffmpeg": "ffmpeg",                  // optional
  "ffprobe": "ffprobe"                 // optional
}
```

### Response

```
{
  "status": "ok",
  "frames": 240,
  "dt": 0.016,
  "manifest": "path/to/manifest.json",
  "preset": "balanced-optics",
  "metrics": {
    "rimMean": 0.41,
    "cohMean": 0.76,
    "indraIndex": 0.67
  }
}
```

## `POST /capture`

Applies the pipeline to each frame of a video.

### Request

```
{
  "input": "path/to/input.mov",
  "output": "path/to/output.mp4",
  "manifest": "path/to/manifest.json", // optional
  "preset": "balanced-optics",         // optional
  "frames": 120,                        // optional limit
  "keepTemp": false,                    // optional (keep decoded PNGs)
  "ffmpeg": "ffmpeg",                  // optional
  "ffprobe": "ffprobe"                 // optional
}
```

### Response

```
{
  "status": "ok",
  "frames": 300,
  "width": 1920,
  "height": 1080,
  "fps": 59.94,
  "durationSeconds": 32.4,
  "output": "<absolute path>/output.mp4"
}
```

Errors follow the `{ "status": "error", "message": "..." }` shape with optional `issues` for validation details.
