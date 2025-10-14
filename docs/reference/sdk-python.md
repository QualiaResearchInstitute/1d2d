# Python SDK

The Python client lives in `sdk/python`. Install it in editable mode:

```bash
pip install -e sdk/python
```

```py
from indra_sdk import IndraClient

client = IndraClient()

render = client.render({
    "input": "./input.png",
    "output": "./output.png",
    "manifest": "./sample-manifest.json",
    "preset": "balanced-optics",
})
print(render["metrics"]["indraIndex"])
```

## API

### `IndraClient(options=None)`

- `base_url` – REST endpoint (default `http://127.0.0.1:8787`)
- `timeout` – request timeout in seconds (default `60`)

Instantiate via `IndraClient(IndraClientOptions(base_url='http://localhost:8787'))` or use keyword arguments.

### `health()`

Returns `True` if the server reports `status: ok`.

### `render(payload)`

Payload mirrors the REST `/render` body. Returns the JSON response as a dict (minus the `status` field).

### `simulate(payload)`

Payload mirrors `/simulate`. Returns aggregated metrics.

### `capture(payload)`

Payload mirrors `/capture`. Returns frame statistics.

### `validate_manifest(payload)`

Include `manifest` (dict) or `manifestPath` (str). Returns manifest metadata and warnings.

## Error handling

REST errors raise `RuntimeError` with the server-provided message. Validation helper methods raise `ValueError` when required keys are missing.
