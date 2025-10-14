# Indra Python SDK

This package provides a lightweight Python client for interacting with the local Indra runtime API that ships with the studio. It is designed for quick scripting workflows and notebook experiments.

## Quick start

```py
from indra_sdk import IndraClient

client = IndraClient()
if client.health():
    result = client.simulate({
        "input": "./fixtures/image.png",
        "frames": 60,
    })
    print(result["metrics"])
```

Refer to the main project documentation for the full API surface and examples.
