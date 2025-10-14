"""Parameter sweep example using the Python SDK.

This script assumes the REST server is running (see docs/reference/rest.md) and that an input
image is available. Update INPUT_IMAGE with a valid path before executing.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from indra_sdk import IndraClient

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR.parent / "public" / "sample-manifest.json"
INPUT_IMAGE = BASE_DIR / "assets" / "sample-input.png"  # create or replace before running
OUTPUT_DIR = BASE_DIR / "artifacts"


def mutate_manifest(blend: float) -> Path:
    data = json.loads(MANIFEST_PATH.read_text())
    presets = data.get("controls", {}).get("presets", [])
    for preset in presets:
        if preset.get("id") == "balanced-optics":
            preset.setdefault("panels", {}).setdefault("compositor", {})["blend"] = blend
    with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="indra-manifest-", delete=False) as tmp:
        json.dump(data, tmp)
        temp_path = Path(tmp.name)
    return temp_path


def main() -> None:
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            f"Input image missing: {INPUT_IMAGE}. Create the file before running this example."
        )

    client = IndraClient()
    if not client.health():
        raise RuntimeError("Local Indra REST server is not reachable on http://127.0.0.1:8787")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "blend_sweep.csv"

    blends = [0.3, 0.4, 0.5, 0.6, 0.7]
    rows = ["blend", "indra_index", "rim_mean", "coherence_mean"]

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(rows)

        for blend in blends:
            manifest = mutate_manifest(blend)
            try:
                result = client.render(
                    {
                        "input": str(INPUT_IMAGE),
                        "output": str(OUTPUT_DIR / f"render_{blend:.2f}.png"),
                        "manifest": str(manifest),
                        "preset": "balanced-optics",
                    }
                )
                metrics = result["metrics"]
                writer.writerow(
                    [blend, metrics["indraIndex"], metrics["rimMean"], metrics["coherenceMean"]]
                )
            finally:
                manifest.unlink(missing_ok=True)

    print(f"Sweep complete. Metrics written to {csv_path}")


if __name__ == "__main__":
    main()
