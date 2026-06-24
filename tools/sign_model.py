#!/usr/bin/env python3
"""P2-E: Compute SHA-256 for a model bundle and update models/model_registry.json.

Usage:
    python tools/sign_model.py models/production/env_classifier.json
    python tools/sign_model.py models/baseline/*.json        # multiple sidecars

The registry lives at models/model_registry.json.
EnvClassifier reads it on load and warns when a digest mismatches — it never blocks
inference, so you can deploy the registry alongside the bundle and rotate it when
you retrain.
"""
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    bundles = [pathlib.Path(a).resolve() for a in sys.argv[1:]]
    missing = [b for b in bundles if not b.is_file()]
    if missing:
        for m in missing:
            print(f"[sign] ERROR: not found — {m}", file=sys.stderr)
        sys.exit(1)

    # Registry sits two levels above each bundle (models/model_registry.json).
    # Use the first bundle to determine the repo root; all bundles must share one registry.
    registry_path = bundles[0].parent.parent / "model_registry.json"
    registry: dict = {}
    if registry_path.is_file():
        with open(registry_path) as fh:
            registry = json.load(fh)

    for bundle in bundles:
        digest = sha256_file(bundle)
        registry[bundle.name] = {
            "sha256": digest,
            "path": str(bundle),
            "signed_at": datetime.now(timezone.utc).isoformat(),
        }
        print(f"[sign] {bundle.name}: sha256={digest[:16]}…")

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as fh:
        json.dump(registry, fh, indent=2)
    print(f"[sign] Registry updated → {registry_path}")


if __name__ == "__main__":
    main()
