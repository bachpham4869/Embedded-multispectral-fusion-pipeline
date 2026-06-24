"""
Integrity test: production joblib SHA-256 vs models/model_registry.json.

Permitted reads: models/model_registry.json (JSON), raw bytes of
    models/production/env_classifier.joblib for SHA-256 only.
No joblib.load — reads raw bytes, so no sklearn version dependency.
No GPU, no network.

Behaviour:
  - If models/production/env_classifier.joblib is missing (gitignored on dev machines),
    the test calls pytest.skip() with an explanatory reason.
  - If the file is present and its SHA-256 does NOT match the registry, the test FAILS.
    CI silence when joblib is absent is intentional and documented here.
"""
import hashlib
import json
import pathlib
import pytest

_REPO = pathlib.Path(__file__).parent.parent
_JOBLIB = _REPO / "models/production/env_classifier.joblib"
_REGISTRY = _REPO / "models/model_registry.json"
_REGISTRY_KEY = "production/env_classifier.joblib"


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_production_joblib_sha256_matches_registry():
    """
    Verify production joblib hash against model_registry.json.

    Skips cleanly if joblib is not present (gitignored).
    Fails if present but hash diverges from registry.
    """
    if not _JOBLIB.exists():
        pytest.skip(
            f"models/production/env_classifier.joblib not found (gitignored on dev machines). "
            f"Run on a machine with the full model bundle to verify integrity."
        )
    if not _REGISTRY.exists():
        pytest.fail(f"model_registry.json not found: {_REGISTRY}")

    registry = json.loads(_REGISTRY.read_text())
    expected = registry.get(_REGISTRY_KEY, {}).get("sha256")
    if not expected:
        pytest.fail(
            f"SHA-256 entry for '{_REGISTRY_KEY}' not found in {_REGISTRY}. "
            f"Run tools/sign_model.py to populate the registry."
        )

    actual = _sha256_file(_JOBLIB)
    assert actual == expected, (
        f"SHA-256 mismatch for {_JOBLIB.name}:\n"
        f"  registry: {expected}\n"
        f"  computed: {actual}\n"
        f"Re-run tools/sign_model.py if the model was legitimately updated."
    )
