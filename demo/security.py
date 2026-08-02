from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

MAX_EDF_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_EDF_EXTENSIONS = {".edf"}
ALLOWED_EDF_CONTENT_TYPES = frozenset(
    {
        "application/octet-stream",
        "application/edf",
        "application/x-edf",
        "binary/octet-stream",
    }
)
EDF_VERSION_MARKERS = (b"0       ", b"255     ")
EDF_HEADER_BYTES = 256


def validate_edf_upload(
    file_bytes: bytes,
    file_name: str,
    *,
    content_type: str | None = None,
    max_bytes: int = MAX_EDF_UPLOAD_BYTES,
) -> None:
    """Validate uploaded EEG file size, extension, MIME type, and EDF header."""
    suffix = Path(file_name).suffix.lower()
    if suffix not in ALLOWED_EDF_EXTENSIONS:
        raise ValueError(
            f"Invalid file extension '{suffix or '(none)'}'. Only .edf files are accepted."
        )

    if content_type:
        normalized_type = content_type.split(";", 1)[0].strip().lower()
        if normalized_type not in ALLOWED_EDF_CONTENT_TYPES:
            raise ValueError(
                f"Unsupported content type '{content_type}'. "
                "Upload an EDF file with a .edf extension."
            )

    if not file_bytes:
        raise ValueError("Empty file upload.")
    if len(file_bytes) > max_bytes:
        max_mb = max_bytes // (1024 * 1024)
        raise ValueError(f"File exceeds the {max_mb} MB upload limit.")

    if len(file_bytes) < EDF_HEADER_BYTES:
        raise ValueError("File is too small to contain a valid EDF header.")

    if file_bytes[:8] not in EDF_VERSION_MARKERS:
        raise ValueError("Invalid EDF header: unrecognized version field.")


def _register_checkpoint_safe_globals() -> None:
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    import numpy as np

    safe_types: list[type] = [np.dtype, np.ndarray]
    for name in ("float64", "float32", "int64", "int32", "bool_"):
        candidate = getattr(np, name, None)
        if isinstance(candidate, type):
            safe_types.append(candidate)
    scalar = getattr(getattr(np, "core", None), "multiarray", None)
    if scalar is not None:
        scalar_type = getattr(scalar, "scalar", None)
        if isinstance(scalar_type, type):
            safe_types.append(scalar_type)
    add_safe_globals(safe_types)


def _checkpoint_payload_from_sidecars(
    safetensors_path: Path,
    config_path: Path,
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config sidecar for safetensors checkpoint: {config_path}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "model_state_dict": load_file(str(safetensors_path), device="cpu"),
        "model_config": config.get("model_config", {}),
        "model_builder": config.get("model_builder"),
        "model_class": config.get("model_class"),
        "input_spec": config.get("input_spec", {}),
        "preprocess": config.get("preprocess", {}),
        "schema_version": config.get("schema_version"),
        "optimal_threshold": config.get("optimal_threshold", 0.5),
        "git_commit": config.get("git_commit"),
        "epoch": config.get("epoch", 0),
        "config": config.get("config", {}),
    }
    if metrics_path is not None and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["val_metrics"] = metrics.get("val", {})
        if "best_threshold" in metrics:
            payload["best_threshold"] = metrics["best_threshold"]
    return payload


def _load_pt_checkpoint_safe(checkpoint_path: Path) -> dict[str, Any]:
    _register_checkpoint_safe_globals()
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not safely load checkpoint '{checkpoint_path.name}'. "
            "Place a safetensors sidecar (model.safetensors + config.json) next to the "
            "checkpoint or re-save the artifact with tensor-safe metadata."
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path.name}' did not deserialize to a metadata dict."
        )
    return payload


def load_trusted_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load a local project checkpoint without unsafe pickle deserialization."""
    checkpoint_path = Path(checkpoint_path)
    safetensors_path = checkpoint_path.with_name("model.safetensors")
    config_path = checkpoint_path.with_name("config.json")
    metrics_path = checkpoint_path.with_name("metrics.json")

    if safetensors_path.exists():
        return _checkpoint_payload_from_sidecars(
            safetensors_path,
            config_path,
            metrics_path if metrics_path.exists() else None,
        )

    tabnet_safetensors = checkpoint_path.with_name("network.safetensors")
    tabnet_config = checkpoint_path.with_name("tabnet_config.json")
    if tabnet_safetensors.exists() and tabnet_config.exists():
        from safetensors.torch import load_file

        config = json.loads(tabnet_config.read_text(encoding="utf-8"))
        return {
            "args": config.get("args", {}),
            "network_state_dict": load_file(str(tabnet_safetensors), device="cpu"),
        }

    return _load_pt_checkpoint_safe(checkpoint_path)
