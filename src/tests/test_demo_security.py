"""Security helpers for the Streamlit demo (issues #41, #42)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from demo.security import (
    EDF_VERSION_MARKERS,
    MAX_EDF_UPLOAD_BYTES,
    load_trusted_checkpoint,
    validate_edf_upload,
)


def _edf_bytes(version: bytes = EDF_VERSION_MARKERS[0], payload_size: int = 256) -> bytes:
    header = version + b"x" * (256 - len(version))
    if payload_size > 256:
        header += b"y" * (payload_size - 256)
    return header


def test_validate_edf_upload_accepts_valid_edf():
    validate_edf_upload(_edf_bytes(), "recording.edf", content_type="application/octet-stream")


def test_validate_edf_upload_rejects_bad_extension():
    with pytest.raises(ValueError, match="Invalid file extension"):
        validate_edf_upload(_edf_bytes(), "recording.csv")


def test_validate_edf_upload_rejects_oversized_file():
    oversized = _edf_bytes(payload_size=MAX_EDF_UPLOAD_BYTES + 1)
    with pytest.raises(ValueError, match="50 MB"):
        validate_edf_upload(oversized, "recording.edf")


def test_validate_edf_upload_rejects_invalid_header():
    with pytest.raises(ValueError, match="Invalid EDF header"):
        validate_edf_upload(b"NOT_AN_EDF_HEADER" + b"x" * 240, "recording.edf")


def test_validate_edf_upload_rejects_unsupported_content_type():
    with pytest.raises(ValueError, match="Unsupported content type"):
        validate_edf_upload(
            _edf_bytes(),
            "recording.edf",
            content_type="text/plain",
        )


def test_load_trusted_checkpoint_from_safetensors_sidecar(tmp_path: Path):
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    tensor = torch.ones(2, 2)
    sidecar_dir = tmp_path / "enhanced_cnn_1d"
    sidecar_dir.mkdir()

    save_file({"layer.weight": tensor}, str(sidecar_dir / "model.safetensors"))
    (sidecar_dir / "config.json").write_text(
        json.dumps(
            {
                "model_config": {"in_channels": 16},
                "optimal_threshold": 0.42,
                "input_spec": {"channels": 16, "sfreq": 256, "window_sec": 1.0},
            }
        ),
        encoding="utf-8",
    )
    (sidecar_dir / "metrics.json").write_text(
        json.dumps({"val": {"f1": 0.9}}),
        encoding="utf-8",
    )

    payload = load_trusted_checkpoint(sidecar_dir / "best_model.pt")

    assert payload["model_state_dict"]["layer.weight"].shape == (2, 2)
    assert payload["model_config"]["in_channels"] == 16
    assert payload["optimal_threshold"] == pytest.approx(0.42)
    assert payload["val_metrics"]["f1"] == pytest.approx(0.9)


def test_load_trusted_checkpoint_weights_only_pt(tmp_path: Path):
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save(
        {
            "model_state_dict": {"layer.weight": torch.zeros(1)},
            "model_config": {"in_channels": 16},
            "optimal_threshold": 0.5,
        },
        checkpoint_path,
    )

    payload = load_trusted_checkpoint(checkpoint_path)

    assert payload["model_state_dict"]["layer.weight"].shape == (1,)
    assert payload["model_config"]["in_channels"] == 16
