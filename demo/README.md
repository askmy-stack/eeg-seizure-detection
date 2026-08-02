# Demo — EEG Seizure Detection

Interactive Streamlit app for real-time seizure detection on raw EEG files.

## Run

```bash
streamlit run demo/app.py
# Opens at http://localhost:8501
```

## Features
- Upload `.edf` EEG files (validated: `.edf` extension, EDF header, 50 MB max)
- CNN-based inference (raw signal)
- Feature-based inference (528-feature pipeline)
- Per-window seizure probability heatmap
- EEG signal visualisation

## Security

- **Upload validation** (`demo/security.py`): local uploads must use a `.edf` extension, pass an EDF/BDF header check, and stay under 50 MB.
- **Checkpoint loading** (`demo/security.py`): prefers `model.safetensors` + `config.json` sidecars when present; otherwise loads `.pt` checkpoints with `torch.load(..., weights_only=True)` and registered safe globals (no `weights_only=False`).

## Figures
Place demo screenshots, GIFs, and video recordings in `demo/fig/`.
