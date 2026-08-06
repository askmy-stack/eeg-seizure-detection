# Contributing

Thanks for helping improve this EEG seizure detection research codebase.

## Getting started

1. Fork and clone the repository.
2. Create a virtual environment and install dependencies (see [Quick Start](README.md#-quick-start) in the README).
3. Create a focused branch from `main`:

   ```bash
   git checkout -b your-branch-name
   ```

## Where to contribute (model families)

Work usually lands in one architecture family. Prefer small PRs that stay inside a single family when possible:

| Family | Typical location under `src/component/` |
| --- | --- |
| Legacy / benchmark LSTMs | `models/` LSTM and m1–m7 paths |
| Improved LSTMs | Improved LSTM / HierarchicalLSTM training entrypoints |
| State-space (Mamba) | EEGMamba / MoE modules |
| HF custom CNNs | Custom CNN training + configs |
| HF pretrained (transformers) | ST-EEGFormer, BENDR, EEGPT, BIOT wrappers |
| Classical ML | Feature pipelines + LightGBM / XGBoost / RF |
| Meta-ensemble | Ensemble strategies and evaluation |

Dataset download, windowing, and preprocessing changes should keep subject-independent splits intact — training subjects must never appear in the test set.

## Development checks

```bash
pytest src/tests/ -q
```

For full CHB-MIT label verification against a local dataset:

```bash
export EEG_RESULTS_ROOT=/path/to/results
export EEG_CSV_BASE_DIR=/path/to/src/data_loader
python3 src/tests/test_labels.py
```

Do not commit secrets, large raw EDF corpora, or trained weight files (`*.pt`, `*.ckpt`, etc.).

## Pull requests

- Keep changes focused and reviewable.
- Describe which model family or pipeline stage you touched.
- Link related issues.
- CI / local tests should pass for the files you changed.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be respectful and constructive.

## Security

Report vulnerabilities privately via [SECURITY.md](SECURITY.md).
