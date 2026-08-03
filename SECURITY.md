# Security Policy

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

- Preferred: use GitHub's private vulnerability reporting for this repository
  (Security tab → **Report a vulnerability**).
- Alternative: email **kamineniabhinaysai@gmail.com** with a description of
  the issue, steps to reproduce, and its potential impact.

We aim to acknowledge reports within 7 days and to keep you updated as we
investigate and fix confirmed issues.

## Supported Versions

eeg-seizure-detection does not yet have tagged releases; security fixes are
applied to the `main` branch. Run the latest `main` to get fixes as soon as
they land.

## Demo / Upload Surface

The Streamlit demo (`demo/app.py`) accepts user-uploaded EDF files and can
load model checkpoints from disk. Uploads are validated for size and type,
and checkpoint loading avoids unsafe pickle paths where possible
(`demo/security.py`). Still treat the demo as a **local research tool**:

- Do not expose it on an untrusted network without additional auth and rate
  limits.
- Do not upload clinical recordings you are not authorized to process.
- Prefer sandboxed or ephemeral environments when sharing demos publicly.

## Data Handled

This project processes EEG signal data (EDF and derived tensors). Clinical
EEG can be sensitive health information. Do not commit real patient
recordings, and avoid publishing checkpoints that may embed identifiable
metadata. Prefer synthetic or de-identified datasets for public demos.

## Reporting Non-Security Bugs

For non-security bugs, please use the normal GitHub issue tracker.
