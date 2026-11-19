"""Central path configuration for the Within-Adapt reproducibility suite.

By default, precomputed result archives are loaded from and written to
``<repo_root>/results`` and raw public EEG datasets are expected under
``<repo_root>/data``. Both locations can be overridden via environment
variables without modifying source code:

- ``WITHIN_ADAPT_RESULTS_DIR``: directory for `.npz` result archives
- ``WITHIN_ADAPT_DATA_ROOT``: root directory holding downloaded public datasets
- ``EEGBCI_DIR``, ``BCIC2A_DIR``, ``CHANG2025_DIR``, ``KAYA2018_DIR``:
  per-dataset overrides
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(os.environ.get("WITHIN_ADAPT_RESULTS_DIR", str(REPO_ROOT / "results"))).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(os.environ.get("WITHIN_ADAPT_DATA_ROOT", str(REPO_ROOT / "data"))).resolve()

EEGBCI_DIR = Path(
    os.environ.get("EEGBCI_DIR", str(DATA_ROOT / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"))
).resolve()
BCIC2A_DIR = Path(
    os.environ.get("BCIC2A_DIR", str(DATA_ROOT / "MNE-bnci-data" / "~bci" / "database" / "001-2014"))
).resolve()
CHANG2025_DIR = Path(
    os.environ.get("CHANG2025_DIR", str(DATA_ROOT / "MNE-chang2025-data"))
).resolve()
KAYA2018_DIR = Path(
    os.environ.get("KAYA2018_DIR", str(DATA_ROOT / "MNE-kaya2018-data"))
).resolve()
