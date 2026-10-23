import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train a small model set once per test session (about 1-2 minutes on CPU)."""
    pytest.importorskip("torch")
    from xgnn_ids.training.experiment import run_experiment

    base = tmp_path_factory.mktemp("run")
    models, results = base / "models", base / "results"
    report = run_experiment(None, str(results), str(models), quick=True, models=["egraphsage", "gat", "mlp"],
                            baselines=["rf"], cross_domain=True, zero_day=False, adversarial=True, xai=True,
                            duration_s=360, log=lambda *_: None)
    os.environ["XGNN_MODELS_DIR"] = str(models)
    os.environ["XGNN_RESULTS_DIR"] = str(results)
    return {"models": models, "results": results, "report": report}
