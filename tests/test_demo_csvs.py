"""The bundled demo captures give the verdicts the README promises, with the shipped models."""
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
MIN_CONF = 0.9          # the console's alert threshold


@pytest.fixture(scope="module")
def detector():
    from xgnn_ids.inference import Detector

    return Detector.load(ROOT / "models")


def _incidents(detector, name):
    res = detector.detect(pd.read_csv(ROOT / "data" / "demo" / name))
    inc = detector.incidents(res, MIN_CONF)
    print(name, {i["class"]: i["flows"] for i in inc})
    return {i["class"] for i in inc}


def test_safe_capture_has_no_incidents(detector):
    assert _incidents(detector, "1_safe.csv") == set()


def test_medium_capture(detector):
    found = _incidents(detector, "2_medium_risk.csv")
    assert {"PortScan", "BruteForce"} <= found and not found & {"DDoS", "DoS", "Botnet", "Infiltration"}


def test_high_capture(detector):
    assert {"DDoS", "Botnet", "WebAttack", "Infiltration"} <= _incidents(detector, "3_high_risk.csv")
