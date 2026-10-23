"""XGNN-IDS: Explainable Graph Neural Network based Network Intrusion Detection.

Network traffic is modelled as a sequence of directed temporal graphs
(hosts = nodes, flows = edges). Edge-level GNNs classify every flow into an
attack family, and an explainability layer (Integrated Gradients, SHAP,
GNNExplainer and graph evidence) tells the analyst *why*.
"""

__version__ = "1.0.0"

from .config import CLASSES, ATTACK_CLASSES  # noqa: F401
