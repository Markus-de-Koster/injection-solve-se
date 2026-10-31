"""Estimator names, meter placements and the labels of the evaluation outputs.

Every table and figure takes its labels from here. The evaluation CSVs identify a row by
``run`` (``"baseline"`` or the experiment name of a network) and ``estimator`` (the decode
variant, see :mod:`se_ablation.evaluate`); :data:`TABLE_ROWS` maps them to the names used
in the paper.

Buses are named as in the CIGRE LV benchmark (``Bus R4`` = bus 4 of the residential feeder,
``I`` industrial, ``C`` commercial). The paper numbers the 44 buses from 0 in node order
(``Bus 0``, ``Bus R0`` … ``Bus R18``, ``Bus I0`` … ``Bus I2``, ``Bus C0`` … ``Bus C20``): bus
``k`` of the paper is node id ``k + 1`` of ``grid.json`` (:func:`bus_number`), so the deployed
meters R4, I2 and C5 are buses 5, 22 and 28 and the alternative ones R6, I2, C4 buses 7, 22
and 27.
"""

from __future__ import annotations

#: named meter placements (the supply is always metered in addition)
PLACEMENTS: dict[str, list[str]] = {
    # one meter per feeder (buses 5, 22, 28), the placement GNN-P is trained for
    "deployed": ["Bus R4", "Bus C5", "Bus I2"],
    # the best of the 720 one-meter-per-feeder placements for LMMSE (buses 7, 22, 27; see
    # se_ablation.evaluate.summarize_search); no network is trained for it
    "alternative": ["Bus R6", "Bus C4", "Bus I2"],
}

#: estimator display names
NAMES = {
    "physics": "Physics",
    "linear": "Linear-P [ours]",
    "lmmse": "LMMSE [ours]",
    "gnn_p": "GNN-P [ours]",
    "gnn_p_r": "GNN-P-R [ours]",
    "mack_dnn": "DNN [Mack]",
    "mack_dnn_i": "DNN+I [Mack]",
    "mack_pann": "PANN [Mack]",
}

#: (name key, run, estimator in the evaluation CSV) of the rows of the paper's table, in its
#: order; ``run = None`` means "the network of this name" (the run column holds the
#: experiment name); the Mack et al. rows (estimator ``None``) come from ``mack table``
TABLE_ROWS = [
    ("physics", "baseline", "physics solve"),
    ("mack_dnn", None, None),
    ("mack_dnn_i", None, None),
    ("mack_pann", None, None),
    ("linear", "baseline", "linear map, best decode per order"),
    ("lmmse", "baseline", "statistics-based (linear MMSE)"),
    ("gnn_p", None, "network, as deployed"),
    ("gnn_p_r", None, "network, as deployed"),
]

#: Mack et al. estimator label in the ``mack table`` outputs -> name key
MACK_KEYS = {"DNN": "mack_dnn", "DNN+I": "mack_dnn_i", "PANN": "mack_pann"}


def label(key: str) -> str:
    """Display name of an estimator key (an unknown key is returned unchanged)."""
    return NAMES.get(key, key)


def bus_number(grid, name: str) -> int:
    """The paper's number of the bus ``name`` (its node id in ``grid`` minus one)."""
    ids = {str(n.name): int(n.id) for n in grid.nodes}
    return ids[name] - 1
