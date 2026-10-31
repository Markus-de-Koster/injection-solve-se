"""The paper's experiments as argument lists of the vendored drivers.

One YAML per experiment in ``experiments/`` holds the arguments of the run that produced
the paper's numbers (``args``) and the arguments appended for a smoke run (``smoke``). A
``train`` experiment runs ``run/examples/injection_hse/train_se_full_workflow.py`` (it
generates its training data, trains, and evaluates the run on its own splits) and, when it
has a ``continuation`` list, then :mod:`se_ablation.continuation` with those options
(``continuation_smoke`` appended for a smoke run); a ``data`` experiment runs
``generate_se_task_datasets.py``. ``{configs}`` in an argument is the repository's
``run/configs`` directory.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
CONFIGS = ROOT / "run" / "configs"
DRIVERS = ROOT / "run" / "examples" / "injection_hse"
DRIVER = {
    "train": DRIVERS / "train_se_full_workflow.py",
    "data": DRIVERS / "generate_se_task_datasets.py",
}

#: options of the training driver that also select the generated data (the subset that
#: ``generate_se_task_datasets.py`` understands with the same meaning)
DATA_OPTIONS = {
    "--scenario-config",
    "--orders",
    "--device-capacitance",
    "--source-impedance",
    "--load-shunt",
    "--load-shunt-basis",
    "--n-random",
    "--n-scenarios",
    "--n-steps",
    "--seed",
    "--chunk-size",
    "--dtype",
}


@dataclass
class Experiment:
    """One experiment: its name, display name, kind and argument lists."""

    name: str
    display: str
    kind: str
    args: list[str]
    smoke: list[str] = field(default_factory=list)
    continuation: list[str] = field(default_factory=list)
    continuation_smoke: list[str] = field(default_factory=list)

    def argv(self, *, smoke: bool = False, extra: list[str] | None = None) -> list[str]:
        """The driver arguments (after the output directory), with ``{configs}`` resolved;
        ``smoke`` appends the smoke sizes, ``extra`` comes last and wins."""
        items = self.args + (self.smoke if smoke else []) + list(extra or [])
        out: list[str] = []
        for item in items:
            out += shlex.split(item.replace("{configs}", str(CONFIGS)))
        return out

    def continuation_argv(
        self, *, smoke: bool = False, extra: list[str] | None = None
    ) -> list[str]:
        """The options of :mod:`se_ablation.continuation` (empty: no continuation)."""
        if not self.continuation:
            return []
        items = self.continuation + (self.continuation_smoke if smoke else []) + list(extra or [])
        return [tok for item in items for tok in shlex.split(item)]

    @property
    def driver(self) -> Path:
        return DRIVER[self.kind]

    def data_argv(self, *, smoke: bool = False, extra: list[str] | None = None) -> list[str]:
        """The training data of a ``train`` experiment as ``generate_se_task_datasets.py``
        arguments: its data-selecting options only."""
        argv = self.argv(smoke=smoke, extra=extra)
        out, i = [], 0
        while i < len(argv):
            opt = argv[i]
            has_value = i + 1 < len(argv) and not argv[i + 1].startswith("--")
            if opt in DATA_OPTIONS:
                out += argv[i : i + 2] if has_value else [opt]
            i += 2 if has_value else 1
        return out


def load(name: str) -> Experiment:
    """The experiment ``experiments/<name>.yaml``."""
    path = EXPERIMENTS / f"{name}.yaml"
    if not path.is_file():
        known = ", ".join(sorted(p.stem for p in EXPERIMENTS.glob("*.yaml")))
        raise SystemExit(f"unknown experiment {name!r}; known: {known}")
    doc = yaml.safe_load(path.read_text())
    return Experiment(
        name=doc["name"],
        display=doc["display"],
        kind=doc["kind"],
        args=[str(a) for a in doc["args"]],
        smoke=[str(a) for a in doc.get("smoke", [])],
        continuation=[str(a) for a in doc.get("continuation", [])],
        continuation_smoke=[str(a) for a in doc.get("continuation_smoke", [])],
    )


def names(kind: str | None = None) -> list[str]:
    """The experiment names (of one ``kind``)."""
    return sorted(
        p.stem
        for p in EXPERIMENTS.glob("*.yaml")
        if kind is None or yaml.safe_load(p.read_text())["kind"] == kind
    )
