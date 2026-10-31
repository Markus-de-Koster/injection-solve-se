"""se_ablation — reproduction of the injection-domain harmonic state estimation study.

The estimators, data generation and training live in :mod:`injection_hse` (a vendored
subset of the study's code base); this package adds what the paper needs on top of it:

- ``experiments``  the paper's experiments as argument lists of the workflow driver;
- ``continuation`` the continued training of a finished run in its last pass;
- ``frame``        the decode frame shared by every scored estimator (placement, the
  anchored network solve, its injection Jacobian, the per-channel reading sigmas, the
  linear measurement map and the linear MMSE estimate);
- ``evaluate``     the error table (nominal and perturbed grids), the meter sweep and the
  exhaustive placement search;
- ``attribution``  integrated gradients and order occlusion through network and solve;
- ``figures``      the data figures (parity, meter sweep, order coupling) and the table;
- ``verify``       re-scoring of a checkpoint against recorded numbers;
- ``reproduce``    the command-line entry (``python -m se_ablation.reproduce``).
"""

__version__ = "1.0.0"
