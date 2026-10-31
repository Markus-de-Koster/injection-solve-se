"""IEEE-conference figure styling and the paper-ready save helper.

Camera-ready output for a double-column IEEE A4 conference template: a temporary
rcParams context (:func:`paper_style`) sized to the IEEE column geometry, and
:func:`save_paper_figure`, which writes every figure as BOTH ``.pdf`` (embedded
TrueType fonts, the format LaTeX includes) and ``.svg`` (text kept editable) via
:func:`pgml.evaluation.style.save_figure`.

The style is applied as a context manager and fully restored on exit, so the
interactive / evaluation plot styling elsewhere in the suite is untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

#: Single-column width [in] for a double-column IEEE A4 conference template.
IEEE_COLUMN_IN: float = 3.5
#: Full two-column span width [in] for a double-column IEEE A4 conference template.
IEEE_PAGE_IN: float = 7.16

#: Colorblind-safe categorical palette (8 hues), ordered to maximize adjacent separation —
#: validated worst adjacent CVD ΔE 24.2, well above the ΔE≥12 target. Assign in FIXED order
#: (never cycle a 9th hue); line style is the secondary encoding for print / grayscale / >8.
SERIES_COLORS: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#4a3aa7",  # violet
    "#eda100",  # yellow
    "#e34948",  # red
    "#008300",  # green
    "#e87ba4",  # magenta
)
_LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":")


def series_style(i: int) -> tuple[str, str]:
    """``(color, linestyle)`` for categorical series ``i``.

    Colors are taken in fixed order from :data:`SERIES_COLORS`; the line style
    cycles as a SECONDARY encoding so series stay distinguishable beyond 8 and
    in grayscale / under color-vision deficiency. Always pair with a legend so
    identity is never carried by color alone.
    """
    return SERIES_COLORS[i % len(SERIES_COLORS)], _LINESTYLES[i % len(_LINESTYLES)]


def categorical_colors(n: int) -> list[str]:
    """The first ``n`` categorical colors from :data:`SERIES_COLORS` (fixed order)."""
    return [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(n)]


def figure_size(*, columns: int = 1, aspect: float = 0.62) -> tuple[float, float]:
    """Figure size in inches for a 1- or 2-column IEEE conference figure.

    Parameters
    ----------
    columns:
        ``1`` (single column, width :data:`IEEE_COLUMN_IN`) or ``2`` (full
        two-column span, width :data:`IEEE_PAGE_IN`).
    aspect:
        Height as a fraction of the width (default ``0.62``, near the golden
        ratio).
    """
    if columns not in (1, 2):
        raise ValueError(f"columns must be 1 or 2, got {columns!r}.")
    width = IEEE_COLUMN_IN if columns == 1 else IEEE_PAGE_IN
    return (width, width * aspect)


@contextmanager
def paper_style(*, columns: int = 1) -> Iterator[None]:
    """rcParams context for camera-ready IEEE figures; restores on exit.

    8 pt serif text with STIX math (Times-compatible without a LaTeX
    dependency), thin axes and ticks, frameless legends, constrained layout,
    and a default figure size of one IEEE column (``columns=2`` spans both).
    Fonts are embedded as TrueType in PDF output (fonttype 42, an IEEE PDF
    requirement) and kept as text in SVG output.

    Use around figure CREATION (the size and layout are consumed when the
    figure is built, not when it is saved)::

        with paper_style(columns=2):
            fig, ax = plt.subplots()
            ...
        save_paper_figure(fig, out_dir, "my_figure")
    """
    import matplotlib as mpl

    rc = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "STIXGeneral",
            "DejaVu Serif",
        ],
        "font.size": 8.0,
        "mathtext.fontset": "stix",
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
        "figure.figsize": figure_size(columns=columns),
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
    with mpl.rc_context(rc):
        yield


def save_paper_figure(fig, out_dir, stem: str) -> tuple[Path, Path]:
    """Write ``<out_dir>/<stem>.pdf`` AND ``<stem>.svg``; return both paths.

    Both files are vector output (tight bounding box) written through
    :func:`pgml.evaluation.style.save_figure`; the PDF is what the paper
    includes, the SVG stays editable for post-processing.
    """
    from pgml.evaluation.style import save_figure

    out = Path(out_dir)
    pdf = Path(save_figure(fig, out / f"{stem}.pdf"))
    svg = Path(save_figure(fig, out / f"{stem}.svg"))
    return pdf, svg


__all__ = [
    "IEEE_COLUMN_IN",
    "IEEE_PAGE_IN",
    "SERIES_COLORS",
    "series_style",
    "categorical_colors",
    "figure_size",
    "paper_style",
    "save_paper_figure",
]
