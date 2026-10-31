"""IEEE paper style: rcParams context behavior + the PDF/SVG save helper."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from injection_hse.analysis import (
    IEEE_COLUMN_IN,
    IEEE_PAGE_IN,
    figure_size,
    paper_style,
    save_paper_figure,
)


def test_figure_size_widths():
    w1, h1 = figure_size(columns=1)
    w2, _ = figure_size(columns=2, aspect=0.4)
    assert w1 == IEEE_COLUMN_IN
    assert w2 == IEEE_PAGE_IN
    assert h1 == pytest.approx(IEEE_COLUMN_IN * 0.62)
    with pytest.raises(ValueError):
        figure_size(columns=3)


def test_paper_style_applies_and_restores():
    before = mpl.rcParams["font.size"]
    with paper_style():
        assert mpl.rcParams["font.size"] == 8.0
        assert mpl.rcParams["font.family"] == ["serif"]
        assert mpl.rcParams["pdf.fonttype"] == 42
        assert tuple(mpl.rcParams["figure.figsize"]) == figure_size(columns=1)
    assert mpl.rcParams["font.size"] == before


def test_save_paper_figure_writes_both_formats(tmp_path):
    with paper_style(columns=2):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 2], label="x")
        ax.legend()
    pdf, svg = save_paper_figure(fig, tmp_path, "styled")
    plt.close(fig)
    assert pdf.name == "styled.pdf" and pdf.stat().st_size > 0
    assert svg.name == "styled.svg" and svg.stat().st_size > 0
