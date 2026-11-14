"""Plot style, shared colours, and saving figures for the paper.

* :func:`use_style` applies one consistent look to every figure (serif
  fonts matching the paper's Palatino, light grid, no top/right spines).
* :data:`COLORS` holds the colour of every recurring category — sexes,
  disruption types, forecasting methods — so a given method or sex has the
  same colour in every chapter, the paper and the app.
* :func:`save_figure` writes the PDF the paper includes
  (``results/figures/<label>.pdf``) and records its provenance.  PDFs are
  written without a creation date so that re-rendering an unchanged figure
  produces a byte-identical file (and an empty git diff).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import paths, provenance

__all__ = ["use_style", "save_figure", "COLORS", "SEX_LABELS", "CLUSTER_COLORS", "EPOCH_COLORS"]

#: Colours for recurring categories.  Change them here, nowhere else.
COLORS: dict[str, str] = {
    # sexes
    "female": "#b2182b", "male": "#2166ac",
    # disruption types (d = 1, 2, 3, 4)
    "war": "#d73027", "respiratory": "#4575b4", "enteric": "#1a9850", "other": "#7f7f7f",
    # forecasting methods
    "MDMx": "#1b9e77", "LC": "#d95f02", "HU": "#7570b3", "BL": "#e7298a", "CFDM": "#66a61e",
    "LQ": "#a6761d", "SVDComp": "#666666",
    # neutral
    "observed": "#333333", "model": "#1b9e77", "highlight": "#e6ab02", "grey": "#9e9e9e",
}

SEX_LABELS = ("Female", "Male")   # index 0 = female, 1 = male, everywhere

#: Colours for mortality-regime clusters (index = cluster number); colour-blind safe.
CLUSTER_COLORS: tuple[str, ...] = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
                                   "#D55E00", "#F0E442", "#000000", "#999999", "#882255",
                                   "#44AA99", "#117733", "#332288", "#AA4499", "#DDCC77")

#: Colours for epoch categories (rapid improvement → rapid worsening), diverging blue–grey–red.
EPOCH_COLORS: tuple[str, ...] = ("#184f95", "#5598e7", "#d6d5ce", "#e8736d", "#a12424")


def use_style() -> None:
    """Apply the project's matplotlib style (call once per chapter)."""
    mpl.rcParams.update({
        "figure.figsize": (8.0, 5.0),
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.family": "serif",
        # First available font wins; Palatino look-alikes first, then fallbacks
        # that exist on every macOS/Windows/Linux installation.
        "font.serif": ["TeX Gyre Pagella", "Palatino", "Palatino Linotype",
                       "Book Antiqua", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "pdf.fonttype": 42,          # embed TrueType: editable text in the PDF
    })


def save_figure(fig: plt.Figure, label: str, *, caption: str | None = None) -> Path:
    """Save ``fig`` as ``results/figures/<label>.pdf`` for the paper.

    The figure stays open, so it still displays inline in the book.

    Parameters
    ----------
    fig : the matplotlib figure
    label : the paper's figure file stem, e.g. ``"fig-s4-age-basis"``
    caption : optional caption recorded in the manifest (the paper keeps its
        own caption; this is for cross-checking)
    """
    paths.ensure(paths.figures())
    path = paths.figures() / f"{label}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    provenance.record_output("figure", path, label=label,
                             extra={"caption": caption} if caption else None)
    print(f"  [figure]   saved   {paths.relative(path)}")
    return path
