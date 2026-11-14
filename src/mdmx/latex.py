"""Everything the book writes *for the paper*: tables and in-text numbers.

The article (``article/``) never contains a hand-typed result.  Instead:

* **Tables** are written here by :func:`save_table` to
  ``results/tables/<label>.tex`` and pulled in with ``\\input``.
* **Numbers quoted in the prose** are registered with :func:`report`
  while a chapter runs, and written by :func:`write_numbers` to
  ``results/numbers/<chapter>.tex`` as LaTeX macros, e.g.::

      \\newcommand{\\tuckerRankAge}{42}

  The paper writes ``\\tuckerRankAge`` where it used to write ``42``.  The
  same values go to ``results/numbers.json``, which the book and the Shiny
  app read, so all three products always quote identical numbers.

``tools/check_paper.py`` fails the build if a result-like number is typed
directly into the paper, so stale numbers cannot creep back in.

Every file is written with ``\\n`` line endings on every platform, so that the
SHA-256 values in ``results/manifest.json`` are the same on macOS and Windows.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import paths, provenance
from ._lock import locked

__all__ = ["save_table", "stack_spanners", "number_word", "report", "write_numbers", "fmt_int", "fmt_num", "fmt_year",
           "latex_escape", "reported", "Rank", "rank_marks"]


# ══════════════════════════════════════════════════════════════════════════════
# Text helpers
# ══════════════════════════════════════════════════════════════════════════════

_LATEX_SPECIAL = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$",
                  "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}

# The characters a column header never uses as markup: each would break the table (``&`` adds a
# column, ``%`` comments out the rest of the row, ``#``, ``_`` and ``^`` are errors outside math).
# ``$``, ``\``, braces and ``~`` are markup that headers use on purpose (``$e_0$``, ``Share (\%)``).
_HEADER_SPECIAL = frozenset("&%#_^")


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters in plain text (not in math)."""
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in str(text))


_HEADER_DASHES = {"–": "--", "—": "---"}


def _escape_header(text: str) -> str:
    """Escape a column header for LaTeX, keeping the LaTeX it contains on purpose.

    Outside math (``$…$``), ``& % # _ ^`` are replaced as :func:`latex_escape` replaces them in text
    cells, and the en and em dashes ``–`` and ``—`` are written ``--`` and ``---``, as the captions write
    them.  Math is left alone, and so is the character after a backslash, so an already escaped
    ``\\%`` or a command such as ``\\,`` passes through.  Examples: ``"held_out: 80 %"`` →
    ``"held\\_out: 80 \\%"``; ``"Share (\\%)"`` and ``"$|\\Delta e_0|$ MAE"`` are unchanged.

    Raises
    ------
    ValueError
        If the header opens math with ``$`` and does not close it (write ``\\$`` for a dollar sign).
    """
    text = str(text)
    out: list[str] = []
    in_math = False
    chars = iter(text)
    for ch in chars:
        if ch == "\\":
            out.append(ch + next(chars, ""))
        elif ch == "$":
            in_math = not in_math
            out.append(ch)
        elif not in_math and ch in _HEADER_SPECIAL:
            out.append(_LATEX_SPECIAL[ch])
        elif not in_math and ch in _HEADER_DASHES:
            out.append(_HEADER_DASHES[ch])
        else:
            out.append(ch)
    if in_math:
        raise ValueError(f"Column header {text!r} has an unbalanced '$' (write '\\$' for a dollar sign).")
    return "".join(out)


_ONES = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
         "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
_TENS = {2: "Twenty", 3: "Thirty", 4: "Forty", 5: "Fifty", 6: "Sixty", 7: "Seventy", 8: "Eighty", 9: "Ninety"}


def number_word(n: int) -> str:
    """An integer 0–99 as letters for a macro name (macro names hold letters only).

    ``5`` → ``"Five"``, ``25`` → ``"Twentyfive"``, ``70`` → ``"Seventy"``.

    Raises
    ------
    ValueError
        If ``n`` is not a whole number from 0 to 99.
    """
    if float(n) != int(n) or not 0 <= int(n) <= 99:
        raise ValueError(f"number_word takes a whole number from 0 to 99, got {n!r}")
    n = int(n)
    return _ONES[n] if n < 20 else _TENS[n // 10] + ("" if n % 10 == 0 else _ONES[n % 10].lower())


def _round_half_up(value: float, digits: int) -> Decimal:
    """``value`` rounded to ``digits`` decimals with halves away from zero (``70.5 → 71``, ``2.5 → 3``).

    Python's own formatting rounds halves to even (``f"{70.5:.0f}" == "70"``), which a reader
    recomputing a share of populations would not expect.  The shortest decimal representation of
    the float is what is rounded, so ``2.675`` gives ``2.68`` although the float is 2.67499…: the
    reader sees decimals, not binary.
    """
    return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


def round_half_up(value: float, digits: int = 0) -> float:
    """``value`` rounded to ``digits`` decimals with halves away from zero, as the report prints them.

    For a chapter that rounds before reporting (a median year of 1948.5 reported as 1949), so that
    it follows the same convention as :func:`fmt_num` rather than Python's half-to-even ``round``.
    """
    return float(_round_half_up(value, digits))


def fmt_int(value: float) -> str:
    """Integer with LaTeX thousands separators: 3648 → ``3{,}648``; halves round up (84.5 → 85)."""
    return f"{int(_round_half_up(value, 0)):,}".replace(",", "{,}")


def fmt_year(value: float) -> str:
    """Calendar year as a plain integer: 1900 or 1900.0 → ``1900`` (no separator, no decimals).

    Use it as the formatter of a year column in :func:`save_table` (``formats={"origin": fmt_year}``):
    integer columns otherwise get thousands separators, which suit counts but not years.

    Raises
    ------
    ValueError
        If the value is not a whole number, so that a fractional value is never rounded into a year.
    """
    year = float(value)
    if not year.is_integer():
        raise ValueError(f"Year {value!r} is not a whole number.")
    return str(int(year))


def fmt_num(value: float, digits: int = 2, signed: bool = False) -> str:
    """Fixed-point number, e.g. ``fmt_num(1.4372, 2) → '1.44'``.

    ``signed=True`` always shows the sign (``+0.35``), used for biases.  A
    small negative value that rounds to zero prints as zero (``-0`` would
    read as a negative number).
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    rounded = _round_half_up(value, digits)
    if rounded == 0:
        rounded = abs(rounded)                      # never "-0.00"
    text = f"{rounded:+.{digits}f}" if signed else f"{rounded:.{digits}f}"
    return text.replace("-", "$-$") if text.startswith("-") else text


# ══════════════════════════════════════════════════════════════════════════════
# Tables
# ══════════════════════════════════════════════════════════════════════════════

def _label_tex(label: str) -> str:
    """'tbl-s4-tucker-ranks' → 'tab:s4_tucker_ranks' (the paper's label style)."""
    stem = label[4:] if label.startswith("tbl-") else label
    return "tab:" + stem.replace("-", "_")


#: A cell that siunitx can align: an optional sign, digits, an optional decimal part.
_NUMBER_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
#: Tables with more display rows than this (group headings included) are set as longtables.
LONG_ROWS = 40
_COLUMN_SPEC_RE = re.compile(r">\{[^{}]*\}[pmb]\{[^{}]*\}|[pmb]\{[^{}]*\}|[lcrSX]")


#: Units and counted nouns that a wrapped table cell keeps on the line of the number before them.
_UNIT_WORDS = ("y", "yr", "year", "years", r"\%", "pp", "ms", "min", "h", "days", "GB", "MB", "cores", "draws",
               "populations", "origins")
_UNIT_TIE = re.compile(r"(?<=[0-9}])[ ]+(?=(?:" + "|".join(re.escape(w) for w in _UNIT_WORDS) + r")(?![A-Za-z]))")


def tie_units(cell: str) -> str:
    """Join a number and the unit after it with a non-breaking space, so that a wrapped cell never
    leaves the unit alone on the next line: ``$\\pm$10 y`` → ``$\\pm$10~y``, ``3{,}648 draws`` →
    ``3{,}648~draws``.

    ``cell`` is LaTeX (already escaped); only a run of spaces between a digit (or the ``}`` of
    ``{,}``) and a whole word of :data:`_UNIT_WORDS` changes.
    """
    return _UNIT_TIE.sub("~", cell)


def _number_text(cell: str) -> str | None:
    """The plain number in a formatted cell (``$-$1.20`` → ``-1.20``, ``2{,}156`` → ``2156``), else None."""
    text = cell.replace("$-$", "-").replace("{,}", "").strip()
    return text if _NUMBER_RE.match(text) else None


def _table_format(numbers: Sequence[str]) -> str:
    """siunitx's ``table-format`` for a column of plain numbers: sign, integer digits, decimals."""
    signed = any(n[0] in "+-" for n in numbers)
    unsigned = [n.lstrip("+-") for n in numbers]
    whole = max(len(n.split(".")[0]) for n in unsigned)
    decimals = max(len(n.split(".")[1]) if "." in n else 0 for n in unsigned)
    return ("-" if signed else "") + f"{whole}" + (f".{decimals}" if decimals else "")


def _header_cell(text: str, align: str, numeric: bool) -> str:
    """A header cell: escaped, two lines where the text has a newline, braced over an ``S`` column."""
    lines = [_escape_header(part) for part in str(text).split("\n")]
    cell = lines[0] if len(lines) == 1 else f"\\makecell[{align}]{{{r'\\'.join(lines)}}}"
    return f"{{{cell}}}" if numeric else cell


#: The caption's key to the marks of :class:`Rank`; ``rank_note`` follows it.
RANK_KEY = "Bold, underlined, italic: the best, second and third value"
#: The key when a table's comparisons are too small for every mark (a mark needs the better half):
#: the most ranks that can be marked → its key.
RANK_KEYS = {1: "Bold: the best value, when it is unique", 2: "Bold, underlined: the best and second value",
             3: RANK_KEY}
#: The marks in a text cell (an ``S`` cell uses font switches, see :func:`save_table`).
_RANK_TEXT = {1: r"\textbf", 2: r"\underline", 3: r"\textit"}


@dataclass(frozen=True)
class Rank:
    """One kind of comparison in a table, for :func:`save_table`'s ``rank``: the best three values of
    each comparison are marked, the best in bold, the second underlined and the third in italic.

    Parameters
    ----------
    columns : the numeric columns compared.
    best : what is better: ``"min"``, ``"max"``, ``"zero"`` (the smallest absolute value) or a number
        (the closest to it, e.g. an interval's nominal coverage); or a dict column → one of these.
    rows : which rows take part, a boolean sequence over the table's rows (default every row).  The
        others are neither ranked nor marked: a yardstick, a reference row.
    across : ``"rows"`` (default): each column is one comparison among the rows, separately within
        each row group of :func:`save_table` and each value of ``within``.  ``"columns"``: each row
        is one comparison among ``columns``.
    within : a column of the table whose values split the rows into separate comparisons.
    """

    columns: Sequence[str]
    best: str | float | dict[str, str | float] = "min"
    rows: Sequence[bool] | None = None
    across: str = "rows"
    within: str | None = None

    def best_of(self, column: str) -> str | float:
        return self.best[column] if isinstance(self.best, dict) else self.best


def rank_marks(values: Sequence[float | None], best: str | float) -> list[int]:
    """The mark of each value of one comparison: 1 (best), 2, 3, or 0 (none).

    Ranks are competition ranks (ties share a rank and the next rank is skipped: 1, 1, 3), so values
    that tie as printed share a mark.  A value is marked only if its rank is at most 3 and no more
    than half of the values compared are as good or better: a mark picks out the better half, and a
    comparison of two marks only the better one, of three only the best.  ``None`` and NaN take no
    part.

    Parameters
    ----------
    values : the comparison's values, as printed.
    best : ``"min"``, ``"max"``, ``"zero"`` or a target number (see :class:`Rank`).
    """
    def loss(v: float) -> float:
        if best == "min":
            return v
        if best == "max":
            return -v
        if best == "zero":
            return abs(v)
        if isinstance(best, (int, float)) and not isinstance(best, bool):
            return abs(v - float(best))
        raise ValueError(f"best must be 'min', 'max', 'zero' or a number, not {best!r}")

    present = [loss(float(v)) for v in values if v is not None and not math.isnan(float(v))]
    n = len(present)
    marks = []
    for v in values:
        if v is None or math.isnan(float(v)):
            marks.append(0)
            continue
        own = loss(float(v))
        rank = 1 + sum(x < own for x in present)
        as_good = sum(x <= own for x in present)
        marks.append(rank if rank <= 3 and as_good <= n / 2 else 0)
    return marks


def _comparisons(df: pd.DataFrame, rank: Sequence[Rank], group_labels: Sequence[Any] | None
                 ) -> list[tuple[list[tuple[int, str]], str | float]]:
    """The comparisons of a table: each a list of (row position, column) cells and what is better."""
    out = []
    n = len(df)
    for spec in rank:
        missing = [c for c in spec.columns if c not in df.columns]
        if missing:
            raise ValueError(f"Rank columns {missing} are not in the table.")
        if spec.across not in ("rows", "columns"):
            raise ValueError(f"Rank.across must be 'rows' or 'columns', not {spec.across!r}")
        take = [True] * n if spec.rows is None else [bool(x) for x in spec.rows]
        if len(take) != n:
            raise ValueError(f"Rank.rows has {len(take)} values, the table {n} rows.")
        if spec.across == "columns":
            for i in (i for i in range(n) if take[i]):
                if isinstance(spec.best, dict) and len({str(spec.best_of(c)) for c in spec.columns}) > 1:
                    raise ValueError("A comparison across columns needs one 'best' for all of them.")
                out.append(([(i, c) for c in spec.columns], spec.best_of(spec.columns[0])))
            continue
        # One comparison per column within each run of a row group and each value of `within`.
        block = [(group_labels[i] if group_labels is not None else None,
                  df[spec.within].iloc[i] if spec.within else None) for i in range(n)]
        keys = list(dict.fromkeys(block[i] for i in range(n) if take[i]))
        for key in keys:
            members = [i for i in range(n) if take[i] and block[i] == key]
            for c in spec.columns:
                out.append(([(i, c) for i in members], spec.best_of(c)))
    return out


def stack_spanners(df: pd.DataFrame, keep: Sequence[str], sep: str = ": ", group: str = "group") -> pd.DataFrame:
    """Move a table's spanner level from the columns to row groups (the guide's "grouped stub").

    A column named ``"<spanner><sep><sub>"`` belongs to the spanner ``<spanner>``.  The result has one
    block of rows per spanner, in the order the spanners first appear: the columns ``keep`` (the
    stub), then one column per sub-header, and the column ``group`` holding the spanner, for
    :func:`save_table`'s ``groups``.  A spanner without some sub-header gets missing cells there.

    Raises
    ------
    ValueError
        If a column outside ``keep`` has no ``sep`` in its name.
    """
    spanned = [c for c in df.columns if c not in keep]
    bad = [c for c in spanned if sep not in c]
    if bad:
        raise ValueError(f"Columns {bad} have no {sep!r}: name them '<spanner>{sep}<sub>' or add them to keep.")
    order: list[str] = []
    subs: list[str] = []
    for c in spanned:
        span, sub = c.split(sep, 1)
        order += [] if span in order else [span]
        subs += [] if sub in subs else [sub]
    blocks = []
    for span in order:
        block = df[list(keep)].copy()
        for sub in subs:
            name = f"{span}{sep}{sub}"
            block[sub] = df[name] if name in df.columns else np.nan
        block.insert(0, group, span)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def save_table(
    df: pd.DataFrame,
    label: str,
    caption: str,
    *,
    short_caption: str | None = None,
    footnote: str | None = None,
    column_format: str | None = None,
    formats: dict[str, Callable[[Any], str] | str] | None = None,
    escape_cells: bool = True,
    group_rows: Sequence[int] = (),
    long: bool | None = None,
    spanners: str | None = None,
    groups: str | None = None,
    wrap: dict[str, str] | None = None,
    colsep: float | None = None,
    rank: Sequence[Rank] = (),
    rank_note: str = "in each comparison",
) -> Path:
    """Write ``df`` as a LaTeX table for the paper, set to the article's table standard.

    The standard (``article/README.md``): booktabs rules only; numbers decimal-aligned in siunitx
    ``S`` columns whose ``table-format`` is computed from the cells; ``@{}`` at both edges; the
    caption above and any note below; no font-size command and no scaling (the article's preamble
    sets one size for every table).  A table too wide for the page is restructured with spanner
    headers, row groups and wrapped text columns, never shrunk.

    Parameters
    ----------
    df : the table to write.  Column names become the header.  They may contain LaTeX on purpose
        (math such as ``r"$\\ezero$"``, an escaped ``r"\\%"``, a command); outside math, ``& % # _ ^``
        are escaped as in text cells, so ``"held_out: 80 %"`` prints as written.  A newline in a name
        sets it on two lines.
    label : file name and LaTeX label stem, e.g. ``"tbl-s4-tucker-ranks"`` →
        ``results/tables/tbl-s4-tucker-ranks.tex`` with ``\\label{tab:s4_tucker_ranks}``.
    caption, short_caption : LaTeX caption (and an optional short form for lists of tables).
    footnote : a note printed under the table, at the table's size.
    column_format : e.g. ``"lrrr"``, one letter (or ``p{…}``) per column of ``df`` other than
        ``groups``; default: ``l`` for text columns and ``r`` for numeric ones.  Every ``r`` column
        whose cells are all plain numbers becomes an ``S`` column; ``l``, ``c`` and ``p`` stay text.
    formats : per-column formatter — a callable, or a format spec such as ``"{:.3f}"``.  Numeric
        columns without a formatter use ``{:.3f}`` (integers: thousands separators, for counts).
        Give year columns :func:`fmt_year` (``formats={"origin": fmt_year}``), so that they print
        as ``1900`` and are not grouped like counts.
    escape_cells : escape LaTeX special characters in text cells.
    group_rows : insert ``\\addlinespace`` before these (0-based) rows.
    long : force ``longtable`` (True) or ``table`` (False).  Default: a longtable when the table
        has more than :data:`LONG_ROWS` display rows.
    spanners : a separator such as ``": "``.  A column named ``"Early: 80 \\%"`` then prints ``80 \\%``
        under the spanner ``Early``, with a trimmed rule; adjacent columns with the same spanner
        share it.  Columns without the separator have no spanner.
    groups : a column of ``df`` whose values label groups of consecutive rows.  It is not printed
        as a column: each group gets an italic heading row and its rows are indented, with space
        between groups.  In a longtable a group's heading stays with its first two rows and its last two
        rows stay together; a page may break between the others (a group longer than the space left on
        a page would otherwise push the whole table to the next page, and longtable would print its
        continuation head above the caption).
    wrap : column → width (``"4cm"``) for text columns that wrap, ragged right.
    colsep : the space between columns in points (LaTeX's default 6; the standard allows down to 4).
    rank : the table's comparisons (:class:`Rank`).  In each, the best value as printed is set in
        bold, the second underlined and the third in italic (:func:`rank_marks`), and the caption
        ends with the key, followed by ``rank_note`` (what one comparison is).  In an ``S`` column
        bold and italic are font switches that siunitx detects (the article's ``\\sisetup`` has
        ``detect-weight`` and ``detect-shape``); an underlined number stays siunitx's, aligned like
        the others, with a rule as wide as ``\\num`` sets it drawn back over it from the text after
        it (the article sets ``table-align-text-after=false``, so that text starts where the number
        ends).
    """
    formats = formats or {}
    wrap = wrap or {}
    shown = [c for c in df.columns if c != groups]
    group_labels = list(df[groups]) if groups else None

    # ── Cell text ─────────────────────────────────────────────────────────────
    def fmt_cell(col: str, value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "--"
        spec = formats.get(col)
        if callable(spec):
            return spec(value)
        if isinstance(spec, str):
            return spec.format(value)
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        if isinstance(value, (int, np.integer)):
            return fmt_int(value)
        if isinstance(value, (float, np.floating)):
            return fmt_num(float(value), 3)
        return latex_escape(value) if escape_cells else str(value)

    # itertuples keeps each column's own type.  iterrows would cast a row of an all-numeric frame
    # to float, and an integer column would print as 2156.000 instead of 2{,}156.
    cells = [[fmt_cell(c, v) for c, v in zip(shown, row, strict=True)]
             for row in df[shown].itertuples(index=False, name=None)]
    printed = [list(row) for row in cells]          # the cells as formatted, before siunitx's plain numbers

    # ── Columns: alignment, S columns and their formats ──────────────────────
    if column_format is None:
        specs = ["r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in shown]
    else:
        specs = _COLUMN_SPEC_RE.findall(column_format)
        if len(specs) != len(shown):
            raise ValueError(f"column_format {column_format!r} has {len(specs)} columns, the table {len(shown)}.")
    for j, c in enumerate(shown):
        if c in wrap:
            specs[j] = f">{{\\raggedright\\arraybackslash}}p{{{wrap[c]}}}"
    numeric = [False] * len(shown)
    for j, c in enumerate(shown):
        if specs[j] != "r" or formats.get(c) is fmt_year:
            continue
        values = [cell for row in cells for cell in [row[j]] if cell != "--"]
        numbers = [_number_text(v) for v in values]
        if values and all(n is not None for n in numbers):
            numeric[j] = True
            specs[j] = f"S[table-format={_table_format(numbers)}]"
            for row in cells:
                row[j] = "{--}" if row[j] == "--" else _number_text(row[j])
    for j in range(len(shown)):
        if not numeric[j]:
            for row in cells:
                row[j] = tie_units(row[j])

    # ── The best three values of each comparison ─────────────────────────────
    n_compared = 0
    for members, best in _comparisons(df, rank, group_labels):
        numbers = [_number_text(printed[i][shown.index(c)]) for i, c in members]
        values = [float(x) if x is not None else None for x in numbers]
        n_compared = max(n_compared, sum(v is not None and not math.isnan(v) for v in values))
        marks = rank_marks(values, best)
        for (i, c), mark in zip(members, marks, strict=True):
            j = shown.index(c)
            if mark == 0:
                continue
            if numeric[j] and mark == 2:
                # A rule as wide as siunitx sets the number, drawn back over it from just after it.
                cells[i][j] = f"{cells[i][j]}{{\\llap{{\\underline{{\\phantom{{\\num{{{cells[i][j]}}}}}}}}}}}"
            elif numeric[j]:
                cells[i][j] = ("\\bfseries " if mark == 1 else "\\itshape ") + cells[i][j]
            else:
                cells[i][j] = f"{_RANK_TEXT[mark]}{{{cells[i][j]}}}"
    if rank and n_compared >= 2:
        # Rank r can be marked only if r ≤ half the values compared (the better half).
        caption = f"{caption}  {RANK_KEYS[min(3, n_compared // 2)]} {rank_note}."
    aligns = ["l" if s.startswith((">", "p", "l")) else "c" if s in ("c", "X") else "r" for s in specs]

    # ── Header: spanners, then the column labels ─────────────────────────────
    def split(name: str) -> tuple[str | None, str]:
        return tuple(name.split(spanners, 1)) if spanners and spanners in name else (None, name)
    heads = [split(c) for c in shown]
    header_rows = []
    if any(span is not None for span, _ in heads):
        cells_top, rules, j = [], [], 0
        while j < len(heads):
            span = heads[j][0]
            k = j
            while k + 1 < len(heads) and span is not None and heads[k + 1][0] == span:
                k += 1
            if span is None:
                cells_top.append("")
            else:
                cells_top.append(f"\\multicolumn{{{k - j + 1}}}{{c}}{{{_header_cell(span, 'c', False)}}}")
                rules.append(f"\\cmidrule(lr){{{j + 1}-{k + 1}}}")
            j = k + 1
        header_rows += [" & ".join(cells_top) + r" \\", "".join(rules)]
    header_rows.append(" & ".join(_header_cell(sub, aligns[j], numeric[j]) for j, (_, sub) in enumerate(heads))
                       + r" \\")

    # ── Body: groups, spacing, page breaks ───────────────────────────────────
    ncol = len(shown)
    n_display = len(cells) + (len(dict.fromkeys(group_labels)) if groups else 0)
    if long is None:
        long = n_display > LONG_ROWS
    spaced = set(group_rows)
    body: list[str] = []
    # Each row's position in its group and the group's size.
    group_pos: list[tuple[int, int]] = []
    if groups:
        sizes = pd.Series(group_labels).groupby((pd.Series(group_labels) != pd.Series(group_labels).shift())
                                                .cumsum()).transform('size').tolist()
        k = 0
        for i in range(len(cells)):
            k = 0 if (i == 0 or group_labels[i] != group_labels[i - 1]) else k + 1
            group_pos.append((k, int(sizes[i])))
    for i, row in enumerate(cells):
        first_in_group = groups and (i == 0 or group_labels[i] != group_labels[i - 1])
        last_in_group = groups and (i == len(cells) - 1 or group_labels[i] != group_labels[i + 1])
        if first_in_group:
            if i > 0:
                body.append(r"\addlinespace")
            heading = _escape_header(group_labels[i]) if escape_cells else str(group_labels[i])
            body.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{heading}}}}}" + (r" \\*" if long else r" \\"))
        elif i in spaced:
            body.append(r"\addlinespace")
        if groups:
            row = [r"\hspace*{1em}" + row[0], *row[1:]]
        # Keep the heading with the first two rows and the last two rows together; allow breaks between.  No
        # break after the table's last row: the closing rule would go to a page of its own under a repeated head.
        if long and i == len(cells) - 1:
            end = r" \\*"
        elif long and groups and not last_in_group:
            k, n = group_pos[i]
            end = r" \\*" if (k == 0 or k >= n - 2) else r" \\"
        else:
            end = r" \\"
        body.append(" & ".join(row) + end)

    # ── The table ─────────────────────────────────────────────────────────────
    # The caption and any note are as wide as the table, and a table narrower than the article's minimum
    # is widened to it (the article's preamble: the mdmxtabular environment, the long-table hook).
    spec_text = "@{}" + "".join(specs) + "@{}"
    sep = f"\\setlength{{\\tabcolsep}}{{{colsep:g}pt}}" if colsep is not None else ""
    lines = [f"% Generated by the MDMx book (chapter: {provenance.current_chapter()}). Do not edit;",
             "% re-render the chapter instead.  See results/manifest.json for provenance."]
    if long:
        cap = f"\\caption[{short_caption}]{{{caption}}}" if short_caption else f"\\caption{{{caption}}}"
        lines += [
            f"{{{sep}" if sep else "",
            f"\\begin{{longtable}}{{@{{\\extracolsep{{\\mdmxLTsep}}}}{spec_text}}}",
            f"{cap}\\label{{{_label_tex(label)}}}\\\\",
            r"\toprule", *header_rows, r"\midrule", r"\endfirsthead",
            f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{Table \\thetable\\ (continued)}}}}\\\\",
            r"\toprule", *header_rows, r"\midrule", r"\endhead",
            r"\midrule", f"\\multicolumn{{{ncol}}}{{@{{}}r}}{{\\textit{{(continued on next page)}}}}\\\\",
            r"\endfoot",
            r"\bottomrule", r"\endlastfoot",
            *body,
            r"\end{longtable}",
            "}" if sep else "",
        ]
        if footnote:
            lines.append(f"\\mdmxLTnote{{{footnote}}}")
    else:
        lines += [r"\begin{table}[!htbp]", r"\centering", sep,
                  f"\\begin{{mdmxtabular}}{{{spec_text}}}{{{short_caption or ''}}}{{{caption}}}"
                  f"{{{_label_tex(label)}}}{{{footnote or ''}}}",
                  r"\toprule", *header_rows, r"\midrule", *body, r"\bottomrule",
                  r"\end{mdmxtabular}", r"\end{table}"]
    paths.ensure(paths.tables())
    path = paths.tables() / f"{label}.tex"
    path.write_text("\n".join(line for line in lines if line != "") + "\n", encoding="utf-8", newline="\n")
    provenance.record_output("table", path, label=label, extra={"n_rows": int(len(df))})
    print(f"  [table]    saved   {paths.relative(path)}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# In-text numbers
# ══════════════════════════════════════════════════════════════════════════════

_NUMBERS: dict[str, dict[str, Any]] = {}
_MACRO_RE = re.compile(r"[A-Za-z]+")
# CONTRIBUTING §4: every macro starts with its chapter's prefix.  The capital letter after the
# prefix keeps the prefix unambiguous (``fcMaeAll``, ``clusK``).
_MACRO_PREFIXES = ("data", "tucker", "clus", "traj", "exc", "fit", "svd", "fc", "bl", "ev")
_PREFIXED_RE = re.compile(rf"(?:{'|'.join(_MACRO_PREFIXES)})[A-Z][A-Za-z]*")


def report(key: str, value: Any, text: str | None = None, *, desc: str = "", digits: int | None = None,
           signed: bool = False) -> None:
    """Register a number that the paper (or app) quotes.

    (Returns nothing, so a notebook cell that ends with a ``report`` call
    does not echo a stray value; use :func:`reported` to read values back.)

    Parameters
    ----------
    key : LaTeX macro name, letters only (LaTeX forbids digits), e.g.
        ``"fcMaeAll"``: a chapter prefix (``data``, ``tucker``, ``clus``,
        ``traj``, ``exc``, ``fit``, ``svd``, ``fc``, ``bl``, ``ev``) followed
        by a capital letter.
    value : the raw value (stored in ``numbers.json`` unrounded).
    text : how it should print in the paper.  If omitted: integers get
        thousands separators; floats use ``digits`` decimals (default 2).
    desc : what the number is, in words (shown in the book's index of numbers).

    Raises
    ------
    ValueError
        If ``key`` is not letters only or does not start with a chapter prefix.
    """
    if not _MACRO_RE.fullmatch(key):
        raise ValueError(f"Macro name {key!r} must contain letters only (no digits, no '_').")
    if not _PREFIXED_RE.fullmatch(key):
        raise ValueError(f"Macro name {key!r} must start with a chapter prefix ({', '.join(_MACRO_PREFIXES)}) "
                         "followed by a capital letter, e.g. 'fcMaeAll'.")
    if text is None:
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            text = fmt_int(value)
        elif isinstance(value, (float, np.floating)):
            text = fmt_num(float(value), 2 if digits is None else digits, signed=signed)
        else:
            text = str(value)
    raw = value.item() if isinstance(value, np.generic) else value
    _NUMBERS[key] = {"value": raw, "text": text, "desc": desc,
                     "chapter": provenance.current_chapter(), "cell": provenance.current_cell()}


def reported() -> dict[str, dict[str, Any]]:
    """The numbers registered so far in this process (key → record)."""
    return dict(_NUMBERS)


def write_numbers() -> Path:
    """Write the current chapter's numbers as LaTeX macros and JSON.

    Produces ``results/numbers/<chapter>.tex`` (only this chapter's macros)
    and updates the combined ``results/numbers.json``.
    """
    chapter = provenance.current_chapter() or "unknown"
    mine = {k: v for k, v in _NUMBERS.items() if v["chapter"] == chapter}
    paths.ensure(paths.numbers())

    tex = [f"% In-text numbers generated by book chapter '{chapter}'. Do not edit;",
           "% re-render the chapter.  Values: results/numbers.json."]
    for key in sorted(mine):
        rec = mine[key]
        comment = f"  % {rec['desc']}" if rec["desc"] else ""
        tex.append(f"\\newcommand{{\\{key}}}{{{rec['text']}}}{comment}")
    tex_path = paths.numbers() / f"{chapter}.tex"
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8", newline="\n")
    provenance.record_output("numbers", tex_path, label=chapter, extra={"n_numbers": len(mine)})

    json_path = paths.results() / "numbers.json"
    with locked(json_path):
        _merge_numbers(json_path, chapter, mine)
    print(f"  [numbers]  saved   {paths.relative(tex_path)}  ({len(mine)} macros)")
    return tex_path


def _merge_numbers(json_path: Path, chapter: str, mine: dict) -> None:
    combined = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    combined = {k: v for k, v in combined.items() if v.get("chapter") != chapter}
    combined.update(mine)
    json_path.write_text(json.dumps(dict(sorted(combined.items())), indent=1, default=str) + "\n",
                         encoding="utf-8", newline="\n")

    # One file the paper \input's: all chapters' macro files, in a fixed order.
    all_tex = sorted(p.name for p in paths.numbers().glob("*.tex") if p.name != "all.tex")
    (paths.numbers() / "all.tex").write_text(
        "% Loads every generated numbers file.  Generated; do not edit.\n"
        + "".join(f"\\input{{\\resultsdir/numbers/{name[:-4]}}}\n" for name in all_tex),
        encoding="utf-8", newline="\n",
    )
