"""The data pipeline: HMD release → curated life tables → mortality tensor.

This module holds the machinery behind Part I of the book (chapters
d1–d6).  Each public function corresponds to one documented step, takes
plain DataFrames/arrays, and returns new ones — it never modifies its
inputs, and every change it makes to the data is also returned as rows of
a *change ledger* so that nothing happens silently.

Steps (book chapter → paper section)
------------------------------------
d1  :func:`hmd_source`, :func:`verify_release`  — which file, which release (§2.1)
d2  :func:`ingest_life_tables`, :func:`ingest_deaths_exposures` (§2.2)
d3  :func:`curate`                               — curation rules (§3.1)
d4  :func:`load_events`, :func:`label_country_years`, :func:`crisis_screen` (§3.2, App. A)
d5  :func:`valley_ages`, :func:`adaptive_pooling` (§3.4)
d6  :func:`build_tensor` (§2.5 and §3.3: the floor on q_x and the observed mask)

Data layout conventions
-----------------------
* Long tables use the columns ``pop`` (HMD code), ``sex`` ("female" /
  "male"), ``year``, ``age`` (0…110; 110 is the open interval 110+).
* Sex order in arrays is always (female, male) — index 0 and 1.
* Tensor axes are (sex, age, country, year), matching the paper.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from . import lifetable, paths

__all__ = [
    "HMD_MEMBERS", "SEXES", "LABELS", "hmd_source", "verify_release", "open_member",
    "read_hmd_text", "ingest_life_tables", "ingest_deaths_exposures", "load_populations",
    "population_names",
    "curate", "load_events", "expand_events", "label_country_years", "crisis_screen",
    "valley_ages", "contiguous_runs", "plan_blocks", "adaptive_pooling", "build_tensor",
]

SEXES = ("female", "male")

#: Disruption labels d_{c,t} (defined with the event dictionaries, §8.2).  "other" (famine, measles,
#: smallpox) is excluded from the baseline and not modelled.
LABELS = {0: "none", 1: "war", 2: "respiratory", 3: "enteric", 4: "other"}
_TYPE_TO_LABEL = {v: k for k, v in LABELS.items()}

#: Files used from the HMD "all statistics" bulk zip (paths inside the zip).
HMD_MEMBERS = {
    "lt_female": "lt_female/fltper_1x1/fltper_1x1.txt",
    "lt_male": "lt_male/mltper_1x1/mltper_1x1.txt",
    "deaths": "deaths/Deaths_1x1/Deaths_1x1.txt",
    "exposures": "exposures/Exposures_1x1/Exposures_1x1.txt",
}


# ══════════════════════════════════════════════════════════════════════════════
# d1 — Locate and verify the HMD release
# ══════════════════════════════════════════════════════════════════════════════

def _manifest() -> dict:
    return json.loads((paths.raw() / "MANIFEST.json").read_text(encoding="utf-8"))


def hmd_source(release: str) -> Path:
    """Return the location of the HMD release on disk.

    Looks first for the bulk zip exactly as downloaded,
    ``raw/hmd/hmd_statistics_<release>.zip``; failing that, for a folder
    ``raw/hmd/<release>/`` holding just the four files this project uses
    (with their paths as inside the zip).  Both are verified against
    ``raw/MANIFEST.json`` by :func:`verify_release`.
    """
    zip_path = paths.raw() / "hmd" / f"hmd_statistics_{release}.zip"
    if zip_path.is_file():
        return zip_path
    folder = paths.raw() / "hmd" / release
    if all((folder / m).is_file() for m in HMD_MEMBERS.values()):
        return folder
    raise FileNotFoundError(
        f"HMD release {release} not found.  Download the HMD bulk 'all statistics' zip "
        f"and save it as {paths.relative(zip_path)} (see MANIFEST.md, section 'Data')."
    )


def open_member(source: Path, member: str) -> io.TextIOBase:
    """Open one of the HMD text files, from the zip or from the folder."""
    if source.suffix == ".zip":
        zf = zipfile.ZipFile(source)
        return io.TextIOWrapper(zf.open(member), encoding="utf-8")
    return open(source / member, encoding="utf-8")


def verify_release(release: str) -> dict:
    """Check the release against ``raw/MANIFEST.json``; return a report.

    Paper: sec:data:hmd

    * If the zip is present, its SHA-256 must match the manifest.
    * The SHA-256 of each of the four member files must match, whether read
      from the zip or from the folder.

    Raises ``ValueError`` on any mismatch, so the pipeline can never run on
    a different HMD release by accident.
    """
    import hashlib

    from .provenance import sha256_file

    man = _manifest()["hmd"][release]
    source = hmd_source(release)
    report = {"release": release, "source": paths.relative(source), "zip_sha256": None, "members": {}}
    if source.suffix == ".zip":
        digest = sha256_file(source)
        report["zip_sha256"] = digest
        if digest != man["zip_sha256"]:
            raise ValueError(f"{source.name}: SHA-256 {digest} ≠ manifest {man['zip_sha256']}")
    for _key, member in HMD_MEMBERS.items():
        h = hashlib.sha256()
        if source.suffix == ".zip":
            with zipfile.ZipFile(source) as zf, zf.open(member) as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
        else:
            with open(source / member, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
        digest = h.hexdigest()
        expected = man["members"][member]
        report["members"][member] = {"sha256": digest, "ok": digest == expected}
        if digest != expected:
            raise ValueError(f"{member}: SHA-256 {digest} ≠ manifest {expected}")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# d2 — Ingest
# ══════════════════════════════════════════════════════════════════════════════

def read_hmd_text(fh: io.TextIOBase) -> tuple[pd.DataFrame, str]:
    """Parse one HMD bulk text file.

    The format is: a title line (with the "Last modified" stamp), a blank
    line, a header line, then whitespace-separated columns.  Missing values
    are written as ``.``.  The open age interval is written ``110+``.

    Returns the table (Age converted to int, 110 for "110+") and the title.
    """
    title = fh.readline().strip()
    df = pd.read_csv(fh, sep=r"\s+", skiprows=1, na_values=".", dtype={"Age": str, "PopName": str})
    df["Age"] = df["Age"].str.rstrip("+").astype(np.int16)
    df["Year"] = df["Year"].astype(np.int16)
    return df, title


def ingest_life_tables(source: Path) -> tuple[pd.DataFrame, dict]:
    """Read the female and male 1×1 period life tables into one long table.

    Returns
    -------
    lt : DataFrame with columns pop, sex, year, age, mx, qx, ax, lx, dx, Lx, Tx, ex
    report : row counts and the files' "Last modified" stamps
    """
    frames, report = [], {}
    for sex in SEXES:
        member = HMD_MEMBERS[f"lt_{sex}"]
        with open_member(source, member) as fh:
            df, title = read_hmd_text(fh)
        report[member] = {"title": title, "rows": int(len(df)),
                          "rows_with_missing": int(df.isna().any(axis=1).sum())}
        df = df.rename(columns={"PopName": "pop", "Year": "year", "Age": "age"})
        df.insert(1, "sex", sex)
        frames.append(df)
    lt = pd.concat(frames, ignore_index=True)
    lt["sex"] = pd.Categorical(lt["sex"], categories=list(SEXES))
    return lt.sort_values(["pop", "sex", "year", "age"], ignore_index=True), report


def ingest_deaths_exposures(source: Path) -> tuple[pd.DataFrame, dict]:
    """Read 1×1 deaths and exposures, reshaped to long by sex.

    Returns a DataFrame with columns pop, sex, year, age, deaths, exposure.
    (The HMD's "Total" column is not needed and is dropped.)
    """
    parts, report = {}, {}
    for key, value_name in (("deaths", "deaths"), ("exposures", "exposure")):
        member = HMD_MEMBERS[key]
        with open_member(source, member) as fh:
            df, title = read_hmd_text(fh)
        report[member] = {"title": title, "rows": int(len(df))}
        long = df.melt(id_vars=["PopName", "Year", "Age"], value_vars=["Female", "Male"],
                       var_name="sex", value_name=value_name)
        long["sex"] = long["sex"].str.lower()
        parts[key] = long.rename(columns={"PopName": "pop", "Year": "year", "Age": "age"})
    de = parts["deaths"].merge(parts["exposures"], on=["pop", "year", "age", "sex"], how="outer")
    de["sex"] = pd.Categorical(de["sex"], categories=list(SEXES))
    return de.sort_values(["pop", "sex", "year", "age"], ignore_index=True)[
        ["pop", "sex", "year", "age", "deaths", "exposure"]], report


def load_populations() -> pd.DataFrame:
    """The population register ``config/populations.csv`` (codes, names, overlaps).

    Paper: sec:data:hmd
    """
    return pd.read_csv(paths.config_dir() / "populations.csv", dtype=str, keep_default_na=False, encoding="utf-8")


def population_names() -> dict[str, str]:
    """HMD code → display name, from the population register (e.g. 'GBR_NP' → 'United Kingdom')."""
    pops = load_populations()
    return dict(zip(pops["code"], pops["name"], strict=True))


# ══════════════════════════════════════════════════════════════════════════════
# d3 — Curation (paper §3.1)
# ══════════════════════════════════════════════════════════════════════════════

def curate(lt: pd.DataFrame, *, populations_excluded: Iterable[str], flat_tail_ages: tuple[int, int],
           age_max: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the curation rules and return what was removed.

    Paper: sec:preprocessing:curation

    Rules, applied in this order, each to whole country-years (both sexes):

    1. **Excluded populations** — the civilian-only duplicates.
    2. **Missing values** — any ``NaN`` in m_x, q_x or a_x (ages 0–110).
    3. **Flat old-age tail** — q_x identical at the two ``flat_tail_ages``
       (e.g. 105 and 109) in either sex.
    4. **Sex pairing** — a country-year must have both sexes.
    5. **Age range** — ages above ``age_max`` (the 110+ interval) dropped.

    Returns
    -------
    curated : the remaining rows, ages 0…age_max
    exclusions : one row per removed country-year with the rule that removed
        it (first matching rule), for the change ledger
    """
    excl_rows = []
    keep = lt.copy()

    # 1. Excluded populations ------------------------------------------------
    dropped = keep[keep["pop"].isin(list(populations_excluded))]
    for (pop, year), _ in dropped.groupby(["pop", "year"], observed=True):
        excl_rows.append((pop, year, "excluded population", "civilian-only duplicate series"))
    keep = keep[~keep["pop"].isin(list(populations_excluded))]

    # 2. Missing values ------------------------------------------------------
    has_missing = keep[["mx", "qx", "ax"]].isna().any(axis=1)
    bad = keep.loc[has_missing, ["pop", "year"]].drop_duplicates()
    for pop, year in bad.itertuples(index=False):
        excl_rows.append((pop, year, "missing values", "HMD reports '.' in m_x/q_x/a_x"))
    bad_keys = set(map(tuple, bad.to_numpy()))

    # 3. Flat old-age tail ---------------------------------------------------
    a1, a2 = flat_tail_ages
    tail = keep[keep["age"].isin([a1, a2])].pivot_table(
        index=["pop", "sex", "year"], columns="age", values="qx", observed=True)
    flat = tail[tail[a1] == tail[a2]].reset_index()[["pop", "year"]].drop_duplicates()
    for pop, year in flat.itertuples(index=False):
        if (pop, year) not in bad_keys:
            excl_rows.append((pop, year, "flat old-age tail", f"q_{a1} = q_{a2}"))
    bad_keys |= set(map(tuple, flat.to_numpy()))

    # 4. Sex pairing ---------------------------------------------------------
    n_sex = keep.groupby(["pop", "year"], observed=True)["sex"].nunique()
    unpaired = n_sex[n_sex < 2].reset_index()[["pop", "year"]]
    for pop, year in unpaired.itertuples(index=False):
        if (pop, year) not in bad_keys:
            excl_rows.append((pop, year, "unpaired sex", "only one sex available"))
    bad_keys |= set(map(tuple, unpaired.to_numpy()))

    key = list(zip(keep["pop"], keep["year"], strict=False))
    keep = keep[[k not in bad_keys for k in key]]

    # 5. Age range -----------------------------------------------------------
    keep = keep[keep["age"] <= age_max].reset_index(drop=True)

    exclusions = pd.DataFrame(excl_rows, columns=["pop", "year", "rule", "detail"])
    return keep, exclusions.sort_values(["rule", "pop", "year"], ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# d4 — Exceptional years (paper §3.2, Appendix A)
# ══════════════════════════════════════════════════════════════════════════════

def load_events() -> pd.DataFrame:
    """The event dictionary ``config/events.csv`` (one row per event entry)."""
    ev = pd.read_csv(paths.config_dir() / "events.csv", dtype={"populations": str},
                     keep_default_na=False, encoding="utf-8")
    unknown = set(ev["type"]) - set(_TYPE_TO_LABEL)
    if unknown:
        raise ValueError(f"Unknown event types in events.csv: {unknown}")
    return ev


def expand_events(events: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Expand dictionary rows into (pop, year, type, event id) pairs.

    ``populations = "ALL"`` means every population with data at any time
    during the event (``coverage`` gives each population's first and last
    year).
    """
    rows = []
    for ev in events.itertuples(index=False):
        if ev.populations.strip() == "ALL":
            pops = coverage.loc[(coverage["first"] <= ev.end) & (coverage["last"] >= ev.start), "pop"]
        else:
            pops = ev.populations.split()
        for pop in pops:
            for year in range(int(ev.start), int(ev.end) + 1):
                rows.append((pop, year, ev.type, ev.id))
    return pd.DataFrame(rows, columns=["pop", "year", "type", "event_id"])


def label_country_years(country_years: pd.DataFrame, expanded: pd.DataFrame) -> pd.DataFrame:
    """Assign the disruption label d_{c,t} to each observed country-year.

    Paper: sec:exceptional:events

    Priority when a country-year is in several dictionaries:
    war > respiratory > enteric > other.  Returns ``country_years`` with
    columns ``label`` (0–4), ``type`` and ``event_id`` (of the winning entry)
    and ``all_events`` (every matching entry, for transparency).
    """
    exp = expanded.copy()
    exp["label"] = exp["type"].map(_TYPE_TO_LABEL)
    exp = exp.sort_values(["pop", "year", "label"])
    win = exp.groupby(["pop", "year"], as_index=False).first()
    allev = exp.groupby(["pop", "year"])["event_id"].agg(lambda s: " ".join(s)).rename("all_events")
    out = country_years.merge(win, on=["pop", "year"], how="left").merge(
        allev.reset_index(), on=["pop", "year"], how="left")
    out["label"] = out["label"].fillna(0).astype(np.int8)
    out["type"] = out["label"].map(LABELS)
    out["event_id"] = out["event_id"].fillna("")
    out["all_events"] = out["all_events"].fillna("")
    return out


def crisis_screen(e0: pd.DataFrame, labels: pd.DataFrame, *, drop: float, halfwidth: int) -> pd.DataFrame:
    """List unlabelled country-years with an unusually low e₀ (diagnostic only).

    For each non-exceptional country-year, compare e₀ (sex average) with the
    median e₀ of the same population in the surrounding ±``halfwidth``
    years (excluding the year itself).  Years more than ``drop`` years below
    that median are returned for human review; nothing is changed.
    """
    df = e0.merge(labels[["pop", "year", "label"]], on=["pop", "year"])
    out = []
    for pop, g in df.groupby("pop"):
        g = g.set_index("year").sort_index()
        for year, row in g.iterrows():
            if row["label"] != 0:
                continue
            window = g.loc[(g.index >= year - halfwidth) & (g.index <= year + halfwidth) & (g.index != year), "e0"]
            if len(window) >= 3 and window.median() - row["e0"] > drop:
                out.append((pop, int(year), float(row["e0"]), float(window.median()),
                            float(window.median() - row["e0"])))
    return pd.DataFrame(out, columns=["pop", "year", "e0", "e0_neighbour_median", "shortfall"])


# ══════════════════════════════════════════════════════════════════════════════
# d5 — Adaptive temporal pooling (paper §3.4)
# ══════════════════════════════════════════════════════════════════════════════

def valley_ages(lt: pd.DataFrame, threshold: float) -> np.ndarray:
    """Ages whose median q_x over all populations, sexes and years < ``threshold``."""
    med = lt.groupby("age")["qx"].median()
    return med.index[med < threshold].to_numpy()


def contiguous_runs(years: Iterable[int], max_gap: int = 0) -> list[list[int]]:
    """Split sorted years into runs, breaking wherever more than ``max_gap`` years are missing.

    >>> contiguous_runs([1990, 1991, 1993, 1994, 1995])
    [[1990, 1991], [1993, 1994, 1995]]
    >>> contiguous_runs([2018, 2019, 2023, 2024], max_gap=3)   # COVID years skipped
    [[2018, 2019, 2023, 2024]]
    """
    runs: list[list[int]] = []
    for y in sorted(years):
        if runs and y - runs[-1][-1] - 1 <= max_gap:
            runs[-1].append(y)
        else:
            runs.append([y])
    return runs


def plan_blocks(run: list[int], positive: np.ndarray) -> list[list[int]]:
    """Partition one contiguous run of years into pooling blocks.

    ``positive[i, j]`` is True if year ``run[i]`` has m_x > 0 at valley age j.
    Years are added to the current block until every valley age has a
    positive rate somewhere in the block; the block is then closed (a year
    with no zeros therefore forms a block on its own).  Years left over at
    the end of the run that never achieve full coverage are merged into the
    previous block of the same run (or form a block of their own if there is
    none).  ``run`` comes from :func:`contiguous_runs`, so a block spans at
    most ``max_gap`` missing or exceptional calendar years (the years
    themselves are never in it).
    """
    blocks: list[list[int]] = []
    current: list[int] = []
    covered = np.zeros(positive.shape[1], dtype=bool)
    for i, year in enumerate(run):
        current.append(year)
        covered |= positive[i]
        if covered.all():
            blocks.append(current)
            current, covered = [], np.zeros_like(covered)
    if current:
        if blocks:
            blocks[-1] = blocks[-1] + current
        else:
            blocks.append(current)
    return blocks


def adaptive_pooling(lt: pd.DataFrame, de: pd.DataFrame, labels: pd.DataFrame, valley: np.ndarray,
                     *, max_gap: int, raw_rate_age_limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool adjacent non-exceptional years where valley-age rates are zero.

    Paper: sec:preprocessing:smoothing, eq:pooling

    Implements the adaptive temporal pooling for each population and sex:

    1. *Flag*: the population–sex needs pooling if any non-exceptional year
       has m_x = 0 at a valley age.  Its *pooling ages* are the valley ages
       plus any other age below ``raw_rate_age_limit`` (where HMD rates are
       raw D/E) at which it has a zero rate in some non-exceptional year —
       in the smallest populations zero counts also occur just above the
       valley.
    2. *Blocks*: the population's non-exceptional years are split into runs
       wherever more than ``max_gap`` calendar years are missing
       (exceptional years such as 2020–2022 are skipped over, never
       pooled); within each run :func:`plan_blocks` forms blocks that are
       zero-free at every pooling age.
    3. *Pool*: for a block of two or more years, the pooled rate at each
       pooling age is ``m_x = Σ_t D_{x,t} / Σ_t E_{x,t}`` from the HMD's death
       counts and exposures.  Below age 80 the published m_x are these
       occurrence–exposure rates rounded to five decimals, so an
       exposure-weighted mean of the published rates would differ a little
       (most at the lowest rates); it is used only at an age where a count or
       an exposure is missing.
    4. *Rebuild*: in every year of the block, q_x at the pooling ages is
       recomputed from the pooled m_x and that year's own a_x
       (``q = m / (1 + (1 − a) m)``).
    5. *Keep everything else*: q_x at all other ages keeps its
       published HMD value in every year, so pooling only touches the ages
       where Poisson noise is the problem.  Single-year blocks and
       unflagged populations are left untouched.

    Exceptional years are never pooled and never part of a block.  A run of
    years is broken wherever more than ``max_gap`` consecutive calendar years
    are missing, excluded or exceptional, so a block may skip such a gap (for
    example 2019 and 2023 around COVID-19 with ``max_gap = 3``) but never a
    longer one; years left at the end of a run join the block before them.

    Returns
    -------
    pooled : ``lt`` with pooled values substituted (same rows and columns)
    blocks : one row per multi-year block (pop, sex, first, last, n_years,
        valley zeros before/after, largest |Δ logit q_x| at valley ages)
    """
    valley = np.asarray(valley)
    ages = np.sort(lt["age"].unique())
    nonexc = labels.loc[labels["label"] == 0, ["pop", "year"]]
    nonexc_keys = set(map(tuple, nonexc.to_numpy()))

    idx_cols = ["pop", "sex", "year"]
    wide = {v: lt.pivot_table(index=idx_cols, columns="age", values=v, observed=True)
            .reindex(columns=ages) for v in ("mx", "ax", "qx")}
    expo = (de.pivot_table(index=idx_cols, columns="age", values="exposure", observed=True)
            .reindex(index=wide["mx"].index, columns=ages))
    deaths = (de.pivot_table(index=idx_cols, columns="age", values="deaths", observed=True)
              .reindex(index=wide["mx"].index, columns=ages))
    valley_pos = np.searchsorted(ages, valley)

    new = {v: wide[v].copy() for v in ("mx", "ax", "qx")}
    block_rows = []
    for (pop, sex), mx_ps in wide["mx"].groupby(level=["pop", "sex"], observed=True):
        years = [y for y in mx_ps.index.get_level_values("year") if (pop, y) in nonexc_keys]
        if not years:
            continue
        mx_years = mx_ps.droplevel(["pop", "sex"]).loc[years]
        positive_all = mx_years.to_numpy()[:, valley_pos] > 0
        if positive_all.all():
            continue                                    # step 1: not flagged
        # Pooling ages: valley ages + other raw-rate ages with zeros here.
        zero_any = (mx_years.to_numpy() <= 0).any(axis=0) & (ages < raw_rate_age_limit) & (ages > 0)
        pool_pos = np.union1d(valley_pos, np.flatnonzero(zero_any))
        pool_ages = ages[pool_pos]
        for run in contiguous_runs(years, max_gap=max_gap):
            pos_run = mx_years.loc[run].to_numpy()[:, pool_pos] > 0
            for block in plan_blocks(run, pos_run):
                if len(block) < 2:
                    continue                            # single year: unchanged
                keys = [(pop, sex, y) for y in block]
                m = wide["mx"].loc[keys].to_numpy()[:, pool_pos]       # (years, pooling ages)
                a = wide["ax"].loc[keys].to_numpy()[:, pool_pos]
                e = expo.loc[keys].to_numpy()[:, pool_pos]
                d = deaths.loc[keys].to_numpy()[:, pool_pos]
                # Pooled rate Σ D / Σ E.  At an age where a count or an exposure is missing, the
                # exposure-weighted mean of the published rates, and failing that their plain mean.
                with np.errstate(invalid="ignore", divide="ignore"):
                    m_de = d.sum(axis=0) / e.sum(axis=0)
                    m_em = np.nansum(e * m, axis=0) / np.nansum(e, axis=0)
                m_pool = np.where(np.isfinite(m_de), m_de, np.where(np.isfinite(m_em), m_em, np.nanmean(m, axis=0)))
                q_old = wide["qx"].loc[keys].to_numpy()[:, pool_pos]
                q_new = lifetable.qx_from_mx(m_pool[None, :], a)        # each year's own a_x
                for i, k in enumerate(keys):
                    new["mx"].loc[k, list(pool_ages)] = m_pool
                    new["qx"].loc[k, list(pool_ages)] = q_new[i]
                dz = np.abs(lifetable.logit(q_new) - lifetable.logit(q_old))
                block_rows.append({
                    "pop": pop, "sex": sex, "first": min(block), "last": max(block),
                    "n_years": len(block), "n_pool_ages": len(pool_pos),
                    "years": " ".join(map(str, block)),
                    "valley_zeros_before": int((m <= 0).sum()),    # zeros at every pooling age
                    "valley_zeros_after": int((m_pool <= 0).sum()) * len(block),
                    "max_abs_dlogit_valley": float(np.nanmax(dz)),
                    "median_abs_dlogit_valley": float(np.nanmedian(dz)),
                })

    # Write the substituted values back into the long table.
    pooled = lt.copy().set_index(idx_cols + ["age"])
    for v in ("mx", "ax", "qx"):
        long = new[v].stack().rename(v)
        pooled[v] = long.reindex(pooled.index).to_numpy()
    pooled = pooled.reset_index()
    blocks = pd.DataFrame(block_rows, columns=[
        "pop", "sex", "first", "last", "n_years", "n_pool_ages", "years",
        "valley_zeros_before", "valley_zeros_after",
        "max_abs_dlogit_valley", "median_abs_dlogit_valley"])
    return pooled, blocks


# ══════════════════════════════════════════════════════════════════════════════
# d6 — The tensor (paper §3.3)
# ══════════════════════════════════════════════════════════════════════════════

def build_tensor(lt: pd.DataFrame, labels: pd.DataFrame, *, q_min: float, min_observed_years: int,
                 populations: pd.DataFrame) -> dict:
    """Assemble the 4-D logit(q_x) tensor and its masks.

    Paper: eq:obs_mask, sec:preprocessing:tensor

    Returns a dict with

    ``M``            (S, A, C, T) logit(q_x); NaN where there is no data
    ``present``      (C, T) bool — both sexes' life tables exist
    ``label``        (C, T) int8 — disruption label (−1 where not present)
    ``observed`` (C, T) bool — present and non-exceptional (the observed mask O)
    ``exceptional``  (C, T) bool — present and exceptional (the set ℰ)
    ``pops``, ``pop_names``, ``years``, ``ages``, ``sexes``
    ``n_floored``, ``n_capped`` — cells changed by the floor/cap q_min of :func:`mdmx.lifetable.logit`
    ``dropped_pops`` — populations removed by the minimum-years rule
    """
    names = dict(zip(populations["code"], populations["name"], strict=False))
    obs_years = labels[labels["label"] == 0].groupby("pop")["year"].nunique()
    pops_all = sorted(lt["pop"].unique())
    dropped = [p for p in pops_all if obs_years.get(p, 0) < min_observed_years]
    pops = [p for p in pops_all if p not in dropped]
    years = np.arange(int(lt["year"].min()), int(lt["year"].max()) + 1)
    ages = np.sort(lt["age"].unique())

    S, A, C, T = len(SEXES), len(ages), len(pops), len(years)
    M = np.full((S, A, C, T), np.nan)
    pop_idx = {p: i for i, p in enumerate(pops)}
    sub = lt[lt["pop"].isin(pops)]
    q = sub["qx"].to_numpy()
    n_floored = int((q < q_min).sum())
    n_capped = int((q > 1 - q_min).sum())
    s_i = sub["sex"].map({s: i for i, s in enumerate(SEXES)}).to_numpy().astype(int)
    a_i = np.searchsorted(ages, sub["age"].to_numpy())
    c_i = sub["pop"].map(pop_idx).to_numpy()
    t_i = sub["year"].to_numpy() - years[0]
    M[s_i, a_i, c_i, t_i] = lifetable.logit(q, q_min)

    present = np.isfinite(M).all(axis=(0, 1))
    label = np.full((C, T), -1, dtype=np.int8)
    lab = labels[labels["pop"].isin(pops)]
    label[lab["pop"].map(pop_idx).to_numpy(), lab["year"].to_numpy() - years[0]] = lab["label"].to_numpy()
    label[~present] = -1
    return {
        "M": M, "present": present, "label": label,
        "observed": present & (label == 0), "exceptional": present & (label > 0),
        "pops": np.array(pops), "pop_names": np.array([names.get(p, p) for p in pops]),
        "years": years, "ages": ages, "sexes": np.array(SEXES),
        "n_floored": n_floored, "n_capped": n_capped, "dropped_pops": dropped,
    }
