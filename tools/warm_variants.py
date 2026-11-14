"""Compute the runs of chapter 12's unranked variants ahead of its render, with a log (development origins only).

    uv run python tools/warm_variants.py --cheap                       # the flow forms, Li–Lee; previews
    uv run python tools/warm_variants.py --mdmx --workers 0            # MDMx's sensitivity runs (the M4)
    uv run python tools/warm_variants.py --mdmx --origins 1980 --workers 4
    nice -n 10 uv run python tools/warm_variants.py --mdmx > work/logs/variants-mdmx.log 2>&1 &

``--cheap`` stages the flow field's variant forms (FR1-50, FR0-all, FR2-all and lab study L07's
ablation) and Li–Lee's two forms at every development origin (``mdmx.variants.run_variants``, parts
``flow`` and ``lilee``; seconds).  It then prints previews of the variants' summaries that need no
other entry: the pooling counts (decision Pool, on the real data), and, on the points that the
variants and the naive share (not yet the comparison's points, which need bayesLife), the variants'
e₀ error by band, their intervals from past errors only against within-era intervals and their counts of measures worse
than the naive and than the windowed naive.  The chapter computes the same summaries for the ranked
entries on the comparison's own points.

``--mdmx`` stages anchored MDMx's sensitivity runs (drift weights and numbers of components; part
``mdmx``), five variants × 21 origins of Kalman fits: hours, for the M4.

The chapter calls the same functions with the same parameters, so every run made here is found in the
artefact store by the render.  Nothing is computed or scored at the held-out origins 2005–2015.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import pandas as pd

from mdmx import artifacts
from mdmx import compare as cp
from mdmx import evaluation as ev
from mdmx import flowfield as ff
from mdmx import forecast as fc
from mdmx import svdcomp as sc
from mdmx import variants as va
from mdmx.config import params


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def previews(ctx, P, runs) -> None:
    """Items 5–7 of the variants on the real data: pooling counts; on the variants' shared points, the e₀
    error by band, intervals from past errors only and the counts of measures worse than either naive."""
    D, F, E = ctx["design"], ctx["F"], ctx["E"]
    M, observed, pops, years = ctx["M"], ctx["observed"], ctx["pops"], ctx["years"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    pooled = va.pooling_counts(artifacts.load("lt_pooled", quiet=True)["blocks"], ctx["jobs"]["full"], D)
    log("Pooling blocks that span an origin (every origin of the design):\n"
        + pooled["summary"].to_string(index=False)
        + f"\n  populations with an overlap: {len(pooled['populations'])}; years past the origin: median "
          f"{pooled['overlaps']['past'].median():.0f}, largest {pooled['overlaps']['past'].max()}"
        + "".join(f"\n  affected populations, {era}: {', '.join(p) or '-'}"
                  for era, p in pooled["affected_populations"].items()))

    L = P.section("lilee")
    res = va.variant_results(runs, n_ages=M.shape[1], ax=ctx["cmp"]["mx_to_qx_ax"],
                             lilee_codes={"group": L["method_group"], "sex": L["method_sex"]})
    labels = list(res)
    truth = fc.observed_e0(M, observed, pops, years)
    levels = cp.quantile_levels(F["pi_levels"])
    naive = cp.naive_e0(truth, va.development_jobs(ctx["jobs"]["full"], D), levels)
    points = pd.concat([ev.long_points(res), naive.assign(entry=ev.NAIVE, h=naive["year"] - naive["origin"])],
                       ignore_index=True)
    keys = ev.paired_keys(points, truth, [*labels, ev.NAIVE])
    keys["era"] = keys["origin"].map(D.era)
    starts = list(F["age_group_starts"]) + [M.shape[1]]
    groups = {lab: list(range(starts[i], starts[i + 1])) for i, lab in enumerate(F["age_group_labels"])}
    old_age = int(E["overall"]["old_age"])
    families = dict(zip(pops, sc.population_families(pops, artifacts.load("populations", quiet=True)), strict=True))
    log(f"Preview on the {len(keys):,} points the {len(labels)} variants and the naive share "
        f"(by era: {keys['era'].value_counts().to_dict()})")

    tab, used = va.band_errors(res, truth, keys, D, M, pops, years, groups, old_age=old_age)
    if (used["scored"] != used["keys"]).any():
        raise RuntimeError(f"a variant misses shared points:\n{used}")
    for measure in ("e0_both", "e0_female", "e0_male", "gap", "schedule_dx"):
        log(f"MAE by era and band, {measure}:\n" + va.band_wide(tab, measure, "mae", D, labels).round(2).to_string())
    log("Mean error by era and band, e0_both:\n" + va.band_wide(tab, "e0_both", "me", D, labels).round(2).to_string())

    min_pts = int(P.section("flowfield")["interval_min_points"])
    cal = ff.within_era_calibration(D.eras, D.held_out, D.held_out_by)
    within, _ = ev.empirical_quantiles(points, truth, keys, labels, calibration=cal, pi_levels=F["pi_levels"],
                                       min_points=min_pts)
    real, sparse = va.past_error_quantiles(points, truth, keys, labels, design=D, pi_levels=F["pi_levels"],
                                         min_points=min_pts)
    cmp_tab = va.interval_comparison({"within era": within, "past errors": real}, keys, truth, D, F["pi_levels"],
                                     {"1–15": (1, 15)})
    both = cmp_tab[cmp_tab["sex"] == "both"].pivot_table(index="entry", columns=["era", "rule"],
                                                          values=["n", "in80", "in95", "wis"], sort=False)
    log(f"Intervals, average of the sexes, horizons 1–15, on the points where both rules give one (past errors "
        f"only: {len(sparse):,} cells without enough errors):\n" + both.round(3).to_string())

    vals = va.variant_values(res, truth, keys, D, M, pops, years, groups, old_age=old_age, levels=levels,
                             pi_levels=F["pi_levels"], calibration=cal, min_points=min_pts, families=families)
    naive_kw = {"old_age": old_age, "levels": levels, "pi_levels": F["pi_levels"], "calibration": cal,
                "min_points": min_pts, "families": families}
    nv = va.naive_values(truth, M, observed, pops, years, ctx["jobs"]["full"], keys, D, groups, window=None,
                         label=ev.NAIVE, **naive_kw)
    nw = va.naive_values(truth, M, observed, pops, years, ctx["jobs"]["full"], keys, D, groups, window=D.window,
                         label=va.NAIVE_WINDOW, **naive_kw)
    EO = E["overall"]
    spec = E["views"]["primary"]
    lo, hi = spec["horizons"]
    allv, n_out = ev.complete(pd.concat([vals, nv, nw], ignore_index=True))
    prim = allv[(allv["era"] == spec["origins"]) & allv["h"].between(lo, hi)]
    worse = va.worse_than_naive(prim[prim["method"] != va.NAIVE_WINDOW], prim[prim["method"] == va.NAIVE_WINDOW],
                                families=EO["families"], composite=EO["composite_full"], methods=labels,
                                naive=ev.NAIVE)
    log(f"Primary view ({prim['point'].nunique():,} points; {n_out} shared points left out for a missing measure): "
        "measures worse than the naive and than the windowed naive, and the ratios to the windowed naive:\n"
        + worse.round(2).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cheap", action="store_true", help="the flow forms and Li–Lee, then the previews")
    mode.add_argument("--mdmx", action="store_true", help="MDMx's sensitivity runs (hours: the M4)")
    ap.add_argument("--origins", nargs="*", type=int, default=None, help="development origins (default: all)")
    ap.add_argument("--workers", type=int, default=0, help="processes for the MDMx fits (0: one per core)")
    ap.add_argument("--no-previews", action="store_true", help="--cheap without the previews")
    args = ap.parse_args()
    t0 = time.perf_counter()
    P = params()
    parts = ("flow", "lilee") if args.cheap else ("mdmx",)
    log(f"parts {parts}; origins {args.origins or 'every development origin'}; workers {args.workers}")
    ctx = ev.context(P)
    D = ctx["design"]
    origins = args.origins or va.development_origins(D)
    log(f"context ready: {len(origins)} origins, {len(va.development_jobs(ctx['jobs']['full'], D, origins))} jobs")
    runs = va.run_variants(ctx, P, parts=parts, workers=args.workers, origins=args.origins, log=log)
    if args.cheap:
        for code, by_origin in runs["lilee"].items():
            fails = sum(len(r["failures"]) for r in by_origin.values())
            warns = pd.concat([r["warnings"] for r in by_origin.values()], ignore_index=True)
            log(f"Li–Lee {code}: {fails} failures, {len(warns)} warnings"
                + (":\n" + warns.groupby("message").size().sort_values(ascending=False).head(8).to_string()
                   if len(warns) else ""))
        secs = pd.DataFrame({o: r["seconds"] for o, r in runs["flow"].items()}).sum(axis=1)
        log("Flow-field forms, run seconds over the origins: " + ", ".join(f"{k} {v:.1f}" for k, v in secs.items()))
        fr2 = pd.DataFrame({o: r["runs"]["FR2-all"]["flow"] for o, r in runs["flow"].items()}).T
        log("FR2-all by origin (v_e, held steps, fallback):\n"
            + fr2[["v_e", "held", "fallback"]].to_string())
        if not args.no_previews:
            previews(ctx, P, runs)
    else:
        for label, by_origin in runs["mdmx"].items():
            not_conv = sum(int((~r["jobs"]["converged"]).sum()) for r in by_origin.values())
            wall = sum(r["seconds"] for r in by_origin.values())
            cpu = sum(r["cpu_seconds"] for r in by_origin.values())
            log(f"MDMx, {label}: {len(by_origin)} origins, fits not converged {not_conv}, "
                f"run time {wall:.0f} s wall, {cpu:.0f} s CPU (as stored)")
    log(f"done in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
