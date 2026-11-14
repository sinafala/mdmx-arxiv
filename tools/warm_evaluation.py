"""Compute chapter 12's runs ahead of its render (every run but bayesLife's), with a log.

    uv run python tools/warm_evaluation.py                          # every part
    uv run python tools/warm_evaluation.py --parts flowfield mdmx   # some parts
    nice -n 10 uv run python tools/warm_evaluation.py --workers 4 --concurrent 2 > work/logs/x.log 2>&1 &

The chapter calls the same functions (``mdmx.evaluation.context`` and ``run_entries``) with the same
parameters, so every run made here is found in the artefact store by the render.  The parts are
listed in ``mdmx.evaluation.PARTS``; a part already computed returns at once.  Nothing is scored here,
the held-out block included.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from mdmx import evaluation as ev
from mdmx.config import params


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=list(ev.PARTS), choices=list(ev.PARTS))
    ap.add_argument("--workers", type=int, default=0, help="processes for the MDMx fits (0: one per core)")
    ap.add_argument("--concurrent", type=int, default=3, help="R processes at once")
    args = ap.parse_args()
    log(f"parts {args.parts}; workers {args.workers}; concurrent {args.concurrent}")
    ctx = ev.context(params())
    log(f"context ready: {len(ctx['design'].origins)} origins, {len(ctx['jobs']['full'])} jobs, "
        f"{len(ctx['forms'])} flow-field forms")
    ev.run_entries(ctx, workers=args.workers, concurrent=args.concurrent, parts=args.parts, log=log)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
