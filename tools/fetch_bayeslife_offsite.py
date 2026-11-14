"""Copy one off-site bayesLife batch into the folder chapter 12 reads.

    uv run python tools/fetch_bayeslife_offsite.py --source work/bayeslife/offsite-batches/2026-11-03_hmd-20261029

bayesLife's runs are made outside a render, in an off-site batch that ``tools/bayeslife_bundle.py``
writes (kept in ``work/bayeslife/offsite-batches/``; its README says how to run it on one or more
machines and how to gather every machine's ``results/`` into one folder).  This copies the whole
batch, the simulations included (about 100 MB per run), into ``work/bayeslife/chapter-12-input/``,
which is git-ignored, with ``rsync``: a second call copies only what changed.  The smoke test's folders are left out.  Chapter 12 reads the copy through ``mdmx.bayeslife.offsite_runs``, which checks it against
the project's manifest and the book's own inputs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from mdmx import bayeslife as bl
from mdmx import paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="the batch, e.g. work/bayeslife/offsite-batches/2026-11-03_hmd-20261029")
    args = ap.parse_args()
    src, dest = args.source.expanduser(), paths.bayeslife() / "chapter-12-input"
    if not (src / "MANIFEST.json").is_file():
        raise SystemExit(f"{src} is not an off-site project (no MANIFEST.json)")
    if shutil.which("rsync") is None:
        raise SystemExit("rsync is needed")
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--delete", "--exclude", "results-smoke/", "--exclude", "sims-smoke/",
                    f"{src}/", f"{dest}/"], check=True)
    bl.verify_offsite(dest)
    results = dest / "results"
    tags = sorted(d.name for d in results.iterdir() if d.is_dir()) if results.is_dir() else []
    done = [t for t in tags if (results / t / "DONE").is_file()]
    failed = [t for t in tags if (results / t / "FAILED").is_file()]
    print(f"Copied {src} to {paths.relative(dest)}; its files match the manifest.")
    print(f"Runs: {len(done)} done, {len(failed)} failed{': ' + ', '.join(failed) if failed else ''}, "
          f"{len(tags) - len(done) - len(failed)} unfinished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
