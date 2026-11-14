"""The ``mdmx`` command line — one entry point for every routine task.

Run ``uv run mdmx --help``.  Commands:

``mdmx data verify``       check the HMD release against raw/MANIFEST.json
``mdmx render [target]``   render the book, or one chapter (path or stem)
``mdmx article``           build the merged article (article/mdmx.pdf) from the companion's results
``mdmx check``             run the tests, the linter, the stand-alone check and every project's
                           results-manifest freshness check
``mdmx clean [--work]``    remove a project's rendered books (and, with --work, its artefact store)

``render`` and ``clean`` take ``--project`` (``book``, the default, or ``companion``): each project
has its own outputs and artefact store (:mod:`mdmx.paths`, "Projects").  ``render --profile NAME``
renders with the project's Quarto profile ``_quarto-NAME.yml`` (the companion's ``public`` profile
leaves out the process material and writes ``companion/_output-public/``).

These are thin wrappers around Quarto, latexmk, pytest and ruff, written in
Python so that they behave identically on macOS, Windows and Linux.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from collections import defaultdict

from . import paths, provenance

__all__ = ["main"]


def _run(cmd: list[str], cwd=None, env=None) -> int:
    print("$ " + " ".join(cmd), flush=True)          # before the command's own output, also in a log
    return subprocess.call(cmd, cwd=cwd, env=env)


def _find_chapter(target: str) -> str:
    """Accept 'book/analysis/04-tucker.qmd', '04-tucker' or 'd2-ingest' (in the project's book)."""
    book = paths.book()
    if target.endswith(".qmd"):
        return str((paths.root() / target).resolve().relative_to(book.resolve()))
    hits = sorted(book.rglob(f"{target}.qmd"))
    if len(hits) != 1:
        raise SystemExit(f"Chapter '{target}' not found (or ambiguous) under {paths.relative(book)}/.")
    return str(hits[0].relative_to(book))


def cmd_render(args) -> int:
    quarto = shutil.which("quarto")
    if quarto is None:
        raise SystemExit("Quarto not found.  Install it (see MANUAL.md) and make sure `quarto` is on PATH.")
    with paths.using_project(args.project):
        cmd = [quarto, "render"]
        if args.target:
            cmd.append(_find_chapter(args.target))
        if args.profile:
            # Quarto renders without complaint under a profile that has no file, so a misspelt name
            # would give the default book in the default folder: stop instead.
            profile_file = paths.book() / f"_quarto-{args.profile}.yml"
            if not profile_file.is_file():
                raise SystemExit(f"Profile '{args.profile}' not found: {paths.relative(profile_file)} does not exist.")
            cmd += ["--profile", args.profile]
        # Make Quarto execute the chapters with *this* Python (the project's .venv), whatever else is
        # installed on the computer; MDMX_PROJECT tells the chapters' mdmx which outputs and store to use.
        env = {**os.environ, "QUARTO_PYTHON": sys.executable, "MDMX_PROJECT": args.project}
        print(f"$ {' '.join(cmd)}    (project {args.project})")
        return subprocess.call(cmd, cwd=paths.book(), env=env)


def _latexmk(folder, env=None) -> int:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise SystemExit("latexmk not found.  Install a TeX distribution (see MANUAL.md).")
    return _run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "mdmx.tex"], cwd=folder, env=env)


def cmd_article(args) -> int:
    """Build ``article/mdmx.pdf`` (or another LaTeX project's, ``--folder article-short``); every figure,
    table and number comes from ``companion/results/``."""
    with paths.using_project("companion"):
        numbers = paths.numbers() / "all.tex"
    if not numbers.is_file():
        raise SystemExit(f"{paths.relative(numbers)} not found: render the companion first "
                         "(uv run mdmx render --project companion).")
    folder = paths.root() / args.folder
    if not (folder / "mdmx.tex").is_file():
        raise SystemExit(f"{args.folder}/mdmx.tex not found")
    # The links to the code and the companion are computed from the sources and results as they are now.
    rc = _run([sys.executable, "tools/paper_links.py", "--folder", args.folder], cwd=paths.root())
    # The folder's latexmkrc writes them too when an editor builds; here they are written, so it skips that.
    return rc or _latexmk(folder, env={**os.environ, "MDMX_LINKS_DONE": "1"})


def _ruff() -> list[str]:
    """The ruff command: ``ruff`` on the PATH, else this Python's ruff module."""
    ruff = shutil.which("ruff")
    if ruff:
        return [ruff]
    if importlib.util.find_spec("ruff") is not None:
        return [sys.executable, "-m", "ruff"]
    raise SystemExit("ruff not found, so the code cannot be linted.  Install the development tools "
                     "with `uv sync --all-extras` and run `uv run mdmx check`.")


_UNVERIFIED_HINTS = {
    "no-manifest": "the results manifest is missing (the project was never rendered?)",
    "no-source-hash": "entries without a chapter source hash (made before source hashes were recorded; "
                      "re-render their chapters)",
    "inputs-missing": "entries whose input artefacts are not in the project's store (a fresh clone? render it)",
    "package-changed": "entries made before the mdmx package last changed (code a chapter calls directly "
                       "may show something else now; re-render their chapters to be sure)",
}


def _report_manifest(problems: list[provenance.ManifestProblem]) -> int:
    """Print the manifest check's problems; return 1 if any file is stale, else 0."""
    stale = [p for p in problems if p.stale]
    unverified: dict[str, list[provenance.ManifestProblem]] = defaultdict(list)
    for p in problems:
        if not p.stale:
            unverified[p.kind].append(p)
    for p in stale:
        print(f"STALE    {p.path}: {p.message}")
    for kind, group in unverified.items():
        print(f"warning: cannot verify {len(group)} {_UNVERIFIED_HINTS.get(kind, kind)}; "
              f"e.g. {group[0].path}: {group[0].message}")
    if stale:
        files = len({p.path for p in stale})
        print(f"manifest: {files} results file(s) stale; re-render their chapters (see above).")
        return 1
    print("manifest: no stale results files" + (" (some could not be verified, see the warnings)"
                                                 if unverified else ""))
    return 0


def cmd_check(args) -> int:
    ruff = _ruff()                      # before the tests, so that a missing linter fails at once
    rc = _run([sys.executable, "-m", "pytest", "-q"], cwd=paths.root())
    rc |= _run([*ruff, "check", "src", "tests", "tools"], cwd=paths.root())
    # The companion and the code make no reference to earlier work (decision SA).
    rc |= _run([sys.executable, "tools/check_standalone.py"], cwd=paths.root())
    for name in paths.PROJECTS:
        with paths.using_project(name):
            if not paths.book().is_dir():
                print(f"(project {name}: no {paths.relative(paths.book())}/ folder in this checkout; not checked)")
                continue
            print(f"$ provenance.check_manifest()    (project {name}: {paths.relative(paths.results())}/)")
            rc |= _report_manifest(provenance.check_manifest())
    return rc


def cmd_data(args) -> int:
    from . import data
    from .config import params

    report = data.verify_release(params().data.hmd_release)
    print(f"HMD release {report['release']} at {report['source']}: all checksums match.")
    return 0


def cmd_clean(args) -> int:
    """Remove the project's rendered books (``_output/`` and each profile's ``_output-<profile>/``) and,
    with ``--work``, its artefact store only: the rest of ``work/`` (another project's store,
    bayesLife's off-site runs, logs) is never touched."""
    with paths.using_project(args.project):
        folders = [paths.book() / "_output", *sorted(paths.book().glob("_output-*")), paths.book() / ".quarto"]
        if args.work:
            folders.append(paths.artifacts())
        for folder in folders:
            if folder.exists():
                shutil.rmtree(folder)
                print(f"removed {paths.relative(folder)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mdmx", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("render", help="render the book or one chapter")
    p.add_argument("target", nargs="?", help="chapter path or stem, e.g. 04-tucker")
    p.add_argument("--project", choices=paths.PROJECTS, default=paths.DEFAULT_PROJECT,
                   help=f"the Quarto book to render (default {paths.DEFAULT_PROJECT})")
    p.add_argument("--profile", help="a Quarto profile of the project (its _quarto-PROFILE.yml), e.g. public: "
                                     "the companion without its process material, in companion/_output-public/")
    p.set_defaults(func=cmd_render)
    p_article = sub.add_parser("article", help="build the merged article's PDF from the companion's results")
    p_article.add_argument("--folder", default="article", help="the LaTeX project: article (default), article-short or article-shortest")
    p_article.set_defaults(func=cmd_article)
    sub.add_parser(
        "check", help="run the tests, the linter and the results-manifest freshness check",
        description="Run pytest, then `ruff check src tests tools` (fails if ruff is not installed), then "
                    "tools/check_standalone.py (no reference to earlier work in the companion and the code), then "
                    "check each project's results manifest (results/manifest.json, "
                    "companion/results/manifest.json): every results file must be the one recorded, its "
                    "chapter's source must not have changed since, and its input artefacts must be the "
                    "ones in the project's store.  Fails on any stale file; warns about files that cannot "
                    "be verified (no artefact store, entries made before source hashes were recorded, or "
                    "made before the mdmx package last changed).",
    ).set_defaults(func=cmd_check)
    p = sub.add_parser("data", help="data utilities")
    p.add_argument("action", choices=["verify"])
    p.set_defaults(func=cmd_data)
    p = sub.add_parser("clean", help="remove a project's rendered output")
    p.add_argument("--project", choices=paths.PROJECTS, default=paths.DEFAULT_PROJECT,
                   help=f"the Quarto book to clean (default {paths.DEFAULT_PROJECT})")
    p.add_argument("--work", action="store_true",
                   help="also delete the project's artefact store (work/artifacts/ for book, "
                        "work/<project>/artifacts/ otherwise); nothing else in work/")
    p.set_defaults(func=cmd_clean)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
