"""Provenance bookkeeping: which chapter made what, from which inputs.

Three things are tracked:

1. **The current chapter.**  Every book chapter starts with
   ``provenance.start_chapter("d2-ingest")``.  All artefacts, figures,
   tables and numbers produced afterwards are attributed to that chapter.

2. **What the chapter has read.**  Each artefact the chapter loads is noted,
   so that a figure or table saved later can list the artefacts (and their
   content hashes) it was derived from.

3. **The results manifest** (``results/manifest.json``).  One entry per
   paper-facing file — figure PDF, LaTeX table, numbers file — with its
   SHA-256, producing chapter, input artefact hashes, the SHA-256 of the
   chapter's source, the git commit and whether the tree was dirty, and a
   timestamp, and the SHA-256 of the ``mdmx`` package's source.  ``mdmx check``
   uses it (:func:`check_manifest`) to check that every file the paper reads is
   the one recorded, that its chapter's source has not changed since, and that
   its input artefacts are the ones in the project's artefact store.  A change
   to the ``mdmx`` package since a file was made gives a warning: library code
   that a chapter runs directly rather than through a staged artefact may have
   changed what the file shows, and only a re-render settles it.  Artefacts
   computed on another machine differ in their recorded run times, so their
   inputs read as changed.  Each project (:data:`mdmx.paths.PROJECTS`) has its
   own manifest in its own results folder.

Environment details (git commit, Python and package versions) are collected
once per process by :func:`environment`.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import cache, lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

from . import paths
from ._lock import locked

__all__ = [
    "start_chapter", "end_chapter", "current_chapter", "note_input", "chapter_inputs",
    "record_output", "environment", "sha256_file", "now", "text_sha256", "lock_sha256", "package_sha256",
    "chapter_source", "chapter_cells", "current_cell",
    "ManifestProblem", "check_manifest", "STALE_KINDS", "UNVERIFIED_KINDS",
]

_STATE: dict[str, Any] = {"chapter": None, "inputs": {}, "started": None, "outputs": [], "cells": {}}

# Packages whose versions are recorded with every artefact.
_TRACKED_PACKAGES = (
    "mdmx", "numpy", "scipy", "pandas", "pyarrow", "duckdb", "scikit-learn",
    "statsmodels", "matplotlib",
)


def now() -> str:
    """Current UTC time as an ISO-8601 string (seconds precision)."""
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, read in 1 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(path: Path) -> str:
    """SHA-256 of a text file with LF line endings, whatever the checkout's line endings are."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def lock_sha256() -> str | None:
    """SHA-256 of the project's ``uv.lock`` (None if there is none), read once per process.

    It stands for the Python environment in every artefact's cache key: after a ``uv lock --upgrade``
    (a new NumPy, SciPy, …) every step recomputes.
    """
    return _lock_sha256(paths.root())


@cache
def _lock_sha256(root: Path) -> str | None:
    lock = root / "uv.lock"
    return text_sha256(lock) if lock.is_file() else None


def package_sha256() -> str:
    """SHA-256 of the ``mdmx`` package's source (every ``.py`` file under ``src/mdmx``, by relative path,
    with LF line endings), read once per process.  Recorded with every results file, so that
    :func:`check_manifest` can warn when the library changed after the file was made."""
    return _package_sha256(Path(__file__).resolve().parent)


@cache
def _package_sha256(package: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(package.rglob("*.py")):
        digest.update(file.relative_to(package).as_posix().encode() + b"\0")
        digest.update(file.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def environment() -> dict[str, Any]:
    """Git commit, Python version and package versions for this process.

    ``git_dirty`` is True if a tracked file differs from the commit, outside every project's results
    folder (``results/``, ``companion/results/``: the chapters write those themselves, so a render
    would otherwise mark every later chapter dirty) and outside ``article/`` and ``docs/`` (text
    that no chapter reads, so editing it during a long render does not make its results unverified).  If git cannot tell (not
    installed, not a repository), ``git_commit`` and ``git_dirty`` are None and ``git_error``
    says why.
    """
    commit, dirty, error = None, None, None
    root = paths.root()
    excluded = [":(exclude)article", ":(exclude)docs"]
    for name in paths.PROJECTS:
        with paths.using_project(name):
            excluded.append(f":(exclude){paths.relative(paths.results())}")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ".", *excluded],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout
        dirty = bool(status.strip())
    except FileNotFoundError as exc:
        error = f"git not found: {exc}"
    except subprocess.CalledProcessError as exc:      # e.g. not a repository; a commit found first is kept
        error = f"{' '.join(exc.cmd[:2])} failed: {(exc.stderr or '').strip() or exc}"
    versions = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_error": error,
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "packages": versions,
        "uv_lock_sha256": lock_sha256(),
    }


# ── Chapter lifecycle ────────────────────────────────────────────────────────

def start_chapter(name: str) -> None:
    """Declare that the code that follows belongs to book chapter ``name``.

    Raises
    ------
    RuntimeError if the working folder is inside one project's folder (Quarto runs each chapter in
    its own folder) while another project is in use: the chapter would otherwise write its
    outputs, store and manifest into the other project's, e.g. the frozen ``results/``.
    """
    _check_project_folder()
    source = chapter_source(name)
    cells: dict[str, str | None] = {}
    for cell in chapter_cells(source) if source else []:
        cells.setdefault(cell["code"], cell["label"])
    _STATE.update(chapter=name, inputs={}, started=now(), outputs=[], cells=cells)
    paths.ensure(paths.logs())
    _log({"event": "chapter-start", "chapter": name, "env": environment()})


def _check_project_folder() -> None:
    """Raise RuntimeError if the working folder belongs to a project other than the one in use."""
    cwd = Path.cwd().resolve()
    root = paths.root()
    for name in paths.PROJECTS:
        folder = (root / name).resolve()
        if cwd == folder or folder in cwd.parents:
            if name != paths.project():
                raise RuntimeError(
                    f"This chapter runs in {paths.relative(cwd)}/, a folder of project {name!r}, but "
                    f"project {paths.project()!r} is in use, so its outputs would go to "
                    f"{paths.relative(paths.results())}/.  Render with "
                    f"'uv run mdmx render --project {name}' (or set MDMX_PROJECT={name})."
                )
            return


def end_chapter() -> dict[str, Any]:
    """Close the chapter: remove its stale outputs, log a summary and return it.

    A paper-facing file that the manifest records for this chapter but that this render did not
    write (a figure or table the chapter no longer makes) is deleted with its manifest entry, so
    that ``results/`` holds only what the chapter now produces.  The chapter's last cell calls
    this, so a render that fails part-way removes nothing.
    """
    summary = {
        "event": "chapter-end",
        "chapter": _STATE["chapter"],
        "started": _STATE["started"],
        "finished": now(),
        "inputs": dict(_STATE["inputs"]),
        "outputs": list(_STATE["outputs"]),
        "removed": _prune_outputs(),
    }
    _log(summary)
    return summary


def _prune_outputs() -> list[str]:
    """Delete the current chapter's recorded outputs that this render did not write; return them."""
    chapter = _STATE["chapter"]
    if chapter is None or not _manifest_path().exists():
        return []
    with locked(_manifest_path()):
        manifest = _read_manifest()
        written = set(_STATE["outputs"])
        stale = sorted(rel for rel, entry in manifest["files"].items()
                       if entry.get("chapter") == chapter and rel not in written)
        for rel in stale:
            (paths.root() / rel).unlink(missing_ok=True)
            del manifest["files"][rel]
        if stale:
            _write_manifest(manifest)
    return stale


def current_chapter() -> str | None:
    return _STATE["chapter"]


_CELL_START = re.compile(r"^```\{python[^}]*\}\s*$")
_CELL_END = re.compile(r"^```\s*$")
_CELL_OPTION = re.compile(r"^#\|\s*([\w-]+)\s*:\s*(.*?)\s*$")


def _normalised_code(text: str) -> str:
    """A cell's code as compared: line ends unified, trailing blanks and blank edge lines removed."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def chapter_cells(source: Path) -> list[dict[str, Any]]:
    """The Python cells of a chapter source (``.qmd``), in order.

    Each is a dict: ``label`` (the ``#| label:`` option, or None), ``first_line`` and ``last_line``
    (1-based, the fences included) and ``code`` (the cell without its option lines, normalised as
    :func:`_normalised_code` does: Quarto sends the kernel the code without the option lines).
    """
    lines = source.read_text(encoding="utf-8").splitlines()
    cells, i = [], 0
    while i < len(lines):
        if not _CELL_START.match(lines[i]):
            i += 1
            continue
        j, options, code = i + 1, {}, []
        while j < len(lines) and not _CELL_END.match(lines[j]):
            option = _CELL_OPTION.match(lines[j])
            if option and not code:                  # options come first; later '#|' lines are code
                options[option.group(1)] = option.group(2)
            else:
                code.append(lines[j])
            j += 1
        cells.append({"label": options.get("label"), "first_line": i + 1, "last_line": j + 1,
                      "code": _normalised_code("\n".join(code))})
        i = j + 1
    return cells


def current_cell() -> str | None:
    """The label of the chapter cell now running, read from the notebook's input history.

    The figure, table and number records carry it, so the article can link each result to the cell
    that made it.  None outside a notebook (a test, a tool), for a cell without a label, or for code
    that is not one of the chapter's cells.
    """
    ipython = sys.modules.get("IPython")            # imported only inside a kernel; never import it here
    shell = ipython.get_ipython() if ipython is not None else None
    history = shell.user_ns.get("In") if shell is not None else None
    if not history:
        return None
    return _STATE["cells"].get(_normalised_code(history[-1]))


def note_input(name: str, content_hash: str) -> None:
    """Remember that the current chapter used artefact ``name``."""
    _STATE["inputs"][name] = content_hash


def chapter_inputs() -> dict[str, str]:
    return dict(_STATE["inputs"])


def _log(entry: dict[str, Any]) -> None:
    chapter = entry.get("chapter") or _STATE["chapter"] or "no-chapter"
    paths.ensure(paths.logs())
    entry = {"time": now(), **entry}
    with open(paths.logs() / f"{chapter}.jsonl", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def log(event: str, **fields: Any) -> None:
    """Append a free-form diagnostic record to the chapter's run log."""
    _log({"event": event, "chapter": _STATE["chapter"], **fields})


# ── Results manifest ─────────────────────────────────────────────────────────

def _manifest_path() -> Path:
    return paths.results() / "manifest.json"


def _read_manifest() -> dict[str, Any]:
    path = _manifest_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"description": "Provenance of every paper-facing file in results/. "
                           "Written automatically; do not edit.", "files": {}}


def _write_manifest(manifest: dict[str, Any]) -> None:
    paths.ensure(paths.results())
    # LF on every platform, so that the file and its git diffs are the same on macOS and Windows.
    _manifest_path().write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")


def chapter_source(chapter: str) -> Path | None:
    """The book file of chapter ``chapter`` (``book/data/<id>.qmd`` or ``book/analysis/<id>.qmd``).

    Returns None if no such file exists (an id that is not a book chapter, e.g. in a test or a tool).

    Raises
    ------
    ValueError if several files under ``book/`` have that name.
    """
    for folder in ("data", "analysis"):
        path = paths.book() / folder / f"{chapter}.qmd"
        if path.is_file():
            return path
    hits = sorted(paths.book().rglob(f"{chapter}.qmd")) if paths.book().is_dir() else []
    if len(hits) > 1:
        raise ValueError(f"chapter {chapter!r} is ambiguous: {[paths.relative(h) for h in hits]}")
    return hits[0] if hits else None


def _source_sha256(chapter: str | None) -> str | None:
    source = chapter_source(chapter) if chapter else None
    return text_sha256(source) if source else None


def record_output(kind: str, path: Path, *, label: str, extra: dict[str, Any] | None = None) -> None:
    """Add/refresh the manifest entry for a paper-facing output file.

    The entry records the file's SHA-256, the chapter, its input artefacts' content hashes, the
    SHA-256 of the chapter's source (``source_sha256``) and of the ``mdmx`` package
    (``package_sha256``), the git commit and whether the tree was dirty (:func:`environment`).

    Parameters
    ----------
    kind : {"figure", "table", "numbers", "bundle"}
    path : the file just written (inside the project's results folder)
    label : the paper label (e.g. ``"fig-s4-age-basis"``)
    extra : any additional fields worth recording (caption, n_rows, …)
    """
    with locked(_manifest_path()):
        _record_output_unlocked(kind, path, label=label, extra=extra)


def _record_output_unlocked(kind: str, path: Path, *, label: str, extra: dict[str, Any] | None) -> None:
    manifest = _read_manifest()
    rel = paths.relative(path)
    env = environment()
    digest = sha256_file(path)
    inputs = chapter_inputs()
    source = _source_sha256(_STATE["chapter"])
    previous = manifest["files"].get(rel)
    _STATE["outputs"].append(rel)
    entry = {
        "kind": kind,
        "label": label,
        "chapter": _STATE["chapter"],
        "cell": current_cell(),
        "sha256": digest,
        "inputs": inputs,
        "created": now(),
        "git_commit": env["git_commit"],
        "git_dirty": env["git_dirty"],
        "source_sha256": source,
        "package_sha256": package_sha256(),
        **({"git_error": env["git_error"]} if env["git_error"] else {}),
        **(extra or {}),
    }
    if previous and previous.get("sha256") == digest and previous.get("inputs") == inputs:
        # Identical file from identical inputs.  Keep the old entry, so that a re-render that changes
        # nothing also leaves git diffs empty, unless the chapter's source or the package changed (the old
        # entry would then look stale or unverified to check_manifest) or a clean tree now vouches for an
        # entry made on a dirty one.
        if (previous.get("source_sha256") == source and previous.get("package_sha256") == entry["package_sha256"]
                and (previous.get("git_dirty") is False or env["git_dirty"] is not False)):
            return
        entry["created"] = previous.get("created", entry["created"])      # the file itself is unchanged
    manifest["files"][rel] = entry
    manifest["files"] = dict(sorted(manifest["files"].items()))
    _write_manifest(manifest)


# ── Freshness check ──────────────────────────────────────────────────────────

#: Problems that make a results file stale: ``mdmx check`` fails on them.
STALE_KINDS = frozenset({"file-missing", "file-changed", "source-changed", "source-missing", "inputs-changed"})
#: Problems that leave a results file unverified: ``mdmx check`` warns about them.
UNVERIFIED_KINDS = frozenset({"no-manifest", "no-source-hash", "inputs-missing", "package-changed"})


@dataclass(frozen=True)
class ManifestProblem:
    """One problem found by :func:`check_manifest`.

    Attributes
    ----------
    path : the results file (relative to the project root), or the manifest itself
    kind : one of :data:`STALE_KINDS` or :data:`UNVERIFIED_KINDS`
    message : what is wrong, in words
    """

    path: str
    kind: str
    message: str

    @property
    def stale(self) -> bool:
        """True if the file is stale; False if it only cannot be verified."""
        return self.kind in STALE_KINDS


def check_manifest() -> list[ManifestProblem]:
    """Check that every file in the project's results manifest (``results/manifest.json`` for
    ``book``) is current.

    For each entry:

    * the file must exist and have the recorded SHA-256 (``file-missing``, ``file-changed``: stale);
    * the chapter's source must have the recorded ``source_sha256`` (``source-changed``, or
      ``source-missing`` if the chapter file is gone: stale); an entry without one (made before
      source hashes were recorded, or outside a book chapter) cannot be verified
      (``no-source-hash``);
    * every input artefact in the project's store (``work/artifacts/<name>/meta.json`` for ``book``)
      must have the recorded content hash (``inputs-changed``: stale); an input whose meta is
      missing, e.g. in a fresh clone without ``work/``, cannot be verified (``inputs-missing``);
    * if the entry records the ``mdmx`` package's hash, a different one now leaves the file
      unverified (``package-changed``): library code that the chapter calls directly, not through
      :func:`mdmx.artifacts.stage`, may have changed what it shows.

    Limit: a ``work/`` rendered on another machine gives ``inputs-changed`` everywhere, because the
    artefacts record their run times.

    Returns
    -------
    The problems found (empty if every file is current), in manifest order.  Nothing is written.
    """
    path = _manifest_path()
    if not path.exists():
        return [ManifestProblem(paths.relative(path), "no-manifest", "there is no results manifest")]
    problems = []
    hashes: dict[str, str | None] = {}

    def artefact_hash(name: str) -> str | None:
        if name not in hashes:
            meta = paths.artifacts() / name / "meta.json"
            hashes[name] = (json.loads(meta.read_text(encoding="utf-8"))["content_hash"]
                            if meta.is_file() else None)
        return hashes[name]

    for rel, entry in _read_manifest()["files"].items():
        file = paths.root() / rel
        if not file.is_file():
            problems.append(ManifestProblem(rel, "file-missing", "the file does not exist"))
        elif sha256_file(file) != entry.get("sha256"):
            problems.append(ManifestProblem(rel, "file-changed", "the file differs from the one recorded "
                                            "(edited by hand, or written without a manifest entry)"))
        chapter = entry.get("chapter")
        if entry.get("source_sha256") is None:
            why = ("made before source hashes were recorded" if "source_sha256" not in entry
                   else "not made by a book chapter")
            problems.append(ManifestProblem(rel, "no-source-hash", f"the entry has no chapter source hash ({why})"))
        else:
            source = chapter_source(chapter) if chapter else None
            if source is None:
                problems.append(ManifestProblem(rel, "source-missing",
                                                f"the source of chapter {chapter!r} no longer exists"))
            elif text_sha256(source) != entry["source_sha256"]:
                problems.append(ManifestProblem(rel, "source-changed", f"{paths.relative(source)} changed "
                                                "after the file was made: re-render the chapter"))
        if entry.get("package_sha256") not in (None, package_sha256()):
            problems.append(ManifestProblem(rel, "package-changed", "the mdmx package changed after the "
                                            "file was made: re-render the chapter to be sure"))
        recorded = entry.get("inputs") or {}
        missing = sorted(n for n in recorded if artefact_hash(n) is None)
        changed = sorted(n for n in recorded if n not in missing and artefact_hash(n) != recorded[n])
        if changed:
            shown = ", ".join(changed[:5]) + (f" and {len(changed) - 5} more" if len(changed) > 5 else "")
            problems.append(ManifestProblem(rel, "inputs-changed", "input artefacts changed since the file was "
                                            f"made: {shown}"))
        if missing:
            problems.append(ManifestProblem(rel, "inputs-missing", f"{len(missing)} input artefacts are not in "
                                            f"{paths.relative(paths.artifacts())}"))
    return problems

