"""Analysis parameters — read once from ``config/params.toml``.

Every tunable value in the pipeline (thresholds, ranks, seeds, window
lengths, …) lives in **one** human-readable file, ``config/params.toml``.
Chapters never hard-code a number that a reader might want to change; they
read it from here.  Because the artefact store hashes the parameters each
step uses (see :mod:`mdmx.artifacts`), changing a value automatically
invalidates exactly the cached results that depend on it.

Usage
-----
>>> from mdmx.config import params
>>> P = params()
>>> P.data.q_min
1e-08
>>> P.section("tucker")          # a plain dict, e.g. for hashing/logging
{'tau': 0.9999, ...}
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

from . import paths

__all__ = ["params", "Params"]


class Params(SimpleNamespace):
    """A read-only-by-convention tree of parameters with attribute access.

    Nested TOML tables become nested :class:`Params`.  ``section(name)``
    returns a plain ``dict`` copy, which is what gets hashed and logged.
    """

    def section(self, name: str) -> dict[str, Any]:
        value = getattr(self, name)
        return _to_dict(value)

    def as_dict(self) -> dict[str, Any]:
        return _to_dict(self)


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Params(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj


@lru_cache(maxsize=1)
def params() -> Params:
    """Load ``config/params.toml`` (cached for the life of the process)."""
    path = paths.config_dir() / "params.toml"
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    # Optional extra files config/params.d/*.toml, each adding its own
    # top-level sections (a section may not be defined twice).
    for extra in sorted((paths.config_dir() / "params.d").glob("*.toml")):
        with open(extra, "rb") as fh:
            more = tomllib.load(fh)
        clash = set(more) & set(raw)
        if clash:
            raise ValueError(f"{extra.name} redefines parameter sections {sorted(clash)}")
        raw.update(more)
    return _to_namespace(raw)
