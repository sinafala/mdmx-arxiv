"""Artefact store: save, load and cache analysis objects — no pickles.

Why not pickle?
---------------
Pickles are tied to the exact Python and library versions that wrote them
(a pickle written under one NumPy may not load under another), they can
execute code when loaded, and a
human cannot open them.  Here every object is stored in open formats:

* NumPy arrays          → one compressed ``arrays.npz`` per artefact
* pandas DataFrames     → one Parquet file each
* everything else       → a JSON tree (``tree.json``) that references the
                          arrays/frames by name

so an artefact is a small folder ``work/artifacts/<name>/`` that any language
can read, plus a ``meta.json`` recording its provenance.

Caching with automatic invalidation
-----------------------------------
:func:`stage` is the workhorse used by every book chapter::

    tensor = artifacts.stage(
        "tensor", build_tensor,
        inputs=["lt_pooled", "event_labels"],   # artefacts it depends on
        params=P.section("tensor"),              # parameters it uses
        code=[mdmx.data],                        # code whose source matters
    )

``stage`` recomputes only if the *content* of an input artefact changed, a
parameter changed, the source code of the listed functions/modules changed,
or the Python environment (``uv.lock``) changed.  Otherwise it loads the
stored result.  This replaces hand-managed cache files and makes the
dependency graph explicit and auditable.

The code in the key is the listed code *and everything it calls*: every
function, class and module-level constant of the ``mdmx`` package (and of
the chapter itself) that the listed code refers to, followed transitively
(:func:`_code_closure`).  Listing the chapter's own function is enough; an
edit to any mdmx function it reaches recomputes the step.

Supported object types
----------------------
``None``, ``bool``, ``int``, ``float``, ``str``, NumPy scalars and arrays
(numeric, boolean or fixed-width string dtypes), pandas DataFrame/Series,
and arbitrarily nested ``dict`` / ``list`` / ``tuple`` of those.  Dicts
with non-string keys (e.g. ``(country, year)`` tuples) are supported too.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import inspect
import json
import os
import shutil
import sys
import textwrap
import types
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import paths, provenance

__all__ = ["save", "load", "exists", "meta", "stage", "register_file", "content_hash", "delete"]


# ══════════════════════════════════════════════════════════════════════════════
# Encoding nested Python objects into (JSON tree, arrays, frames)
# ══════════════════════════════════════════════════════════════════════════════

class _Encoder:
    """Walk an object, pulling arrays and DataFrames out into side stores."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.frames: dict[str, pd.DataFrame] = {}

    def encode(self, obj: Any, where: str = "root") -> Any:
        if obj is None or isinstance(obj, (bool, str)):
            return obj
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if isinstance(obj, float) and not np.isfinite(obj):
                return {"__float__": repr(obj)}          # nan / inf survive JSON
            return obj
        if isinstance(obj, np.generic):
            return self.encode(obj.item(), where)
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                raise TypeError(f"{where}: object arrays cannot be stored safely; "
                                "convert to a numeric or string dtype first.")
            key = f"a{len(self.arrays)}"
            self.arrays[key] = obj
            return {"__ndarray__": key}
        if isinstance(obj, pd.Series):
            obj = obj.to_frame(name=obj.name if obj.name is not None else "value")
            key = f"f{len(self.frames)}"
            self.frames[key] = obj
            return {"__series__": key}
        if isinstance(obj, pd.DataFrame):
            key = f"f{len(self.frames)}"
            self.frames[key] = obj
            return {"__frame__": key}
        if isinstance(obj, tuple):
            return {"__tuple__": [self.encode(v, f"{where}[{i}]") for i, v in enumerate(obj)]}
        if isinstance(obj, (set, frozenset)):
            items = sorted(obj, key=repr)
            return {"__set__": [self.encode(v, f"{where}{{}}") for v in items]}
        if isinstance(obj, list):
            return [self.encode(v, f"{where}[{i}]") for i, v in enumerate(obj)]
        if isinstance(obj, dict):
            if all(isinstance(k, str) for k in obj):
                return {"__dict__": {k: self.encode(v, f"{where}.{k}") for k, v in obj.items()}}
            return {"__items__": [[self.encode(k, f"{where}.key"), self.encode(v, f"{where}.val")]
                                  for k, v in obj.items()]}
        if isinstance(obj, Path):
            return {"__path__": paths.relative(obj)}
        raise TypeError(f"{where}: cannot store object of type {type(obj).__name__}")


def _decode(node: Any, arrays: dict[str, np.ndarray], frames: dict[str, pd.DataFrame]) -> Any:
    if isinstance(node, list):
        return [_decode(v, arrays, frames) for v in node]
    if not isinstance(node, dict):
        return node
    if "__ndarray__" in node:
        return arrays[node["__ndarray__"]]
    if "__frame__" in node:
        return frames[node["__frame__"]]
    if "__series__" in node:
        frame = frames[node["__series__"]]
        return frame.iloc[:, 0]
    if "__tuple__" in node:
        return tuple(_decode(v, arrays, frames) for v in node["__tuple__"])
    if "__set__" in node:
        return {_decode(v, arrays, frames) for v in node["__set__"]}
    if "__float__" in node:
        return float(node["__float__"])
    if "__path__" in node:
        return paths.root() / node["__path__"]
    if "__dict__" in node:
        return {k: _decode(v, arrays, frames) for k, v in node["__dict__"].items()}
    if "__items__" in node:
        return {_decode(k, arrays, frames): _decode(v, arrays, frames) for k, v in node["__items__"]}
    raise ValueError(f"Unrecognised node in artefact tree: {list(node)[:3]}")


# ══════════════════════════════════════════════════════════════════════════════
# Hashing
# ══════════════════════════════════════════════════════════════════════════════

def _hash_parts(tree: Any, arrays: dict[str, np.ndarray], frames: dict[str, pd.DataFrame]) -> str:
    """Content hash of an encoded object (independent of file formats)."""
    digest = hashlib.sha256()
    digest.update(json.dumps(tree, sort_keys=True).encode())
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        digest.update(f"{key}:{arr.dtype.str}:{arr.shape}".encode())
        digest.update(arr.tobytes())
    for key in sorted(frames):
        frame = frames[key]
        digest.update(f"{key}:{list(map(str, frame.columns))}:{list(map(str, frame.dtypes))}".encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def content_hash(obj: Any) -> str:
    """SHA-256 of an object's content, as stored by :func:`save`."""
    enc = _Encoder()
    tree = enc.encode(obj)
    return _hash_parts(tree, enc.arrays, enc.frames)


def _hash_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ── Which code a key covers: the listed code and, transitively, the project code it refers to ──

def _is_mdmx_module(name: object) -> bool:
    return isinstance(name, str) and (name == "mdmx" or name.startswith("mdmx."))


def _is_project_module(name: object) -> bool:
    """Code whose source belongs in cache keys: the mdmx package, and the chapter or script that runs
    (``__main__``: a Quarto/Jupyter cell's functions, a ``tools/`` script's)."""
    return name == "__main__" or _is_mdmx_module(name)


def _unwrap(obj: Any) -> Any:
    """The function behind a ``functools.partial``, a bound, static or class method, or a decorator.

    A partial's arguments are data (they belong in ``params``); its function is the code.
    """
    seen = set()
    while id(obj) not in seen:
        seen.add(id(obj))
        if isinstance(obj, functools.partial):
            obj = obj.func
        elif isinstance(obj, (staticmethod, classmethod, types.MethodType)):
            obj = obj.__func__
        elif not isinstance(obj, (type, types.ModuleType)) and "__wrapped__" in getattr(obj, "__dict__", ()):
            obj = obj.__dict__["__wrapped__"]           # functools.wraps, lru_cache
        else:
            break
    return obj


def _is_code(obj: Any) -> bool:
    return isinstance(obj, (types.ModuleType, type)) or inspect.isfunction(obj)


def _qualified_name(obj: Any) -> str:
    if isinstance(obj, types.ModuleType):
        return obj.__name__
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None) or repr(obj)
    module = getattr(obj, "__module__", None)
    return f"{module}.{name}" if module else name


def _constant_text(value: Any, *, containers: bool = False) -> str | None:
    """Canonical text of a constant (``None`` for anything else).

    Immutable values always qualify.  Lists and dicts (and read-only mappings) of constants qualify
    only with ``containers``, which the caller sets for public module-level names (``REFITS``,
    ``LABELS``): a private one may be state that changes while the process runs
    (``provenance._STATE``), which would make keys differ from call to call.
    """
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return repr(value)
    if isinstance(value, (tuple, frozenset)) or (containers and isinstance(value, list)):
        parts = [_constant_text(v, containers=containers) for v in value]
        if any(p is None for p in parts):
            return None
        if isinstance(value, frozenset):
            return "frozenset({" + ", ".join(sorted(parts)) + "})"     # set order varies between processes
        if isinstance(value, list):
            return "[" + ", ".join(parts) + "]"
        return "(" + ", ".join(parts) + ",)"
    if containers and isinstance(value, (dict, types.MappingProxyType)):
        items = [(_constant_text(k, containers=True), _constant_text(v, containers=True)) for k, v in value.items()]
        if any(k is None or v is None for k, v in items):
            return None
        return "{" + ", ".join(f"{k}: {v}" for k, v in sorted(items)) + "}"   # insertion order is not content
    return None


def _ast_chains(tree: ast.AST) -> set[tuple[str, ...]]:
    """Every name and dotted name (``a``, ``a.b``, ``a.b.c``) that the code mentions."""
    chains = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            chains.add((node.id,))
        elif isinstance(node, ast.Attribute):
            attrs, value = [], node
            while isinstance(value, ast.Attribute):
                attrs.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                chains.add((value.id, *reversed(attrs)))
    return chains


def _code_names(code: types.CodeType) -> set[str]:
    """The global, attribute and free-variable names a code object (and its nested code) loads."""
    names = set(code.co_names) | set(code.co_freevars)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _code_names(const)
    return names


@dataclasses.dataclass
class _CodeInfo:
    """The source of one function, class or module, read once per process."""

    anchor: Any                          # the object keyed by ``id``; held so that the id stays unique
    source: str | None                   # None: no source (e.g. a lambda typed at the console)
    key: str | None                      # what the cache key hashes (:func:`_key_text`)
    file: str | None
    chains: frozenset[tuple[str, ...]]   # the names it mentions (:func:`_ast_chains`)
    defined: bool = False                # :func:`_check_definition` has passed


_INFO: dict[int, _CodeInfo] = {}


def _info(obj: Any) -> _CodeInfo:
    """The source of ``obj`` and the names it mentions, cached per code object, class or module.

    Functions are keyed by their code object: the many closures one ``def`` makes in a loop share
    its source.  What a name refers to is not cached (:func:`_refers_to`), because a notebook can
    rebind it.
    """
    anchor = obj.__code__ if inspect.isfunction(obj) else obj
    info = _INFO.get(id(anchor))
    if info is not None and info.anchor is anchor:
        return info
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError):
        source = None
    if isinstance(obj, types.ModuleType):
        file = getattr(obj, "__file__", None)
    else:
        try:
            file = inspect.getsourcefile(obj)
        except TypeError:                          # a built-in: there is no file to check
            file = None
    tree = None
    if source is not None:
        try:
            tree = ast.parse(textwrap.dedent(source))
        except SyntaxError:     # e.g. a lambda inside a longer expression: inspect returns part of it
            tree = None
    if tree is not None:
        chains = _ast_chains(tree)
        if inspect.isfunction(obj) and obj.__code__.co_name == "<lambda>":
            # A lambda's source is the whole statement around it; keep the names the lambda uses.
            used = _code_names(obj.__code__)
            chains = {c for c in chains if c[0] in used}
    elif inspect.isfunction(obj):
        names = _code_names(obj.__code__)
        chains = {(n,) for n in names} | {(a, b) for a in names for b in names}
    else:
        chains = set()
    info = _CodeInfo(anchor=anchor, source=source, key=_key_text(source), file=file, chains=frozenset(chains))
    _INFO[id(anchor)] = info
    return info


def _key_text(source: str | None) -> str | None:
    """What a cache key hashes for ``source``: its syntax tree without docstrings.

    Comments, docstrings, blank lines and layout do not change what code computes, so editing them
    (a ``Paper:`` tag, a reworded comment) must not invalidate a cached result.  ``ast.dump`` leaves
    out comments and positions; module, class and function docstrings are removed first.  Source
    that does not parse on its own (part of a longer expression) is hashed as it is.
    """
    if source is None:
        return None
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def _closure_cells(func: types.FunctionType) -> dict[str, Any]:
    """The values of a nested function's free variables (an enclosing function's locals)."""
    cells = {}
    for name, cell in zip(func.__code__.co_freevars, func.__closure__ or (), strict=False):
        try:
            cells[name] = cell.cell_contents
        except ValueError:      # an empty cell: the variable is not bound yet, so it names no code
            continue
    return cells


_MISSING = object()


def _refers_to(obj: Any, info: _CodeInfo, constants: dict[str, str]) -> list[Any]:
    """The project functions and classes that ``obj``'s source refers to directly.

    Names are resolved as Python would at run time: a nested function's free variables, then its
    globals (``func.__globals__``, which for a function defined in a notebook cell is the notebook's
    namespace), then attribute by attribute through modules (``tz.unfold``, ``mdmx.tensor.unfold``).
    A name bound to an instance of a project class brings in the class.  Constants found in an mdmx
    module's namespace (numbers, strings and tuples such as ``compare.A0``, and public dicts and
    lists) are added to ``constants``.  For a module, the
    functions and classes of the module itself are left out: its source covers them.
    """
    if isinstance(obj, types.ModuleType):
        namespace, cells, own = vars(obj), {}, obj.__name__
    elif inspect.isfunction(obj):
        namespace, cells, own = obj.__globals__, _closure_cells(obj), obj.__module__
    elif isinstance(obj, type):
        module = sys.modules.get(obj.__module__)
        namespace, cells, own = (vars(module) if module else {}), {}, obj.__module__
    else:
        return []
    found = []
    for chain in info.chains:
        if chain[0] in cells:
            value, owner = cells[chain[0]], None
        elif chain[0] in namespace:
            value, owner = namespace[chain[0]], own
        else:
            continue                                # a local variable, a builtin or an attribute of one
        value, i = _unwrap(value), 1
        while isinstance(value, types.ModuleType) and i < len(chain):
            # vars(), not getattr(): a module's __getattr__ may import things (lazy loading)
            value, owner, i = _unwrap(vars(value).get(chain[i], _MISSING)), value.__name__, i + 1
        if value is _MISSING or isinstance(value, types.ModuleType):
            continue
        name = chain[i - 1]
        if inspect.isfunction(value) or isinstance(value, type):
            target = value
        elif (text := _constant_text(value, containers=not name.startswith("_"))) is not None:
            # Dunders (__file__, __name__) describe where the code lives, not what it computes: an
            # absolute path in the key would make it differ from one clone to another.
            if owner is not None and _is_mdmx_module(owner) and not name.startswith("__"):
                constants[f"{owner}.{name}"] = text
            continue
        else:
            target = type(value)
        if not _is_project_module(getattr(target, "__module__", None)):
            continue
        if isinstance(obj, types.ModuleType) and target.__module__ == obj.__name__:
            continue
        found.append(target)
    return found


def _mdmx_callees(obj: Any) -> list[Any]:
    """The project functions and classes that ``obj`` refers to directly (one step of
    :func:`_code_closure`)."""
    obj = _unwrap(obj)
    return _refers_to(obj, _info(obj), {}) if _is_code(obj) else []


def _code_closure(code: Iterable[Any]) -> tuple[list[Any], dict[str, str]]:
    """The code a cache key covers: ``code`` and, transitively, the project code it refers to.

    Parameters
    ----------
    code : functions, classes, modules (or partials and methods of them) — the ``code`` argument
        of :func:`stage`.

    Returns
    -------
    (items, constants): the functions, classes and modules, sorted by qualified name, so that the
    key does not depend on the order they were listed or found in (a function or class whose module
    is itself an item is left out: the module's source covers it); and the mdmx
    module-level constants they use, ``"module.NAME" → canonical text``, sorted by name.
    """
    pending = [_unwrap(item) for item in code]
    seen: dict[int, Any] = {}
    constants: dict[str, str] = {}
    while pending:
        obj = pending.pop()
        if id(obj) in seen:
            continue
        seen[id(obj)] = obj
        if _is_code(obj):
            pending.extend(f for f in _refers_to(obj, _info(obj), constants) if id(f) not in seen)
    modules = {o.__name__ for o in seen.values() if isinstance(o, types.ModuleType)}
    items = [o for o in seen.values()
             if isinstance(o, types.ModuleType) or getattr(o, "__module__", None) not in modules]
    items.sort(key=lambda o: (_qualified_name(o), (_info(o).source or "") if _is_code(o) else ""))
    return items, dict(sorted(constants.items()))


def _code_hash(code: Iterable[Any]) -> tuple[str, list[str]]:
    """Hash the source of ``code`` and of everything it calls (:func:`_code_closure`).

    Returns
    -------
    (SHA-256 hex digest, the qualified names of the items hashed, then ``"module.NAME = value"`` for
    each constant).

    Raises
    ------
    RuntimeError if the source of an item may not be the code this process runs
    (:func:`_check_file`, :func:`_check_definition`).
    """
    items, constants = _code_closure(code)
    digest = hashlib.sha256()
    names = []
    checked: set[str] = set()
    for item in items:
        name = _qualified_name(item)
        names.append(name)
        info = _info(item) if _is_code(item) else None
        if info is None or info.source is None:
            source = name                      # e.g. a lambda typed at the console
        else:
            source = info.key
            if info.file and info.file not in checked:
                _check_file(item, info.file)
                checked.add(info.file)
            if not info.defined:
                _check_definition(item, info.source, name)
                info.defined = True
        digest.update(f"{name}\0{source}\0".encode())
    for name, text in constants.items():
        names.append(f"{name} = {text}")
        digest.update(f"{name}\0{text}\0".encode())
    return digest.hexdigest(), names


# Source file → (module object id, mtime in ns, size) when this process first hashed code from it.
_SOURCE_STATE: dict[str, tuple[int, int, int]] = {}

_EDITED = ("a file was edited after this process imported it.  Restart the process, so that cache "
           "keys describe the code that runs.")


def _check_file(item: Any, file: str) -> None:
    """Raise if ``file`` changed on disk since this process first hashed code from it.

    ``inspect`` reads source from the file *as it is now*, at the line where a function began when it
    was imported.  If the file is edited while a long run is going, later cache keys would describe
    code that did not run (or other lines of the file altogether).  The file's size and modification
    time must be those it had when this process first hashed code from it (a module object other
    than the one first seen, i.e. a fresh import, starts afresh).  Run for every hashed item, on
    every call.
    """
    path = Path(file)
    if not path.is_file():                 # e.g. a notebook cell: its source lives in memory only
        return
    module = item if isinstance(item, types.ModuleType) else sys.modules.get(getattr(item, "__module__", ""))
    st = path.stat()
    state = (id(module), st.st_mtime_ns, st.st_size)
    seen = _SOURCE_STATE.setdefault(file, state)
    if seen[0] != state[0]:
        _SOURCE_STATE[file] = state
    elif seen != state:
        raise RuntimeError(f"{file} changed on disk after this process first read it: {_EDITED}")


def _check_definition(item: Any, source: str, name: str) -> None:
    """Raise if a function's or class's source does not parse as its own definition.

    Catches a file edited between the import and the first read of the source, when the lines
    ``inspect`` reads now belong to other code.  Run once per source read (the source is cached).
    """
    defined = getattr(getattr(item, "__code__", None), "co_name", None) or (
        item.__name__ if isinstance(item, type) else None)
    if not defined or defined == "<lambda>":
        return
    try:
        first = ast.parse(textwrap.dedent(source)).body[0]
    except (SyntaxError, IndexError):
        first = None
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    if not (isinstance(first, kinds) and first.name == defined):
        raise RuntimeError(f"the source of {name} on disk is no longer its definition: {_EDITED}")


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def _folder(name: str) -> Path:
    if not name or any(c in name for c in r'\/:*?"<>| '):
        raise ValueError(f"Invalid artefact name {name!r}: use letters, digits, '-', '_' and '.'.")
    return paths.artifacts() / name


def exists(name: str) -> bool:
    """True if artefact ``name`` has been saved."""
    return (_folder(name) / "meta.json").is_file()


def meta(name: str) -> dict[str, Any]:
    """Return the provenance record of artefact ``name``."""
    return json.loads((_folder(name) / "meta.json").read_text(encoding="utf-8"))


def save(name: str, obj: Any, *, inputs: Iterable[str] = (), params: Any = None,
         code: Iterable[Any] = (), desc: str = "", quiet: bool = False) -> dict[str, Any]:
    """Store ``obj`` as artefact ``name`` and return its provenance record.

    Parameters
    ----------
    name : artefact name (also the folder name under ``work/artifacts``)
    obj : the object to store (see module docstring for supported types)
    inputs : names of the artefacts ``obj`` was computed from
    params : the parameter values used (any JSON-serialisable structure)
    code : functions/modules whose source code determines the result (with everything they call,
        :func:`_code_closure`)
    desc : one-line human description, shown in the provenance appendix
    """
    return _write(name, obj, inputs=list(inputs), params=params, code_key=_code_hash(code), desc=desc,
                  quiet=quiet)


def _write(name: str, obj: Any, *, inputs: list[str], params: Any, code_key: tuple[str, list[str]],
           desc: str, quiet: bool) -> dict[str, Any]:
    """:func:`save` with the code's key already computed (:func:`stage` computes it before the step runs,
    so that the key describes the code that ran even if a file is edited during a long run)."""
    enc = _Encoder()
    tree = enc.encode(obj)
    folder = _folder(name)
    tmp = folder.with_name(folder.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    if enc.arrays:
        np.savez_compressed(tmp / "arrays.npz", **enc.arrays)
    for key, frame in enc.frames.items():
        frame.to_parquet(tmp / f"{key}.parquet", compression="zstd")
    (tmp / "tree.json").write_text(json.dumps(tree, indent=1), encoding="utf-8", newline="\n")

    input_hashes = {i: meta(i)["content_hash"] for i in inputs}
    code_hash, code_names = code_key
    record = {
        "name": name,
        "desc": desc,
        "chapter": provenance.current_chapter(),
        "created": provenance.now(),
        "content_hash": _hash_parts(tree, enc.arrays, enc.frames),
        "inputs": input_hashes,
        "params": params,
        "params_hash": _hash_json(params),
        "code": code_names,
        "code_hash": code_hash,
        "uv_lock_sha256": provenance.lock_sha256(),
        "environment": provenance.environment(),
        "size_bytes": sum(p.stat().st_size for p in tmp.iterdir()),
    }
    (tmp / "meta.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8", newline="\n")

    # Atomic-ish replace: never leave a half-written artefact behind.
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(tmp, folder)
    provenance.note_input(name, record["content_hash"])
    if not quiet:
        print(f"  [artefact] saved   {name:<28s} {record['size_bytes'] / 1e6:8.2f} MB")
    return record


def load(name: str, *, quiet: bool = False) -> Any:
    """Load artefact ``name`` (and note it as an input of the current chapter)."""
    folder = _folder(name)
    if not (folder / "meta.json").is_file():
        raise FileNotFoundError(
            f"Artefact '{name}' not found in {paths.relative(paths.artifacts())}. "
            "Render the chapter that produces it first (see the provenance appendix)."
        )
    record = meta(name)
    if record.get("kind") == "file":
        provenance.note_input(name, record["content_hash"])
        return paths.root() / record["path"]
    tree = json.loads((folder / "tree.json").read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    if (folder / "arrays.npz").exists():
        with np.load(folder / "arrays.npz", allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
    frames = {p.stem: pd.read_parquet(p) for p in folder.glob("*.parquet")}
    provenance.note_input(name, record["content_hash"])
    if not quiet:
        print(f"  [artefact] loaded  {name}")
    return _decode(tree, arrays, frames)


def delete(name: str) -> None:
    """Remove artefact ``name`` (used by ``mdmx clean``)."""
    folder = _folder(name)
    if folder.exists():
        shutil.rmtree(folder)


def register_file(name: str, path: Path, *, desc: str = "", expected_sha256: str | None = None) -> dict[str, Any]:
    """Record an external input file (e.g. the HMD zip) as an artefact.

    The file itself is not copied; its SHA-256 becomes the artefact's content
    hash, so everything computed from it is invalidated if the file changes.
    If ``expected_sha256`` is given and does not match, an error is raised.
    """
    path = Path(path)
    digest = provenance.sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(
            f"{paths.relative(path)} has SHA-256 {digest}, but {expected_sha256} was expected. "
            "This is not the data release the analysis was built on."
        )
    folder = _folder(name)
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "name": name, "kind": "file", "desc": desc, "path": paths.relative(path),
        "content_hash": digest, "size_bytes": path.stat().st_size,
        "chapter": provenance.current_chapter(), "created": provenance.now(),
    }
    old = folder / "meta.json"
    if not (old.exists() and json.loads(old.read_text(encoding="utf-8"))["content_hash"] == digest):
        old.write_text(json.dumps(record, indent=1), encoding="utf-8", newline="\n")
    provenance.note_input(name, digest)
    return record


def stage(name: str, compute: Callable[[], Any], *, inputs: Iterable[str] = (),
          params: Any = None, code: Iterable[Any] | None = None, desc: str = "",
          force: bool = False) -> Any:
    """Return artefact ``name``, recomputing it only when something changed.

    The stored result is reused if, and only if,

    * every input artefact has the same content hash as when it was stored,
    * ``params`` are identical,
    * the source code of ``code`` (default: ``compute`` itself) and of every mdmx function,
      class and constant it reaches (:func:`_code_closure`) is unchanged, and
    * the Python environment is unchanged (the SHA-256 of ``uv.lock``).

    Set ``force=True`` or the environment variable ``MDMX_FORCE=1`` to
    recompute regardless.
    """
    inputs = list(inputs)
    code = [compute] if code is None else list(code)
    force = force or os.environ.get("MDMX_FORCE") == "1"
    code_key = _code_hash(code)
    if exists(name) and not force:
        old = meta(name)
        current_inputs = {i: meta(i)["content_hash"] for i in inputs}
        reasons = []
        if old.get("inputs") != current_inputs:
            reasons.append("inputs changed")
        if old.get("params_hash") != _hash_json(params):
            reasons.append("parameters changed")
        if old.get("code_hash") != code_key[0]:
            reasons.append("code changed")
        if old.get("uv_lock_sha256") != provenance.lock_sha256():
            reasons.append("environment changed")
        if not reasons:
            for i, h in current_inputs.items():
                provenance.note_input(i, h)
            return load(name)
        print(f"  [artefact] rebuild {name} ({', '.join(reasons)})")
    for i in inputs:                      # record the inputs as used
        provenance.note_input(i, meta(i)["content_hash"])
    result = compute()
    _write(name, result, inputs=inputs, params=params, code_key=code_key, desc=desc, quiet=False)
    return result
