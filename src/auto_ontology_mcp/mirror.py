"""SQLite mirrors that make MongoDB databases and HTTP API collections speakable by Auto Ontology.

Auto Ontology reads schemas, profiles, validates and executes in SQL; MongoDB and REST APIs
have no SQL. For those two source families the server keeps a **mirror**: a local SQLite file,
rebuilt from the source on demand, holding one table per collection or endpoint. Everything
upstream of the connection then works unchanged — the scan, the profiler, the join inference,
`validate_sql`, `run_sql` — because they run on the mirror, which is an ordinary SQLite file.
The mirror is regenerated, not maintained: refresh re-fetches; nothing is ever written to the
source.

Flattening is deterministic and needs no schema knowledge:

- `_id` (or `id`) becomes the TEXT primary key column `id`; ObjectId is stored as its 24-char hex.
- Keys are sanitized to `[A-Za-z0-9_]`; keys that differ only in punctuation share a column
  (the profile shows the mix). `__v` is dropped.
- Scalars keep their kind (int/bool -> INTEGER, float -> REAL, str -> TEXT). Dates and
  datetimes become ISO 8601 text; Decimal128 and Int64 become REAL/INTEGER.
- Nested objects and arrays of scalars are stored as JSON text and stay queryable through
  SQLite's JSON functions (json_extract and friends).
- An array of documents becomes a child table `<parent>__<key>` with `parent_id`, `_idx` and one
  column per sub-document key, so the one-to-many join the array carries is visible to the
  ontology. Arrays of documents one level below that are kept as JSON text.
- Keys whose name matches a deny substring (password, token, secret, ...) are not written at all.

A column whose observed kinds differ is declared TEXT and the values keep their natural
storage class (SQLite is dynamically typed); profiling makes the mixing visible.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .errors import OntologyError

# Field-name substrings whose keys are never written to a mirror: credentials, tokens and raw
# payment-gateway payloads. Overridable and extendable per datasource with config["deny"].
DEFAULT_DENY: tuple[str, ...] = (
    "password", "token", "secret", "sessionid", "otp", "refresh",
    "initresponse", "cancelresponse", "checkouturl", "forgotpassword",
    "logincode", "cardbrand", "cardexpiry", "cardlastfour", "recoverpaymentlink",
)

# Not data fields: Mongo's document version key.
_SKIP_KEYS = frozenset({"__v"})

# BSON classes are matched by type name, never imported, so flattening also runs when pymongo
# is not installed (which is the case for API-only setups and for the unit tests).
_BSON_HEX = frozenset({"ObjectId", "Binary"})
_BSON_REAL = frozenset({"Decimal128"})
_BSON_INT = frozenset({"Int64", "Long"})

# Type promotion: INTEGER < REAL < TEXT. NULL never promotes anything.
_RANK = {"INTEGER": 0, "REAL": 1, "TEXT": 2}


def sanitize_key(key: Any) -> str:
    out = re.sub(r"[^A-Za-z0-9_]", "_", str(key))
    if not out:
        return "f_"
    if out[0].isdigit():
        out = "f_" + out
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, (bytes, bytearray)):
        return f"<{len(obj)} bytes>"
    return str(obj)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def coerce_value(value: Any) -> tuple[str | None, Any]:
    """(declared type family, stored value) for one BSON/JSON value. None family = NULL."""
    if value is None:
        return None, None
    if isinstance(value, bool):
        return "INTEGER", int(value)
    if isinstance(value, int):
        return "INTEGER", value
    if isinstance(value, float):
        return ("REAL", value) if math.isfinite(value) else ("TEXT", str(value))
    if isinstance(value, str):
        return "TEXT", value
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return "TEXT", value.isoformat()
    name = type(value).__name__
    if name in _BSON_HEX:
        return "TEXT", str(value)
    if name in _BSON_REAL:
        try:
            return "REAL", float(value)
        except Exception:  # noqa: BLE001 - not a number after all
            return "TEXT", str(value)
    if name in _BSON_INT:
        try:
            return "INTEGER", int(value)
        except Exception:  # noqa: BLE001
            return "TEXT", str(value)
    if isinstance(value, (bytes, bytearray)):
        return "TEXT", f"<{len(value)} bytes>"
    if isinstance(value, (list, tuple, dict)):
        return "TEXT", _json_text(value)
    return "TEXT", str(value)


def _promote(current: str | None, observed: str | None) -> str | None:
    if observed is None:
        return current
    if current is None:
        return observed
    return current if _RANK[current] >= _RANK[observed] else observed


class TableBuilder:
    """Columns and rows of one mirror table, filled as documents stream in."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.columns: dict[str, str | None] = {}  # first-seen order; value = type family
        self.rows: list[dict[str, Any]] = []

    def add_row(self, cells: dict[str, tuple[str | None, Any]]) -> None:
        for key, (family, stored) in cells.items():
            self.columns[key] = _promote(self.columns.get(key), family)
        self.rows.append({k: v[1] for k, v in cells.items()})

    @property
    def spec(self) -> tuple[list[tuple[str, str]], list[str]]:
        cols = [(k, v or "TEXT") for k, v in self.columns.items()]
        names = {k for k, _ in cols}
        if "id" in names and self.name.count("__") == 0:
            pk = ["id"]
        elif {"parent_id", "_idx"} <= names:
            pk = ["parent_id", "_idx"]
        else:
            pk = []
        return cols, pk


def _denied(sane_key: str, deny: tuple[str, ...]) -> bool:
    folded = sane_key.casefold()
    return any(d.casefold() in folded for d in deny)


def flatten_document(
    table: str,
    doc: dict[str, Any],
    builders: dict[str, TableBuilder],
    *,
    deny: tuple[str, ...] = DEFAULT_DENY,
    parent: tuple[str, int] | None = None,
    depth: int = 0,
) -> None:
    """Flatten one document (or sub-document, with parent=) into the builder for *table*."""
    if not isinstance(doc, dict):
        doc = {"value": doc}
    raw_id = doc.get("_id")
    if raw_id is None:
        raw_id = doc.get("id")
    row_key = None if raw_id is None else str(coerce_value(raw_id)[1])
    cells: dict[str, tuple[str | None, Any]] = {}
    if parent is not None:
        pid, pidx = parent
        cells["parent_id"] = ("TEXT", str(pid))
        cells["_idx"] = ("INTEGER", int(pidx))
    if table not in builders:
        builders[table] = TableBuilder(table)
    for raw_key, value in doc.items():
        if raw_key in _SKIP_KEYS:
            continue
        # _id and id name the same anchor column; a document with both merges them.
        key = "id" if raw_key in ("_id", "id") else sanitize_key(raw_key)
        if _denied(key, deny):
            continue
        if (
            depth == 0
            and isinstance(value, list)
            and value
            and all(isinstance(v, dict) for v in value)
        ):
            child = f"{table}__{key}"
            if child not in builders:
                builders[child] = TableBuilder(child)
            anchor = row_key if row_key is not None else str(len(builders[table].rows))
            for idx, sub in enumerate(value):
                flatten_document(child, sub, builders, deny=deny, parent=(anchor, idx), depth=1)
            continue
        cells[key] = coerce_value(value)
    builders[table].add_row(cells)


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def write_mirror_sqlite(path: Path, builders: dict[str, TableBuilder]) -> dict[str, int]:
    """Write all tables to a fresh SQLite file and swap it in atomically. Returns {table: rows}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    counts: dict[str, int] = {}
    try:
        for name, builder in builders.items():
            cols, pk = builder.spec
            if not cols:
                continue  # a collection whose every field was denied: nothing to mirror
            col_defs = ", ".join(f"{_quote(c)} {t}" for c, t in cols)
            if pk:
                col_defs += f", PRIMARY KEY ({', '.join(_quote(c) for c in pk)})"
            con.execute(f"DROP TABLE IF EXISTS {_quote(name)}")
            con.execute(f"CREATE TABLE {_quote(name)} ({col_defs})")
            order = [c for c, _ in cols]
            values = [[row.get(c) for c in order] for row in builder.rows]
            if values:
                con.executemany(
                    f"INSERT INTO {_quote(name)} ({', '.join(_quote(c) for c in order)}) "
                    f"VALUES ({', '.join('?' * len(order))})",
                    values,
                )
            counts[name] = len(values)
        con.commit()
    finally:
        con.close()
    os.replace(tmp, path)
    return counts


def _dig(obj: Any, dotted: str | None) -> Any:
    cur = obj
    for part in (dotted or "").split("."):
        if not part:
            continue
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _resolve_env(value: str) -> str:
    if isinstance(value, str) and value.startswith("env:"):
        name = value[4:]
        got = os.environ.get(name)
        if got is None:
            raise OntologyError(f"The datasource references the environment variable {name}, but it is not set.")
        return got
    return value


# ------------------------------------------------------------------ MongoDB
def mongo_connect(url: str, timeout_s: int) -> Any:
    """pymongo client for a mongodb:// or mongodb+srv:// URL (installed with the `mongo` extra)."""
    try:
        import pymongo
    except ModuleNotFoundError as exc:
        raise OntologyError(
            "MongoDB datasources need pymongo: install the extra, e.g. "
            "`pip install 'auto-ontology-mcp[mongo]'` (or uv add), then retry. "
            "Nothing else about the server changes."
        ) from exc
    return pymongo.MongoClient(url, serverSelectionTimeoutMS=timeout_s * 1000, tz_aware=True)


def mongo_ping(client: Any) -> None:
    client.admin.command("ping")


def mongo_source_names(client: Any, database: str) -> list[str]:
    names = client[database].list_collection_names()
    return sorted(n for n in names if not n.startswith("system."))


def mongo_sources(client: Any, database: str, collections: list[str]) -> Iterator[tuple[str, Iterable[dict[str, Any]]]]:
    db = client[database]
    for name in collections:
        yield name, db[name].find({})


# ------------------------------------------------------------------ API collections
def api_sources(config: dict[str, Any], timeout_s: int) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Fetch every endpoint of an API collection; yields (name, docs).

    Endpoint config: {name, url (or base_url+path), headers?, params?, items_path?,
    pagination: {"type": "none"|"page"|"cursor", ...}, max_pages?}. Header values may be
    "env:VAR_NAME" so that secrets stay out of the stored config.
    """
    try:
        import httpx
    except ModuleNotFoundError as exc:
        raise OntologyError(
            "API collection datasources need httpx: install the `api` extra "
            "(`pip install 'auto-ontology-mcp[api]'`) and retry."
        ) from exc
    base = _resolve_env(str(config.get("base_url") or ""))
    default_headers = {k: _resolve_env(str(v)) for k, v in (config.get("headers") or {}).items()}
    with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0))) as client:
        for ep in config.get("endpoints") or []:
            yield ep["name"], _fetch_endpoint(client, ep, base, default_headers, timeout_s)


def _fetch_endpoint(client: Any, ep: dict[str, Any], base: str, default_headers: dict[str, str], timeout_s: int) -> list[dict[str, Any]]:
    url = _resolve_env(str(ep.get("url") or "")) or (base + str(ep.get("path") or ""))
    headers = dict(default_headers)
    headers.update({k: _resolve_env(str(v)) for k, v in (ep.get("headers") or {}).items()})
    params = dict(ep.get("params") or {})
    pagination = ep.get("pagination") or {"type": "none"}
    ptype = pagination.get("type", "none")
    max_pages = min(int(pagination.get("max_pages", 200)), 2000)
    out: list[dict[str, Any]] = []
    page = int(pagination.get("start", 1))
    cursor: Any = pagination.get("start_cursor")
    cursor_param = pagination.get("param", "cursor")
    page_param = pagination.get("param", "page")
    size_param = pagination.get("size_param", "limit")
    size = int(pagination.get("size", 0)) or None
    first_page = True
    for _page_no in range(max_pages):
        call_params = dict(params)
        if ptype == "page":
            call_params[page_param] = page
            if size:
                call_params[size_param] = size
        elif ptype == "cursor":
            if not first_page and not cursor:
                break
            if cursor:
                call_params[cursor_param] = cursor
        resp = client.get(url, headers=headers, params=call_params)
        if resp.status_code >= 400:
            raise OntologyError(f"Endpoint {ep.get('name')!r} returned HTTP {resp.status_code} for {url}.")
        try:
            payload = resp.json()
        except Exception as exc:
            raise OntologyError(f"Endpoint {ep.get('name')!r} did not return JSON ({type(exc).__name__}).") from exc
        items = _dig(payload, ep.get("items_path"))
        if items is None:
            items = payload if isinstance(payload, list) else ([payload] if isinstance(payload, dict) else None)
        if isinstance(items, dict):
            items = [items]  # single-object endpoint: the object is the one record
        if not isinstance(items, list):
            raise OntologyError(
                f"Endpoint {ep.get('name')!r}: items_path {ep.get('items_path')!r} did not select records. "
                "Point it at the array (or object) of documents in the response."
            )
        out.extend(item if isinstance(item, dict) else {"value": item} for item in items)
        if ptype == "none":
            break
        has_more = _dig(payload, pagination.get("has_more_path"))
        if has_more is False:
            break
        if not items or (size and len(items) < size):
            break
        if ptype == "page":
            page += int(pagination.get("step", 1))
        else:
            cursor = _dig(payload, pagination.get("next_path", "next_cursor"))
            if not cursor:
                break
        first_page = False
    return out


def source_type_of(url: str | None, config: dict[str, Any] | None) -> str:
    """database (SQLAlchemy), mongo (mongodb:// URL) or api (endpoint config)."""
    if config is not None:
        return "api"
    low = (url or "").strip().lower()
    if low.startswith(("mongodb://", "mongodb+srv://")):
        return "mongo"
    if low.startswith("api://"):
        return "api"
    return "database"


def validate_api_config(config: Any, datasource: str) -> dict[str, Any]:
    """Check an API collection config and return the storable form."""
    if not isinstance(config, dict):
        raise OntologyError(
            "An API collection datasource needs a config dict, e.g. "
            "{'endpoints': [{'name': 'deals', 'url': 'https://api.example.com/v1/deals', "
            "'items_path': 'results', 'pagination': {'type': 'cursor', 'next_path': 'paging.next'}}]}."
        )
    endpoints = config.get("endpoints")
    if not isinstance(endpoints, list) or not endpoints:
        raise OntologyError("The API collection config needs a non-empty 'endpoints' list.")
    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, ep in enumerate(endpoints):
        if not isinstance(ep, dict):
            raise OntologyError(f"Endpoint {i + 1} of {datasource!r} is not an object.")
        name = sanitize_key(ep.get("name") or f"endpoint_{i + 1}")
        if name in seen:
            raise OntologyError(f"Endpoint name {name!r} appears twice in {datasource!r}; names become table names.")
        seen.add(name)
        if not (ep.get("url") or ep.get("path")):
            raise OntologyError(f"Endpoint {name!r} needs a url (or a path against base_url).")
        clean.append({k: v for k, v in ep.items() if k in {"name", "url", "path", "headers", "params", "items_path", "pagination"}})
        clean[-1]["name"] = name
    deny = config.get("deny")
    if deny is not None and (not isinstance(deny, list) or not all(isinstance(d, str) for d in deny)):
        raise OntologyError("config['deny'] must be a list of field-name substrings.")
    return {
        "base_url": str(config.get("base_url") or ""),
        "headers": config.get("headers") or {},
        "endpoints": clean,
        **({"deny": list(deny)} if deny else {}),
    }


def mongo_config(url: str) -> dict[str, Any]:
    """Split a mongodb URL into its parts and check what a mirror needs from it."""
    from urllib.parse import urlsplit
    parsed = urlsplit(url)
    if not parsed.scheme.lower().startswith("mongodb"):
        raise OntologyError("MongoDB datasources take a mongodb:// or mongodb+srv:// connection URL.")
    if not parsed.hostname:
        raise OntologyError(f"The MongoDB URL has no host: {mask_host(url)}.")
    database = parsed.path.lstrip("/")
    if not database:
        raise OntologyError("The MongoDB URL must name the database, e.g. mongodb://host:27017/mydb.")
    return {"url": url, "database": database}


def mask_host(url: str) -> str:
    """A MongoDB URL with password (and host) hidden, safe for logs and tool output."""
    from urllib.parse import urlsplit, urlunsplit
    parsed = urlsplit(url)
    host = "***" if parsed.hostname else ""
    netloc = f"{parsed.username}:***@{host}" if parsed.username else host
    if parsed.port and not parsed.username:
        netloc = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, ""))


# ------------------------------------------------------------------ materialize
def materialize(sources: Iterable[tuple[str, Iterable[dict[str, Any]]]], *, deny: list[str] | None = None) -> dict[str, TableBuilder]:
    deny_effective = DEFAULT_DENY if deny is None else tuple(dict.fromkeys([*DEFAULT_DENY, *deny]))
    builders: dict[str, TableBuilder] = {}
    for name, docs in sources:
        table = sanitize_key(name)
        if table not in builders:
            builders[table] = TableBuilder(table)
        for doc in docs:
            flatten_document(table, doc, builders, deny=deny_effective)
    return builders
