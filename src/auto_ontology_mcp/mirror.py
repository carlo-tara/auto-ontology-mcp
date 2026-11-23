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
from collections import defaultdict
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
        # _id is the anchor; a document's own id joins it only when identical, else the
        # legacy value survives under "legacy_id": merging would crash the PK on duplicate
        # legacy ids and null out the anchor when the id field is absent-valued.
        if raw_key == "_id":
            key = "id"
        elif raw_key == "id":
            if row_key is None:
                key = "id"
            elif str(coerce_value(value)[1]) == row_key:
                key = "id"
            else:
                key = "legacy_id"
        else:
            key = sanitize_key(raw_key)
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


def mongo_sources(
    client: Any,
    database: str,
    collections: list[str],
    filters: dict[str, Any] | None = None,
    dropped: dict[str, int] | None = None,
) -> Iterator[tuple[str, Iterable[dict[str, Any]]]]:
    db = client[database]
    if not filters:
        for name in collections:
            yield name, db[name].find({})
        return
    cache: dict[str, dict[str, set[str]]] = {}

    def parent_values(of: str) -> dict[str, set[str]]:
        if of not in cache:
            want = _wanted_parent_fields(filters).get(of, set())
            sets = {f: set() for f in want}
            projection = {f: 1 for f in want} or None
            for doc in db[of].find(filters[of]["match"], projection):
                for f in want:
                    v = doc.get(f)
                    if v is not None:
                        sets[f].add(str(v))
            cache[of] = sets
        return cache[of]

    def keep(name: str, doc: dict[str, Any]) -> bool:
        spec = filters.get(name)
        if not spec:
            return True
        drop = ("match" in spec and _match_doc(doc, spec["match"])) or any(
            doc.get(link["field"]) is not None
            and str(doc.get(link["field"])) in parent_values(p["of"]).get(link["parent_field"], set())
            for p in spec.get("parents", [])
            for link in p["links"]
        )
        if drop and dropped is not None:
            dropped[name] = dropped.get(name, 0) + 1
        return not drop

    for name in collections:
        yield name, (d for d in db[name].find({}) if keep(name, d))


# ------------------------------------------------------------------ document filters
# Policy rules of the "internal accounts never appear in an analysis" kind are per-document
# and parent-based: a child document must go when the parent it references is excluded.
# `match` is a subset of the Mongo query language (bare equality, $regex/$options, $in/$nin,
# $ne/$eq, $exists) that is valid both as a pymongo query and in the client-side _match_doc,
# so parent key-sets are computed at the source. Unknown operators fail loudly: a security
# filter that silently matches nothing is worse than no migration.
_FILTER_OPS = frozenset({"$regex", "$options", "$in", "$nin", "$ne", "$eq", "$exists"})


def _regex_flags(options: Any) -> int:
    flags = 0
    for ch in str(options or ""):
        if ch == "i":
            flags |= re.IGNORECASE
        elif ch == "s":
            flags |= re.DOTALL
        elif ch == "m":
            flags |= re.MULTILINE
        elif ch == "x":
            flags |= re.VERBOSE
        else:
            raise OntologyError(f"filter $options: unsupported flag {ch!r}.")
    return flags


def _validate_match(query: Any, where: str) -> dict[str, Any]:
    if not isinstance(query, dict) or not query:
        raise OntologyError(f"{where}: 'match' needs a non-empty object of field conditions.")
    for field, cond in query.items():
        if str(field).startswith("$"):
            raise OntologyError(f"{where}: top-level operators are not supported; use field conditions.")
        if isinstance(cond, dict):
            bare = {op for op in cond if not str(op).startswith("$")}
            if bare:
                raise OntologyError(f"{where}: condition on {field!r} mixes bare keys with operators: {sorted(bare)}.")
            unknown = {op for op in cond if op not in _FILTER_OPS}
            if unknown:
                raise OntologyError(f"{where}: unsupported operator(s) {sorted(unknown)} on {field!r} (supported: {sorted(_FILTER_OPS)}).")
            if "$regex" in cond and not isinstance(cond["$regex"], str):
                raise OntologyError(f"{where}: $regex on {field!r} needs a string pattern.")
        elif isinstance(cond, (list, tuple)):
            raise OntologyError(f"{where}: bare equality on {field!r} needs a scalar.")
    return {str(k): v for k, v in query.items()}


def _match_doc(doc: dict[str, Any], query: dict[str, Any]) -> bool:
    """Client-side evaluation of the subset accepted by _validate_match."""
    for field, cond in query.items():
        value = doc.get(field)
        if isinstance(cond, dict):
            for op, arg in cond.items():
                if op == "$regex":
                    if not isinstance(value, str) or not re.search(arg, value, _regex_flags(cond.get("$options"))):
                        return False
                elif op == "$options":
                    continue
                elif op in ("$in", "$nin"):
                    members = {str(a) for a in arg}
                    hit = value is not None and str(value) in members
                    if hit == (op == "$nin"):
                        return False
                elif op in ("$ne", "$eq"):
                    hit = value is not None and str(value) == str(arg)
                    if hit == (op == "$ne"):
                        return False
                elif op == "$exists":
                    if (value is not None) != bool(arg):
                        return False
        else:
            if value is None or str(value) != str(cond):
                return False
    return True


def validate_doc_filters(filters: Any, datasource: str) -> dict[str, dict[str, Any]]:
    """Validate {name: {"match"?: query, "parents"?: [{"of", "links": [{"field", "parent_field"}]}]}}."""
    if not isinstance(filters, dict) or not filters:
        raise OntologyError(f"filters of {datasource!r} must be a non-empty object keyed by collection/endpoint name.")
    out: dict[str, dict[str, Any]] = {}
    for name, spec in filters.items():
        where = f"Filter on {str(name)!r} of {datasource!r}"
        if not isinstance(spec, dict):
            raise OntologyError(f"{where}: expected an object with 'match' and/or 'parents'.")
        unknown = {k for k in spec if k not in {"match", "parents"}}
        if unknown:
            raise OntologyError(f"{where}: unknown keys {sorted(unknown)}.")
        clean: dict[str, Any] = {}
        if "match" in spec:
            clean["match"] = _validate_match(spec["match"], where)
        parents: list[dict[str, Any]] = []
        for p in spec.get("parents") or []:
            if not isinstance(p, dict) or not p.get("of"):
                raise OntologyError(f"{where}: each 'parents' entry needs an 'of' naming the parent collection/endpoint.")
            of = str(p["of"])
            parent_spec = filters.get(of)
            if not isinstance(parent_spec, dict) or "match" not in parent_spec:
                raise OntologyError(
                    f"{where}: parent {of!r} has no 'match', so nothing could ever be excluded through it "
                    "(fail-safe: refusing a filter that drops nothing)."
                )
            links = p.get("links")
            if not isinstance(links, list) or not links:
                raise OntologyError(f"{where}: parent {of!r} needs a non-empty 'links' list: [{{'field', 'parent_field'}}].")
            clean_links = []
            for link in links:
                if not isinstance(link, dict) or not link.get("field"):
                    raise OntologyError(f"{where}: a link of parent {of!r} needs a 'field' (the child's reference field).")
                clean_links.append({"field": str(link["field"]), "parent_field": str(link.get("parent_field") or "_id")})
            parents.append({"of": of, "links": clean_links})
        if parents:
            clean["parents"] = parents
        if not clean:
            raise OntologyError(f"{where}: needs 'match' and/or 'parents'.")
        out[str(name)] = clean
    return out


def _wanted_parent_fields(filters: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    fields: dict[str, set[str]] = defaultdict(set)
    for spec in filters.values():
        for p in spec.get("parents", []):
            for link in p["links"]:
                fields[p["of"]].add(link["parent_field"])
    return fields


def validate_mongo_config(config: Any, datasource: str) -> dict[str, Any]:
    """Mirror options of a MongoDB datasource: {"deny": [...], "filters": {...}}."""
    if not isinstance(config, dict):
        raise OntologyError(f"The config of {datasource!r} must be an object (keys: 'deny', 'filters').")
    unknown = {k for k in config if k not in {"deny", "filters"}}
    if unknown:
        raise OntologyError(
            f"The config of {datasource!r} has unknown keys {sorted(unknown)}; "
            "a MongoDB mirror takes 'deny' (extra field substrings) and 'filters' (per-document exclusions)."
        )
    out: dict[str, Any] = {}
    if config.get("deny") is not None:
        deny = config["deny"]
        if not isinstance(deny, list) or not all(isinstance(d, str) for d in deny):
            raise OntologyError(f"config['deny'] of {datasource!r} must be a list of field-name substrings.")
        out["deny"] = deny
    if config.get("filters") is not None:
        out["filters"] = validate_doc_filters(config["filters"], datasource)
    return out


# ------------------------------------------------------------------ API collections
def api_filtered_sources(
    sources: list[tuple[str, list[dict[str, Any]]]],
    filters: dict[str, Any] | None,
    dropped: dict[str, int] | None = None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Apply document filters to fetched endpoint lists (see validate_doc_filters)."""
    if not filters:
        return sources
    items = dict(sources)
    cache: dict[str, dict[str, set[str]]] = {}

    def parent_values(of: str) -> dict[str, set[str]]:
        if of not in cache:
            want = _wanted_parent_fields(filters).get(of, set())
            query = filters[of]["match"]
            sets = {f: set() for f in want}
            for doc in items.get(of, []):
                if _match_doc(doc, query):
                    for f in want:
                        v = doc.get(f)
                        if v is not None:
                            sets[f].add(str(v))
            cache[of] = sets
        return cache[of]

    def keep(name: str, doc: dict[str, Any]) -> bool:
        spec = filters.get(name)
        if not spec:
            return True
        drop = ("match" in spec and _match_doc(doc, spec["match"])) or any(
            doc.get(link["field"]) is not None
            and str(doc.get(link["field"])) in parent_values(p["of"]).get(link["parent_field"], set())
            for p in spec.get("parents", [])
            for link in p["links"]
        )
        if drop and dropped is not None:
            dropped[name] = dropped.get(name, 0) + 1
        return not drop

    return [(name, [d for d in docs if keep(name, d)]) for name, docs in sources]


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
    """database (SQLAlchemy), mongo (mongodb:// URL) or api (endpoint config).

    The URL wins over the presence of a config: a mongodb:// URL may carry a config with
    mirror options (deny, filters), and only api:// / bare names rely on the config."""
    low = (url or "").strip().lower()
    if low.startswith(("mongodb://", "mongodb+srv://")):
        return "mongo"
    if config is not None or low.startswith("api://"):
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
        "filters": validate_doc_filters(config["filters"], datasource) if config.get("filters") is not None else None,
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
