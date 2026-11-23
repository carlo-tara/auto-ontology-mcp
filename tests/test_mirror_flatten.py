"""Deterministic document → mirror-table flattening rules."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3

import pytest

from auto_ontology_mcp import mirror


class FakeObjectId(str):
    """Any class named ObjectId is hex-mapped without importing bson."""


def test_coerce_families():
    assert mirror.coerce_value(1) == ("INTEGER", 1)
    assert mirror.coerce_value(True) == ("INTEGER", 1)
    assert mirror.coerce_value(1.5) == ("REAL", 1.5)
    assert mirror.coerce_value("x") == ("TEXT", "x")
    assert mirror.coerce_value(None) == (None, None)
    assert mirror.coerce_value(dt.datetime(2026, 1, 2, 3, 4)) == ("TEXT", "2026-01-02T03:04:00")
    oid = FakeObjectId("507f1f77bcf86cd799439011")
    assert mirror.coerce_value(oid) == ("TEXT", "507f1f77bcf86cd799439011")
    fam, stored = mirror.coerce_value({"a": [1, None]})
    assert fam == "TEXT"
    assert json.loads(stored) == {"a": [1, None]}
    assert mirror.coerce_value(b"ab") == ("TEXT", "<2 bytes>")


def test_type_promotion_and_order():
    builders = {}
    mirror.flatten_document("t", {"_id": "1", "a": 1}, builders)
    mirror.flatten_document("t", {"_id": "2", "a": 1.5, "b": "z"}, builders)
    cols, pk = builders["t"].spec
    assert ("a", "REAL") in cols and ("b", "TEXT") in cols and pk == ["id"]
    # first-seen order is kept
    assert [c for c, _ in cols] == ["id", "a", "b"]


def test_mixed_types_fall_to_text():
    builders = {}
    mirror.flatten_document("t", {"_id": "1", "v": "s"}, builders)
    mirror.flatten_document("t", {"_id": "2", "v": 4}, builders)
    assert dict(builders["t"].spec[0])["v"] == "TEXT"


def test_array_of_documents_becomes_child_table():
    builders = {}
    doc = {"_id": "t1", "user": "u1", "products": [{"product_id": "p1", "qty": 2}, {"product_id": "p2"}]}
    mirror.flatten_document("transactions", doc, builders)
    child = builders["transactions__products"]
    cols, pk = child.spec
    assert pk == ["parent_id", "_idx"]
    assert [c for c, _ in cols] == ["parent_id", "_idx", "product_id", "qty"]
    rows = child.rows
    assert rows[0] == {"parent_id": "t1", "_idx": 0, "product_id": "p1", "qty": 2}
    assert rows[1]["_idx"] == 1 and "qty" not in rows[1]


def test_nested_array_under_child_stays_json():
    builders = {}
    mirror.flatten_document("t", {"_id": "1", "items": [{"sub": [{"deep": 1}]}]}, builders)
    child = builders["t__items"]
    assert json.loads(child.rows[0]["sub"]) == [{"deep": 1}]  # kept as JSON text, not a grandchild table
    assert "t__items__sub" not in builders


def test_denied_and_skipped_keys():
    builders = {}
    mirror.flatten_document(
        "t",
        {"_id": "1", "password_hash": "x", "stripeToken": "y", "__v": 0, "ok": 1},
        builders,
    )
    assert dict(builders["t"].spec[0]) == {"id": "TEXT", "ok": "INTEGER"}


def test_sanitize_key():
    assert mirror.sanitize_key("a.b-c") == "a_b_c"
    assert mirror.sanitize_key("2fa") == "f_2fa"


def test_no_anchor_falls_back_to_row_index():
    builders = {}
    mirror.flatten_document("t", {"k": 1, "items": [{"a": 1}]}, builders)
    mirror.flatten_document("t", {"k": 2, "items": [{"a": 2}, {"a": 3}]}, builders)
    assert [r["parent_id"] for r in builders["t__items"].rows] == ["0", "1", "1"]


def test_write_mirror_sqlite(tmp_path):
    builders = {}
    mirror.flatten_document("t", {"_id": "1", "v": 1, "items": [{"a": 1}]}, builders)
    counts = mirror.write_mirror_sqlite(tmp_path / "m.sqlite", builders)
    assert counts == {"t": 1, "t__items": 1}
    con = sqlite3.connect(tmp_path / "m.sqlite")
    assert con.execute("SELECT v FROM t").fetchone() == (1,)
    assert con.execute("SELECT parent_id, _idx, a FROM t__items").fetchone() == ("1", 0, 1)


def test_source_type_of():
    assert mirror.source_type_of("mongodb://h/db", None) == "mongo"
    assert mirror.source_type_of("mongodb+srv://u:p@h/db", None) == "mongo"
    assert mirror.source_type_of("sqlite:///x.db", None) == "database"
    assert mirror.source_type_of("api://name", None) == "api"
    assert mirror.source_type_of("", {"endpoints": []}) == "api"


def test_mongo_config_and_mask():
    cfg = mirror.mongo_config("mongodb+srv://u:pw@cluster0.mongodb/mydb?retryWrites=true")
    assert cfg["database"] == "mydb"
    with pytest.raises(Exception, match="name the database"):
        mirror.mongo_config("mongodb://h/")
    masked = mirror.mask_host("mongodb+srv://u:pw@cluster0.mongodb/mydb")
    assert "pw" not in masked and "cluster0" not in masked and "mydb" in masked


def test_validate_api_config():
    cfg = mirror.validate_api_config(
        {"base_url": "https://x", "endpoints": [{"name": "deals", "path": "/deals"}]}, "crm"
    )
    assert cfg["endpoints"][0]["name"] == "deals" and cfg["base_url"] == "https://x"
    with pytest.raises(Exception, match="non-empty"):
        mirror.validate_api_config({"endpoints": []}, "crm")
    with pytest.raises(Exception, match="twice"):
        mirror.validate_api_config(
            {"endpoints": [{"name": "a", "url": "u"}, {"name": "a", "url": "v"}]}, "crm"
        )
