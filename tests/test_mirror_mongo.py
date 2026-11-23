"""MongoDB datasources end to end: mirror fetch, scan, SQL — against a fake pymongo client."""

from __future__ import annotations

import datetime as dt

import pytest

from auto_ontology_mcp import mirror
from auto_ontology_mcp.service import OntologyError


class FakeCollection:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs

    def find(self, _filter: dict) -> list[dict]:
        return self.docs


class FakeDB:
    def __init__(self, cols: dict[str, list[dict]]) -> None:
        self.cols = cols

    def list_collection_names(self) -> list[str]:
        return sorted(self.cols) + ["system.indexes"]

    def __getitem__(self, name: str) -> FakeCollection:
        return FakeCollection(self.cols[name])


class FakeClient:
    def __init__(self, db: FakeDB) -> None:
        self._db = db
        self.closed = False

    def __getitem__(self, _name: str) -> FakeDB:
        return self._db

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def mongo_db():
    return {
        "users": [
            {"_id": "u1", "name": "Ann", "email": "ann@acme.io", "password": "x", "age": 44},
            {"_id": "u2", "name": "Bo", "email": "bo@acme.io", "tags": ["vip", "early"]},
        ],
        "transactions": [
            {
                "_id": "t1",
                "user": "u1",
                "finalTotal": 99.0,
                "status": 2,
                "createdAt": dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
                "products": [{"product_id": "p1", "qty": 2, "price": 10.0}, {"product_id": "p2", "qty": 1}],
            },
            {"_id": "t2", "user": "u2", "finalTotal": 50.0, "status": 1, "products": []},
        ],
    }


@pytest.fixture()
def fake_mongo(monkeypatch, mongo_db):
    state = {"db": mongo_db}
    clients: list[FakeClient] = []

    def connect(url: str, timeout_s: int) -> FakeClient:
        client = FakeClient(FakeDB(state["db"]))
        clients.append(client)
        return client

    monkeypatch.setattr(mirror, "mongo_connect", connect)
    return state, clients


def test_add_mongo_datasource_materializes_and_scans(settings, fake_mongo):
    from auto_ontology_mcp.service import OntologyService

    svc = OntologyService(settings)
    res = svc.add_datasource("mongo_live", "mongodb://user:pw@localhost:27017/testdb", description="live db")
    assert res["source_type"] == "mongo"
    assert "pw" not in res["url"]
    assert "fetches the mirror" in res["next_step"]
    entry = next(d for d in svc.check_readiness()["datasources"] if d["datasource"] == "mongo_live")
    assert entry["verdict"] == "not_materialized"
    scan = svc.scan_datasource("mongo_live")  # the first scan fetches the mirror
    assert scan["tables"] == 3  # users, transactions, transactions__products
    ds = svc.ds("mongo_live")
    assert ds["dialect"] == "sqlite"
    assert ds["source_type"] == "mongo"
    assert fake_mongo[1][0].closed is True  # the client is released after the fetch

    out = svc.run_sql(
        "SELECT u.name AS name, SUM(t.finalTotal) AS spend "
        "FROM transactions t JOIN users u ON t.user = u.id GROUP BY u.name ORDER BY name"
    )
    assert out["rows"] == [["Ann", 99.0], ["Bo", 50.0]]

    child = svc.run_sql("SELECT COUNT(*) AS n FROM transactions__products")
    assert child["rows"] == [[2]]

    # credential-shaped fields never reach the mirror
    with pytest.raises(OntologyError, match="password"):
        svc.run_sql("SELECT password FROM users")
    # dates land as ISO text; containers as JSON text
    r = svc.run_sql("SELECT createdAt FROM transactions WHERE id = 't1'")
    assert r["rows"][0][0].startswith("2026-01-02")
    r = svc.run_sql("SELECT tags FROM users WHERE id = 'u2'")
    assert "vip" in r["rows"][0][0]


def test_readiness_before_first_materialize(settings, fake_mongo):
    from auto_ontology_mcp.service import OntologyService

    svc = OntologyService(settings)
    svc.add_datasource("m", "mongodb://h/db", scan=False)
    entry = next(d for d in svc.check_readiness()["datasources"] if d["datasource"] == "m")
    assert entry["verdict"] == "not_materialized"
    assert "refresh_datasource" in entry["next_step"]

    svc.refresh_datasource("m")
    entry = next(d for d in svc.check_readiness()["datasources"] if d["datasource"] == "m")
    assert entry["source_type"] == "mongo"
    assert entry["materialized_at"]


def test_refresh_re_fetches_new_documents(settings, fake_mongo):
    from auto_ontology_mcp.service import OntologyService

    svc = OntologyService(settings)
    svc.add_datasource("m", "mongodb://h/db", scan=False)
    svc.refresh_datasource("m")  # scan=True: the catalog comes with the first fetch
    assert svc.run_sql("SELECT COUNT(*) AS n FROM users")["rows"] == [[2]]

    fake_mongo[0]["db"]["users"].append({"_id": "u3", "name": "Cy"})
    out = svc.refresh_datasource("m", scan=False)
    assert out["rows"] == 7  # users 3 + transactions 2 + transactions__products 2
    assert svc.run_sql("SELECT COUNT(*) AS n FROM users")["rows"] == [[3]]


def test_table_filter_excludes_collections(settings, fake_mongo, monkeypatch):
    from auto_ontology_mcp.service import OntologyService

    svc = OntologyService(settings)
    svc.add_datasource("m", "mongodb://h/db", scan=False, tables=["users"])
    svc.refresh_datasource("m", scan=False)
    svc.scan_datasource("m")
    names = {t["name"] for t in svc.store.tables(svc.ds("m")["id"])}
    assert names == {"users"}


def test_refresh_on_sql_database_fails(svc):
    with pytest.raises(OntologyError, match="live SQL database"):
        svc.refresh_datasource("demo")


def test_bad_mongo_urls_rejected(settings):
    from auto_ontology_mcp.service import OntologyService

    svc = OntologyService(settings)
    with pytest.raises(OntologyError, match="name the database"):
        svc.add_datasource("m", "mongodb://hostonly/", scan=False)
