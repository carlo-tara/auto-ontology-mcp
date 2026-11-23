"""API-collection datasources end to end: paginated endpoints, auth, nesting, mirrors."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from auto_ontology_mcp.service import OntologyError

PRODUCTS = [
    {"id": "p1", "name": "Widget", "price": 12.5, "apiToken": "zzz"},
    {"id": "p2", "name": "Gadget", "price": 7.0},
]
ORDERS = [
    {"id": 100, "email": "a@acme.io", "items": [{"sku": "p1", "qty": 2}, {"sku": "p2", "qty": 1}]},
    {"id": 101, "email": "b@acme.io", "items": [{"sku": "p2", "qty": 5}]},
]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a):  # keep the test output clean
        pass

    def _json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/products":
            if self.headers.get("X-Api-Key") != "secret-key":
                self.send_response(403)
                self.end_headers()
                return
            page = int(query.get("page", ["1"])[0])
            if page == 1:
                self._json({"results": [PRODUCTS[0]], "next": 2, "hasMore": True})
            elif page == 2:
                self._json({"results": [PRODUCTS[1]], "next": None, "hasMore": False})
            else:
                self._json({"results": [], "next": None})
        elif parsed.path == "/orders":
            self._json({"data": {"rows": ORDERS}})
        elif parsed.path == "/campaigns":
            after = int(query.get("after", ["0"])[0])
            rows = [c for c in CAMPAIGNS if c["id"] > after][:1]
            nxt = rows[-1]["id"] if len(rows) == 1 and any(c["id"] > rows[-1]["id"] for c in CAMPAIGNS) else None
            self._json({"items": rows, "next_cursor": nxt})
        else:
            self.send_response(404)
            self.end_headers()


CAMPAIGNS = [{"id": 1, "name": "spring"}, {"id": 2, "name": "summer"}, {"id": 3, "name": "fall"}]


@pytest.fixture(scope="module")
def api_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_api_collection_e2e(settings, api_server, monkeypatch):
    monkeypatch.setenv("API_TEST_KEY", "secret-key")
    from auto_ontology_mcp.service import OntologyService

    svc = OntologyService(settings)
    config = {
        "headers": {"X-Api-Key": "env:API_TEST_KEY"},
        "deny": ["sessionid"],
        "endpoints": [
            {"name": "products", "url": api_server + "/products", "items_path": "results",
             "pagination": {"type": "page", "param": "page", "size": 1, "has_more_path": "hasMore"}},
            {"name": "orders", "url": api_server + "/orders", "items_path": "data.rows"},
            {"name": "campaigns", "url": api_server + "/campaigns", "items_path": "items",
             "pagination": {"type": "cursor", "param": "after", "next_path": "next_cursor"}},
        ],
    }
    res = svc.add_datasource("crm", "api://crm", description="CRM API", config=config)
    assert res["source_type"] == "api"
    assert res["endpoints"] == ["products", "orders", "campaigns"]
    scan = svc.scan_datasource("crm")
    assert scan["tables"] == 4  # products, orders, orders__items, campaigns

    # paginated products came through both pages
    assert svc.run_sql("SELECT COUNT(*) AS n FROM products")["rows"] == [[2]]
    # credential-shaped field dropped
    with pytest.raises(OntologyError, match="apiToken"):
        svc.run_sql("SELECT apiToken FROM products")

    # nested items_path selected the array
    assert svc.run_sql("SELECT COUNT(*) AS n FROM orders")["rows"] == [[2]]
    # cursor pagination walked every campaign
    assert svc.run_sql("SELECT COUNT(*) AS n FROM campaigns")["rows"] == [[3]]

    out = svc.run_sql(
        "SELECT p.name, o.email, oi.qty FROM orders__items oi "
        "JOIN orders o ON oi.parent_id = CAST(o.id AS TEXT) "
        "JOIN products p ON oi.sku = p.id ORDER BY p.name, oi.qty"
    )
    assert out["rows"] == [["Gadget", "a@acme.io", 1], ["Gadget", "b@acme.io", 5], ["Widget", "a@acme.io", 2]]

    # the stored config keeps the env reference, not the resolved secret value semantics
    ds = svc.ds("crm")
    assert ds["source_config"]["headers"]["X-Api-Key"] == "env:API_TEST_KEY"

    # refresh re-fetches
    stats = svc.refresh_datasource("crm", scan=False)
    assert stats["tables"]["campaigns"] == 3


def test_api_config_validation(settings):
    from auto_ontology_mcp.service import OntologyService

    svc = OntologyService(settings)
    with pytest.raises(OntologyError, match="endpoint"):
        svc.add_datasource("x", "api://x", config={"endpoints": []})
    with pytest.raises(OntologyError, match="url"):
        svc.add_datasource("y", "api://y", config={"endpoints": [{"name": "a"}]})


def test_http_error_surfaces(settings, api_server):
    from auto_ontology_mcp.service import OntologyService

    svc = OntologyService(settings)
    svc.add_datasource("z", "api://z", scan=False, config={"endpoints": [{"name": "bad", "url": api_server + "/nope"}]})
    with pytest.raises(OntologyError, match="HTTP 404"):
        svc.refresh_datasource("z", scan=False)
