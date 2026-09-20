"""The GUI's HTTP routes, pinned, and what the API promises to whom.

A3-8 measured 22 routes, FastAPI's default version and no snapshot. U6's
ruling is that this API is INTERNAL: it is versioned with the React client
bundled beside it, not on its own, and carries no compatibility promise to any
other consumer.

An internal API still needs a snapshot, for a reason that is almost the
opposite of a public one's. A public contract is pinned so it cannot change
without a version bump. This is pinned so a route arriving or leaving is
VISIBLE -- because the client that consumes it is checked into this repository
and rebuilt by hand, and a route removed without rebuilding the bundle is a
404 that the e2e suite can only catch if the bundle happens to exercise it.

The `info.version` in the OpenAPI document was FastAPI's default `0.1.0`,
which reads as a stated API version and was only a placeholder. It now reports
the package's version and the description says the API is internal, so nobody
reads a promise into it.

**Two routes left on 2026-09-20 and this file is how that is visible.** `POST
/api/snapshot` and `PATCH /api/nodes/{node_id}` existed only for the config-GUI
spike (A10-5, A10-6); removing the spike removed them, and the snapshot went
red naming both before anything else noticed. A3-8's count of 22 is now 20.
That is the whole argument for pinning an internal API: not that it cannot
change, but that it cannot change quietly.
"""

from __future__ import annotations

import json
import pathlib

GOLDEN = pathlib.Path(__file__).resolve().parent / "golden" / "http_routes.json"


def _live() -> list[dict[str, str]]:
    from rheplicant.gui.api import create_app

    app = create_app()
    routes = [
        {"method": method, "path": route.path}
        for route in app.routes
        for method in (getattr(route, "methods", None) or {"MOUNT"})
        if not route.path.startswith(("/openapi", "/docs", "/redoc"))
    ]
    routes.sort(key=lambda row: (row["path"], row["method"]))
    return routes


def test_the_route_table_matches_its_snapshot():
    stored = json.loads(GOLDEN.read_text(encoding="utf-8"))
    live = _live()
    if live == stored:
        return
    as_pairs = {(row["method"], row["path"]) for row in live}
    was_pairs = {(row["method"], row["path"]) for row in stored}
    raise AssertionError(
        "the GUI's HTTP routes have changed.\n"
        f"  added:   {sorted(as_pairs - was_pairs)}\n"
        f"  removed: {sorted(was_pairs - as_pairs)}\n"
        "The React client bundled under src/rheplicant/gui/static/ is the only "
        "supported consumer and it is rebuilt by hand -- a removed route is a "
        "404 in that bundle. Regenerate this snapshot in the same commit that "
        "rebuilds the bundle."
    )


def test_the_openapi_document_does_not_advertise_a_placeholder_version():
    """FastAPI's default is `0.1.0`, which looks like a declared API version."""
    from rheplicant.gui.api import create_app

    app = create_app()
    assert app.version != "0.1.0", (
        "the OpenAPI document reports FastAPI's default version, which reads "
        "as a stated API version and is not one"
    )
    assert "internal" in (app.description or "").lower(), (
        "the OpenAPI description does not say the API is internal, so a reader "
        "of the generated documentation has no way to know it carries no "
        "compatibility promise"
    )


def test_every_route_is_under_the_api_prefix_or_is_a_mount():
    """A route outside `/api` is served alongside the client's own paths.

    Not a style rule: the frontend is mounted at `/`, so a new bare path can
    shadow a client route and the failure is a page that stops loading.
    """
    stray = [
        row for row in _live()
        if not row["path"].startswith("/api") and row["method"] != "MOUNT"
    ]
    assert not stray, (
        f"these routes are outside /api and are not mounts: {stray}. The "
        "client is mounted at / and a bare path can shadow one of its routes"
    )


def test_the_stability_page_states_the_real_route_count():
    """The page's number, against the snapshot rather than against memory.

    It said twenty-two for as long as there were twenty-two, and nothing would
    have told anyone when that stopped being true. Removing the spike's two
    routes is exactly the edit that makes a written-out count wrong.
    """
    words = {
        18: "Eighteen", 19: "Nineteen", 20: "Twenty", 21: "Twenty-one",
        22: "Twenty-two", 23: "Twenty-three", 24: "Twenty-four",
    }
    count = len(json.loads(GOLDEN.read_bytes()))
    page = (
        pathlib.Path(__file__).resolve().parents[2] / "docs" / "stability.md"
    ).read_text(encoding="utf-8")
    assert f"{words[count]} routes under `/api`" in page, (
        f"docs/stability.md does not say there are {count} routes"
    )
