"""Browser tests for the exported site, with real WebAssembly inference.

These tests need the export in ``dist/``, the Playwright browsers, and network access::

    uv run python -m scripts.export_site
    uv run playwright install chromium firefox
    uv run pytest -m browser --no-cov
"""

import json
import math
import threading
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import BrowserContext, Page, Route, sync_playwright

from app import EMBEDDER_JS, QUERY_DTYPE, TRANSFORMERS_JS_URL, load_index

pytestmark = pytest.mark.browser

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "reference_vectors.json").read_text("utf-8"))
INDEX = load_index((ROOT / "public" / "embeddings.json").read_text("utf-8"))

# Hosts that supply the runtime, the packages, and the model files. None is an inference service.
ALLOWED_HOSTS = {"cdn.jsdelivr.net", "files.pythonhosted.org", "pypi.org", "wasm.marimo.app"}
ALLOWED_HOST_SUFFIXES = ("huggingface.co", ".hf.co")
LOAD_TIMEOUT = 180_000
MODEL_TIMEOUT = 420_000


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        """Keep the request lines out of the test output."""


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Serve ``dist/`` below ``/moritz-says/``, as GitHub Pages does."""
    if not (DIST / "index.html").exists():
        pytest.fail("dist/index.html is missing. Run the marimo export first.")
    root = tmp_path_factory.mktemp("pages")
    (root / "moritz-says").symlink_to(DIST, target_is_directory=True)
    handler = partial(QuietHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://localhost:{server.server_port}/moritz-says/"
    server.shutdown()


@pytest.fixture(scope="module", params=["chromium", "firefox"])
def context(request: pytest.FixtureRequest) -> Iterator[BrowserContext]:
    """One browser context for each browser. The tests share its model cache."""
    with sync_playwright() as playwright:
        browser = getattr(playwright, request.param).launch()
        yield browser.new_context()
        browser.close()


def open_page(context: BrowserContext, hosts: set[str], errors: list[str]) -> Page:
    page = context.new_page()
    page.on("request", lambda request: hosts.add(urlsplit(request.url).hostname or ""))
    page.on("pageerror", lambda error: errors.append(str(error)))
    return page


def search(page: Page, query: str) -> None:
    page.get_by_role("textbox").fill(query)
    page.get_by_role("button", name="Search").click()


def is_allowed(host: str) -> bool:
    local = host in {"", "localhost", "127.0.0.1"}
    return local or host in ALLOWED_HOSTS or host.endswith(ALLOWED_HOST_SUFFIXES)


def test_search_end_to_end(context: BrowserContext, site: str) -> None:
    """A1, A2, A3, and the empty query of A6."""
    hosts: set[str] = set()
    errors: list[str] = []
    page = open_page(context, hosts, errors)
    page.goto(site)
    page.get_by_text("Ready. The index holds").wait_for(timeout=LOAD_TIMEOUT)
    page.reload()
    page.get_by_text("Ready. The index holds").wait_for(timeout=LOAD_TIMEOUT)

    search(page, "   ")
    page.get_by_text("Write a query.").wait_for(timeout=LOAD_TIMEOUT)
    assert not any(host.endswith(ALLOWED_HOST_SUFFIXES) for host in hosts), "blank query"

    reference = FIXTURE["reference"]
    search(page, reference["query"])
    items = page.locator("ol.results li")
    items.first.wait_for(timeout=MODEL_TIMEOUT)
    texts = [" ".join(text.split()) for text in items.all_inner_texts()]
    assert 1 <= len(texts) <= 5
    expected = next(chunk for chunk in INDEX.chunks if chunk.id == reference["expected_id"])
    start = " ".join(expected.text.split())[:80]
    assert any(start in text for text in texts[:3]), texts[:3]

    assert not errors
    assert all(is_allowed(host) for host in hosts), sorted(hosts)
    page.close()


def test_browser_vectors_match_python(context: BrowserContext, site: str) -> None:
    """A4: the browser model and the Python model give the same vectors."""
    page = context.new_page()
    page.goto(site)
    model = FIXTURE["model"]
    sentences = [item["text"] for item in FIXTURE["sentences"]]
    vectors = page.evaluate(
        f"""async ([url, modelId, revision, dtype, texts]) => {{
            const embed = await ({EMBEDDER_JS})(url, modelId, revision, dtype);
            const vectors = [];
            for (const text of texts) vectors.push(await embed(text));
            return vectors;
        }}""",
        [
            TRANSFORMERS_JS_URL,
            model["query_model"],
            model["query_revision"],
            QUERY_DTYPE,
            sentences,
        ],
    )
    for item, vector in zip(FIXTURE["sentences"], vectors, strict=True):
        assert len(vector) == len(item["embedding"]) == model["dimensions"] == 384
        assert math.hypot(*vector) == pytest.approx(1.0, abs=0.01)
        assert math.hypot(*item["embedding"]) == pytest.approx(1.0, abs=0.01)
        assert math.sumprod(vector, item["embedding"]) >= 0.98, item["text"]
    page.close()


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (404, "not found", "The search is not available."),
        (200, "{not json", "The search is not available."),
        (200, json.dumps({"schema_version": 2}), "The search is not available."),
        (
            200,
            json.dumps({"schema_version": 1, "model": FIXTURE["model"], "chunks": []}),
            "The search index holds no passage.",
        ),
    ],
)
def test_bad_index_shows_message(
    context: BrowserContext, site: str, status: int, body: str, message: str
) -> None:
    """A6: a missing, an invalid, and an empty index give a message and no search field."""
    errors: list[str] = []
    page = open_page(context, set(), errors)
    page.route("**/public/embeddings.json", lambda route: route.fulfill(status=status, body=body))
    page.goto(site)
    page.get_by_text(message).wait_for(timeout=LOAD_TIMEOUT)
    assert page.get_by_role("textbox").count() == 0
    assert not errors
    page.close()


def test_model_failure_shows_retry(context: BrowserContext, site: str) -> None:
    """A6: a failed model download gives an error with a retry that works."""
    assert context.browser is not None
    fresh = context.browser.new_context()  # An empty cache, so that the model must download.
    errors: list[str] = []
    page = open_page(fresh, set(), errors)

    def block(route: Route) -> None:
        route.abort()

    page.route("https://huggingface.co/**", block)
    page.goto(site)
    page.get_by_text("Ready. The index holds").wait_for(timeout=LOAD_TIMEOUT)
    search(page, FIXTURE["reference"]["query"])
    page.get_by_text("The search failed.").wait_for(timeout=MODEL_TIMEOUT)
    assert page.locator("ol.results li").count() == 0

    page.unroute("https://huggingface.co/**", block)
    page.get_by_role("button", name="Retry").click()
    page.locator("ol.results li").first.wait_for(timeout=MODEL_TIMEOUT)
    assert not errors
    fresh.close()


def test_page_explains_the_author_and_the_search(context: BrowserContext, site: str) -> None:
    """The page shows the About section, the diagram, and the list of works."""
    hosts: set[str] = set()
    errors: list[str] = []
    page = open_page(context, hosts, errors)
    page.goto(site)
    page.get_by_role("heading", name="About me").wait_for(timeout=LOAD_TIMEOUT)
    assert page.get_by_role("img", name="Portrait of Moritz Mähr").is_visible()
    page.get_by_role("heading", name="How it works").wait_for(timeout=LOAD_TIMEOUT)
    page.locator("svg[aria-roledescription='flowchart-v2']").wait_for(timeout=LOAD_TIMEOUT)
    works = page.locator("ul.works li")
    works.first.wait_for(timeout=LOAD_TIMEOUT)
    assert works.count() == len({chunk.source for chunk in INDEX.chunks})
    assert not errors
    assert all(is_allowed(host) for host in hosts), sorted(hosts)
    page.close()
