import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Moritz says",
    css_file="site.css",
    html_head_file="head.html",
)

with app.setup:
    import html
    import math
    import sys
    from collections.abc import Sequence
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any, Literal
    from urllib.parse import urlsplit

    import marimo as mo
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        FiniteFloat,
        ValidationError,
        field_validator,
        model_validator,
    )

    # Pinned browser runtime for query embeddings. The model ID and revision come from the index.
    TRANSFORMERS_JS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2"
    QUERY_DTYPE = "q8"

    INDEX_PATH = ("public", "embeddings.json")
    TOP_WORKS = 5
    PASSAGES_PER_WORK = 5
    NORM_TOLERANCE = 0.01
    APPROVED_URL_HOSTS = frozenset({"doi.org"})
    IS_WASM = sys.platform == "emscripten"

    DIAGRAM_THEME = {
        "primaryColor": "#e9eee9",
        "primaryBorderColor": "#385344",
        "primaryTextColor": "#222222",
        "lineColor": "#555555",
        "clusterBkg": "#fcfcfa",
        "clusterBorder": "#385344",
        "fontFamily": "IBM Plex Sans, sans-serif",
        "fontSize": "16px",
    }

    # Keep this diagram the same as the one in README.md.
    DIAGRAM = """
    flowchart TB
      accTitle: How the search works
      accDescr: At build time, a script turns the texts into passages with embeddings. In the browser, the page loads them, embeds your query, ranks the passages, and shows the best ones.
      subgraph build["Build time, on the author's computer"]
        A["content/*.md"] --> B["build_index.py"] --> C["public/embeddings.json"]
      end
      subgraph browser["In your browser: your query never leaves your device"]
        D["Load and check the index"] --> E["Embed your query<br/>Transformers.js, ONNX"] --> F["Rank the passages<br/>Python, dot product"] --> G["Show the best passages"]
      end
      C --> D
    """

    # The only JavaScript in the application: load the model, embed one text, return a plain array.
    # The text is an argument. The source below is a constant; no value is interpolated into it.
    EMBEDDER_JS = """
    (async (moduleUrl, modelId, revision, dtype) => {
      const { pipeline } = await import(moduleUrl);
      const extractor = await pipeline("feature-extraction", modelId, {
        revision,
        dtype,
        device: "wasm",
      });
      return async (text) => {
        const output = await extractor(text, { pooling: "mean", normalize: true });
        return Array.from(output.data);
      };
    })
    """


@app.function
def is_safe_url(url: str) -> bool:
    """Accept a site-relative link or an https link to an approved host.

    >>> is_safe_url("https://doi.org/10.5281/zenodo.11124720")
    True
    >>> is_safe_url("docs/intro.html")
    True
    >>> is_safe_url("javascript:alert(1)")
    False
    >>> is_safe_url("//example.org/x")
    False
    """
    if not url or url != url.strip() or "\\" in url or any(ord(char) < 32 for char in url):
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme == "https":
        return parts.hostname in APPROVED_URL_HOSTS
    return not parts.scheme and not parts.netloc


@app.function
def check_vector(vector: Sequence[float], dimensions: int) -> None:
    """Raise ValueError unless the vector is finite, has the right size, and has unit length.

    >>> check_vector([0.6, 0.8], 2)
    >>> check_vector([1.0, 1.0], 2)
    Traceback (most recent call last):
    ValueError: vector norm 1.4142 is not 1
    """
    if len(vector) != dimensions:
        raise ValueError(f"vector has {len(vector)} dimensions, expected {dimensions}")
    norm = math.hypot(*vector)
    if not math.isfinite(norm):
        raise ValueError("vector has a non-finite value")
    if abs(norm - 1.0) > NORM_TOLERANCE:
        raise ValueError(f"vector norm {norm:.4f} is not 1")


@app.class_definition
class ModelInfo(BaseModel):
    """Embedding model settings that the index records.

    >>> ModelInfo(
    ...     document_model="a", document_revision="1", query_model="b", query_revision="2", dimensions=2
    ... ).pooling
    'mean'
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_model: str = Field(min_length=1)
    document_revision: str = Field(min_length=1)
    query_model: str = Field(min_length=1)
    query_revision: str = Field(min_length=1)
    dimensions: int = Field(gt=0)
    pooling: Literal["mean"] = "mean"
    normalized: Literal[True] = True
    document_prefix: str = ""
    query_prefix: str = ""


@app.class_definition
class Chunk(BaseModel):
    """One searchable passage with its embedding.

    >>> Chunk(id="a:0001", title="A", text="Text.", source="a.md", embedding=[1.0, 0.0]).url is None
    True
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    url: str | None = None
    embedding: list[FiniteFloat]

    @field_validator("title", "text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("url")
    @classmethod
    def _safe_url(cls, value: str | None) -> str | None:
        if value is not None and not is_safe_url(value):
            raise ValueError("must be a relative link or an https link to an approved host")
        return value


@app.class_definition
class SearchIndex(BaseModel):
    """The contract of ``public/embeddings.json``.

    >>> model = ModelInfo(
    ...     document_model="a", document_revision="1", query_model="b", query_revision="2", dimensions=2
    ... )
    >>> SearchIndex(schema_version=1, model=model, chunks=[]).chunks
    []
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    model: ModelInfo
    chunks: list[Chunk]

    @model_validator(mode="after")
    def _check_chunks(self) -> "SearchIndex":
        seen: set[str] = set()
        for chunk in self.chunks:
            if chunk.id in seen:
                raise ValueError(f"duplicate chunk id {chunk.id!r}")
            seen.add(chunk.id)
            try:
                check_vector(chunk.embedding, self.model.dimensions)
            except ValueError as exc:
                raise ValueError(f"chunk {chunk.id!r}: {exc}") from exc
        return self


@app.function
def load_index(raw: str) -> SearchIndex:
    """Parse and validate the index JSON text. Raise ValueError with a short message.

    >>> load_index('{"schema_version": 2}')
    Traceback (most recent call last):
    ValueError: Invalid search index at schema_version: Input should be 1 (3 problems)
    """
    try:
        return SearchIndex.model_validate_json(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "top level"
        raise ValueError(
            f"Invalid search index at {where}: {first['msg']} ({exc.error_count()} problems)"
        ) from exc


@app.function
def rank(
    query: Sequence[float], chunks: Sequence[Chunk], top_k: int | None = None
) -> list[tuple[Chunk, float]]:
    """Return the chunks by dot product, best first. Break a tie by chunk ID. Keep ``top_k``, if set.

    The vectors have unit length, so the dot product is the cosine similarity.

    >>> a = Chunk(id="a", title="A", text="x", source="a.md", embedding=[1.0, 0.0])
    >>> b = Chunk(id="b", title="B", text="y", source="b.md", embedding=[0.0, 1.0])
    >>> [(chunk.id, value) for chunk, value in rank([0.0, 1.0], [a, b])]
    [('b', 1.0), ('a', 0.0)]
    >>> [chunk.id for chunk, _ in rank([0.0, 1.0], [a, b], top_k=1)]
    ['b']
    """
    scored = [(chunk, math.sumprod(query, chunk.embedding)) for chunk in chunks]
    scored.sort(key=lambda pair: (-pair[1], pair[0].id))
    return scored[:top_k]


@app.class_definition
@dataclass(frozen=True)
class WorkResult:
    """The best passages of one work, best first.

    >>> WorkResult(title="A", url=None, hits=()).title
    'A'
    """

    title: str
    url: str | None
    hits: tuple[tuple[Chunk, float], ...]


@app.function
def group_by_work(
    ranked: Sequence[tuple[Chunk, float]],
    top_works: int = TOP_WORKS,
    per_work: int = PASSAGES_PER_WORK,
) -> list[WorkResult]:
    """Group ranked passages by work. The work of the best passage comes first.

    >>> a1 = Chunk(id="a:1", title="A", text="x", source="a.md", embedding=[1.0])
    >>> a2 = Chunk(id="a:2", title="A", text="y", source="a.md", embedding=[1.0])
    >>> b1 = Chunk(id="b:1", title="B", text="z", source="b.md", embedding=[1.0])
    >>> [(w.title, [c.id for c, _ in w.hits]) for w in group_by_work([(b1, 0.9), (a1, 0.8), (a2, 0.7)])]
    [('B', ['b:1']), ('A', ['a:1', 'a:2'])]
    >>> [w.title for w in group_by_work([(b1, 0.9), (a1, 0.8)], top_works=1)]
    ['B']
    >>> [len(w.hits) for w in group_by_work([(a1, 0.9), (a2, 0.8)], per_work=1)]
    [1]
    """
    groups: dict[str, list[tuple[Chunk, float]]] = {}
    for chunk, value in ranked:
        hits = groups.get(chunk.source)
        if hits is None:
            if len(groups) >= top_works:
                continue
            hits = groups[chunk.source] = []
        if len(hits) < per_work:
            hits.append((chunk, value))
    return [
        WorkResult(title=hits[0][0].title, url=hits[0][0].url, hits=tuple(hits))
        for hits in groups.values()
    ]


@app.function
def render_results(results: Sequence[WorkResult]) -> str:
    """Render the works with their passages as HTML. Escape every value that comes from the index.

    >>> chunk = Chunk(id="a", title="<A>", text="x & y", source="a.md", embedding=[1.0])
    >>> render_results([WorkResult(title=chunk.title, url=None, hits=((chunk, 0.5),))])
    '<section class="work"><h2>&lt;A&gt;</h2><ol class="results"><li><small>Ranking score 0.500</small><p>x &amp; y</p></li></ol></section>'
    >>> render_results([])
    '<p role="status">No passage found.</p>'
    """
    if not results:
        return '<p role="status">No passage found.</p>'
    sections: list[str] = []
    for work in results:
        link = ""
        if work.url is not None:
            href = html.escape(work.url, quote=True)
            link = f'<p><a href="{href}" target="_blank" rel="noopener noreferrer">Source</a></p>'
        items = "".join(
            f"<li><small>Ranking score {value:.3f}</small><p>{html.escape(chunk.text)}</p></li>"
            for chunk, value in work.hits
        )
        sections.append(
            f'<section class="work"><h2>{html.escape(work.title)}</h2>{link}'
            f'<ol class="results">{items}</ol></section>'
        )
    return "".join(sections)


@app.function
def render_corpus(index: SearchIndex) -> str:
    """List each searchable work with its link and passage count. Escape every value.

    >>> a = Chunk(id="a:1", title="Beta <1>", text="x", source="a.md", url="https://doi.org/1", embedding=[1.0])
    >>> b = Chunk(id="a:2", title="Beta <1>", text="y", source="a.md", url="https://doi.org/1", embedding=[1.0])
    >>> c = Chunk(id="c:1", title="Alpha", text="z", source="c.md", embedding=[1.0])
    >>> model = ModelInfo(
    ...     document_model="a", document_revision="1", query_model="b", query_revision="2", dimensions=1
    ... )
    >>> render_corpus(SearchIndex(schema_version=1, model=model, chunks=[a, b, c]))
    '<ul class="works"><li>Alpha <small>1 passage</small></li><li><a href="https://doi.org/1" target="_blank" rel="noopener noreferrer">Beta &lt;1&gt;</a> <small>2 passages</small></li></ul>'
    """
    works: dict[str, list[Chunk]] = {}
    for chunk in index.chunks:
        works.setdefault(chunk.source, []).append(chunk)
    items: list[tuple[str, str]] = []
    for chunks in works.values():
        first = chunks[0]
        title = html.escape(first.title)
        if first.url is not None:
            href = html.escape(first.url, quote=True)
            title = f'<a href="{href}" target="_blank" rel="noopener noreferrer">{title}</a>'
        count = len(chunks)
        items.append(
            (
                first.title.casefold(),
                f"<li>{title} <small>{count} passage{'s' * (count != 1)}</small></li>",
            )
        )
    return f'<ul class="works">{"".join(item for _, item in sorted(items))}</ul>'


@app.cell
def _():
    _nav = mo.nav_menu(
        {
            "#/": "Search",
            "#/works": "Works",
            "#/how-it-works": "How it works",
            "#/about": "About",
        }
    )
    mo.Html(f"""
    <header class="site-header">
    <a class="wordmark" href="#/">Moritz says</a>
    <nav aria-label="Main navigation">{_nav.text}</nav>
    </header>
    """)
    return


@app.cell
def _():
    intro = mo.md("""
    <p class="eyebrow">History of computing · Digital history</p>

    # Moritz says

    Search the publications of Moritz Mähr by meaning. Write a question or a topic in German or English.

    The search runs in your browser. The first search downloads a language model of about 120 MB.
    """)
    return (intro,)


@app.cell
async def _():
    index: SearchIndex | None = None
    index_error = ""
    try:
        with mo.status.spinner(title="Loading the search index"):
            _base = mo.notebook_location()
            if _base is None:
                raise ValueError("The notebook location is unknown.")
            _location = _base.joinpath(*INDEX_PATH)
            if IS_WASM:
                from pyodide.http import pyfetch  # ty: ignore[import-not-found]

                _response = await pyfetch(str(_location))
                if not _response.ok:
                    raise ValueError(f"The server answered with status {_response.status}.")
                _raw = await _response.string()
            else:
                _raw = Path(str(_location)).read_text(encoding="utf-8")
            index = load_index(_raw)
    except Exception as _exc:
        index_error = str(_exc) or type(_exc).__name__
    return index, index_error


@app.cell
def _(index, index_error):
    if index is None:
        status = mo.callout(
            mo.Html(
                "<strong>The search is not available.</strong> The search index did not load. "
                + html.escape(index_error)
            ),
            kind="danger",
        )
    elif not index.chunks:
        status = mo.callout(mo.Html("The search index holds no passage."), kind="warn")
    else:
        _works = len({chunk.source for chunk in index.chunks})
        status = mo.callout(
            mo.Html(f"Ready. The index holds {len(index.chunks)} passages from {_works} works."),
            kind="success",
        )
    return (status,)


@app.cell
def _(index):
    form = None
    retry = mo.ui.run_button(label="Retry")
    if index is not None and index.chunks:
        form = mo.ui.text(
            label="Query",
            placeholder="Wie veränderte der Computer die Verwaltung?",
            full_width=True,
        ).form(submit_button_label="Search", bordered=False)
    return form, retry


@app.cell
def _():
    # Holds the JavaScript embedding function, so that each search uses the same model instance.
    session: dict[str, Any] = {}
    return (session,)


@app.cell
async def _(form, index, retry, session):
    retry.value  # A click on Retry runs this cell again.
    results_view = mo.md("")
    _query = (form.value or "").strip() if form is not None else ""
    if form is None or index is None:
        pass
    elif not _query:
        results_view = mo.md("Write a query. Then select **Search**.")
    elif not IS_WASM:
        results_view = mo.callout(
            mo.md(
                "The query model runs only in the WebAssembly export. "
                "Run `uv run python -m scripts.export_site` and serve `dist/`."
            ),
            kind="warn",
        )
    else:
        try:
            if "embed" not in session:
                with mo.status.spinner(
                    title="Loading the language model",
                    subtitle="The first search downloads about 120 MB.",
                ):
                    from pyodide.code import run_js  # ty: ignore[import-not-found]

                    session["embed"] = await run_js(EMBEDDER_JS)(
                        TRANSFORMERS_JS_URL,
                        index.model.query_model,
                        index.model.query_revision,
                        QUERY_DTYPE,
                    )
            with mo.status.spinner(title="Searching"):
                _vector = (await session["embed"](index.model.query_prefix + _query)).to_py()
                check_vector(_vector, index.model.dimensions)
                results_view = mo.Html(render_results(group_by_work(rank(_vector, index.chunks))))
        except Exception as _exc:
            results_view = mo.vstack(
                [
                    mo.callout(
                        mo.Html(
                            "<strong>The search failed.</strong> "
                            "The language model did not load or did not run. "
                            + html.escape(str(_exc) or type(_exc).__name__)
                        ),
                        kind="danger",
                    ),
                    retry,
                ]
            )
    return (results_view,)


@app.cell
def _():
    about = mo.Html("""
    <section class="about" aria-labelledby="about-heading">
    <img class="portrait" src="public/img/portrait-560.webp"
      srcset="public/img/portrait-280.webp 280w, public/img/portrait-560.webp 560w"
      sizes="(max-width: 30em) 12rem, 15rem" width="560" height="560" alt="Portrait of Moritz Mähr">
    <div>
    <h2 id="about-heading">About me</h2>
    <p>I am a historian of technology. I study how computers change institutions and everyday life,
    especially in Swiss public administration and migration governance. I am a Research Associate in
    Digital History at the <a href="https://www.hist.unibe.ch/">Department of History</a> of the
    <a href="https://www.unibe.ch/index_eng.html">University of Bern</a> and an Information and Library
    Science Specialist with
    <a href="https://ethz.ch/staffnet/en/organisation/departments/eth-library/research-support-services/research-analytics-service.html">Research Analytics Services</a>
    at <a href="https://ethz.ch/en.html">ETH Zurich</a>, where I also earned my doctorate.
    I teach digital history and critical AI literacy, and I build open tools for working with
    historical sources.</p>
    <p>More at <a href="https://moritzmaehr.ch/">moritzmaehr.ch</a>, including my
    <a href="https://moritzmaehr.ch/academic/">CV and publications</a>.</p>
    </div>
    </section>
    """)
    return (about,)


@app.cell
def _():
    how = mo.vstack(
        [
            mo.md("""
    ## How it works

    This search finds passages. It does not write answers, and it does not check facts.
    Read each passage in the original work before you rely on it.

    1. A script splits each text into short passages and turns each passage into numbers, an embedding.
       It runs once, on my computer. The result is a public file.
    2. Your browser loads that file. It also loads a small language model of about 120 MB.
       Only the first search downloads the model.
    3. The model turns your query into an embedding. This happens on your device. Your query is not sent anywhere.
    4. Python compares your embedding with each passage. The best matches come first.
       The score is a ranking score, not a probability.
            """),
            mo.mermaid(DIAGRAM, theme="base", theme_variables=DIAGRAM_THEME),
            mo.md("""
    **Text version of the diagram.** At build time, `build_index.py` reads `content/*.md` and writes
    `public/embeddings.json`. In your browser, the page loads and checks that file, embeds your query with
    Transformers.js, ranks the passages in Python by dot product, and shows the best passages.

    The software is open source under AGPL-3.0-only. The code is on
    [GitHub](https://github.com/maehr/moritz-says). All texts and embeddings are public.
            """),
        ]
    )
    return (how,)


@app.cell
def _(index):
    if index is None or not index.chunks:
        works = mo.md("## What you can search\n\nThe search index is not available.")
    else:
        works = mo.vstack(
            [
                mo.md(
                    "## What you can search\n\nThese works are in the index. The links lead to the originals."
                ),
                mo.Html(render_corpus(index)),
            ]
        )
    return (works,)


@app.cell
def _(about, form, how, intro, results_view, status, works):
    # The search page is the catch-all, so that an empty hash and "#/" both show it.
    mo.routes(
        {
            "#/works": lambda: works,
            "#/how-it-works": lambda: how,
            "#/about": lambda: about,
            mo.routes.CATCH_ALL: lambda: mo.vstack(
                [intro, status, form if form is not None else mo.md(""), results_view]
            ),
        }
    )
    return


@app.cell
def _():
    mo.Html("""
    <footer class="site-footer">
    <a href="https://moritzmaehr.ch/">Moritz Mähr</a>
    <a href="https://moritzmaehr.ch/academic/">CV &amp; Publications</a>
    <a href="https://orcid.org/0000-0002-1367-1618">ORCID</a>
    <a href="https://github.com/maehr">GitHub</a>
    <a href="https://github.com/maehr/moritz-says">Source code</a>
    </footer>
    """)
    return


if __name__ == "__main__":
    app.run()
