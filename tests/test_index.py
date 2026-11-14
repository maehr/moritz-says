"""Unit tests for index validation, ranking, rendering, and offline indexing."""

import json
from typing import Any

import pytest

from app import (
    NORM_TOLERANCE,
    TOP_K,
    Chunk,
    ModelInfo,
    SearchIndex,
    check_vector,
    is_safe_url,
    load_index,
    rank,
    render_results,
)
from scripts.build_index import (
    Document,
    build_chunks,
    chunk_text,
    clean_markdown,
    parse_document,
    serialize_index,
    split_long,
)

MODEL = ModelInfo(
    document_model="a", document_revision="1", query_model="b", query_revision="2", dimensions=2
)


def words(text: str) -> int:
    return len(text.split())


def chunk_data(**override: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "a:0001",
        "title": "A",
        "text": "Text.",
        "source": "a.md",
        "url": None,
        "embedding": [0.6, 0.8],
    }
    return data | override


def index_text(chunks: list[dict[str, Any]] | None = None, **top: Any) -> str:
    data = {
        "schema_version": 1,
        "model": MODEL.model_dump(),
        "chunks": [chunk_data()] if chunks is None else chunks,
    }
    return json.dumps(data | top)


def make_chunk(id: str, embedding: list[float]) -> Chunk:
    return Chunk(id=id, title="T", text="x", source="a.md", embedding=embedding)


def sentence(label: str, count: int) -> str:
    return " ".join([label, *["w"] * (count - 1)]) + "."


# Index validation


def test_load_index_accepts_a_valid_index() -> None:
    index = load_index(index_text([chunk_data(url="https://doi.org/10.1/x")]))
    assert isinstance(index, SearchIndex)
    assert index.model == MODEL
    assert [(c.id, c.url, c.embedding) for c in index.chunks] == [
        ("a:0001", "https://doi.org/10.1/x", [0.6, 0.8])
    ]


BAD_INDEXES = {
    "schema version": index_text(schema_version=2),
    "extra key": index_text(extra=1),
    "extra chunk key": index_text([chunk_data(extra=1)]),
    "not json": "not json",
    "duplicate ids": index_text([chunk_data(), chunk_data(embedding=[1.0, 0.0])]),
    "empty title": index_text([chunk_data(title="")]),
    "blank title": index_text([chunk_data(title="  \n")]),
    "empty text": index_text([chunk_data(text="")]),
    "blank text": index_text([chunk_data(text=" \t ")]),
    "too few dimensions": index_text([chunk_data(embedding=[1.0])]),
    "too many dimensions": index_text([chunk_data(embedding=[0.6, 0.8, 0.0])]),
    "norm too high": index_text([chunk_data(embedding=[1.0 + NORM_TOLERANCE + 0.001, 0.0])]),
    "norm too low": index_text([chunk_data(embedding=[1.0 - NORM_TOLERANCE - 0.001, 0.0])]),
    "NaN": index_text([chunk_data(embedding=[float("nan"), 0.0])]),
    "Infinity": index_text([chunk_data(embedding=[float("inf"), 0.0])]),
    "unsafe url": index_text([chunk_data(url="javascript:alert(1)")]),
}


@pytest.mark.parametrize("raw", BAD_INDEXES.values(), ids=BAD_INDEXES.keys())
def test_load_index_rejects(raw: str) -> None:
    with pytest.raises(ValueError, match=r"^Invalid search index at"):
        load_index(raw)


def test_load_index_names_the_first_problem() -> None:
    with pytest.raises(ValueError, match=r"^Invalid search index at schema_version: "):
        load_index(index_text(schema_version=2))
    with pytest.raises(ValueError, match=r"duplicate chunk id 'a:0001'"):
        load_index(BAD_INDEXES["duplicate ids"])


def test_load_index_accepts_a_norm_just_inside_the_tolerance() -> None:
    for scale in (1.0 + NORM_TOLERANCE - 0.001, 1.0 - NORM_TOLERANCE + 0.001):
        assert load_index(index_text([chunk_data(embedding=[scale, 0.0])])).chunks[0].embedding[
            0
        ] == pytest.approx(scale)


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([1.0], "1 dimensions, expected 2"),
        ([1.0, 0.0, 0.0], "3 dimensions, expected 2"),
        ([float("inf"), 0.0], "non-finite"),
        ([float("nan"), 0.0], "non-finite"),
        ([1.0, 1.0], "norm 1.4142 is not 1"),
        ([0.0, 0.0], "norm 0.0000 is not 1"),
    ],
)
def test_check_vector_rejects(vector: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_vector(vector, 2)


def test_check_vector_accepts_a_unit_vector() -> None:
    check_vector([0.6, 0.8], 2)
    check_vector([1.0 + NORM_TOLERANCE - 0.001, 0.0], 2)


@pytest.mark.parametrize(
    "url",
    ["https://doi.org/10.5281/zenodo.1", "docs/intro.html", "/docs/intro.html", "#top", "?q=1"],
)
def test_is_safe_url_accepts(url: str) -> None:
    assert is_safe_url(url)
    assert Chunk(**chunk_data(url=url)).url == url


@pytest.mark.parametrize(
    "url",
    [
        "http://doi.org/10.1/x",
        "https://example.org/x",
        "javascript:alert(1)",
        "//example.org/x",
        "docs\\intro.html",
        "docs/in\x00tro.html",
        "docs/in\ntro.html",
        " docs/intro.html",
        "docs/intro.html ",
        "",
        "https://[invalid",
    ],
)
def test_is_safe_url_rejects(url: str) -> None:
    assert not is_safe_url(url)
    with pytest.raises(ValueError, match="approved host"):
        Chunk(**chunk_data(url=url))


def test_chunk_without_url_is_valid() -> None:
    assert Chunk(**chunk_data()).url is None


# Ranking (A5)


def test_rank_scores_an_equal_vector_as_one() -> None:
    vector = [0.6, 0.8, 0.0]
    ((best, score),) = rank(vector, [make_chunk("a", vector)])
    assert best.id == "a"
    assert score == pytest.approx(1.0)


def test_rank_orders_by_descending_score_and_is_repeatable() -> None:
    chunks = [
        make_chunk("c", [0.0, 1.0]),
        make_chunk("a", [0.8, 0.6]),
        make_chunk("d", [-1.0, 0.0]),
        make_chunk("b", [1.0, 0.0]),
    ]
    runs = [rank([1.0, 0.0], chunks) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
    assert [c.id for c, _ in runs[0]] == ["b", "a", "c", "d"]
    scores = [score for _, score in runs[0]]
    assert scores == sorted(scores, reverse=True)
    assert scores == pytest.approx([1.0, 0.8, 0.0, -1.0])


def test_rank_breaks_ties_by_chunk_id() -> None:
    chunks = [make_chunk(id, [1.0, 0.0]) for id in ("b", "c", "a")]
    assert [c.id for c, _ in rank([1.0, 0.0], chunks)] == ["a", "b", "c"]
    assert [c.id for c, _ in rank([1.0, 0.0], chunks[::-1])] == ["a", "b", "c"]


def test_rank_limits_the_result_count() -> None:
    chunks = [make_chunk(f"id{n}", [1.0, 0.0]) for n in range(8)]
    assert TOP_K == 5
    assert len(rank([1.0, 0.0], chunks)) == TOP_K
    assert [c.id for c, _ in rank([1.0, 0.0], chunks, top_k=2)] == ["id0", "id1"]
    assert len(rank([1.0, 0.0], chunks, top_k=20)) == 8


def test_rank_edge_cases() -> None:
    assert rank([1.0, 0.0], []) == []
    with pytest.raises(ValueError, match="."):
        rank([1.0, 0.0, 0.0], [make_chunk("a", [1.0, 0.0])])


# Rendering


def test_render_results_escapes_values() -> None:
    chunk = Chunk(**chunk_data(title="<script>alert(1)</script>", text="<script> & </b>"))
    output = render_results([(chunk, 0.5)])
    assert "<script>" not in output
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in output
    assert "&lt;script&gt; &amp; &lt;/b&gt;" in output


def test_render_results_links_only_when_url_is_set() -> None:
    plain = render_results([(Chunk(**chunk_data()), 0.25)])
    assert "<a " not in plain
    assert "Ranking score 0.250" in plain
    linked = render_results([(Chunk(**chunk_data(url='p.html?a=1&b="2"')), 0.25)])
    assert 'href="p.html?a=1&amp;b=&quot;2&quot;"' in linked
    assert 'rel="noopener noreferrer"' in linked


def test_render_results_without_results() -> None:
    assert render_results([]) == '<p role="status">No passage found.</p>'


# Offline indexing: parse_document


def test_parse_document_reads_front_matter() -> None:
    quoted = parse_document("a.md", '---\ntitle: "A: B"\nurl: "https://doi.org/1"\n---\nBody.')
    assert (quoted.source, quoted.title, quoted.url, quoted.body) == (
        "a.md",
        "A: B",
        "https://doi.org/1",
        "Body.",
    )
    unquoted = parse_document("a.md", "---\ntitle: Plain: Title\nurl: https://doi.org/1\n---\nB")
    assert (unquoted.title, unquoted.url, unquoted.body) == (
        "Plain: Title",
        "https://doi.org/1",
        "B",
    )


def test_parse_document_defaults() -> None:
    document = parse_document("my_file-name.md", "---\nnote\nauthor: X\n---\nBody.")
    assert (document.title, document.url, document.body) == ("My file name", None, "Body.")
    assert parse_document("a.md", "---\ntitle: T\nurl:\n---\nB").url is None


@pytest.mark.parametrize("raw", ["Just text.", "---\ntitle: T\nNo closing line.", "---\n"])
def test_parse_document_keeps_text_without_complete_front_matter(raw: str) -> None:
    document = parse_document("some-file.txt", raw)
    assert (document.title, document.url, document.body) == ("Some file", None, raw)


# Offline indexing: clean_markdown and split_long


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a<!-- note -->b", "ab"),
        ("a<!--\nline\nline\n-->b", "ab"),
        ("Text ![alt](img.png) end", "Text  end"),
        ("See [the site](https://example.org).", "See the site."),
        ("<p>one<br/>two <span class=x>three</span></p>", "onetwo three"),
        ("## Head\n###### Deep\nText # not a mark", "Head\nDeep\nText # not a mark"),
        ("**bold** *it* _u_ ***both***", "bold it u both"),
        ("if a < b and c > d", "if a < b and c > d"),
    ],
)
def test_clean_markdown(raw: str, expected: str) -> None:
    assert clean_markdown(raw) == expected


def test_split_long_respects_the_limit_and_keeps_order() -> None:
    text = " ".join(f"w{n}" for n in range(11))
    pieces = split_long(text, words, 4)
    assert [words(piece) for piece in pieces] == [4, 4, 3]
    assert " ".join(pieces) == text
    assert split_long("", words, 4) == []


def test_split_long_keeps_a_word_that_exceeds_the_limit() -> None:
    assert split_long("a abcdef gh", len, 3) == ["a", "abcdef", "gh"]


# Offline indexing: chunk_text


def test_chunk_text_never_exceeds_max_tokens() -> None:
    text = " ".join(sentence(f"S{n}", 4) for n in range(10))
    chunks = chunk_text(text, words, min_tokens=6, max_tokens=10, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(words(chunk) <= 10 for chunk in chunks)
    assert " ".join(chunks) == text


def test_chunk_text_closes_a_passage_at_a_paragraph_end() -> None:
    first = f"{sentence('A1', 5)} {sentence('A2', 5)}"
    second = sentence("B1", 5)
    assert chunk_text(f"{first}\n\n{second}", words, 8, 100, 0) == [first, second]
    assert chunk_text(f"{first}\n{second}", words, 8, 100, 0) == [f"{first} {second}"]


def test_chunk_text_joins_a_short_paragraph_to_the_next() -> None:
    first, second = sentence("A1", 3), sentence("B1", 6)
    assert chunk_text(f"{first}\n\n  \n{second}", words, 8, 100, 0) == [f"{first} {second}"]


def test_chunk_text_repeats_a_short_last_sentence() -> None:
    s1, s2, s3, s4 = (sentence(f"S{n}", size) for n, size in ((1, 6), (2, 6), (3, 3), (4, 6)))
    text = " ".join([s1, s2, s3, s4])
    assert chunk_text(text, words, 100, 16, 4) == [f"{s1} {s2} {s3}", f"{s3} {s4}"]
    assert chunk_text(text, words, 100, 16, 2) == [f"{s1} {s2} {s3}", s4]


def test_chunk_text_splits_a_sentence_longer_than_max() -> None:
    text = sentence("Long", 25)
    chunks = chunk_text(text, words, 5, 10, 3)
    assert [words(chunk) for chunk in chunks] == [10, 10, 5]
    assert " ".join(chunks) == text


@pytest.mark.parametrize("text", ["", "  \n\n \t "])
def test_chunk_text_of_blank_text_is_empty(text: str) -> None:
    assert chunk_text(text, words) == []


def test_chunk_text_is_deterministic() -> None:
    text = "\n\n".join(" ".join(sentence(f"P{p}S{s}", 7) for s in range(6)) for p in range(4))
    assert chunk_text(text, words, 10, 20, 8) == chunk_text(text, words, 10, 20, 8)


# Offline indexing: build_chunks and serialize_index


def test_build_chunks_numbers_passages_per_document() -> None:
    long_body = f"{sentence('A', 130)}\n\n{sentence('B', 130)}"
    documents = [
        Document("intro.md", "Intro", "https://doi.org/1", long_body),
        Document("notes/b.txt", "B", None, "**Bold** text."),
    ]
    records = build_chunks(documents, words)
    assert [r["id"] for r in records] == ["intro:0001", "intro:0002", "b:0001"]
    assert [(r["title"], r["source"], r["url"]) for r in records] == [
        ("Intro", "intro.md", "https://doi.org/1"),
        ("Intro", "intro.md", "https://doi.org/1"),
        ("B", "notes/b.txt", None),
    ]
    assert records[2]["text"] == "Bold text."
    assert build_chunks([], words) == []


def index_records() -> list[dict[str, Any]]:
    return [
        chunk_data(id="a:0001", title="Mähr", embedding=[1.0, 0.0]),
        chunk_data(id="a:0002", url="docs/a.html", embedding=[0.0, 1.0]),
    ]


def test_serialize_index_is_deterministic_and_one_chunk_per_line() -> None:
    text = serialize_index(MODEL, index_records())
    assert text == serialize_index(MODEL, index_records())
    lines = text.splitlines()
    assert len(lines) == 4 + 2
    assert [json.loads(line.rstrip(",")) for line in lines[3:5]] == index_records()
    assert "Mähr" in text
    assert text.endswith("]}\n")


def test_serialize_index_round_trips_through_load_index() -> None:
    index = load_index(serialize_index(MODEL, index_records()))
    assert index.model == MODEL
    assert [c.model_dump() for c in index.chunks] == index_records()


def test_serialize_index_of_no_chunks_is_valid() -> None:
    text = serialize_index(MODEL, [])
    assert json.loads(text)["chunks"] == []
    assert load_index(text).chunks == []


@pytest.mark.parametrize(
    ("raw", "title", "body"),
    [
        ("---\r\ntitle: T\r\n---\r\nB", "T", "B"),
        ("---\ntitle: T\n---", "T", ""),
        ("---\n---\nText", "A", "Text"),
    ],
)
def test_parse_document_front_matter_edge_cases(raw: str, title: str, body: str) -> None:
    document = parse_document("a.md", raw)
    assert (document.title, document.body) == (title, body)


def test_clean_markdown_keeps_marks_inside_words_and_numbers() -> None:
    assert clean_markdown("use snake_case_name and 2*3*4") == "use snake_case_name and 2*3*4"
    assert clean_markdown("a **bold** and _soft_ word") == "a bold and soft word"


def test_clean_markdown_decodes_html_entities() -> None:
    assert clean_markdown("Arts &amp; Humanities") == "Arts & Humanities"


def test_clean_markdown_keeps_only_the_number_of_a_footnote_mark() -> None:
    assert clean_markdown("Ein Satz.$^{12}$ Weiter.") == "Ein Satz. 12 Weiter."


def test_clean_markdown_removes_markdown_escapes() -> None:
    assert clean_markdown(r"ILO-SR\_N1\_engl.pdf and 1\. item") == "ILO-SR_N1_engl.pdf and 1. item"
