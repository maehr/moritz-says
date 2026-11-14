# Specification: Static semantic search with marimo WASM and GitHub Pages

**Status:** Proposed MVP  
**Target:** Public, read-only search over a small, curated document collection  
**Hosting:** GitHub Pages; no application server

## 1. Goal

Publish a search page that retrieves relevant passages by meaning rather than only by exact keyword. All document processing and embedding generation happens before deployment. On the page, marimo runs Python in Pyodide; a browser-compatible ONNX model computes query embeddings; Python ranks the passages.

**Success condition:** A visitor can open a GitHub Pages URL, enter a natural-language query, and see relevant passages with document attribution without sending the query to an inference API.

## 2. Scope and assumptions

**In scope**
- A manually curated collection of Markdown or plain-text documents.
- Up to approximately 500 searchable passages in MVP; linear ranking, no vector database.
- Static document passages and precomputed embeddings committed as a versioned JSON asset.
- One query field, submit action, results grouped by work (up to five works with up to five passages each), source title, excerpt, and optional source link.
- GitHub Actions validation and automatic publication from `main`.

**Out of scope**
- User uploads, authentication, editing, live indexing, crawling, analytics, server APIs, generative answers, and chat.
- Guaranteed offline support on a first visit. The WASM runtime and model can require external downloads.
- Search over confidential or access-restricted documents. Published assets are publicly inspectable.

**Default model assumption:** English-language passages. Use `sentence-transformers/all-MiniLM-L6-v2` for build-time vectors and its browser-compatible `Xenova/all-MiniLM-L6-v2` ONNX variant for query vectors. If the corpus is German or multilingual, choose and verify a multilingual model before building the index. Both model IDs and immutable revisions must be pinned in configuration.

## 3. Architecture

```text
Offline / developer machine                    Static website / visitor's browser
content/*.md                                   GitHub Pages
    |                                               |
scripts/build_index.py                         index.html + marimo assets
    |                                               |
chunk + embed (Python)                         marimo UI (Pyodide / WASM)
    |                                               |
public/embeddings.json  --------------------> fetch static JSON
                                                    |
                                              query -> Transformers.js / ONNX WASM
                                                    |
                                              normalized query vector
                                                    |
                                              Python dot-product ranking
                                                    |
                                              top five results
```

The only ongoing host is GitHub Pages. External hosting may supply the marimo/Pyodide runtime, JavaScript package, and ONNX model; none of these is an inference service. The model executes locally in the browser.

## 4. Repository layout

```text
.
├── AGENTS.md                         # Copy the supplied Python stack specification
├── CLAUDE.md                         # Contains only: @AGENTS.md
├── app.py                            # marimo UI and ranking
├── site.css                          # Page styles
├── head.html                         # Page metadata
├── content/                          # Markdown/plain-text sources
├── public/
│   ├── embeddings.json              # Generated, checked-in search index
│   ├── fonts/                       # Self-hosted fonts
│   └── img/                         # Self-hosted images
├── scripts/
│   └── build_index.py               # Offline chunking and embedding
├── tests/
│   ├── test_index.py                 # Schema, chunking, score tests
│   └── test_integration.py           # Compatibility fixtures / contracts
├── pyproject.toml
├── uv.lock
└── .github/workflows/pages.yml
```

Use the repository's provided Python tooling convention: `uv`, `ruff`, `ty`, `pytest`, and `marimo`. Keep native Python embedding dependencies **out** of browser installation. If embedded in notebook metadata, mark incompatible dependencies as non-WASM (`sys_platform != 'emscripten'`).

## 5. Index contract

File: `public/embeddings.json` (UTF-8, deterministic JSON).

```json
{
  "schema_version": 1,
  "model": {
    "document_model": "sentence-transformers/all-MiniLM-L6-v2",
    "document_revision": "PINNED_COMMIT",
    "query_model": "Xenova/all-MiniLM-L6-v2",
    "query_revision": "PINNED_COMMIT",
    "dimensions": 384,
    "pooling": "mean",
    "normalized": true
  },
  "chunks": [
    {
      "id": "intro:0001",
      "title": "Introduction",
      "text": "The full searchable passage.",
      "source": "intro.md",
      "url": null,
      "embedding": [0.012, -0.031]
    }
  ]
}
```

The two-element embedding above is **illustrative only**; actual embeddings must contain exactly 384 finite numbers.

**Validation rules**
- Require `schema_version == 1`; reject unknown versions.
- Require unique, stable chunk IDs; nonempty `title` and `text`.
- Require every embedding to have the configured dimensionality, only finite numeric values, and L2 norm approximately 1 (tolerance 0.01).
- Optional links must be relative to the site or use approved `https://` URLs; do not render arbitrary HTML from source text.
- Refuse to start search on a malformed index. Display a clear load error.
- The index file is regenerated from sources, never manually edited.

## 6. Offline indexing

1. Read the curated text files in stable filename order.
2. Extract title and optional canonical URL from front matter, or use a deterministic filename-based title.
3. Split at natural paragraph/sentence boundaries into passages targeted at 120–180 model tokens, with at most approximately 20-token overlap. Never exceed the selected model's effective input length; the MiniLM sentence-transformer configuration uses a 256-token maximum.
4. Generate embeddings using the pinned Python model, with normalization enabled.
5. Serialize metadata, passages, and embedding vectors in stable order to `public/embeddings.json`.
6. Validate the generated index before committing it.

Document changes require re-running the index generator and committing the updated JSON. The website does not embed documents during visits. Keep a fixture containing several sample sentences and their reference vectors to detect accidental changes to normalization, model revisions, tokenization, or pooling.

## 7. Browser application

**Framework:** marimo in `html-wasm` **run** mode, with `app.py` as the sole notebook/application entry point.

**Behavior**
1. Render a heading, explanation, search field, and submit button.
2. Load `public/embeddings.json` from the exported notebook location using `mo.notebook_location()` so the app works under the GitHub Pages repository subpath.
3. Validate the index and display a loading/ready/error state.
4. Lazy-load the pinned JavaScript `feature-extraction` model on the first submitted query (or explicitly initialize it with a visible loading state). Use the WASM execution provider and a verified quantization setting such as `q8`.
5. Embed the query with `pooling: 'mean'` and `normalize: true`.
6. Convert the resulting typed vector to a Python sequence through a tested Pyodide/JavaScript bridge. Do not assume an arbitrary CDN import or `js.eval` example works without a browser integration test.
7. Compute `score = sum(query[i] * chunk.embedding[i])` for every passage; because vectors are normalized, this is cosine similarity.
8. Sort by descending score, breaking ties by chunk ID. Group the passages by work. Show the work of the best passage first, at most five works, and at most five passages for each work.
9. Render source titles and text safely as plain text, and provide clickable source URLs when present. Label similarity as a ranking score, not a probability.

**Edge cases:** Blank or whitespace-only queries do not invoke the model. A missing/empty index shows a message. If model download or inference fails, show an error with a retry option. Repeated searches reuse the model instance. Disable duplicate submissions while a query runs.

**Implementation boundary:** Keep the marimo application and ranking in Python. Limit JavaScript to model initialization/inference and conversion of the resulting vector. The JS module bridge must be verified in the exported WASM page, not only in a normal local Python process.

## 8. Export and deployment

**Local check:**

```bash
uv sync --locked
uv run marimo check app.py --select MW
uv run marimo export html-wasm app.py -o dist --mode run
python -m http.server --directory dist 8000
```

Test using `http://localhost:8000/`, never by opening `dist/index.html` with `file://`.

**GitHub Actions workflow** (`.github/workflows/pages.yml`):
- Run checks and export on pull requests and pushes to `main`; allow optional manual dispatch. Deploy only on `main`.
- Use pinned actions (full commit SHAs), pinned Python dependencies via `uv.lock`, and the configured runtime version.
- Run the repository validation gate and `marimo check --select MW`.
- Export with `uv run python -m scripts.export_site`. It runs `marimo export html-wasm` and removes the default description.
- Ensure `dist/index.html`, `dist/public/embeddings.json`, and required marimo assets exist; place `.nojekyll` in `dist/` if needed.
- Upload `dist/` with `actions/upload-pages-artifact`; deploy via `actions/deploy-pages` in a dependent job.
- Set `contents: read` globally; grant `pages: write` and `id-token: write` to the deployment job only. Use the `github-pages` environment.
- In repository settings, set Pages source to **GitHub Actions**.

All application asset references must work under `https://OWNER.github.io/REPO/`, not only `/` at a domain root.

## 9. Security, reliability, and quality

- No inference API, application credentials, database, or runtime write endpoint.
- Queries and document text must not be transmitted to an inference provider; model *files* may be downloaded from approved external hosts. If strict no-external-request operation is needed, add an explicit later requirement to self-host and cache all assets.
- Treat documents, their URLs, and generated JSON as untrusted input. Validate and escape content; avoid raw HTML interpolation and unsafe dynamic JavaScript evaluation.
- Pin marimo, Transformers.js, model revisions, GitHub Actions, and Python dependencies to reviewed versions.
- Show a distinct loading state for the first model download. Never imply the application works offline on first visit.
- Prefer deterministic builds. Commit the generated index, but not the exported `dist/` directory.
- Keep all search logic in a small pure Python function so it can be unit-tested without marimo or a browser.

## 10. Acceptance criteria

- **A1 Static:** After deployment, the public Pages URL works with no custom backend; browser network inspection shows no request to an embedding inference endpoint.
- **A2 Deployment:** Direct navigation and page reload work at the `/REPO/` URL, including loading `public/embeddings.json`.
- **A3 Search:** A reference semantic query (different wording from the target passage) ranks the expected passage within the top three passages. The page shows it in the results.
- **A4 Compatibility:** For fixture sentences, Python and browser embeddings both have 384 dimensions, near-unit norm, and cosine similarity at least 0.98 to each other. A mismatch blocks release.
- **A5 Ranking:** Equal normalized vectors score approximately 1; a fixed test set ranks deterministically; no vector database is used.
- **A6 Failure states:** Empty query, missing/invalid JSON, failed model download, and empty result set all produce usable UI states without uncaught exceptions.
- **A7 Privacy:** Visitors can inspect all published texts and embeddings; the README explicitly warns not to publish confidential content.
- **A8 Browser:** The exported page is tested in current desktop Chromium and Firefox with real WASM inference, not only with mocked model output.
- **A9 CI:** Checks and export run on PRs; only `main` deploys; deployment requires the build job to succeed.

## 11. Implementation order

1. Create a 3-document fixture corpus and generate a valid index with pinned model revisions.
2. Build the marimo search UI and pure Python ranker; test locally.
3. Prove the JS/ONNX embedding bridge works in an actual `html-wasm` export; verify cross-runtime vector equivalence before scaling the corpus.
4. Add failure states and safe rendering.
5. Add unit and browser smoke tests, CI export, and GitHub Pages deployment.
6. Replace fixture content with the real public corpus and regenerate the index.

## 12. References

- marimo WebAssembly export: https://docs.marimo.io/guides/exporting/webassembly_html/
- marimo WASM data files and `public/`: https://docs.marimo.io/guides/wasm/
- marimo GitHub Pages publishing: https://docs.marimo.io/guides/publishing/github/
- Transformers.js embedding pipeline: https://huggingface.co/docs/transformers.js/api/pipelines
- Browser ONNX model: https://huggingface.co/Xenova/all-MiniLM-L6-v2
- GitHub Pages custom workflows: https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages
