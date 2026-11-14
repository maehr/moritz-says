# Agents Tooling Specification — moritz-says

This repository is a static semantic search page for the publications of Moritz Mähr. It is a marimo WebAssembly notebook on GitHub Pages.

It follows the shared Python specification, `10-AGENTS.python.md`. `SPEC.md` is the product specification.
Sections 1, 3, and 6 name only the tools that this repository uses.
These parts of the shared specification are not set up: prek, commitizen, git-cliff, CodeQL, dependency review, and `CODEOWNERS`.

## 1. Tooling

Pin versions in `pyproject.toml`. Commit `uv.lock`. Install with `uv sync --locked`. Never ship a development tool to runtime.

**Development.** [uv](https://docs.astral.sh/uv/) dependencies, environments, and the lockfile · [ruff](https://docs.astral.sh/ruff/) lint and format · [ty](https://docs.astral.sh/ty/) types, pinned exactly while it stays in beta · [pytest](https://docs.pytest.org/) with [pytest-cov](https://pytest-cov.readthedocs.io/) tests, doctests, and coverage · [Playwright](https://playwright.dev/python/) browser tests.

**Runtime.** [Pydantic](https://pydantic.dev/) v2 validation · [marimo](https://docs.marimo.io/) reactive notebook, exported with `html-wasm`.

**Index build.** [sentence-transformers](https://sbert.net/) in the `index` dependency group. The default environment and CI do not install this group.

## 2. Standards

* [SemVer 2.0.0](https://semver.org/) for versions.
* [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages.
* [Contributor Covenant 3.0](https://www.contributor-covenant.org/version/3/0/code_of_conduct/) as `CODE_OF_CONDUCT.md`.
* [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html) as `LICENSE`. SPDX identifier `AGPL-3.0-only`.

## 3. Code

* Use a type hint everywhere.
* Use Pydantic at an I/O boundary. The index file is the I/O boundary of this repository.
* Use a dataclass for an internal carrier. Keep core logic in pure functions.
* Put a runnable example in each docstring.
* Require 100% coverage on core logic. Keep each exclusion narrow and explicit.
* The browser gets `app.py`, `site.css`, `head.html`, and `public/`. Put browser logic in a top-level function of `app.py`, not in a cell.
* Host every font and image in `public/`. Never link a third-party host.
* Never edit `public/embeddings.json` or `tests/fixtures/reference_vectors.json` by hand. Run the index build.
* Convert a PDF for `content/` with the `pdf2md` MCP server.

Run this gate before each commit. CI runs the same gate.

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
uv run marimo check app.py --select MW
```

`uv run pytest` includes the doctests and requires 100% coverage.

Other commands:

```bash
uv run --group index python -m scripts.build_index          # Build the index and the fixture.
uv run python -m scripts.export_site                        # Export the site to dist/.
uv run pytest -m browser --no-cov                           # Browser tests. They need dist/ and a network.
```

## 4. GitHub Workflow

Topic branch and pull request, unless this file says `Git workflow: main`. No push access: `gh repo fork OWNER/REPO --clone`.

PR title: a Conventional Commit. One logical change per PR.

## 5. CI/CD Security

Follow [secure use of Actions](https://docs.github.com/en/actions/reference/security/secure-use).

* Set `permissions: contents: read` at the top level. Widen it per job only where a job needs more.
* Default the repository token to read: `gh api -X PUT repos/OWNER/REPO/actions/permissions/workflow -f default_workflow_permissions=read`.
* Pin an action to a full commit SHA. Verify the SHA against the upstream repository, not a fork. Let Dependabot bump it.
* Never check out fork code under `pull_request_target`. Prefer `workflow_run`, and treat its artifacts as untrusted.
* Never interpolate `github.event.*` into `run:`. Pass the value through `env:` and quote `"$VAR"`.
* Use OIDC and a short-lived cloud role. Do not store a long-lived secret. Keep a secret a scalar, never a JSON blob. Rotate it.
* Gate a deploy on an Environment with required reviewers. Prefer an environment secret over a repository secret.
* Do not use a self-hosted runner on a public repository.
* Cover `.github/workflows/**` in `CODEOWNERS`.
* Require the section 3 gate, CodeQL, and `dependency-review-action`. Run the same gate locally, so CI gives no surprise. On a private repository, confirm the plan covers CodeQL and dependency review first.

## 6. Browser and WASM

`app.py` runs in a browser with Pyodide.

* Ship pure Python, or a verified Pyodide or PyPI WASM wheel. Never ship a development tool.
* Do not use an OS socket, a subprocess, or a thread.
* Use `pyodide.http.pyfetch` for HTTP.
* Keep JavaScript to the `EMBEDDER_JS` constant. Pass each value as an argument. Never put a value into JavaScript source.
* Verify each change of the JavaScript bridge with the browser tests, not only with the unit tests.
