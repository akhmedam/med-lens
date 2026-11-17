# Contributing

Use Python 3.11. Follow the [installation guide](README.md#installation) for the
macOS/Apple Silicon application, then install the development dependencies:

```sh
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

Node.js 22 is required for the annotation page's JavaScript regression check.
Run the same checks as CI before opening a pull request:

```sh
.venv/bin/python -m pip check
.venv/bin/python -m ruff check .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m build --no-isolation
git diff --check
```

Ruff checks syntax, undefined names and invalid control flow. Keep edits focused;
avoid unrelated formatting changes. Use descriptive branch names such as `fix/`
or `feat/` and commit messages that explain the behavior changed.

CI installs the package on Linux, runs offline tests including HTTP, annotation
JavaScript and native grammar fixtures, and verifies the built wheel's resources
and commands outside the checkout. It downloads no models or PubMed data. These
checks do not establish support for inference on Linux or validate answer quality;
the application requires macOS/Apple Silicon and its isolated MLX runtime.

Keep models, caches, source snapshots and raw results out of Git. Preserve frozen
evaluation inputs, hashes and human/AI provenance. Report any inference or
quality checks separately with their exact configuration and limitations.
