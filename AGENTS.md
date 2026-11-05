# Project instructions

## Python

Always use `uv` for Python execution, environments and dependencies in this repository.

- Run scripts with `uv run python scripts/example.py`.
- Run tests with `uv run python -m unittest discover -s tests -v`.
- Use `uv sync`, `uv add`, or `uv pip install` to manage dependencies.
- Use `uv venv` when an explicit environment is needed.
- Do not invoke `python`, `python3`, `pip`, or `pip3` directly from the shell.
- Keep documentation and command examples consistent with this rule.

## Data and attribution

Keep source books, book-derived pairs, credentials and training artifacts local and ignored by Git. Publish code, documentation, configuration and invented examples only. Credit Lochana Perera as the project author. Mention Dan MacKinlay as inspiration and preserve accurate dependency attribution.
