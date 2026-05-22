# Project Guidelines

## Package Manager

Use `uv` as the default package manager for all Python operations in this project.

- Install packages: `uv add <package>`
- Install from requirements.txt: `uv pip install -r requirements.txt`
- Run scripts: `uv run <script.py>`
- Create venv: `uv venv`
- Sync environment: `uv sync`

Do NOT use `pip` or `pip install` directly. Always prefer `uv`.

## Project Structure

All source code lives under `src/core/`. Run scripts from the project root as modules:

```
python -m src.core.run
```

## Code Requirements

### Required
- Type hints on all function parameters and return types.
- Docstrings in English for every class and every method.
- Descriptive names for methods, variables, and attributes (e.g. `training_node`, not `node1`).

### Forbidden
- Inline comments in the code.
- Type hints on simple assignments (write `self.model = model`, not `self.model: BaseEstimator = model`).
- Mathematical notation for variable names (use `input_data`, not `X`; use `output_data`, not `y`).

## Tests

Every time a new file is created or an existing one is modified, write or update a corresponding unit test file under `tests/` with the highest possible code coverage.

- Mirror the `src/` structure inside `tests/` (e.g. `src/core/agents/trainer_agent.py` → `tests/core/agents/test_trainer_agent.py`).
- Use `pytest` as the test framework.
- Run tests with: `uv run pytest` or `python -m pytest`.
