# Kripa

Kripa is a personal AI assistant project. This repository currently contains
only a small Python application scaffold; AI and assistant capabilities have
not been implemented yet.

## Getting started

Use Python 3.10 or newer. From the project root, install the (currently empty)
dependency list and run the application:

```bash
python -m pip install -r requirements.txt
python -m kripa
```

The starter prints a short message to confirm the application entry point is
wired up. Add application behavior to the `kripa` package as the project
develops.

## Project layout

- `kripa/` contains the application package.
- `kripa/app.py` holds the top-level application function.
- `kripa/__main__.py` makes the package runnable with `python -m kripa`.
- `kripa/__init__.py` identifies the directory as a package and exposes its
  version.
- `requirements.txt` is the place to list third-party runtime dependencies
  when they are needed.
- `.gitignore` excludes Python-generated files and common local environments.

## Development

Keep modules focused on one responsibility and keep optional integrations out
of the core package until they are needed. There are no third-party runtime
dependencies at this stage.
