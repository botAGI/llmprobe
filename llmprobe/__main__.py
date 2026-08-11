"""Allow ``python -m llmprobe`` to run the CLI.

Delegates to the typer application so the module can be launched directly,
matching the README-promoted ``--safe`` surface and the acceptance check
``python -m llmprobe --help``.
"""

from llmprobe.cli import app

if __name__ == "__main__":
    app()
