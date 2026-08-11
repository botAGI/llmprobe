"""Packaging test: build the wheel and verify the installed console command.

Builds the project into a temporary directory, installs it into a clean
virtualenv, and runs the ``llmprobe --help`` console command, asserting that
the process exits 0 and that stdout is non-empty. The install step is hermetic
(``--no-index --find-links`` against the wheels built into the temp dir), so
no network or live server is needed at install time.

The build step deliberately runs with dependency wheels (``pip wheel .``) so
the resulting ``--find-links`` directory is a complete, self-contained install
source for the fresh virtualenv.
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WHEEL_GLOB = "llmprobe-*.whl"


def _run(cmd: list[str], *args, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, *args, capture_output=True, text=True, **kwargs)


def test_packaged_llmprobe_runs_help(tmp_path: Path) -> None:
    """A built wheel installs cleanly and ``llmprobe --help`` exits 0."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    build = _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "-w",
            str(build_dir),
        ],
        cwd=str(PROJECT_ROOT),
    )
    assert build.returncode == 0, f"wheel build failed:\n{build.stdout}\n{build.stderr}"

    wheels = list(build_dir.glob(WHEEL_GLOB))
    assert len(wheels) == 1, f"expected exactly one llmprobe wheel, got {wheels}"
    wheel = wheels[0]

    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True)
    venv_python = venv_dir / ("bin" if sys.platform != "win32" else "Scripts") / "python"

    install = _run(
        [str(venv_python), "-m", "pip", "install", "--no-index", "--find-links", str(build_dir), str(wheel)]
    )
    assert install.returncode == 0, (
        f"wheel install failed:\n{install.stdout}\n{install.stderr}"
    )

    console = venv_dir / ("bin" if sys.platform != "win32" else "Scripts") / "llmprobe"
    assert console.is_file(), f"console script not installed at {console}"

    result = _run([str(console), "--help"])
    assert result.returncode == 0, f"llmprobe --help exited {result.returncode}"
    assert result.stdout.strip(), "llmprobe --help produced empty output"
