"""Packaging test: build the wheel and verify the installed console command.

Builds the project into a temporary directory, installs it into a clean
virtualenv without dev dependencies, and exercises the installed ``llmprobe``
console script: the ``--help`` command exits 0, and a ``--safe`` run against a
nonexistent address fails with exit code 2 and a clear message (never a raw
traceback). The install step is hermetic (``--no-index --find-links`` against
the wheels built into the temp dir), so no network or live server is needed at
install time.

The build step deliberately runs with dependency wheels (``pip wheel .``) so
the resulting ``--find-links`` directory is a complete, self-contained install
source for the fresh virtualenv. Only runtime dependencies are installed — the
dev extras (pytest, fastapi, …) are intentionally excluded, proving the console
script works with the production dependency set alone.
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WHEEL_GLOB = "llmprobe-*.whl"
UNREACHABLE_ADDR = "http://127.0.0.1:9"


def _run(cmd: list[str], *args, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, *args, capture_output=True, text=True, **kwargs)


def _build_and_install(tmp_path: Path) -> Path:
    """Build a wheel and install it (no dev deps) into a fresh venv.

    Returns the path to the installed ``llmprobe`` console script. The venv
    is created with only the wheel's runtime dependencies — no dev extras —
    exactly the surface an end user would get from ``pip install llmprobe``.
    """
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
    return console


def test_packaged_llmprobe_runs_help(tmp_path: Path) -> None:
    """A built wheel installs cleanly and ``llmprobe --help`` exits 0."""
    console = _build_and_install(tmp_path)

    result = _run([str(console), "--help"])
    assert result.returncode == 0, f"llmprobe --help exited {result.returncode}"
    assert result.stdout.strip(), "llmprobe --help produced empty output"


def test_packaged_llmprobe_reports_unreachable_server(tmp_path: Path) -> None:
    """A ``--safe`` run against an unreachable server exits 2 with a clear message.

    The installed console script must fail cleanly against a nonexistent
    address: exit code 2 (an operational failure, distinguishable from the
    exit-1 mismatch severity) and an explanatory line on stderr — not a Python
    traceback leaking to the user.
    """
    console = _build_and_install(tmp_path)

    result = _run([str(console), UNREACHABLE_ADDR, "--safe", "--timeout", "2"])
    assert result.returncode == 2, (
        f"expected exit code 2 against unreachable server, got "
        f"{result.returncode}:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Traceback" not in result.stderr, (
        f"unreachable server leaked a traceback:\n{result.stderr}"
    )
    assert result.stderr.strip(), "llmprobe produced no error message on stderr"
