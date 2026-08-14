#!/usr/bin/env python3
"""Gradmotion entry point.

Installs the whole_body_tracking extension via ``pip install -e`` then hands
off to ``rsl_rl/train.py`` via :func:`os.execv` so that Isaac Lab's
``AppLauncher`` sees a clean process.

All CLI arguments are forwarded to train.py, e.g.::

    python scripts/entry_train.py \
        --task=Tracking-Flat-G1-v0 \
        --motion_file motions/dance_zui.npz \
        --headless --num_envs 4096
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Flux nodes sometimes time out on files.pythonhosted.org; retry with mirrors.
_PIP_MIRRORS: list[list[str]] = [
    [],
    [
        "-i",
        "https://pypi.tuna.tsinghua.edu.cn/simple",
        "--trusted-host",
        "pypi.tuna.tsinghua.edu.cn",
    ],
    [
        "-i",
        "https://mirrors.aliyun.com/pypi/simple/",
        "--trusted-host",
        "mirrors.aliyun.com",
    ],
]


def _pip_install_editable(ext_path: str) -> None:
    """Install extension with timeout bumps, mirrors, and no-build-isolation fallback."""
    env = os.environ.copy()
    env.setdefault("PIP_DEFAULT_TIMEOUT", "120")
    last_err: BaseException | None = None

    for mirror_idx, mirror_args in enumerate(_PIP_MIRRORS):
        for no_iso in (False, True):
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                ext_path,
                "--default-timeout",
                "120",
            ]
            if no_iso:
                cmd.append("--no-build-isolation")
            cmd.extend(mirror_args)
            label = f"mirror={mirror_idx} no_build_isolation={no_iso}"
            print(f"[entry] pip install -e ({label})")
            print(f"[entry] {' '.join(cmd)}")
            try:
                subprocess.check_call(cmd, env=env)
                return
            except subprocess.CalledProcessError as exc:
                last_err = exc
                print(f"[entry] pip failed exit={exc.returncode}; retrying...")
                time.sleep(2)

    assert last_err is not None
    raise last_err


def main() -> None:
    # Change working directory to whole_body_tracking/ so relative paths resolve
    os.chdir(REPO_ROOT)

    # Step 1 — install the extension (creates an importable egg-link)
    ext_path = os.path.join(REPO_ROOT, "source", "whole_body_tracking")
    _pip_install_editable(ext_path)

    # Step 2 — hand off to train.py (replaces this process)
    train_script = os.path.join(SCRIPT_DIR, "rsl_rl", "train.py")
    cli_args = sys.argv[1:]
    print(f"[entry] launching {train_script} {' '.join(cli_args)}")
    os.execv(sys.executable, [sys.executable, train_script] + cli_args)


if __name__ == "__main__":
    main()
