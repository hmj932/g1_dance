#!/usr/bin/env python3
"""Gradmotion play entry point.

Installs the whole_body_tracking extension via ``pip install -e`` then hands
off to ``rsl_rl/play.py`` via :func:`os.execv`.

For play-only Flux tasks, Gradmotion mounts ``checkPointFilePath`` under
``checkPointMountPath`` (often ``g1_dance/``). ``play.py`` only looks under
``logs/rsl_rl/<experiment>/``, so this entry stages a found ``model_*.pt``
into that layout before launch.

All CLI arguments are forwarded to play.py, e.g.::

    python scripts/entry_play.py \
        --task=Tracking-Flat-G1-v0 \
        --motion_file motions/dance_zui.npz \
        --num_envs 2 --video --headless
"""

from __future__ import annotations

import glob
import os
import shutil
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


def _find_mounted_checkpoint() -> str | None:
    """Locate a Gradmotion-mounted model_*.pt near the repo / personal mount."""
    clone_root = os.path.dirname(REPO_ROOT)  # .../g1_dance
    search_roots = [
        REPO_ROOT,
        clone_root,
        os.path.join(clone_root, "personal"),
        "/personal",
        "/workspace/isaaclab/g1_dance",
        "/workspace/isaaclab",
    ]
    candidates: list[str] = []
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        for path in glob.glob(os.path.join(root, "model_*.pt")):
            candidates.append(path)
        for path in glob.glob(os.path.join(root, "**", "model_*.pt"), recursive=True):
            # Prefer shallow mounts; skip already-staged logs copies later via dedupe.
            candidates.append(path)

    # De-dupe while preferring non-logs paths (fresh mounts).
    uniq: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        uniq.append(path)

    if not uniq:
        return None

    def score(path: str) -> tuple[int, int, float]:
        name = os.path.basename(path)
        # Prefer model_4999 over model_500; prefer outside logs/
        try:
            iter_n = int(name.replace("model_", "").replace(".pt", ""))
        except ValueError:
            iter_n = -1
        in_logs = 1 if "/logs/" in path.replace("\\", "/") else 0
        return (in_logs, -iter_n, -os.path.getmtime(path))

    uniq.sort(key=score)
    chosen = uniq[0]
    print(f"[entry] found mounted checkpoint candidates={len(uniq)}; chosen={chosen}")
    return chosen


def _stage_checkpoint_for_play(src_pt: str) -> list[str]:
    """Copy mounted pt into logs/rsl_rl/g1_flat/gm_mounted/ and return play CLI extras."""
    fname = os.path.basename(src_pt)
    # Normalize upload names like model_4999_20260811....pt -> model_4999.pt if possible
    if fname.startswith("model_") and fname.endswith(".pt"):
        body = fname[len("model_") : -len(".pt")]
        digits = "".join(ch for ch in body.split("_")[0] if ch.isdigit())
        if digits:
            fname = f"model_{digits}.pt"

    dest_dir = os.path.join(REPO_ROOT, "logs", "rsl_rl", "g1_flat", "gm_mounted")
    os.makedirs(dest_dir, exist_ok=True)
    dest_pt = os.path.join(dest_dir, fname)
    if os.path.realpath(src_pt) != os.path.realpath(dest_pt):
        print(f"[entry] staging checkpoint {src_pt} -> {dest_pt}")
        shutil.copy2(src_pt, dest_pt)
    else:
        print(f"[entry] checkpoint already staged at {dest_pt}")

    return ["--load_run", "gm_mounted", "--checkpoint", fname]


def main() -> None:
    # Change working directory to whole_body_tracking/ so relative paths resolve
    os.chdir(REPO_ROOT)

    # Step 1 — install the extension (creates an importable egg-link)
    ext_path = os.path.join(REPO_ROOT, "source", "whole_body_tracking")
    _pip_install_editable(ext_path)

    # Step 2 — stage Gradmotion-mounted checkpoint if present
    cli_args = sys.argv[1:]
    has_load = any(a == "--load_run" or a.startswith("--load_run=") for a in cli_args)
    has_ckpt = any(a == "--checkpoint" or a.startswith("--checkpoint=") for a in cli_args)
    if not (has_load and has_ckpt):
        mounted = _find_mounted_checkpoint()
        if mounted:
            cli_args = cli_args + _stage_checkpoint_for_play(mounted)
        else:
            print("[entry] WARN: no mounted model_*.pt found; play.py will use default log lookup")

    # Step 3 — hand off to play.py (replaces this process)
    play_script = os.path.join(SCRIPT_DIR, "rsl_rl", "play.py")
    print(f"[entry] launching {play_script} {' '.join(cli_args)}")
    os.execv(sys.executable, [sys.executable, play_script] + cli_args)


if __name__ == "__main__":
    main()
