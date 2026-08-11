#!/usr/bin/env python3
"""Gradmotion entry: train, then Isaac play with video.

``gm-run`` only allows a single start script (no ``&&`` chaining). This wrapper
mirrors the old X1 "hang play.py after train" workflow:

1. ``pip install -e`` the extension
2. run ``rsl_rl/train.py`` to completion
3. run ``rsl_rl/play.py --video --headless`` on the latest checkpoint

Forward shared CLI args to both scripts. Play-only extras (``--video``,
``--video_length``, ``--num_envs`` for play) can be overridden via env::

    GM_PLAY_NUM_ENVS=2
    GM_PLAY_VIDEO_LENGTH=500
"""

from __future__ import annotations

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Args that are train-specific and should not be forwarded to play.
_TRAIN_ONLY = {
    "--max_iterations",
    "--seed",
    "--video_interval",
    "--registry_name",
    "--resume",
    "--logger",
    "--log_project_name",
    "--run_name",
    "--experiment_name",
}


def _strip_train_only(argv: list[str]) -> list[str]:
    """Keep shared flags for play; drop train-only knobs."""
    out: list[str] = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        key = arg.split("=", 1)[0]
        if key in _TRAIN_ONLY:
            if "=" not in arg and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                skip_next = True
            continue
        # Drop bare --video from train args; play always records.
        if arg == "--video":
            continue
        if key == "--video_length":
            if "=" not in arg and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                skip_next = True
            continue
        if key == "--num_envs":
            if "=" not in arg and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                skip_next = True
            continue
        out.append(arg)
    return out


def main() -> None:
    os.chdir(REPO_ROOT)

    ext_path = os.path.join(REPO_ROOT, "source", "whole_body_tracking")
    print(f"[entry] pip install -e {ext_path}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ext_path])

    train_script = os.path.join(SCRIPT_DIR, "rsl_rl", "train.py")
    play_script = os.path.join(SCRIPT_DIR, "rsl_rl", "play.py")
    cli_args = sys.argv[1:]

    print(f"[entry] TRAIN: {train_script} {' '.join(cli_args)}")
    subprocess.check_call([sys.executable, train_script] + cli_args)

    play_args = _strip_train_only(cli_args)
    play_num_envs = os.environ.get("GM_PLAY_NUM_ENVS", "2")
    play_video_length = os.environ.get("GM_PLAY_VIDEO_LENGTH", "500")
    play_cmd = [
        sys.executable,
        play_script,
        *play_args,
        "--num_envs",
        play_num_envs,
        "--video",
        "--video_length",
        play_video_length,
        "--headless",
    ]
    # Ensure headless is present even if train already passed it.
    if "--headless" not in play_args:
        pass  # already appended
    print(f"[entry] PLAY: {' '.join(play_cmd)}")
    subprocess.check_call(play_cmd)
    print("[entry] train + play finished")


if __name__ == "__main__":
    main()
