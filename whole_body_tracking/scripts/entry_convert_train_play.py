#!/usr/bin/env python3
"""Gradmotion entry: CSV→NPZ convert, then train, then Isaac play.

``gm-run`` only allows a single start script (no ``&&``). This wrapper:

1. ``pip install -e`` the extension (mirrors + retries)
2. run ``csv_to_npz.py`` → local ``--output_file`` (+ optional ``/personal`` copy)
3. run ``rsl_rl/train.py`` to completion
4. run ``rsl_rl/play.py --video --headless`` on the latest checkpoint

Convert-specific CLI flags are stripped before train/play::

    --input_file --input_fps --frame_range --output_name --output_file
    --no_wandb --output_fps

Training still needs ``--motion_file`` pointing at the converted npz (same path
as ``--output_file``).

Example::

    gm-run g1_dance/whole_body_tracking/scripts/entry_convert_train_play.py \\
        --task=Tracking-Flat-G1-v0 \\
        --input_file motions/csv/dance2_subject3.csv --input_fps 30 \\
        --output_name dance2_subject3 \\
        --output_file motions/dance2_subject3.npz --no_wandb \\
        --motion_file motions/dance2_subject3.npz \\
        --headless --num_envs 4096 --max_iterations 5000
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

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

# Convert-only flags (not forwarded to train/play).
_CONVERT_FLAGS = {
    "--input_file",
    "--input_fps",
    "--frame_range",
    "--output_name",
    "--output_file",
    "--no_wandb",
    "--output_fps",
}

# Train-only flags (not forwarded to play). Same as entry_train_and_play.
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


def _pip_install_editable(ext_path: str) -> None:
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
            try:
                subprocess.check_call(cmd, env=env)
                return
            except subprocess.CalledProcessError as exc:
                last_err = exc
                print(f"[entry] pip failed exit={exc.returncode}; retrying...")
                time.sleep(2)

    raise RuntimeError(f"pip install -e failed after retries: {last_err}")


def _split_argv(argv: list[str]) -> tuple[list[str], list[str], str | None]:
    """Return (convert_args, train_args, output_file)."""
    convert_args: list[str] = []
    train_args: list[str] = []
    output_file: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        key = arg.split("=", 1)[0]
        if key in _CONVERT_FLAGS:
            if key == "--no_wandb":
                convert_args.append(arg)
                i += 1
                continue
            if "=" in arg:
                convert_args.append(arg)
                if key == "--output_file":
                    output_file = arg.split("=", 1)[1]
                i += 1
                continue
            # flag + value(s)
            convert_args.append(arg)
            i += 1
            if key == "--frame_range":
                # two ints
                for _ in range(2):
                    if i < len(argv) and not argv[i].startswith("-"):
                        convert_args.append(argv[i])
                        i += 1
            elif i < len(argv) and not argv[i].startswith("-"):
                convert_args.append(argv[i])
                if key == "--output_file":
                    output_file = argv[i]
                i += 1
            continue
        train_args.append(arg)
        i += 1
    return convert_args, train_args, output_file


def _strip_train_only(argv: list[str]) -> list[str]:
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


def _persist_npz(output_file: str) -> None:
    """Best-effort copy to /personal so later Flux tasks can remount it."""
    if not output_file or not os.path.isfile(output_file):
        print(f"[entry] skip /personal copy; missing {output_file}")
        return
    try:
        os.makedirs("/personal/motions", exist_ok=True)
    except OSError as exc:
        print(f"[entry] cannot mkdir /personal/motions: {exc}")
        return
    dest = os.path.join("/personal", "motions", os.path.basename(output_file))
    try:
        shutil.copy2(output_file, dest)
        print(f"[entry] copied npz → {dest}")
    except OSError as exc:
        print(f"[entry] /personal copy failed: {exc}")


def main() -> None:
    os.chdir(REPO_ROOT)

    ext_path = os.path.join(REPO_ROOT, "source", "whole_body_tracking")
    _pip_install_editable(ext_path)

    convert_args, train_args, output_file = _split_argv(sys.argv[1:])
    if not convert_args:
        raise SystemExit(
            "[entry] missing convert args; need at least "
            "--input_file ... --output_name ... --output_file ... --no_wandb"
        )

    # Ensure headless for convert Isaac app.
    if "--headless" not in convert_args and "--headless" not in train_args:
        convert_args.append("--headless")
    elif "--headless" not in convert_args and "--headless" in train_args:
        convert_args.append("--headless")

    convert_script = os.path.join(SCRIPT_DIR, "csv_to_npz.py")
    print(f"[entry] CONVERT: {convert_script} {' '.join(convert_args)}")
    subprocess.check_call([sys.executable, convert_script] + convert_args)

    if output_file:
        _persist_npz(output_file)
        if not os.path.isfile(output_file):
            raise SystemExit(f"[entry] convert finished but npz missing: {output_file}")

    train_script = os.path.join(SCRIPT_DIR, "rsl_rl", "train.py")
    print(f"[entry] TRAIN: {train_script} {' '.join(train_args)}")
    subprocess.check_call([sys.executable, train_script] + train_args)

    play_script = os.path.join(SCRIPT_DIR, "rsl_rl", "play.py")
    play_args = _strip_train_only(train_args)
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
    print(f"[entry] PLAY: {' '.join(play_cmd)}")
    subprocess.check_call(play_cmd)
    print("[entry] convert + train + play finished")


if __name__ == "__main__":
    main()
