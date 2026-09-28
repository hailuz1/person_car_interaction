"""CLI entry point for the causal, real-time person-car interaction system.

Examples:
    python run.py --clip clip1                 # live cv2 window ('q' quits) + mp4
    python run.py --clip clip1 --no-display    # headless, still writes json + mp4
    python run.py --clip clip1 --no-save       # window only, no mp4
    python run.py --all                        # run every clip in data/given
"""

import argparse
from dataclasses import replace
from pathlib import Path

from config import Config
from src.runner import run


def main():
    """Parse CLI arguments and run one clip (or every clip with --all).

    Args:
        None (reads sys.argv via argparse).
    Returns:
        None.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", default="clip1", help="clip stem in data/given (e.g. clip1)")
    p.add_argument("--data-dir", default=None,
                   help="override the input video folder (e.g. ../data/test)")
    p.add_argument("--all", action="store_true", help="process every clip in data_dir")
    p.add_argument("--no-display", action="store_true", help="do not open a cv2 window")
    p.add_argument("--no-save", action="store_true", help="do not write an annotated mp4")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--no-json", action="store_true",
                   help="do not write event.json / diagnostics.json")
    a = p.parse_args()

    cfg = Config()
    if a.data_dir:
        cfg = replace(cfg, data_dir=Path(a.data_dir).expanduser().resolve())
    common = dict(cfg=cfg, display=not a.no_display, save=not a.no_save,
                  max_frames=a.max_frames,
                  write_json=not a.no_json)

    if a.all:
        stems = sorted(v.stem for v in cfg.data_dir.iterdir()
                       if v.suffix.lower() in cfg.video_exts)
        if not stems:
            raise SystemExit(f"no videos found in {cfg.data_dir}")
        for stem in stems:
            run(stem=stem, **common)
    else:
        run(stem=a.clip, **common)


if __name__ == "__main__":
    main()
