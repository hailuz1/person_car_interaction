"""Configuration for the causal, real-time person-car interaction system.

All tunable settings live here as a single dataclass. Time-based thresholds are
given in SECONDS and converted to a per-clip frame count with the clip's FPS
(see :meth:`Config.resolve_for_fps`), so they mean the same real-world duration
regardless of frame rate.

The values below are copied verbatim from the proven offline/POC pipeline so the
streaming results stay identical; only the fields the causal SAM3-seg + provisional
path actually uses are kept.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

# Project root = the folder that contains this package.
_PKG_ROOT = Path(__file__).resolve().parent.parent
# The workspace root (holds data/, track_cache/, src/, runs/ folders).
_WORKSPACE = _PKG_ROOT


@dataclass
class Config:
    """All tunable settings for the causal person-car interaction system."""

    # --- data / IO ---------------------------------------------------------- #
    data_dir: Path = field(default_factory=lambda: _WORKSPACE / "data" / "given")
    output_dir: Path = field(default_factory=lambda: _PKG_ROOT / "runs")
    video_exts: frozenset = frozenset({".mp4", ".avi", ".mov", ".mkv"})

    # Track cache. SAM3 tracking is the only non-deterministic / GPU-heavy step;
    # caching its per-frame boxes + mask descriptors lets every run reuse the
    # exact same tracks (stable ids, reproducible events) with no GPU required.
    # Points at the shared workspace cache so existing seg caches are reused.
    use_track_cache: bool = True
    refresh_tracks: bool = False   # force re-run SAM3 and overwrite the cache
    cache_dir: Path = field(default_factory=lambda: _WORKSPACE / "track_cache")

    # ===================================================================== #
    #  Detection thresholds, grouped by the interaction STATE they gate.
    #  Fields ending in `_sec` are in seconds and resolved to a per-clip frame
    #  count by `resolve_for_fps`; everything else is a ratio/threshold.
    # ===================================================================== #

    # --- SHARED: gate enter AND exit AND loiter ----------------------------- #
    min_track_sec: float = 0.20    # ignore person tracks shorter than this (pre-gate for all)
    near_dist: float = 1.3         # distance (car diagonals) at which closeness -> 0
    car_max_speed_per_sec: float = 0.25  # max ground-relative car speed (diag/sec); `car_stationary`

    # --- ENTER + EXIT: shared by approach and departure --------------------- #
    window_sec: float = 0.17       # start/end span used to judge proximity
    near_th: float = 0.35          # closeness above this = person is "at" the vehicle (`at_car_*`)
    delta: float = 0.18            # min closeness change: approach (enter) / departure (exit)
    on_car_overlap_th: float = 0.15  # min mean person-over-car overlap in the window (emerged/reached)
    big_car_frame_ratio: float = 0.45  # car-box / frame diagonal ratio counted as "big"
    on_car_overlap_th_big: float = 0.03  # relaxed emerged/reached overlap for big cars
    on_car_near_th: float = 0.60  # closeness that alone satisfies emerged/reached
    #   segmentation major-axis gate (drops enter/exit off the car body):
    seg_major_reach: float = 1.35  # allowed |major-axis| offset in car semi-major lengths
    seg_minor_band: float = 1.4    # allowed |minor-axis| offset in car semi-minor lengths
    seg_axis_frac: float = 0.40    # min fraction of window frames that must pass
    seg_strict: bool = False       # reject events whose window has no mask data

    # --- EXIT only ---------------------------------------------------------- #
    edge_margin_sec: float = 0.10  # ignore exit events this soon after clip start (`not_at_clip_edge`)
    exit_diou_drop: float = 0.045  # min start->end DIoU drop that alone confirms departure

    # --- ENTER only --------------------------------------------------------- #
    min_absence_sec: float = 0.2   # person must vanish this long before clip end (`vanishes_mid_clip`)
    move_min: float = 0.35         # min net radial approach toward the car (`moved_closer`)

    # --- LOITER only -------------------------------------------------------- #
    loiter_min_sec: float = 2.0    # min time spent near a car to count as loitering
    loiter_near_th: float = 0.60   # closeness above this counts as "near" for loitering
    loiter_max_net_drift: float = 0.6  # net person displacement allowed for a loiter
    loiter_straight_th: float = 0.85   # net/path straightness above which it's a walk-through

    # --- SHARED infra: camera ego-motion (feeds `car_stationary` for all) --- #
    ego_max_corners: int = 500     # feature points tracked for camera-motion estimation
    ego_quality: float = 0.01      # goodFeaturesToTrack quality level
    ego_min_dist: int = 8          # min pixel distance between tracked features
    ego_box_pad: float = 0.15      # fractional padding around dynamic boxes when masking
    ego_min_pts: int = 8           # min matched points required to accept a transform

    # --- SHARED infra: SAM3 open-vocabulary detector / tracker -------------- #
    sam3_src: Path = field(default_factory=lambda: _WORKSPACE / "src" / "sam3-main")
    sam3_checkpoint: str = "../sam3.pt"
    sam3_bpe: str = ""             # BPE tokenizer path; empty => packaged vocab
    sam3_person_prompt: str = "person"
    sam3_car_prompt: str = "car"
    sam3_score_th: float = 0.5     # min per-instance probability to keep a SAM3 detection
    sam3_gpu: int = 0              # cuda device index for the SAM3 model

    # --- derived per-clip frame values (AUTO-filled by resolve_for_fps) ----- #
    #   annotated with the state(s) that consume each one.
    window: int = 5                # enter + exit
    edge_margin: int = 3           # exit
    min_track_len: int = 6         # shared (all)
    min_absence: int = 8           # enter
    loiter_min_frames: int = 60    # loiter
    car_max_speed: float = 0.009   # shared (all); max ground-relative car speed (diag/frame),
    #                                error caused by registration / detector-tracker jitter, etc.

    def resolve_for_fps(self, fps: float) -> "Config":
        """Return a copy with the seconds-based settings converted to frames.

        Frame counts depend on the clip's frame rate, so a threshold expressed in
        seconds maps to a different number of frames per clip. Per-frame speeds
        are likewise scaled by 1/fps. Called once per video before detection.

        Args:
            fps: the clip's frame rate (<=0 is treated as 30.0).
        Returns:
            A new Config with the derived per-frame fields (window, edge_margin,
            min_track_len, min_absence, loiter_min_frames, car_max_speed) filled.
        """
        f = fps if fps and fps > 0 else 30.0
        return replace(
            self,
            window=max(1, round(self.window_sec * f)),
            edge_margin=max(0, round(self.edge_margin_sec * f)),
            min_track_len=max(2, round(self.min_track_sec * f)),
            min_absence=max(0, round(self.min_absence_sec * f)),
            loiter_min_frames=max(1, round(self.loiter_min_sec * f)),
            car_max_speed=self.car_max_speed_per_sec / f,
        )
