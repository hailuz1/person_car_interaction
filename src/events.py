"""The interaction event record produced by the pipeline."""

from dataclasses import dataclass, field


@dataclass
class Event:
    """A single detected enter/exit/loiter interaction."""

    person_id: int
    car_id: int
    type: str          # "enter", "exit" or "loiter"
    frame: int         # frame at which the event is considered to occur
    frame_start: int = 0  # first frame of the state's evidence window
    frame_end: int = 0    # last frame of the state's evidence window
    c_start: float = 0.0   # closeness to the car at the start of the person track
    c_end: float = 0.0     # closeness to the car at the end of the person track
    motion: float = 0.0    # net radial approach (enter) / departure (exit), car diagonals
    duration: int = 0  # loiter: number of frames spent near the car (0 otherwise)
    # every condition value the gate evaluated to accept this event (metric ->
    # {"value": ..., "th": ..., "pass": ...} or a raw scalar), for diagnostics.
    metrics: dict = field(default_factory=dict)
