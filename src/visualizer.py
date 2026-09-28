"""Frame renderer for the causal stream (boxes + event banner).

The Visualizer draws, for one frame at a time: boxes with ids, a persistent banner
of the provisional signals emitted so far, and a small HUD.

Kept in its own module so rendering is fully decoupled from the detection logic.
"""

import cv2

VERB = {"enter": "ENTERING", "exit": "EXITING", "loiter": "LOITERING near"}
BANNER_BG = {"enter": (0, 140, 255), "exit": (0, 0, 200), "loiter": (0, 215, 255)}


class Visualizer:
    """Draws masks, boxes, ids, the running event banner and a HUD onto frames."""

    # BGR base colours.
    CAR_COLOR = (255, 128, 0)
    PERSON_COLOR = (0, 200, 0)
    LOITER_COLOR = (0, 215, 255)
    CAR_EVENT_COLOR = (0, 140, 255)
    PERSON_EVENT_COLOR = (0, 0, 255)

    def __init__(self, cfg, detector, mask_alpha: float = 0.45):
        """Initialize the renderer.

        Args:
            cfg: Config (colors/behavior).
            detector: Detector holding the per-frame mask/poly descriptors to draw.
            mask_alpha: blend weight for the translucent mask fills.
        Returns:
            None.
        """
        self.cfg = cfg
        self.detector = detector
        self.mask_alpha = mask_alpha

    # --- low-level drawing helpers ---
    @staticmethod
    def _draw_label(img, text, org, bg):
        """Draw a filled text label with a colored background.

        Args:
            img: BGR image to draw on (in place).
            text: label string.
            org: (x, y) bottom-left anchor of the label box.
            bg: BGR background color.
        Returns:
            None.
        """
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        (tw, th), base = cv2.getTextSize(text, font, scale, thick)
        x, y = org
        cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 6, y), bg, -1)
        cv2.putText(img, text, (x + 3, y - base - 2), font, scale, (255, 255, 255), thick)

    def _car_color(self, evt):
        """Pick a car's draw color from its current event highlight.

        Args:
            evt: event type ("enter"/"exit"/"loiter") or None/falsy for idle.
        Returns:
            A BGR color tuple.
        """
        if evt == "loiter":
            return self.LOITER_COLOR
        return self.CAR_EVENT_COLOR if evt else self.CAR_COLOR

    def _person_color(self, evt):
        """Pick a person's draw color from their current event highlight.

        Args:
            evt: event type ("enter"/"exit"/"loiter") or None/falsy for idle.
        Returns:
            A BGR color tuple.
        """
        if evt == "loiter":
            return self.LOITER_COLOR
        return self.PERSON_EVENT_COLOR if evt else self.PERSON_COLOR

    # --- full-frame draw ---
    def draw(self, frame, f_idx, persons, cars, car_hl, person_hl,
             banner_items, total_frames, hud_text):
        """Render one frame in place.

        `persons` / `cars`: [(id, box), ...] for this frame.
        `car_hl` / `person_hl`: {id: event_type} highlight maps.
        `banner_items`: iterable of {person_id, car_id, type} signals to list.

        Args:
            frame: BGR image to draw on (in place).
            f_idx: current frame index (selects the mask descriptors to draw).
            persons: [(person_id, box), ...] for this frame.
            cars: [(car_id, box), ...] for this frame.
            car_hl: {car_id: event_type} highlight map.
            person_hl: {person_id: event_type} highlight map.
            banner_items: iterable of {person_id, car_id, type} to list in banner.
            total_frames: total frames (for context; HUD text is passed in).
            hud_text: the HUD string drawn at the bottom.
        Returns:
            The same `frame`, drawn in place.
        """
        # boxes + labels
        for cid, box in cars:
            x1, y1, x2, y2 = map(int, box[:4])
            color = self._car_color(car_hl.get(cid))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"car {cid}" + (f" {box[4]:.2f}" if len(box) > 4 else "")
            self._draw_label(frame, label, (x1, y1), color)
        for pid, box in persons:
            x1, y1, x2, y2 = map(int, box[:4])
            color = self._person_color(person_hl.get(pid))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"person {pid}" + (f" {box[4]:.2f}" if len(box) > 4 else "")
            self._draw_label(frame, label, (x1, y1), color)

        # persistent banner of emitted (past, causal) provisional signals
        y = 26
        for it in sorted(banner_items, key=lambda d: (d["person_id"], d["type"])):
            text = f"Person {it['person_id']} {VERB[it['type']]} car {it['car_id']}"
            self._draw_label(frame, text, (10, y), BANNER_BG[it["type"]])
            y += 30

        # HUD
        h = frame.shape[0]
        self._draw_label(frame, hud_text, (10, h - 12), (40, 40, 40))
        return frame
