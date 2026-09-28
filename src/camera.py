"""Causal camera ego-motion: transform (f-1 -> f) from two consecutive frames.

This is the offline background-registration step fed one frame at a time, so at
frame `f` it uses only frames `<= f`. The dynamic-object mask uses the previous
frame's boxes (already known) and the fitted transform maps background points
from frame f-1 into frame f. Copied faithfully from the proven causal POC.
"""

import cv2
import numpy as np


class CamMotion:
    """Estimate the frame-to-frame background transform from the last two frames."""

    def __init__(self, cfg):
        """Initialize the ego-motion estimator.

        Args:
            cfg: Config holding the goodFeaturesToTrack / RANSAC parameters.
        Returns:
            None.
        """
        self.cfg = cfg
        self.prev_gray = None
        self.prev_boxes = []

    def step(self, gray, cur_boxes):
        """Return the 2x3 affine (f-1 -> f) for this frame, or None (identity).

        Args:
            gray: current frame as a grayscale image.
            cur_boxes: dynamic-object boxes in this frame (masked out of the
                background feature search, and stored for the next call).
        Returns:
            2x3 numpy affine transform mapping frame f-1 -> f, or None when it
            cannot be estimated (treated as identity by callers).
        """
        cfg = self.cfg
        m = None
        if self.prev_gray is not None:
            h, w = self.prev_gray.shape
            mask = np.full((h, w), 255, dtype=np.uint8)
            for box in self.prev_boxes:
                bw, bh = box[2] - box[0], box[3] - box[1]
                x1 = max(0, int(box[0] - cfg.ego_box_pad * bw))
                y1 = max(0, int(box[1] - cfg.ego_box_pad * bh))
                x2 = min(w, int(box[2] + cfg.ego_box_pad * bw))
                y2 = min(h, int(box[3] + cfg.ego_box_pad * bh))
                mask[y1:y2, x1:x2] = 0
            pts = cv2.goodFeaturesToTrack(
                self.prev_gray, maxCorners=cfg.ego_max_corners,
                qualityLevel=cfg.ego_quality, minDistance=cfg.ego_min_dist, mask=mask,
            )
            if pts is not None and len(pts) >= cfg.ego_min_pts:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, pts, None
                )
                good_prev = pts[status == 1]
                good_next = nxt[status == 1]
                if len(good_prev) >= cfg.ego_min_pts:
                    m, _ = cv2.estimateAffinePartial2D(
                        good_prev, good_next, method=cv2.RANSAC
                    )
        self.prev_gray = gray
        self.prev_boxes = cur_boxes
        return m
