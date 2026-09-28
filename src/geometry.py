"""Pure geometry helpers: box proximity and mask-ellipse descriptors.

These are stateless functions shared by the detector, the driver and the
visualizer. They are copied verbatim from the proven pipeline so the numeric
results stay byte-for-byte identical.
"""

import math

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Box geometry
# --------------------------------------------------------------------------- #
def box_center(box):
    """Center of an xyxy box.

    Args:
        box: [x1, y1, x2, y2(, ...)].
    Returns:
        (cx, cy) center in pixels.
    """
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def box_diag(box) -> float:
    """Diagonal length of an xyxy box.

    Args:
        box: [x1, y1, x2, y2(, ...)].
    Returns:
        float diagonal in pixels (+1e-6 to stay non-zero).
    """
    return math.hypot(box[2] - box[0], box[3] - box[1]) + 1e-6


def center_distance_in_diagonals(person, car) -> float:
    """Person-to-car center distance, normalized by the car's diagonal.

    Args:
        person: person box [x1, y1, x2, y2(, ...)].
        car: car box [x1, y1, x2, y2(, ...)].
    Returns:
        float center distance expressed in car-diagonals.
    """
    pcx, pcy = box_center(person)
    ccx, ccy = box_center(car)
    return math.hypot(pcx - ccx, pcy - ccy) / box_diag(car)


def closeness(person, car, near_dist: float) -> float:
    """Closeness in [0, 1] of a person box to a car box.

    Combines overlap (fraction of the person over the car) with a distance
    score normalized by the car's diagonal, so a person on/next to a car scores
    high and one far away scores ~0.

    Args:
        person: person box [x1, y1, x2, y2(, ...)].
        car: car box [x1, y1, x2, y2(, ...)].
        near_dist: distance (in car-diagonals) at which the distance score -> 0.
    Returns:
        float in [0, 1]; max of overlap and distance score.
    """
    ix1, iy1 = max(person[0], car[0]), max(person[1], car[1])
    ix2, iy2 = min(person[2], car[2]), min(person[3], car[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    person_area = max(1e-6, (person[2] - person[0]) * (person[3] - person[1]))
    overlap = inter / person_area

    dist = center_distance_in_diagonals(person, car)
    dist_score = max(0.0, 1.0 - dist / near_dist)
    return max(overlap, dist_score)


def diou(person, car) -> float:
    """Distance-IoU of a person box and a car box, in [-1, 1].

    Uses intersection over *union* (unlike ``closeness``'s intersection over the
    person area, which saturates when the person sits fully inside the car), so a
    small person inside a large near-camera car scores low and drops further as
    the person recedes.

    Args:
        person: person box [x1, y1, x2, y2(, ...)].
        car: car box [x1, y1, x2, y2(, ...)].
    Returns:
        float distance-IoU in [-1, 1] (IoU minus the normalized center-distance).
    """
    ix1, iy1 = max(person[0], car[0]), max(person[1], car[1])
    ix2, iy2 = min(person[2], car[2]), min(person[3], car[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    pa = (person[2] - person[0]) * (person[3] - person[1])
    ca = (car[2] - car[0]) * (car[3] - car[1])
    iou = inter / (pa + ca - inter + 1e-6)
    pcx, pcy = box_center(person)
    ccx, ccy = box_center(car)
    rho2 = (pcx - ccx) ** 2 + (pcy - ccy) ** 2
    ex1, ey1 = min(person[0], car[0]), min(person[1], car[1])
    ex2, ey2 = max(person[2], car[2]), max(person[3], car[3])
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-6
    return iou - rho2 / c2


# --------------------------------------------------------------------------- #
# Mask geometry (equivalent-ellipse descriptors)
# --------------------------------------------------------------------------- #
# A descriptor is a 6-tuple [cx, cy, area, a, b, theta]:
#   cx, cy  centroid of the mask, in pixels
#   area    mask pixel area
#   a, b    semi-major / semi-minor axis length of the equivalent ellipse
#   theta   orientation of the major axis, radians
def mask_ellipse(mask) -> list | None:
    """Equivalent-ellipse descriptor of a boolean mask, or None if empty.

    Args:
        mask: 2D boolean/0-1 mask array.
    Returns:
        [cx, cy, area, a, b, theta] (center, pixel area, semi-major/minor axes,
        major-axis angle in radians), or None when the mask is empty.
    """
    m = cv2.moments(np.ascontiguousarray(mask).astype(np.uint8), binaryImage=True)
    area = m["m00"]
    if area <= 0:
        return None
    cx = m["m10"] / area
    cy = m["m01"] / area
    mu20 = m["mu20"] / area
    mu02 = m["mu02"] / area
    mu11 = m["mu11"] / area
    common = math.sqrt(max(0.0, (mu20 - mu02) ** 2 + 4.0 * mu11 * mu11))
    l1 = 0.5 * (mu20 + mu02 + common)
    l2 = 0.5 * (mu20 + mu02 - common)
    a = 2.0 * math.sqrt(max(l1, 0.0))
    b = 2.0 * math.sqrt(max(l2, 0.0))
    theta = 0.5 * math.atan2(2.0 * mu11, (mu20 - mu02))
    return [float(cx), float(cy), float(area), float(a), float(b), float(theta)]


def axis_offsets(px: float, py: float, ell) -> tuple[float, float]:
    """Offset of a point from an ellipse centre, in the ellipse's own frame.

    Args:
        px: point x in pixels.
        py: point y in pixels.
        ell: ellipse descriptor [cx, cy, area, a, b, theta].
    Returns:
        (u, v): the offset along the ellipse major and minor axes.
    """
    cx, cy, _area, _a, _b, theta = ell
    dx, dy = px - cx, py - cy
    c, s = math.cos(theta), math.sin(theta)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    return u, v


def on_major_axis(px: float, py: float, ell, major_reach: float,
                  minor_band: float) -> bool:
    """Whether a point lies within a car ellipse's elongated major-axis region.

    Args:
        px: point x in pixels.
        py: point y in pixels.
        ell: car ellipse descriptor [cx, cy, area, a, b, theta].
        major_reach: allowed |major-axis| offset in semi-major lengths.
        minor_band: allowed |minor-axis| offset in semi-minor lengths.
    Returns:
        True if the point falls inside the reach x band region, else False.
    """
    _cx, _cy, _area, a, b, _theta = ell
    a = max(a, b, 1.0)
    b = max(b, 1.0)
    #convert the point to the ellipse's own frame, u is the offset along the major axis, v is the offset along the minor axis
    u, v = axis_offsets(px, py, ell) 
    return abs(u) <= major_reach * a and abs(v) <= minor_band * b


def mask_polygons(mask, epsilon_frac: float = 0.008, min_area: float = 25.0) -> list:
    """Simplified external contour polygons of a boolean mask (pixel points).

    Args:
        mask: 2D boolean/0-1 mask array.
        epsilon_frac: contour-approximation tolerance as a fraction of perimeter.
        min_area: minimum contour area (px) to keep.
    Returns:
        list of polygons, each a list of [x, y] integer pixel points.
    """
    mask_u8 = np.ascontiguousarray(mask).astype(np.uint8)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        eps = max(1.5, epsilon_frac * cv2.arcLength(c, True))
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) >= 3:
            polys.append([[int(x), int(y)] for x, y in approx])
    return polys
