from typing import Tuple

import cv2
import numpy as np
import supervision as sv


def draw_circle(
    scene: np.ndarray, center: sv.Point, color: sv.Color, radius: int = 2
) -> np.ndarray:
    cv2.circle(
        scene,
        center=center.as_xy_int_tuple(),
        radius=radius,
        color=color.as_bgr(),
        thickness=-1,
    )
    return scene


def calculate_dynamic_circle_radius(resolution_wh: Tuple[int, int]) -> int:
    min_dimension = min(resolution_wh)
    if min_dimension < 480:
        return 4
    if min_dimension < 720:
        return 8
    if min_dimension < 1080:
        return 8
    if min_dimension < 2160:
        return 16
    else:
        return 16


def calculate_crop_dim(a, b):
    # Calculates the crop dimensions of the image resultant
    if a > b:
        width = a
        height = a
    else:
        width = b
        height = b

    return width, height
