#!/usr/bin/env python
# coding: utf-8

# # Depth Mapping
# 

# ## Imports
# 

import numpy as np
import cv2 as cv
from skimage import io
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from math import sin, cos, radians, sqrt, ceil, floor
from os import cpu_count
from threading import Thread
from typing import List, Tuple
from logging import warning


# ## Declarations
# 

def is_notebook() -> bool:
    """Checks whether the script is started from inside a jupyter notebook.
    From https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook

    Returns:
        bool: True if the script is started from inside a jupyter notebook.
    """
    try:
        shell = get_ipython().__class__.__name__
        return True if shell == "ZMQInteractiveShell" else False
    except NameError:
        return False


IS_NOTEBOOK: bool = is_notebook()
PATH_PREFIX: str = (
    "https://raw.githubusercontent.com/YertleTurtleGit/photometric-stereo-mappings/main/test_dataset/output/"
    if IS_NOTEBOOK
    else "./../test-dataset/output/"
)
NORMAL_MAP_PATH: str = PATH_PREFIX + "normal.png"
MASK_PATH: str = PATH_PREFIX + "opacity.png"
OUTPUT_PATH = None if IS_NOTEBOOK else PATH_PREFIX + "./test_dataset/height_map.png"


def _read_image(
    image_path: str, color: bool = True, target_dtype: np.dtype = np.dtype("float64")
) -> np.ndarray:
    """Reads an image from URI and converts it to an array with specified bit depth.

    Args:
        image_path (str): The path to the image file.
        color (bool, optional): Read image as color image. Defaults to True.
        target_dtype (np.dtype, optional): The target bit depth. Defaults to np.dtype("float64").

    Returns:
        np.ndarray: The output array with shape (w,h,3) for color or (w,h) for grayscale images.
    """
    image = io.imread(image_path)
    image_dtype: np.dtype = image.dtype
    image = image.astype(target_dtype)

    if image_dtype == np.dtype("uint8"):
        image /= pow(2, 8) - 1
    elif image_dtype == np.dtype("uint16"):
        image /= pow(2, 16) - 1
    elif image_dtype == np.dtype("uint32"):
        image /= pow(2, 32) - 1

    if color:
        if len(image.shape) == 3:
            return image
        elif len(image.shape) == 2:
            return np.array([image, image, image])
        elif len(image.shape) == 4:
            return np.array([image[:, :, 0], image[:, :, 1], image[:, :, 2]])
        else:
            warning(
                "Image channel count of "
                + str(len(image.shape))
                + " with shape "
                + str(image.shape)
                + " is unknown: "
                + image_path
            )
    else:
        if len(image.shape) == 2:
            return image
        elif len(image.shape) == 3 or len(image.shape) == 4:
            return (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3
        else:
            warning(
                "Image channel count of "
                + str(len(image.shape))
                + " with shape "
                + str(image.shape)
                + " is unknown: "
                + image_path
            )

    return image


# # Logic
# 

def _get_atlas_countries(
    mask: np.ndarray, albedo: np.ndarray
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:

    image_parts: List[np.ndarray] = []
    image_parts_rectangles: List[Tuple[int, int, int, int]] = []

    for x in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            if mask[x, y] > 0:
                mask_before = mask.copy()
                _, mask, _, rect = cv.floodFill(np.float32(mask), None, (y, x), 0)

                part_albedo = albedo.copy()
                part_albedo[mask_before == mask] = 0

                r_y, r_x, r_h, r_w = rect
                image_part = part_albedo[r_x : r_x + r_w, r_y : r_y + r_h]

                image_parts.append(image_part)
                image_parts_rectangles.append((r_x, r_y, r_w, r_h))

    return image_parts, image_parts_rectangles


def _integrate(
    normal_map: np.ndarray,
    normal_is_open_gl: bool,
    normal_is_pseudo_compressed: bool,
    target_iteration_count: int,
    max_thread_count: int,
) -> np.ndarray:
    normal_map = normal_map.astype("float64")

    if normal_is_pseudo_compressed:
        # TODO Fix line.
        normal_map[:, :2] *= 2
        normal_map[:, :2] -= 1

    # TODO Handle non-open_gl.

    SLOPE_SHIFT: float = -0.5
    normal_map[:, :, 0][normal_map[:, :, 2] != 0] /= normal_map[:, :, 2][
        normal_map[:, :, 2] != 0
    ]
    normal_map[:, :, 1][normal_map[:, :, 2] != 0] /= normal_map[:, :, 2][
        normal_map[:, :, 2] != 0
    ]
    normal_map[normal_map[:, :, 2] == 0] = [0, 0, 0]
    normal_map[normal_map[:, :, 2] != 0] += SLOPE_SHIFT
    normal_map[:, :, 0] *= -1

    n_w, n_h, _ = normal_map.shape

    diagonal_length: int = ceil(sqrt(pow(n_w, 2) + pow(n_h, 2)))
    isotropic_integral = np.zeros((diagonal_length, diagonal_length))

    i_w, i_h = isotropic_integral.shape

    fx: np.ndarray = np.zeros(isotropic_integral.shape)
    fy: np.ndarray = np.zeros(isotropic_integral.shape)

    fx[
        floor((i_w - n_w) / 2) : floor((i_w + n_w) / 2),
        floor((i_h - n_h) / 2) : floor((i_h + n_h) / 2),
    ] = normal_map[:, :, 0]
    fy[
        floor((i_w - n_w) / 2) : floor((i_w + n_w) / 2),
        floor((i_h - n_h) / 2) : floor((i_h + n_h) / 2),
    ] = normal_map[:, :, 1]

    def integrate_anisotropic(angles: List[float]):
        for angle in angles:
            x_factor: float = cos(radians(angle))
            y_factor: float = sin(radians(angle))

            factor_norm: float = abs(x_factor) + abs(y_factor)
            x_factor /= factor_norm
            y_factor /= factor_norm

            fxy = fx * x_factor + fy * y_factor
            fxy = rotate(fxy, angle, axes=(1, 0), reshape=False)

            anisotropic_integral = rotate(
                np.cumsum(fxy, axis=1), -angle, axes=(1, 0), reshape=False
            )
            isotropic_integral[~np.isnan(anisotropic_integral)] += anisotropic_integral[
                ~np.isnan(anisotropic_integral)
            ]

    anisotropic_integral_threads: List[Thread] = []
    angle_per_thread: float = 360 / max_thread_count
    iterations_per_thread: int = round(target_iteration_count / max_thread_count)

    for thread_id in range(0, max_thread_count):
        angles = np.linspace(
            thread_id * angle_per_thread,
            (thread_id + 1) * angle_per_thread,
            iterations_per_thread,
            endpoint=False,
        )
        thread = Thread(target=integrate_anisotropic, args=(angles,))
        thread.start()
        anisotropic_integral_threads.append(thread)

    for thread in anisotropic_integral_threads:
        thread.join()

    height_map = isotropic_integral[
        floor((i_w - n_w) / 2) : floor((i_w + n_w) / 2),
        floor((i_h - n_h) / 2) : floor((i_h + n_h) / 2),
    ]

    height_map[
        np.sqrt(
            np.abs(normal_map[:, :, 0])
            + np.abs(normal_map[:, :, 1])
            + np.abs(normal_map[:, :, 2])
        )
        == 0
    ] = None

    return height_map


def height_map(
    normal_map_path: str,
    mask_path: str,
    output_path: str = None,
    normal_is_open_gl: bool = True,
    normal_is_pseudo_compressed: bool = False,
    target_iteration_count: int = 100,
    max_thread_count: int = max(int(cpu_count() or 1), 1),
):
    normal_map = _read_image(normal_map_path)
    mask = _read_image(mask_path, color=False)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    countries, country_bounding_boxes = _get_atlas_countries(mask, normal_map)

    height_map = np.zeros((normal_map.shape[0], normal_map.shape[1]))

    for i in range(0, len(countries)):
        p_x, p_y, p_w, p_h = country_bounding_boxes[i]
        part_height_map = _integrate(
            countries[i],
            normal_is_open_gl,
            normal_is_pseudo_compressed,
            target_iteration_count,
            max_thread_count,
        )
        height_map[p_x : p_x + p_w, p_y : p_y + p_h][
            ~np.isnan(part_height_map)
        ] = part_height_map[~np.isnan(part_height_map)]

    height_map -= np.min(height_map)
    height_map /= np.max(height_map)
    height_map *= pow(2, 8) - 1

    if OUTPUT_PATH:
        cv.imwrite(output_path, height_map)
    else:
        plt.imshow(height_map)


height_map(NORMAL_MAP_PATH, MASK_PATH, OUTPUT_PATH)

