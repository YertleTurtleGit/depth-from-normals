import numpy as np
import cv2 as cv
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import cpu_count
from typing import Tuple, List, Union
import numba


def calculate_gradients(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normals = normals.astype(np.float64)

    horizontal_angle_map = np.arccos(np.clip(normals[..., 0], -1, 1))
    left_gradients = np.sign(horizontal_angle_map - np.pi / 2) * (
        1 - np.sin(horizontal_angle_map)
    )

    vertical_angle_map = np.arccos(np.clip(normals[..., 1], -1, 1))
    top_gradients = -np.sign(vertical_angle_map - np.pi / 2) * (
        1 - np.sin(vertical_angle_map)
    )

    return left_gradients, top_gradients


@numba.jit(nopython=True)
def integrate_gradient_field(gradient_field: np.ndarray, axis: int) -> np.ndarray:
    heights = np.zeros(gradient_field.shape)

    for d1 in numba.prange(heights.shape[1 - axis]):
        sum_value = 0
        for d2 in range(heights.shape[axis]):
            coordinates = (d1, d2) if axis == 1 else (d2, d1)
            sum_value = sum_value + gradient_field[coordinates]
            heights[coordinates] = sum_value

    return heights


def calculate_heights(
    left_gradients: np.ndarray, top_gradients
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_heights = integrate_gradient_field(left_gradients, 1)
    right_heights = np.fliplr(integrate_gradient_field(np.fliplr(-left_gradients), 1))
    top_heights = integrate_gradient_field(top_gradients, 0)
    bottom_heights = np.flipud(integrate_gradient_field(np.flipud(-top_gradients), 0))
    return left_heights, right_heights, top_heights, bottom_heights


def combine_heights(*heights: np.ndarray) -> np.ndarray:
    return np.mean(np.stack(heights, axis=0), axis=0)


def rotate(matrix: np.ndarray, angle: float) -> np.ndarray:
    h, w = matrix.shape[:2]
    center = (w / 2, h / 2)

    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    corners = cv.transform(
        np.array([[[0, 0], [w, 0], [w, h], [0, h]]]), rotation_matrix
    )[0]

    _, _, w, h = cv.boundingRect(corners)

    rotation_matrix[0, 2] += w / 2 - center[0]
    rotation_matrix[1, 2] += h / 2 - center[1]
    result = cv.warpAffine(matrix, rotation_matrix, (w, h), flags=cv.INTER_LINEAR)

    return result


def rotate_vector_field_normals(normals: np.ndarray, angle: float) -> np.ndarray:
    angle = np.radians(angle)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)

    return np.stack(
        [
            normals[..., 0] * cos_angle - normals[..., 1] * sin_angle,
            normals[..., 0] * sin_angle + normals[..., 1] * cos_angle,
            normals[..., 2],
        ],
        axis=-1,
    )


def centered_crop(image: np.ndarray, target_resolution: Tuple[int, int]) -> np.ndarray:
    return image[
        (image.shape[0] - target_resolution[0]) // 2 : (
            image.shape[0] - target_resolution[0]
        )
        // 2
        + target_resolution[0],
        (image.shape[1] - target_resolution[1]) // 2 : (
            image.shape[1] - target_resolution[1]
        )
        // 2
        + target_resolution[1],
    ]


def integrate_vector_field(
    vector_field: np.ndarray,
    target_iteration_count: int,
    thread_count: int,
) -> np.ndarray:
    shape = vector_field.shape[:2]
    angles = np.linspace(0, 90, target_iteration_count, endpoint=False)

    def integrate_vector_field_angles(angles: List[float]) -> np.ndarray:
        all_combined_heights = np.zeros(shape)

        for angle in angles:
            rotated_vector_field = rotate_vector_field_normals(
                rotate(vector_field, angle), angle
            )

            left_gradients, top_gradients = calculate_gradients(rotated_vector_field)
            (
                left_heights,
                right_heights,
                top_heights,
                bottom_heights,
            ) = calculate_heights(left_gradients, top_gradients)

            combined_heights = combine_heights(
                left_heights, right_heights, top_heights, bottom_heights
            )
            combined_heights = centered_crop(rotate(combined_heights, -angle), shape)
            all_combined_heights += combined_heights / len(angles)

        return all_combined_heights

    with Pool(processes=thread_count) as pool:
        heights = pool.map(
            integrate_vector_field_angles,
            np.array(
                np.array_split(angles, thread_count),
                dtype=object,
            ),
        )
        pool.close()
        pool.join()

    isotropic_height = np.zeros(shape)
    for height in heights:
        isotropic_height += height / thread_count

    return isotropic_height


def estimate_height_map(
    normal_map: np.ndarray,
    mask: Union[np.ndarray, None] = None,
    height_divisor: float = 1,
    target_iteration_count: int = 250,
    thread_count: int = max(cpu_count(), 1),
    raw_values: bool = False,
) -> np.ndarray:
    if mask is None and normal_map.shape[2] == 3:
        normal_map = np.pad(normal_map, ((0, 0), (0, 0), (0, 1)), constant_values=255)

    if mask is not None:
        normal_map = np.stack(
            [normal_map[..., 0], normal_map[..., 1], normal_map[..., 2], mask], axis=-1
        )

    normals = ((normal_map.astype(np.float64) / 255) - 0.5) * 2
    heights = integrate_vector_field(normals, target_iteration_count, thread_count)

    if raw_values:
        return heights

    heights /= height_divisor
    heights *= 2**16 - 1

    if np.min(heights) < 0 or np.max(heights) > 2**16 - 1:
        raise OverflowError("Height values are clipping.")

    heights = np.clip(heights, 0, 2**16 - 1)
    heights = heights.astype(np.uint16)

    return heights
