#!/usr/bin/env python
# coding: utf-8

from typing import List
from logging import warning
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import io


IS_NOTEBOOK: bool = False  # @param {type: "boolean"}
AMBIENT_OCCLUSION_MAP_FILE_NAME: str = "ambient_occlusion.png"  # @param {type: "string"}

PATH_PREFIX: str = (
    "https://raw.githubusercontent.com/YertleTurtleGit/photometric-stereo-mappings/main/test_dataset/"
    if IS_NOTEBOOK
    else "./../test_dataset/"
)

HEIGHT_MAP_PATH:str = PATH_PREFIX + "output/height.png"
MASK_PATH = PATH_PREFIX + "output/opacity.png"

OUTPUT_PATH = None if IS_NOTEBOOK else PATH_PREFIX + "output/" + AMBIENT_OCCLUSION_MAP_FILE_NAME


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


def ambient_occlusion_map(
    height_map_path: str, output_path: str, mask_path: str = None
):
    """Calculates the ambient occlusion.

    Args:
        height_map_path (str): _description_
        output_path (str): _description_
        mask_path (str, optional): _description_. Defaults to None.
    """
    height_map = _read_image(height_map_path, color=False)
    blurred_height_map = cv.blur(height_map, (3, 3))

    ambient_occlusion_map = height_map - blurred_height_map
    # TODO Fix high ao at the edges because of masked height mapping.
    ambient_occlusion_map[ambient_occlusion_map > 0.025] = 0

    if mask_path:
        mask_image = _read_image(mask_path, color=False)
        ambient_occlusion_map[mask_image == 0] = 0

    if output_path:
        cv.imwrite(output_path, ambient_occlusion_map)
    else:
        plt.imshow(ambient_occlusion_map)


if __name__=="__main__":
    ambient_occlusion_map(HEIGHT_MAP_PATH, OUTPUT_PATH, MASK_PATH)

