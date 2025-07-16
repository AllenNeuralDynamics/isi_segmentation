"""Helper data process functions"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np


@dataclass
class ISIData:
    date: any
    retinotopy_altitude: np.ndarray
    retinotopy_azimuth: np.ndarray
    visual_sign: np.ndarray
    vasculature_image: np.ndarray
    defocus_image: np.ndarray
    visual_sign_pixel_size: float
    vasculature_pixel_size: float
    retinotopy_altitude_shape: tuple
    retinotopy_azimuth_shape: tuple
    visual_sign_shape: tuple
    vasculature_image_shape: tuple
    defocus_image_shape: tuple
    label_map_image: np.ndarray = None

    @classmethod
    def from_files(cls, hdf5_file_path):
        with h5py.File(hdf5_file_path, "r") as f:
            date = f.attrs["date"]
            retinotopy_altitude = f["retinotopy_altitude"][:]
            retinotopy_azimuth = f["retinotopy_azimuth"][:]
            visual_sign = f["visual_sign"][:]
            vasculature_image = f["vasculature_image"][:]
            defocus_image = f["defocus_image"][:]
            visual_sign_pixel_size = f["visual_sign"].attrs["pixel_size_x_um"]
            vasculature_pixel_size = f["vasculature_image"].attrs["pixel_size_x_um"]
            retinotopy_altitude_shape = f["retinotopy_altitude"].shape
            retinotopy_azimuth_shape = f["retinotopy_azimuth"].shape
            visual_sign_shape = f["visual_sign"].shape
            vasculature_image_shape = f["vasculature_image"].shape
            defocus_image_shape = f["defocus_image"].shape
            label_map_image = None
        return cls(
            date,
            retinotopy_altitude,
            retinotopy_azimuth,
            visual_sign,
            vasculature_image,
            defocus_image,
            visual_sign_pixel_size,
            vasculature_pixel_size,
            retinotopy_altitude_shape,
            retinotopy_azimuth_shape,
            visual_sign_shape,
            vasculature_image_shape,
            defocus_image_shape,
            label_map_image,
        )


""" constant variables for prediction"""
# the shape of input of the UNet should be (512, 512)
IMAGE_W = 512
IMAGE_H = 512


def print_arr_inf(array: np.ndarray) -> None:
    """Print the intensity information given an array"""
    logging.info(
        "Intensity info: {:.2f} ± {:.2f}, max={:.2f}, min={:.2f}, median={:.2f}".format(
            np.mean(array),
            np.std(array),
            np.max(array),
            np.min(array),
            np.median(array),
        )
    )


def Normalized(x: np.ndarray) -> np.ndarray:
    """Normalize the value of input array to (0, 1)"""
    normalized = (x - np.min(x)) / (np.max(x) - np.min(x))

    return normalized


def normalize_sign_map_forpred(img: np.ndarray) -> np.ndarray:
    """normalize and save the sign map"""

    # the intensity of sign map should be in range of -1.0 and 1.0
    assert np.min(img) >= -1.0
    assert np.max(img) <= 1.0

    # after normalization, the intensity of sign map should be in range of 0.0 and 1.0
    img = Normalized(img)

    assert np.min(img) >= 0.0
    assert np.max(img) <= 1.0

    return np.multiply(img, 255).astype(np.uint8)


def read_img_forpred(image_path: Path) -> np.ndarray:
    """Read and preprocess the sign map.

    Args:
        image_path: path to input image
    Return:
        numpy array for input image
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # image shape: (540, 640)
    image = cv2.resize(image, (IMAGE_W, IMAGE_H))  # image shape: (512, 512)
    image = image / 255.0

    # the intensity of input sign map should be in range of 0.0 and 1.0 for prediction
    assert np.min(image) >= 0.0
    assert np.max(image) <= 1.0

    image = np.expand_dims(image, axis=0)  # [1, H, W]
    image = image.astype(np.float32)

    return image


def verify_image_shape(input_shape: tuple, expected_shape: tuple) -> None:
    """Verify the image shape"""
    if input_shape != expected_shape:
        s = f"Image shape: {input_shape}, not equal to the expected shape: {expected_shape}!"
        raise ValueError(s)
