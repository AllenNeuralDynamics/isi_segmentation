"""Helper plot functions"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

""" constant variables for class and color definition"""
CLASS_COLOR_MAP = {
    0: [256, 256, 256],
    1: [80, 80, 255],
    2: [0, 255, 0],
    3: [255, 165, 0],
    4: [255, 0, 0],
    5: [0, 159, 172],
    6: [255, 255, 0],
    7: [0, 255, 255],
    8: [100, 55, 200],
    9: [66, 204, 255],
    10: [24, 128, 100],
    11: [201, 147, 153],
    12: [200, 109, 172],
    13: [255, 127, 80],
    14: [204, 255, 66],
}

CLASS_NAME_MAP = {
    0: "N/A",
    1: "VISp",
    2: "VISam",
    3: "VISal",
    4: "VISl",
    5: "VISrl",
    6: "VISpl",
    7: "VISpm",
    8: "VISli",
    9: "VISpor",
    10: "VISrll",
    11: "VISlla",
    12: "VISmma",
    13: "VISmmp",
    14: "VISm",
}

import numpy as np


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    """
    Convert a 2D label map (class IDs) into a uint8 RGB visualization using the
    global CLASS_COLOR_MAP.

    Parameters
    ----------
    label_map : np.ndarray
        2D array of shape (H, W). Values are interpreted as class IDs.

    Returns
    -------
    np.ndarray
        RGB image of shape (H, W, 3) and dtype uint8, suitable for visualization.

    Raises
    ------
    KeyError
        If a label value is not present in CLASS_COLOR_MAP.
    """
    label_map = label_map.astype(np.uint8)

    label_map_3d = np.ndarray(
        shape=(label_map.shape[0], label_map.shape[1], 3),
        dtype=np.int32,
    )

    for i in range(label_map.shape[0]):
        for j in range(label_map.shape[1]):
            label_map_3d[i, j] = CLASS_COLOR_MAP[label_map[i, j]]

    max_val = label_map_3d.max()
    if max_val > 0:
        label_map_3d = (label_map_3d / max_val * 255).astype(np.uint8)
    else:
        label_map_3d = label_map_3d.astype(np.uint8)

    return label_map_3d



def colorize_and_annotate_label_map(label_map_ids_path: Path, label_map_path: Path) -> None:
    """
    Load a grayscale label-map (class IDs), colorize it using `colorize_label_map`,
    annotate each (non-background) class with its name at the class mask centroid,
    and save the resulting visualization to `label_map_path`.

    Assumes global mappings exist:
      - CLASS_NAME_MAP: dict[int, str]
      - colorize_label_map(label_map: np.ndarray) -> np.ndarray  (returns uint8 RGB image)

    Parameters
    ----------
    label_map_ids_path : Path
        Path to grayscale image where each pixel value is a class ID (typically uint8).
    label_map_path : Path
        Output path for the annotated RGB visualization (PNG recommended).

    Raises
    ------
    ValueError
        If the label-map image cannot be read.
    KeyError
        If a class id is missing from CLASS_NAME_MAP.
    """
    # Load class-id label map
    label_map_ids = cv2.imread(str(label_map_ids_path), cv2.IMREAD_GRAYSCALE)
    if label_map_ids is None:
        raise ValueError(f"Failed to read label map ids at {label_map_ids_path}")

    # Colorize (expects class IDs, not RGB)
    label_map_rgb = colorize_label_map(label_map_ids)

    # Build figure and show RGB (matplotlib expects RGB)
    fig, ax = plt.subplots(1, 1, figsize=(label_map_rgb.shape[1] / 100, label_map_rgb.shape[0] / 100), dpi=100)
    ax.imshow(label_map_rgb)
    ax.set_title("Label map")
    ax.axis("off")

    # Compute which classes exist
    # Use np.unique for determinism vs set(flatten()).
    classes = np.unique(label_map_ids)

    # Annotate each class except background (assumes background is 0; matches your classes[1:] pattern)
    for cur_class in classes[1:]:
        # segmented region mask for current class
        mask = (label_map_ids == cur_class)

        count = int(mask.sum())
        if count == 0:
            continue

        # centroid (y, x) in image coordinates
        yx_sum = np.argwhere(mask).sum(axis=0).astype(np.float64)
        y_center, x_center = (yx_sum / count).tolist()

        ax.text(
            x_center,
            y_center,
            CLASS_NAME_MAP[int(cur_class)],
            ha="center",
            va="center",
            fontsize=8,
        )

    # Save annotated visualization
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(label_map_path), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_img_label(
    sign_map_path: Path, label_map_path: Path, savefig_path: Path
) -> None:
    """Visualize the sign map and label map

    Args:
        sign_map_path: path to the sign map
        label_map_path: path to the label map
        savefig_path: path to save plot
    """
    assert os.path.isfile(sign_map_path), "sign_map_path not a valid file"
    assert os.path.isfile(label_map_path), "label_map_path not a valid file"

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # -------------------------------------
    # show sign map
    # -------------------------------------

    sign_map = cv2.imread(sign_map_path, cv2.IMREAD_GRAYSCALE)
    sign_map = sign_map.astype(np.float32)

    ax[0].imshow(sign_map, cmap="jet")
    ax[0].set_title("Sign map")

    # -------------------------------------
    # show label map
    # -------------------------------------

    label_map = cv2.imread(label_map_path)
    # label_map = cv2.cvtColor(label_map_bgr, cv2.COLOR_BGR2RGB)

    ax[1].imshow(label_map)
    ax[1].set_title("Label map")

    # plt.show()
    plt.savefig(savefig_path, bbox_inches="tight", pad_inches=0.01)
    plt.close()
