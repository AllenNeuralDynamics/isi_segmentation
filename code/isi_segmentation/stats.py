import logging
import math
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

from isi_segmentation.plot import CLASS_NAME_MAP
from isi_segmentation.utils import ISIData


def eccentricity(
    az: np.ndarray, alt: np.ndarray, az_center: float, alt_center: float
) -> np.ndarray:
    """Compute the eccentricity map given azimuth, altitude, and their centers."""
    daz = az - az_center
    dalt = alt - alt_center
    ecc = np.arctan(
        np.sqrt(
            np.square(np.tan(dalt)) + np.square(np.tan(daz)) / np.square(np.cos(dalt))
        )
    )
    return ecc


def retinotopy_metric(
    mask: np.ndarray, map: np.ndarray
) -> Tuple[float, float, float, float]:
    """Calculate min, max, range, and bias for a retinotopy map within a mask."""
    ind = np.where(mask > 0)
    vals = map[ind]
    maxv = np.degrees(np.max(vals))
    minv = np.degrees(np.min(vals))
    return (minv, maxv, maxv - minv, abs(minv + maxv))


def window_level(input: np.ndarray, cmin: float, cmax: float) -> np.ndarray:
    """Apply window leveling to an input array between cmin and cmax."""
    data = np.copy(input)
    data[data < cmin] = cmin
    data[data > cmax] = cmax
    data -= cmin
    data /= cmax - cmin
    return data


def outline_mask(mask: np.ndarray) -> np.ndarray:
    """Generate an outline mask from a binary mask using Sobel edge detection."""
    edge_horizont = ndimage.sobel(mask, 0)
    edge_vertical = ndimage.sobel(mask, 1)
    magnitude = np.hypot(edge_horizont, edge_vertical)
    outline = np.zeros(mask.shape)
    outline[magnitude > 0] = 1
    outline = np.uint8(outline)
    return outline


class ISIMetricsAndVisualization:
    """Provides metrics calculation and visualization methods for ISI data using a data loader."""

    def __init__(
        self,
        data_loader: ISIData,
        altitude_scale: float = 0.322,
        azimuth_scale: float = 0.383,
        line_alpha: float = 100.0,
        ecc_zero_max: float = 50.0,
        ecc_v1_max: float = 50.0,
    ) -> None:
        """Initialize with a loaded ISIDataLoader instance and set up image dimensions."""
        self.data = data_loader
        self.label_map_image = data_loader.label_map_image
        # Set up image dimensions
        self.vasculature_image_width = self.data.vasculature_image_shape[1]
        self.vasculature_image_height = self.data.vasculature_image_shape[0]
        self.visual_sign_width = self.data.visual_sign_shape[1]
        self.visual_sign_height = self.data.visual_sign_shape[0]
        self.retinotopy_altitude_width = self.data.retinotopy_altitude_shape[1]
        self.retinotopy_altitude_height = self.data.retinotopy_altitude_shape[0]
        self.retinotopy_azimuth_width = self.data.retinotopy_azimuth_shape[1]
        self.retinotopy_azimuth_height = self.data.retinotopy_azimuth_shape[0]
        self.defocus_image_width = self.data.defocus_image_shape[1]
        self.defocus_image_height = self.data.defocus_image_shape[0]

        self.altitude_scale = altitude_scale
        self.azimuth_scale = azimuth_scale
        self.line_alpha = line_alpha
        self.ecc_zero_max = ecc_zero_max
        self.ecc_v1_max = ecc_v1_max

    def generate_qc_metric_and_images(
        self,
    ) -> Tuple[Dict[str, Any], Image.Image, Image.Image, Image.Image]:
        """Generate QC metrics and images for ISI segmentation and targeting using only the label_map_image.
        Returns:
            all_region_data (dict): Region metrics.
            eccentricity_retinotopic_zero_im (PIL.Image): Eccentricity retinotopic zero overlay image.
            eccentricity_v_one_centroid_im (PIL.Image): Eccentricity from V1 centroid overlay image.
            target_map_im (PIL.Image): Target map overlay image.
        """
        if self.label_map_image is None:
            raise Exception(
                "label_map_image must be provided to ISIDataLoader for region extraction."
            )

        # Build annotated_regions_data from label_map_image only, using full-size masks
        annotated_regions_data = []
        label_map = self.label_map_image
        structure_ids = np.unique(label_map)
        logging.info(f"Found {len(structure_ids)} unique structure IDs in label map.")
        for structure_id in structure_ids:
            if structure_id == 0:
                continue  # skip background
            mask = (label_map == structure_id).astype(np.uint8)
            annotated_regions_data.append(
                {"structure_id": int(structure_id), "region_mask": mask}
            )

        altitude_phase = self.data.retinotopy_altitude
        azimuth_phase = self.data.retinotopy_azimuth
        vasculature = self.data.vasculature_image.astype(float)

        altitude_scale = self.altitude_scale
        azimuth_scale = self.azimuth_scale
        altitude = altitude_phase * altitude_scale
        azimuth = azimuth_phase * azimuth_scale

        eccentricity_ret_zero = np.degrees(eccentricity(azimuth, altitude, 0.0, 0.0))

        VISp_blob = None
        outlines = []
        all_region_data = {}

        for annotated_region in annotated_regions_data:
            structure_id = annotated_region["structure_id"]
            region_mask = annotated_region["region_mask"]

            region_data = {}

            # compute centroid
            centroid = [np.mean(x) for x in np.where(region_mask)]
            region_data["y_centroid"] = float(centroid[0])
            region_data["x_centroid"] = float(centroid[1])

            # compute azimuth/altitude min, max, range, and bias
            (
                region_data["azimuth_min"],
                region_data["azimuth_max"],
                region_data["azimuth_range"],
                region_data["azimuth_bias"],
            ) = map(float, retinotopy_metric(region_mask, azimuth))
            (
                region_data["altitude_min"],
                region_data["altitude_max"],
                region_data["altitude_range"],
                region_data["altitude_bias"],
            ) = map(float, retinotopy_metric(region_mask, altitude))

            # eccentricity at centroid
            region_data["eccentricity_at_centroid"] = float(
                eccentricity_ret_zero[
                    round(region_data["y_centroid"]),
                    round(region_data["x_centroid"]),
                ]
            )

            outlines.append(outline_mask(region_mask))

            if structure_id == 1:
                VISp_blob = region_data

            all_region_data[CLASS_NAME_MAP[structure_id]] = region_data

        if VISp_blob is None:
            raise Exception(
                "Expected to find VISp region with structure id of 385 but did not find one"
            )

        azimuth_target = azimuth[
            round(VISp_blob["y_centroid"]), round(VISp_blob["x_centroid"])
        ]
        altitude_target = altitude[
            round(VISp_blob["y_centroid"]), round(VISp_blob["x_centroid"])
        ]
        eccentricity_V1_centroid = np.degrees(
            eccentricity(azimuth, altitude, azimuth_target, altitude_target)
        )

        #
        # Create higher resolution vasculature map background image
        data = window_level(vasculature, np.min(vasculature), np.max(vasculature))
        background_im = Image.fromarray(np.uint8(plt.cm.gray(data) * 255))

        # ----------------------------
        # This creates eccentricity retintopic zero map
        # ----------------------------

        # foreground - eccentricity map
        data = window_level(eccentricity_ret_zero, 0, self.ecc_zero_max)
        foreground_im = Image.fromarray(np.uint8(plt.cm.hsv(data) * 255))

        # resize foreground image to background
        foreground_im = foreground_im.resize(background_im.size)

        # alpha blend images
        composite_zero_im = Image.blend(foreground_im, background_im, 0.65)

        # --------------------------
        # This creates eccentricity from V1 centroid map and targeting map
        # ---------------------------
        data = window_level(eccentricity_V1_centroid, 0, self.ecc_v1_max)
        foreground_im = Image.fromarray(np.uint8(plt.cm.hsv(data) * 255))

        # resize foreground image to background
        foreground_im = foreground_im.resize(background_im.size)

        # alpha blend images
        composite_v1_im = Image.blend(foreground_im, background_im, 0.65)

        # erosion structure element
        structure = ndimage.generate_binary_structure(2, 1)
        structure = ndimage.iterate_structure(structure, 3)

        line_alpha = self.line_alpha

        target_mask_ten = eccentricity_V1_centroid < 10.0
        target_mask_five = eccentricity_V1_centroid < 5.0
        target_mask_one = eccentricity_V1_centroid < 1.0

        target_mask_ten = outline_mask(target_mask_ten)
        target_mask_five = outline_mask(target_mask_five)
        target_mask_one = outline_mask(target_mask_one)

        # resize mask
        outline_im_ten = Image.fromarray(np.uint8(target_mask_ten * line_alpha))
        outline_im_ten = outline_im_ten.resize(background_im.size)

        outline_im_five = Image.fromarray(np.uint8(target_mask_five * line_alpha))
        outline_im_five = outline_im_five.resize(background_im.size)

        outline_im_one = Image.fromarray(np.uint8(target_mask_one * line_alpha))
        outline_im_one = outline_im_one.resize(background_im.size)

        red_im = Image.fromarray(
            np.uint8(plt.cm.Reds_r(np.zeros(vasculature.shape)) * 255)
        )
        green_im = Image.fromarray(
            np.uint8(plt.cm.Greens_r(np.zeros(vasculature.shape)) * 255)
        )

        # erode mask
        arr_ten = np.asarray(outline_im_ten, dtype=np.uint8)
        arr_ten = ndimage.binary_erosion(arr_ten, structure).astype(arr_ten.dtype)
        outline_im_ten = Image.fromarray(np.uint8(arr_ten * line_alpha))

        arr_five = np.asarray(outline_im_five, dtype=np.uint8)
        arr_five = ndimage.binary_erosion(arr_five, structure).astype(arr_five.dtype)
        outline_im_five = Image.fromarray(np.uint8(arr_five * line_alpha))

        arr_one = np.asarray(outline_im_one, dtype=np.uint8)
        arr_one = ndimage.binary_erosion(arr_one, structure).astype(arr_one.dtype)
        outline_im_one = Image.fromarray(np.uint8(arr_one * line_alpha))

        target_im = Image.composite(foreground_im, background_im, outline_im_ten)
        target_im = Image.composite(green_im, target_im, outline_im_five)
        target_im = Image.composite(red_im, target_im, outline_im_one)

        # generate segmentation outline image
        region_outlines = np.zeros(altitude.shape)
        # for idx, r in enumerate(output_json['annotated_regions']) :
        # region_outlines[r['outline_mask'] > 0] = 1.0

        for outline in outlines:
            region_outlines[outline > 0] = 1.0

        # resize mask
        outline_im = Image.fromarray(np.uint8(region_outlines * line_alpha))
        outline_im = outline_im.resize(background_im.size)

        # erode mask
        arr = np.asarray(outline_im, dtype=np.uint8)
        arr = ndimage.binary_erosion(arr, structure).astype(arr.dtype)
        outline_im = Image.fromarray(np.uint8(arr * line_alpha))

        # create a blue image
        blue_im = Image.fromarray(
            np.uint8(plt.cm.Blues_r(np.zeros(vasculature.shape)) * 255)
        )

        # composite the region outline with the target image
        target_im = Image.composite(blue_im, target_im, outline_im)

        # Return all images and metrics
        return all_region_data, composite_zero_im, composite_v1_im, target_im

    def create_visual_sign_image(self) -> Tuple[Image.Image, Image.Image]:
        """Create a visual sign image using a colormap and alpha mask and return the PIL image."""
        arr = self.data.visual_sign
        alpha = 85.0
        threshold = 0.25
        mask = np.uint8((abs(arr) > threshold) * alpha)
        arr = (arr + 1.0) / 2.0
        sign_im = Image.fromarray(np.uint8(plt.cm.jet(arr) * 255))
        mask_im = Image.fromarray(mask)
        return sign_im, mask_im

    def create_retinotopy_altitude_image(self) -> Image.Image:
        """Create a retinotopy altitude image from the input file and return the PIL image."""
        arr = self.data.retinotopy_altitude
        arr = (arr + math.pi) / (2.0 * math.pi)
        im = Image.fromarray(np.uint8(plt.cm.hsv(arr) * 255))
        return im

    def create_retinotopy_azimuth_image(self) -> Image.Image:
        """Create a retinotopy azimuth image from the input file and return the PIL image."""
        arr = self.data.retinotopy_azimuth
        arr = (arr + math.pi) / (2.0 * math.pi)
        im = Image.fromarray(np.uint8(plt.cm.hsv(arr) * 255))
        return im

    def create_vasculature_image(self) -> Image.Image:
        """Create a vasculature image from the input file and return the PIL image."""
        arr = self.data.vasculature_image.astype(float)
        arr = arr / arr.max()
        im = Image.fromarray(np.uint8(plt.cm.gray(arr) * 255))
        return im

    def create_defocus_image(self) -> Image.Image:
        """Create a defocus image from the input file and return the PIL image."""
        arr = self.data.defocus_image.astype(float)
        arr = arr / arr.max()
        im = Image.fromarray(np.uint8(plt.cm.gray(arr) * 255))
        return im

    def create_isi_overlay_image(
        self, vasculature_im: Image.Image, sign_im: Image.Image, mask_im: Image.Image
    ) -> Image.Image:
        """Create an ISI overlay image (visual sign over vasculature) with alpha mask and return the PIL image.
        Parameters:
            vasculature_im (PIL.Image): Vasculature image (already processed, window-leveled and colormapped here).
            sign_im (PIL.Image): Visual sign image (already processed and resized).
            mask_im (PIL.Image): Alpha mask image (already processed and resized).
        """
        # Window-level the vasculature image to [0.1, 1] and apply gray colormap
        vasculature_arr = np.array(vasculature_im)
        # If the input is RGBA or RGB, convert to grayscale first
        if vasculature_arr.ndim == 3:
            # Only use the first channel if it's multi-channel (e.g., RGB)
            vasculature_arr = vasculature_arr[..., 0]
        vasculature_arr = vasculature_arr.astype(float)
        if vasculature_arr.max() > 0:
            vasculature_arr = (vasculature_arr - 0.1 * vasculature_arr.max()) / (
                0.9 * vasculature_arr.max()
            )
            vasculature_arr = np.clip(vasculature_arr, 0, 1)
            vasculature_im_proc = Image.fromarray(
                np.uint8(plt.cm.gray(vasculature_arr)[:, :, 0] * 255)
                if vasculature_arr.ndim == 3
                else np.uint8(plt.cm.gray(vasculature_arr) * 255)
            )
        else:
            vasculature_im_proc = vasculature_im

        # Resize vasculature to match sign image size
        vasculature_im_proc = vasculature_im_proc.resize(sign_im.size)
        mask_im = mask_im.resize(sign_im.size)

        # Composite overlay
        composite_im = Image.composite(sign_im, vasculature_im_proc, mask_im)
        return composite_im
