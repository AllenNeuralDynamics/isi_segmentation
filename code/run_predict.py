"""Run inference on a sign map"""

import argparse
from isi_segmentation.prediction import predict
from isi_segmentation.stats import ISIData, ISIMetricsAndVisualization
from isi_segmentation.metadata import (
    make_data_description,
    make_processing,
    make_quality_control,
)
import os, glob
import logging
from pathlib import Path


if __name__ == "__main__":
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="../data/**/*_processed.hdf5",
        help="pattern to the hdf5 file which contains the testing sign map",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../data/isi_segmentation_model/isi_segmentation_model.h5",
        help="path to the trained isi-segmentation model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results/",
        help="folder to store outputs",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument('--altitude_scale', type=float, default=0.322, help='Scaling factor for altitude phase')
    parser.add_argument('--azimuth_scale', type=float, default=0.383, help='Scaling factor for azimuth phase')
    parser.add_argument('--line_alpha', type=float, default=100.0, help='Alpha value for outline overlays')
    parser.add_argument('--ecc_zero_max', type=float, default=50.0, help='Max value for eccentricity_ret_zero windowing')
    parser.add_argument('--ecc_v1_max', type=float, default=50.0, help='Max value for eccentricity_V1_centroid windowing')

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    hdf5_path = glob.glob(args.input_pattern)[0]

    logging.info(f"Using sample file: {hdf5_path}")

    segmentation_dir = os.path.join(args.output_dir, "segmentation")
    os.makedirs(segmentation_dir, exist_ok=True)

    qc_dir = os.path.join(args.output_dir, "qc")
    os.makedirs(qc_dir, exist_ok=True)

    # predict the label map for the sign map.
    data = ISIData.from_files(hdf5_path)

    sign_map_path = os.path.join(segmentation_dir, "sign_map.png")
    label_map_path = os.path.join(segmentation_dir, "label_map.png")

    data.label_map_image = predict(
        data=data,
        sign_map_path=sign_map_path,
        label_map_path=label_map_path,
        model_path=args.model_path,
    )

    logging.info(f"Using label map: {label_map_path}")

    # Instantiate loader and metrics/visualization
    metrics = ISIMetricsAndVisualization(
        data,
        altitude_scale=args.altitude_scale,
        azimuth_scale=args.azimuth_scale,
        line_alpha=args.line_alpha,
        ecc_zero_max=args.ecc_zero_max,
        ecc_v1_max=args.ecc_v1_max,
    )

    retinotopy_vertical_path = os.path.join(qc_dir, "retinotopy_vertical.png")
    retinotopy_horizontal_path = os.path.join(
        qc_dir, "retinotopy_horizontal.png"
    )
    vasculature_path = os.path.join(qc_dir, "vasculature.png")
    isi_imaging_plane_path = os.path.join(qc_dir, "defocus.png")
    isi_overlay_path = os.path.join(qc_dir, "overlay.png")
    eccentricity_retinotopic_zero_path = os.path.join(
        qc_dir, "eccentricity_retinotopic_zero.png"
    )
    eccentricity_v_one_centroid_path = os.path.join(
        qc_dir, "eccentricity_v_one_centroid.png"
    )
    target_map_path = os.path.join(qc_dir, "target_map.png")

    metrics.create_visual_sign_image(sign_map_path)
    metrics.create_retinotopy_altitude_image(retinotopy_vertical_path)
    metrics.create_retinotopy_azimuth_image(retinotopy_horizontal_path)
    metrics.create_vasculature_image(vasculature_path)
    # TODO: missing overlay.png right now (isi_overlay_path)
    metrics.create_defocus_image(isi_imaging_plane_path)

    # Try to run QC metrics and target map generation
    try:
        metrics.generate_qc_metric_and_images(
            eccentricity_retinotopic_zero_path,
            eccentricity_v_one_centroid_path,
            target_map_path,
        )
        logging.info("QC metrics and target map generated successfully.")
    except Exception as e:
        logging.error(f"Error running generate_qc_metric_and_images: {e}")

    dd = make_data_description()
    if dd:
        dd.write_standard_file(output_directory="../results")

    processing = make_processing()
    if processing:
        processing.write_standard_file(output_directory="../results")

    qc = make_quality_control(
        sign_map_path,
        label_map_path,
        retinotopy_vertical_path,
        retinotopy_horizontal_path,
        vasculature_path,
        isi_imaging_plane_path,
        isi_overlay_path,
        eccentricity_retinotopic_zero_path,
        eccentricity_v_one_centroid_path,
        target_map_path,
    )
    if qc:
        qc.write_standard_file(output_directory="../results")

    logging.info("Run complete.")
