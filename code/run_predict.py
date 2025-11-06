"""Run inference on a sign map"""

import argparse
import glob
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from isi_segmentation.metadata import (make_data_description, make_processing,
                                       make_quality_control)
from isi_segmentation.prediction import predict
from isi_segmentation.stats import ISIData, ISIMetricsAndVisualization


def copy_schema_file(filepath: Path, output_directory: Path):
    """Copy a schema file to the output directory if it exists."""
    if os.path.exists(filepath):
        shutil.copy2(filepath, output_directory)
        print(f"Copied {filepath.name} to {output_directory}")

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
    parser.add_argument(
        "--altitude_scale",
        type=float,
        default=0.322,
        help="Scaling factor for altitude phase",
    )
    parser.add_argument(
        "--azimuth_scale",
        type=float,
        default=0.383,
        help="Scaling factor for azimuth phase",
    )
    parser.add_argument(
        "--line_alpha",
        type=float,
        default=100.0,
        help="Alpha value for outline overlays",
    )
    parser.add_argument(
        "--ecc_zero_max",
        type=float,
        default=50.0,
        help="Max value for eccentricity_ret_zero windowing",
    )
    parser.add_argument(
        "--ecc_v1_max",
        type=float,
        default=50.0,
        help="Max value for eccentricity_V1_centroid windowing",
    )
    start_time = datetime.now()
    logging.info(f"Script started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
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
    region_metrics_path = os.path.join(segmentation_dir, "region_metrics.json")

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
    retinotopy_horizontal_path = os.path.join(qc_dir, "retinotopy_horizontal.png")
    vasculature_path = os.path.join(qc_dir, "vasculature.png")
    isi_imaging_plane_path = os.path.join(qc_dir, "defocus.png")
    isi_overlay_path = os.path.join(qc_dir, "isi_overlay.png")
    eccentricity_retinotopic_zero_path = os.path.join(
        qc_dir, "eccentricity_retinotopic_zero.png"
    )
    eccentricity_v_one_centroid_path = os.path.join(
        qc_dir, "eccentricity_v_one_centroid.png"
    )
    target_map_path = os.path.join(segmentation_dir, "target_map.png")

    # Create and save images using new non-IO create_* methods
    sign_im, mask_im = metrics.create_visual_sign_image()
    sign_im.save(sign_map_path)
    retinotopy_altitude_im = metrics.create_retinotopy_altitude_image()
    retinotopy_altitude_im.save(retinotopy_vertical_path)
    retinotopy_azimuth_im = metrics.create_retinotopy_azimuth_image()
    retinotopy_azimuth_im.save(retinotopy_horizontal_path)
    vasculature_im = metrics.create_vasculature_image()
    vasculature_im.save(vasculature_path)
    defocus_im = metrics.create_defocus_image()
    defocus_im.save(isi_imaging_plane_path)
    # ISI overlay
    isi_overlay_im = metrics.create_isi_overlay_image(vasculature_im, sign_im, mask_im)
    isi_overlay_im.save(isi_overlay_path)

    # Try to run QC metrics and target map generation
    region_metrics, ecc_zero_im, ecc_v1_im, target_map_im = (
        metrics.generate_qc_metric_and_images()
    )
    ecc_zero_im.save(eccentricity_retinotopic_zero_path)
    ecc_v1_im.save(eccentricity_v_one_centroid_path)
    target_map_im.save(target_map_path)
    logging.info("QC metrics and target map generated successfully.")

    with open(region_metrics_path, "w") as f:
        json.dump(region_metrics, f, indent=3)

    dd = make_data_description()
    if dd:
        dd.write_standard_file(output_directory="../results")

    processing = make_processing(
        start_time=start_time,
        input_path=hdf5_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        altitude_scale=args.altitude_scale,
        azimuth_scale=args.azimuth_scale,
        line_alpha=args.line_alpha,
        ecc_zero_max=args.ecc_zero_max,
        ecc_v1_max=args.ecc_v1_max,
    )
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

    data_pattern = Path("../data/")
    results_dir = Path("../results")

    rig_json = next(data_pattern.rglob("rig.json"),"")
    if rig_json:
        copy_schema_file(rig_json, results_dir)
    instrument_json = next(data_pattern.rglob("instrument.json"),"")
    if instrument_json:
        copy_schema_file(instrument_json, results_dir)
    
    session_json = next(data_pattern.rglob("session.json"),"")
    if session_json:
        copy_schema_file(session_json, results_dir)
    acquisition_json = next(data_pattern.rglob("acquisition.json"),"")
    if acquisition_json:
        copy_schema_file(acquisition_json, results_dir)
    logging.info("Run complete.")
