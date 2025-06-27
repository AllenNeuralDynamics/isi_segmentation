"""Run inference on a sign map """
import argparse
from isi_segmentation.prediction import predict
from isi_segmentation.stats import ISIData, ISIMetricsAndVisualization
import os

if __name__ == "__main__":
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_path', type=str, default=None, required=True, 
                        help='path to the hdf5 file which contains the testing sign map')
    parser.add_argument('--sign_map_path', type=str, default='../results/sign_map.jpg',
                        help='path to the sign map')
    parser.add_argument('--label_map_path', type=str, default='../results/label_map.png',
                        help='path to save the label map')
    parser.add_argument('--model_path', type=str, default='../data/isi_segmentation_model/isi_segmentation_model.h5',
                        help='path to the trained isi-segmentation model')
    
    args = parser.parse_args()
    
    # predict the label map for the sign map.
    label_map = predict(**vars(args))

    hdf5_file = args.hdf5_path
    label_map_path = args.label_map_path
    print(f"Using sample file: {hdf5_file}")
    print(f"Using label map: {label_map_path}")

    # Instantiate loader and metrics/visualization
    data = ISIData.from_files(hdf5_file, label_map_path=label_map_path)
    metrics = ISIMetricsAndVisualization(data)

    # Example output paths (these would be replaced with real ones in production)
    output_dir_path = "/results/qc_images"
    os.makedirs(output_dir_path, exist_ok=True)
    sign_map_path = os.path.join(output_dir_path, "sign_map.png")
    retinotopy_vertical_path = os.path.join(output_dir_path, "retinotopy_vertical.png")
    retinotopy_horizontal_path = os.path.join(output_dir_path, "retinotopy_horizontal.png")
    vasculature_path = os.path.join(output_dir_path, "vasculature.png")
    isi_imaging_plane_path = os.path.join(output_dir_path, "defocus.png")
    isi_overlay_path = os.path.join(output_dir_path, "overlay.png")
    eccentricity_retinotopic_zero_path = os.path.join(output_dir_path, "eccentricity_retinotopic_zero.png")
    eccentricity_v_one_centroid_path = os.path.join(output_dir_path, "eccentricity_v_one_centroid.png")
    target_map_path = os.path.join(output_dir_path, "target_map.png")

    # Call representative methods
    metrics.create_visual_sign_image(sign_map_path)
    metrics.create_retinotopy_altitude_image(retinotopy_vertical_path)
    metrics.create_retinotopy_azimuth_image(retinotopy_horizontal_path)
    metrics.create_vasculature_image(vasculature_path)
    metrics.create_defocus_image(isi_imaging_plane_path)

    # Try to run QC metrics and target map generation
    try:
        metrics.generate_qc_metric_and_images(
            eccentricity_retinotopic_zero_path,
            eccentricity_v_one_centroid_path,
            target_map_path
        )
        print("QC metrics and target map generated successfully.")
    except Exception as e:
        print(f"Error running generate_qc_metric_and_images: {e}")

    print("Test run complete.")
    
