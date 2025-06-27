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
    parser.add_argument('--model_path', type=str, default='../data/isi_segmentation_model/isi_segmentation_model.h5',
                        help='path to the trained isi-segmentation model')    
    parser.add_argument('--output_dir', type=str, default='../results/', help='folder to store outputs')
    
    args = parser.parse_args()
    
    print(f"Using sample file: {args.hdf5_path}")
    
    segmentation_dir = os.path.join(args.output_dir, 'segmentation')
    os.makedirs(segmentation_dir, exist_ok=True)

    qc_dir = os.path.join(args.output_dir, 'qc')
    os.makedirs(qc_dir, exist_ok=True)
    
    # predict the label map for the sign map.
    data = ISIData.from_files(args.hdf5_path)

    sign_map_path = os.path.join(segmentation_dir, 'sign_map.png')
    label_map_path = os.path.join(segmentation_dir, 'label_map.png')

    data.label_map_image = predict(
        data=data,
        sign_map_path=sign_map_path,
        label_map_path=label_map_path,
        model_path=args.model_path
    )

    print(f"Using label map: {label_map_path}")

    # Instantiate loader and metrics/visualization
    metrics = ISIMetricsAndVisualization(data)

    # Example output paths (these would be replaced with real ones in production)
    retinotopy_vertical_path = os.path.join(qc_dir, "retinotopy_vertical.png")
    retinotopy_horizontal_path = os.path.join(qc_dir, "retinotopy_horizontal.png")
    vasculature_path = os.path.join(qc_dir, "vasculature.png")
    isi_imaging_plane_path = os.path.join(qc_dir, "defocus.png")
    isi_overlay_path = os.path.join(qc_dir, "overlay.png")
    eccentricity_retinotopic_zero_path = os.path.join(qc_dir, "eccentricity_retinotopic_zero.png")
    eccentricity_v_one_centroid_path = os.path.join(qc_dir, "eccentricity_v_one_centroid.png")
    target_map_path = os.path.join(qc_dir, "target_map.png")

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
    
