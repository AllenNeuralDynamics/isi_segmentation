import datetime
import json
from pathlib import Path

from aind_data_schema.core.data_description import DataDescription, DataLevel, DerivedDataDescription
from aind_data_schema.core.processing import (DataProcess, PipelineProcess,
                                              Processing)
from aind_data_schema.core.quality_control import (QCEvaluation, QCMetric,
                                                   QCStatus, QualityControl,
                                                   Stage, Status)
from aind_data_schema_models.modalities import Modality


def make_data_description(input_dir="../data") -> DerivedDataDescription:
    """Read data_description.json from input directory.

    Parameters
    ----------
    input_dir : str
        Directory where data_description.json is located. Defaults to "../data".

    Returns
    -------
    DerivedDataDescription
        An upgraded DerivedDataDescription object based on the existing data description.
    """
    data_description = None

    if input_dir:
        data_desc_path = next(Path(input_dir).rglob("data_description.json"))
        if data_desc_path.exists():
            with open(data_desc_path, "r") as f:
                data_description = json.load(f)
    data_description = DataDescription(**data_description)
    data_description.modality = [Modality.ISI]
    derived_data_description = DerivedDataDescription.from_data_description(
                                data_description, process_name="processed"
                            )

    return derived_data_description


def make_processing(
    start_time=None,
    input_path=None,
    model_path=None,
    output_dir=None,
    altitude_scale=0.322,
    azimuth_scale=0.383,
    line_alpha=100.0,
    ecc_zero_max=50.0,
    ecc_v1_max=50.0,
) -> Processing:
    """Create a Processing object for ISI segmentation pipeline."""

    current_time = datetime.datetime.now()

    # Create the main ISI segmentation data process
    isi_segmentation_process = DataProcess(
        name="Image atlas alignment",  # TODO: Review if this is the correct name
        software_version="1.0.0",  # Update this to match your actual version
        start_date_time=start_time or current_time,
        end_date_time=current_time,
        input_location=str(Path(input_path)) if input_path else "../data",
        output_location=str(Path(output_dir)) if output_dir else "../results",
        code_url="https://github.com/AllenNeuralDynamics/isi_segmentation",  # Update if different
        code_version="main",  # Update to actual commit hash or version
        parameters={
            "altitude_scale": altitude_scale,
            "azimuth_scale": azimuth_scale,
            "line_alpha": line_alpha,
            "ecc_zero_max": ecc_zero_max,
            "ecc_v1_max": ecc_v1_max,
            "model_path": str(model_path) if model_path else None,
        },
        outputs={
            "sign_map": "segmentation/sign_map.png",
            "label_map": "segmentation/label_map.png",
            "target_map": "segmentation/target_map.png",
            "region_metrics": "segmentation/region_metrics.json",
            "retinotopy_vertical": "qc/retinotopy_vertical.png",
            "retinotopy_horizontal": "qc/retinotopy_horizontal.png",
            "vasculature": "qc/vasculature.png",
            "defocus": "qc/defocus.png",
            "isi_overlay": "qc/isi_overlay.png",
            "eccentricity_retinotopic_zero": "qc/eccentricity_retinotopic_zero.png",
            "eccentricity_v_one_centroid": "qc/eccentricity_v_one_centroid.png",
        },
        notes="Automated segmentation of intrinsic signal imaging data to identify visual cortical areas",
    )

    # Create the PipelineProcess object
    pipeline_process = PipelineProcess(
        data_processes=[isi_segmentation_process],
        processor_full_name="ISI Segmentation Pipeline",
        pipeline_version="1.0.0",
        pipeline_url="https://github.com/AllenNeuralDynamics/isi_segmentation",
        note="Automated segmentation pipeline for intrinsic signal imaging data",
    )

    # Create the Processing object
    processing = Processing(
        processing_pipeline=pipeline_process,
        notes="Processing pipeline for ISI segmentation and quality control",
    )

    return processing


def make_quality_control(
    sign_map_path: Path,
    label_map_path: Path,
    retinotopy_vertical_path: Path,
    retinotopy_horizontal_path: Path,
    vasculature_path: Path,
    isi_imaging_plane_path: Path,
    isi_overlay_path: Path,
    eccentricity_retinotopic_zero_path: Path,
    eccentricity_v_one_centroid_path: Path,
    target_map_path: Path,
):

    t = datetime.datetime.now()
    passed = QCStatus(evaluator="Automated", status=Status.PASS, timestamp=t)

    segmentation_eval = QCEvaluation(
        name="Sign map segmentation",
        description="Check areas were segmented properly",
        modality=Modality.ISI,
        stage=Stage.PROCESSING,
        metrics=[
            QCMetric(
                name="Sign map",
                description="Qualitative evaluation of the sign map",
                value="passed",
                reference=sign_map_path,
                status_history=[passed],
            ),
            QCMetric(
                name="Label map",
                description="Qualitative check of predicted area labels",
                value="passed",
                reference=label_map_path,
                status_history=[passed],
            ),
            QCMetric(
                name="Target map",
                description="Qualitative evaluation of ISI target map",
                value="passed",
                reference=target_map_path,
                status_history=[passed],
            ),
        ],
        notes="",
        created=t,
    )

    raw_qc_eval = QCEvaluation(
        name="Raw ISI QC",
        description="Check quality of raw ISI data",
        modality=Modality.ISI,
        stage=Stage.PROCESSING,
        metrics=[
            QCMetric(
                name="Horizontal retinotopy",
                description="Qualitative evaluation of horizontal retinotopy image",
                value="passed",
                reference=retinotopy_horizontal_path,
                status_history=[passed],
            ),
            QCMetric(
                name="Vertical retinotopy",
                description="Qualitative evaluation of vertical retinotopy image",
                value="passed",
                reference=retinotopy_vertical_path,
                status_history=[passed],
            ),
            QCMetric(
                name="Vasculature",
                description="Qualitative evaluation of vasculature in cranial window",
                value="passed",
                reference=vasculature_path,
                status_history=[passed],
            ),
            QCMetric(
                name="Imaging plane (defocus)",
                description="Qualitative evaluation of ISI imaging plane via defocused image",
                value="passed",
                reference=isi_imaging_plane_path,
                status_history=[passed],
            ),
            QCMetric(
                name="Imaging plane (overlay)",
                description="Qualitative evaluation of ISI imaging plane via overlaid sign map",
                value="passed",
                reference=isi_overlay_path,
                status_history=[passed],
            ),
            QCMetric(
                name="Eccentricity (retinotopic zero)",
                description="Qualitative evaluation of eccentricity relative to retinotopic zero",
                value="passed",
                reference=eccentricity_retinotopic_zero_path,
                status_history=[passed],
            ),
            QCMetric(
                name="Eccentricity (VISp zero)",
                description="Qualitative evaluation of eccentricity relative to primary visual cortex",
                value="passed",
                reference=eccentricity_v_one_centroid_path,
                status_history=[passed],
            ),
        ],
        notes="",
        created=t,
    )

    return QualityControl(evaluations=[segmentation_eval, raw_qc_eval])
