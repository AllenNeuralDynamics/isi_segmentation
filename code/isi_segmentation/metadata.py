from aind_data_schema.core.quality_control import (
    QCEvaluation,
    QualityControl,
    QCMetric,
    Stage,
    Status,
    QCStatus,
)
from aind_data_schema.core.data_description import DataDescription, DataLevel
from aind_data_schema.core.processing import Processing
from aind_data_schema_models.modalities import Modality
from pathlib import Path

import datetime


def make_data_description():
    return None


def make_processing():
    return None


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

    s = QCStatus(evaluator="Automated", status=Status.PASS, timestamp=t)
    segmentation_eval = QCEvaluation(
        name="Sign map segmentation",
        description="Check areas were segmented properly",
        modality=Modality.POPHYS,
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
        ],
        notes="",
        created=t,
    )

    raw_qc_eval = QCEvaluation(
        name="Raw ISI QC",
        description="Check quality of raw ISI data",
        modality=Modality.POPHYS,
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
            QCMetric(
                name="Target map",
                description="Qualitative evaluation of ISI target map",
                value="passed",
                reference=target_map_path,
                status_history=[passed],
            )
        ],
        notes="",
        created=t,
    )

    return QualityControl(evaluations=[segmentation_eval, raw_qc_eval])
