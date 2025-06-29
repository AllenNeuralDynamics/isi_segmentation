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
from isi_segmentation.isi_types import PathLike

import datetime


def make_data_description():
    return None


def make_processing():
    return None


def make_quality_control(
    sign_map_path: PathLike,
    label_map_path: PathLike,
    retinotopy_vertical_path: PathLike,
    retinotopy_horizontal_path: PathLike,
    vasculature_path: PathLike,
    isi_imaging_plane_path: PathLike,
    isi_overlay_path: PathLike,
    eccentricity_retinotopic_zero_path: PathLike,
    eccentricity_v_one_centroid_path: PathLike,
    target_map_path: PathLike,
):

    t = datetime.datetime.now()
    passed = QCStatus(evaluator="Automated", status=Status.PASS, timestamp=t)

    s = QCStatus(evaluator="Automated", status=Status.PASS, timestamp=t)
    segmentation_eval = QCEvaluation(
        name="Sign map segmentation",
        description="Check areas were segmented properly",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSED,
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

    return QualityControl(evaluations=[segmentation_eval])
