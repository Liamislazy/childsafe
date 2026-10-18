"""Named contracts at runtime boundaries; CSVs remain plain, backward-compatible records."""
from typing import TypedDict,TYPE_CHECKING

if TYPE_CHECKING:
    from childsafe.alerts.processor import AlertProcessor
    from childsafe.vision.geometry import GeometryExtractor
    from childsafe.vision.elevated import ElevatedQuality
    from childsafe.vision.objects import MaskEvidence
    from childsafe.vision.landmarks import LandmarkDetector


class FeatureRow(TypedDict,total=False):
    t: float
    person_id: int
    session_id: int
    schema_version: int
    source_ok: bool
    frame_valid: bool
    mouth_visible: bool
    hand_ok: bool
    detector_ok: bool
    classifier_ok: bool
    calibrated: bool
    object_coverage: bool
    association_ambiguous: bool
    history_reset: bool
    object_present: bool
    mask_reliable: bool
    mask_temporally_stable: bool
    proximity_evidence: bool
    quality_reasons: list[str]
    evidence_source: str
    p_unsafe_raw: float
    p_unsafe_cal: float
    obj_class: str
    obj_conf: float


class PersonStateRequired(TypedDict):
    processor: 'AlertProcessor'
    geometry: 'GeometryExtractor'
    quality: 'ElevatedQuality'
    mask_evidence: 'MaskEvidence'
    landmarks: 'LandmarkDetector | None'
    landmark_failed: bool
    error: str
    classifier_error: str
    features: FeatureRow
    object_box: tuple | None


class PersonState(PersonStateRequired,total=False):
    object_label: str
    classified: tuple
    classified_hand: tuple
    obj_result: tuple
    box: tuple
