"""Context policy, separate from the checkpoint's historical safe/unsafe labels."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HazardAssessment:
    status: str
    kind: str
    policy_version: str = 'mouth-proximity-v1'


def strong_detector_category(features, cfg):
    """Raw category evidence only, never a calibrated probability or contact claim."""
    policy = cfg.get('hazard_policy', {})
    label, score = features.get('object_label', ''), features.get('object_conf', math.nan)
    try:
        strong = math.isfinite(score) and score >= policy.get('detector_min_conf', .75)
    except TypeError:
        strong = False
    usable = (features.get('frame_valid', True) and features.get('source_ok', True)
              and features.get('detector_ok', False) and features.get('object_present', False)
              and not features.get('association_ambiguous', False))
    return label if usable and strong and isinstance(label, str) and label else None


def detector_hazard(features, cfg):
    """Independent category evidence; caller still requires visible stable mask+dwell.

    Boxes and unknown classes cannot supply confirmed localization regardless of score.
    """
    policy = cfg.get('hazard_policy', {})
    kinds = dict(policy.get('mouth_hazards', {}))
    kinds.update(policy.get('supplemental_hazards', {}))
    label = strong_detector_category(features, cfg)
    return HazardAssessment('known_hazard', kinds[label], 'mouth-proximity-v2') if label in kinds else None


def assess_hazard(category, dataset_status, cfg, object_gate=True):
    if not object_gate:
        return HazardAssessment('known_hazard','ablation_any_object')
    if dataset_status in ('none','uncertain'):
        return HazardAssessment('unclassified','unknown')
    kinds = cfg.get('hazard_policy',{}).get('mouth_hazards',{})
    kind = kinds.get(category)
    if kind and dataset_status == 'unsafe':
        return HazardAssessment('known_hazard',kind)
    # A toy/fruit/spoon label says nothing about size, fragments or suitability.
    return HazardAssessment('context_unknown','unknown')
