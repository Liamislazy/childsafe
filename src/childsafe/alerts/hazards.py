"""Context policy, separate from the checkpoint's historical safe/unsafe labels."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HazardAssessment:
    status: str
    kind: str
    policy_version: str = 'mouth-proximity-v1'


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
