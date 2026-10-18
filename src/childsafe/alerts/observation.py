"""Observation quality is independent of risk. Missing evidence is never green."""
from __future__ import annotations


def image_observation(f: dict) -> tuple[str, list[str]]:
    """Image usability only; does not certify models, hazard scope or calibration."""
    if 'schema_version' not in f:
        return 'degraded', ['legacy: no observation quality data']
    if not f.get('source_ok', True):
        return 'unobservable', ['source disconnected']
    if not f.get('frame_valid', True):
        return 'unobservable', ['stale or invalid frame']
    reasons = [r for r in f.get('quality_reasons',[]) if r in {
        'poor exposure','blurred image','face too small','mouth too small','hand too small',
        'no observable hand','head orientation unavailable','mouth hidden by head orientation',
        'unstable mouth landmarks','landmark model unavailable'}]
    if not f.get('mouth_visible', False):
        return 'unobservable', reasons + ['mouth not observable']
    if not f.get('hand_ok',False) and 'no observable hand' not in reasons:
        reasons.append('no observable hand')
    return ('degraded' if reasons else 'observable'), reasons


def observation(f: dict) -> tuple[str, list[str]]:
    image_state, _ = image_observation(f)
    if 'schema_version' not in f or image_state == 'unobservable':
        return image_observation(f)
    reasons = list(f.get('quality_reasons', []))
    if image_state == 'degraded':
        reasons.extend(r for r in image_observation(f)[1] if r not in reasons)
    if f.get('association_ambiguous', False):
        reasons.append('ambiguous association')
    if not f.get('detector_ok', False):
        reasons.append('detector unavailable')
    if not f.get('classifier_ok', False):
        reasons.append('classifier unavailable')
    if not f.get('calibrated', False):
        reasons.append('elevated-view thresholds unvalidated')
    if not f.get('object_coverage', False):
        reasons.append('object class coverage limited')
    return ('degraded' if reasons else 'observable'), reasons
