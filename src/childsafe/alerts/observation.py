"""Observation quality is independent of risk. Missing evidence is never green."""
from __future__ import annotations


def observation(f: dict) -> tuple[str, list[str]]:
    if 'schema_version' not in f:
        return 'degraded', ['legacy: no observation quality data']
    reasons = list(f.get('quality_reasons', []))
    if not f.get('source_ok', True):
        return 'unobservable', ['source disconnected']
    if not f.get('frame_valid', True):
        return 'unobservable', ['stale or invalid frame']
    if not f.get('mouth_visible', False):
        return 'unobservable', reasons + ['mouth not observable']
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
