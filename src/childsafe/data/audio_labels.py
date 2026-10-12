"""Audit COUGHVID acoustic annotations without converting them to FBAO diagnoses."""
import re

import pandas as pd


def _checkbox(value):
    if pd.isna(value) or value == "":
        return None
    token = str(value).strip().lower()
    if token in {"true", "1", "1.0"}:
        return True
    if token in {"false", "0", "0.0"}:
        return False
    raise ValueError("unsupported expert checkbox value")


def audit_coughvid_metadata(metadata: pd.DataFrame) -> dict:
    """Aggregate only; a recording UUID is not a verified participant identity.

    Counts refer to audible_choking checkbox annotations on cough recordings,
    never confirmed toy obstruction events. Unlabeled rows remain unlabeled.
    """
    if not {"uuid", "age"}.issubset(metadata.columns):
        raise ValueError("metadata requires uuid and age columns")
    if metadata.uuid.isna().any() or metadata.uuid.astype(str).str.strip().eq("").any():
        raise ValueError("missing recording UUID")
    if metadata.uuid.duplicated().any():
        raise ValueError("duplicate recording UUID; recordings must not be double counted")
    reviewers = sorted({int(m.group(1)) for c in metadata.columns
                        if (m := re.fullmatch(r"choking_(\d+)", c))})
    if not reviewers or any(f"quality_{i}" not in metadata for i in reviewers):
        raise ValueError("expected expert choking_N and quality_N columns")
    age = pd.to_numeric(metadata.age, errors="coerce")
    malformed_age = metadata.age.notna() & age.isna()
    if malformed_age.any() or ((age < 0) | (age > 120)).any():
        raise ValueError("invalid self-reported age")
    by_age = {name: {"recordings": 0, "any_positive": 0, "usable_any_positive": 0}
              for name in ("under_18_self_reported", "adult_self_reported", "age_unknown")}
    counts = dict(recordings=len(metadata), annotated_recordings=0, any_positive=0,
                  all_available_reviewers_positive=0, reviewer_disagreement=0,
                  at_least_two_reviewers_all_positive=0,
                  single_reviewer_positive=0,
                  usable_any_positive=0, usable_all_available_reviewers_positive=0,
                  usable_reviewer_disagreement=0, explicitly_negative_only=0,
                  unannotated_recordings=0)
    per_reviewer = {str(i): {"annotations": 0, "positive": 0, "usable_positive": 0}
                    for i in reviewers}
    for position, (_, row) in enumerate(metadata.iterrows()):
        values, usable = [], []
        for i in reviewers:
            value = _checkbox(row[f"choking_{i}"])
            if value is None:
                continue
            values.append(value)
            per_reviewer[str(i)]["annotations"] += 1
            per_reviewer[str(i)]["positive"] += int(value)
            q = row[f"quality_{i}"]
            quality = "" if pd.isna(q) else str(q).strip().lower()
            if quality not in {"", "good", "ok", "poor", "no_cough"}:
                raise ValueError("unsupported expert audio quality")
            if quality in {"good", "ok"}:
                usable.append(value)
                per_reviewer[str(i)]["usable_positive"] += int(value)
        a = age.iloc[position]
        group = by_age["age_unknown" if pd.isna(a) else
                       "under_18_self_reported" if a < 18 else "adult_self_reported"]
        group["recordings"] += 1
        group["any_positive"] += int(any(values))
        group["usable_any_positive"] += int(any(usable))
        if not values:
            counts["unannotated_recordings"] += 1
            continue
        counts["annotated_recordings"] += 1
        counts["any_positive"] += int(any(values))
        counts["all_available_reviewers_positive"] += int(all(values))
        counts["at_least_two_reviewers_all_positive"] += int(len(values) >= 2 and all(values))
        counts["single_reviewer_positive"] += int(len(values) == 1 and values[0])
        counts["reviewer_disagreement"] += int(any(values) and not all(values))
        counts["explicitly_negative_only"] += int(not any(values))
        counts["usable_any_positive"] += int(any(usable))
        counts["usable_all_available_reviewers_positive"] += int(bool(usable) and all(usable))
        counts["usable_reviewer_disagreement"] += int(any(usable) and not all(usable))
    return {
        "target": "audible_choking_on_cough_recordings",
        "counts": counts, "by_self_reported_age": by_age, "per_reviewer": per_reviewer,
        "expert_columns": [f"choking_{i}" for i in reviewers],
        "confirmed_pediatric_toy_obstruction_events": None,
        "clinical_target_validation": "not established by these acoustic annotations",
        "subject_independent_split_possible_from_uuid": False,
        "definitions": {
            "any_positive": "at least one reviewer marked the acoustic checkbox true",
            "all_available_reviewers_positive": "all nonmissing checkboxes true; may be one reviewer",
            "usable": "only checkboxes whose same reviewer rated quality good or ok",
            "explicitly_negative_only": "all available checkboxes false; not a clinical non-choking label",
            "age": "self-reported, not verified pediatric cohort membership",
        },
        "limitations": [
            "No synchronized room-camera video or confirmed toy cause in this audit.",
            "Missing checkbox is not a negative label; poor/no_cough excluded from usable counts.",
            "No onset/offset labels for clinical choking events are established here.",
            "A silent severe event cannot be learned from audible positives alone.",
            "status_SSL and cough_detected are not choking ground truth.",
        ],
    }
