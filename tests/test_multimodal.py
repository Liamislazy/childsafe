from dataclasses import replace

import pytest

from childsafe.runtime.multimodal import FeatureTimeline, SignalFeatures


def sample(t, *, modality="visual", subject="p1", **kwargs):
    defaults = dict(source_id="cam" if modality == "visual" else "mic",
                    generation=1, epoch="e1", modality=modality, start=t,
                    end=t if modality == "visual" else t + .1,
                    subject_id=subject, features=(0.,), clock_verified=True,
                    attribution_verified=subject is not None)
    defaults.update(kwargs)
    return SignalFeatures(**defaults)


def timeline(**kwargs):
    result = FeatureTimeline("e1", **kwargs)
    result.register_source("cam", 1)
    result.register_source("mic", 1)
    return result


def populate(buf, *, audio_subject="p1", **kwargs):
    for t in (0., .2, .4, .6, .8):
        assert buf.append(sample(t), now=1.)
        assert buf.append(sample(t, modality="audio", subject=audio_subject, **kwargs), now=1.)


def test_zero_audio_features_are_valid_but_missing_microphone_is_not_silence():
    buf = timeline()
    populate(buf)
    w = buf.window("p1", start=0., end=1., now=1.)
    assert w.aligned and len(w.audio) == 5
    # Explicit device failure cannot be hidden by otherwise valid windows.
    buf.append(sample(.9, modality="audio", available=False, features=()), now=1.)
    w = buf.window("p1", start=0., end=1., now=1.)
    assert not w.aligned and "audio_source_unavailable" in w.reasons
    with pytest.raises(ValueError):
        sample(.9, modality="audio", available=False)


def test_room_audio_cannot_be_assigned_to_a_visible_person():
    buf = timeline()
    populate(buf, audio_subject=None)
    w = buf.window("p1", start=0., end=1., now=1.)
    assert w.audio == () and len(w.room_audio) == 5
    assert "audio_subject_unverified" in w.reasons
    assert not w.aligned
    assert not buf.window("p2", start=0., end=1., now=1.).visual


def test_unverified_capture_clock_and_other_person_are_never_fused():
    buf = timeline()
    populate(buf, clock_verified=False)
    w = buf.window("p1", start=0., end=1., now=1.)
    assert not w.audio and "audio_clock_unverified" in w.reasons
    buf = timeline()
    populate(buf, audio_subject="p2")
    w = buf.window("p1", start=0., end=1., now=1.)
    assert not w.audio and not w.room_audio and not w.aligned


def test_reconnect_rejects_late_callbacks_and_clears_only_that_source():
    buf = timeline()
    populate(buf)
    buf.register_source("mic", 2)
    assert not buf.append(sample(.8, modality="audio"), now=1.)
    w = buf.window("p1", start=0., end=1., now=1.)
    assert len(w.visual) == 5 and not w.audio
    with pytest.raises(ValueError):
        buf.register_source("mic", 1)
    buf.reset("e2")
    buf.register_source("cam", 1)
    assert len(buf) == 0
    assert not buf.append(sample(.8), now=1.)
    assert buf.append(replace(sample(.8), epoch="e2"), now=1.)


def test_gaps_expiry_capacity_future_and_backwards_clock():
    buf = timeline(horizon_s=2., max_samples=3)
    for t in (.1, .2, .3, .4):
        buf.append(sample(t), now=1.)
    assert len(buf) == 3
    w = buf.window("p1", start=0., end=1., now=1.)
    assert "visual_gap" in w.reasons
    assert not buf.append(sample(2.), now=1.)
    assert not buf.append(sample(.1), now=3.)
    assert len(buf) == 0
    assert "stale_window" in buf.window("p1", start=0., end=1., now=3.).reasons
    with pytest.raises(ValueError):
        buf.append(sample(2.), now=2.)


def test_out_of_order_packets_are_sorted_without_future_or_boundary_audio_leakage():
    buf = timeline()
    for t in (.8, .6, .4, .2, 0.):
        buf.append(sample(t), now=1.)
        buf.append(sample(t, modality="audio"), now=1.)
    buf.append(sample(.95, modality="audio", end=1.2), now=1.2)
    w = buf.window("p1", start=0., end=1., now=1.2)
    assert w.aligned
    assert [s.start for s in w.visual] == [0., .2, .4, .6, .8]
    assert all(s.end <= 1. for s in w.audio)
    w = buf.window("p1", start=.05, end=1., now=1.2)
    assert all(s.start >= .05 for s in w.audio)


@pytest.mark.parametrize("kwargs", [dict(start=float("nan")), dict(features=(float("inf"),)),
                                    dict(features=()), dict(features=(0.,) * 4097),
                                    dict(subject_id=None), dict(end=-1.), dict(generation=-1)])
def test_invalid_contracts_rejected(kwargs):
    with pytest.raises(ValueError):
        sample(0., **kwargs)


def test_unknown_source_and_expired_window_cannot_be_aligned():
    buf = timeline()
    assert not buf.append(sample(0., source_id="unregistered"), now=1.)
    populate(buf)
    w = buf.window("p1", start=0., end=1., now=2.)
    assert "stale_window" in w.reasons and not w.aligned
