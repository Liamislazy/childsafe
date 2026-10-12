"""Bounded, in-memory feature alignment for research; never diagnoses choking.

Producers must supply capture timestamps on one verified monotonic clock. Arrival
timestamps from VideoSource are insufficient for audiovisual synchronization.
No capture hardware, recording, resampling, or identity inference happens here.
"""
from collections import deque
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class SignalFeatures:
    source_id: str
    generation: int
    epoch: str
    modality: str
    start: float
    end: float
    subject_id: str | None
    features: tuple[float, ...] = ()
    available: bool = True
    clock_verified: bool = False
    attribution_verified: bool = False

    def __post_init__(self):
        if not self.source_id or not self.epoch or self.modality not in {"visual", "audio"}:
            raise ValueError("source, epoch and supported modality required")
        if self.generation < 0 or not isinstance(self.generation, int):
            raise ValueError("generation must be a nonnegative integer")
        if not all(isfinite(t) for t in (self.start, self.end)) or self.end < self.start:
            raise ValueError("invalid capture interval")
        if self.modality == "audio" and self.end == self.start:
            raise ValueError("audio needs a nonzero capture interval")
        if (not isinstance(self.features, tuple) or len(self.features) > 4096
                or not all(isfinite(v) for v in self.features)):
            raise ValueError("features must be a finite immutable tuple")
        if self.available and not self.features:
            raise ValueError("available samples need features; missing data is not silence")
        if not self.available and self.features:
            raise ValueError("unavailable samples must not contain features")
        if self.attribution_verified and not self.subject_id:
            raise ValueError("verified attribution needs an explicit subject")


@dataclass(frozen=True)
class EvidenceWindow:
    epoch: str
    subject_id: str
    start: float
    end: float
    visual: tuple[SignalFeatures, ...]
    audio: tuple[SignalFeatures, ...]
    room_audio: tuple[SignalFeatures, ...]
    reasons: tuple[str, ...]

    @property
    def aligned(self) -> bool:
        """Transport/coverage check only, not medical or model validity."""
        return not self.reasons


class FeatureTimeline:
    """Single-owner timeline. Caller serializes producers and resets recording epochs.

    Every source is explicitly registered with a generation. Reconnect must use a
    larger generation; old callbacks cannot reintroduce pre-reconnect evidence.
    Source timestamps may arrive out of order and are sorted at query time.
    """
    def __init__(self, epoch: str, *, horizon_s: float = 10.0,
                 max_samples: int = 2048, max_gap_s: float = .25):
        if not epoch or not isfinite(horizon_s) or horizon_s <= 0:
            raise ValueError("positive horizon and epoch required")
        if not isinstance(max_samples, int) or max_samples < 1:
            raise ValueError("positive sample capacity required")
        if not isfinite(max_gap_s) or not 0 <= max_gap_s < horizon_s:
            raise ValueError("invalid gap tolerance")
        self.epoch = epoch
        self.horizon_s, self.max_samples, self.max_gap_s = horizon_s, max_samples, max_gap_s
        self._sources: dict[str, int] = {}
        self._samples: deque[SignalFeatures] = deque(maxlen=max_samples)
        self._watermark: float | None = None

    def reset(self, epoch: str):
        if not epoch or epoch == self.epoch:
            raise ValueError("reset requires a new recording epoch")
        self.epoch = epoch
        self._sources.clear()
        self._samples.clear()
        self._watermark = None

    def register_source(self, source_id: str, generation: int):
        if not source_id or not isinstance(generation, int) or generation < 0:
            raise ValueError("invalid source generation")
        old = self._sources.get(source_id)
        if old is not None and generation <= old:
            raise ValueError("reconnect generation must increase")
        self._sources[source_id] = generation
        self._samples = deque((s for s in self._samples if s.source_id != source_id),
                              maxlen=self.max_samples)

    def append(self, sample: SignalFeatures, *, now: float) -> bool:
        if not isfinite(now):
            raise ValueError("finite monotonic now required")
        if self._watermark is not None and now < self._watermark:
            raise ValueError("timeline clock must not move backwards")
        self._watermark = now
        self._prune(now)
        if sample.epoch != self.epoch or self._sources.get(sample.source_id) != sample.generation:
            return False
        if sample.end > now or sample.start < now - self.horizon_s:
            return False
        self._samples.append(sample)
        return True

    def _prune(self, now: float):
        self._samples = deque((s for s in self._samples if s.end >= now - self.horizon_s),
                              maxlen=self.max_samples)

    def window(self, subject_id: str, *, start: float, end: float, now: float) -> EvidenceWindow:
        if (not subject_id or not all(isfinite(t) for t in (start, end, now))
                or not start < end <= now or end - start > self.horizon_s):
            raise ValueError("invalid subject/window")
        if self._watermark is not None and now < self._watermark:
            raise ValueError("timeline clock must not move backwards")
        self._watermark = now
        self._prune(now)
        # Do not leak future features or a summary of audio outside this window.
        candidates = sorted((s for s in self._samples if s.start >= start and s.end <= end),
                            key=lambda s: (s.start, s.end, s.source_id))
        def valid(s):
            return s.available and s.clock_verified
        visual = tuple(s for s in candidates if s.modality == "visual" and valid(s)
                       and s.subject_id == subject_id and s.attribution_verified)
        audio = tuple(s for s in candidates if s.modality == "audio" and valid(s)
                      and s.subject_id == subject_id and s.attribution_verified)
        room = tuple(s for s in candidates if s.modality == "audio" and valid(s)
                     and not s.attribution_verified)
        reasons = []
        if start < now - self.horizon_s or end < now - self.max_gap_s:
            reasons.append("stale_window")
        for name, samples in (("visual", visual), ("audio", audio)):
            if not samples:
                reasons.append(f"{name}_unavailable")
            elif not self._covers(samples, start, end):
                reasons.append(f"{name}_gap")
        if room and not audio:
            reasons.append("audio_subject_unverified")
        # Invalid packets are retained as metadata; never substitute zero features.
        for s in candidates:
            if s.subject_id == subject_id or (s.modality == "audio" and s.subject_id is None):
                if not s.clock_verified:
                    reasons.append(f"{s.modality}_clock_unverified")
                if not s.available:
                    reasons.append(f"{s.modality}_source_unavailable")
        return EvidenceWindow(self.epoch, subject_id, start, end, visual, audio, room,
                              tuple(dict.fromkeys(reasons)))

    def _covers(self, samples, start, end):
        cursor = start
        for s in samples:
            if s.start - cursor > self.max_gap_s + 1e-9:
                return False
            cursor = max(cursor, s.end)
        return end - cursor <= self.max_gap_s + 1e-9

    def __len__(self):
        return len(self._samples)
