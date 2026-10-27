package vn.childsafe.core

import org.junit.Assert.*
import org.junit.Test

class MonitorStateTest {
    private fun event(seq: Long = 0, risk: Risk = Risk.DANGER, observation: Observation = Observation.DEGRADED,
                      id: String = "event-$seq", epoch: String = "one", alert: Boolean = true) =
        CameraResult(epoch, seq, id, risk, observation, "Test event", alert)

    @Test fun startsUnknownAndBootstrapDoesNotReplayAlarm() {
        val s = MonitorState()
        assertFalse(s.view(0).green)
        val v = s.accept(event(), "one", 0, 1, bootstrap = true)
        assertEquals(Risk.DANGER, v.risk)
        assertFalse(v.notify)
    }

    @Test fun separateObservationFromRiskAndFreshness() {
        val s = MonitorState()
        assertFalse(s.accept(event(risk = Risk.NORMAL), "one", 0, 0).green)
        assertTrue(s.accept(event(1, Risk.NORMAL, Observation.OBSERVABLE), "one", 0, 1).green)
        assertFalse(s.view(2501).green)
        assertEquals(Connection.STALE, s.view(2501).connection)
    }

    @Test fun staleWrongEpochAndDuplicateResultsAreIgnored() {
        val s = MonitorState()
        assertFalse(s.accept(event(), "one", 2500, 0).notify)
        assertFalse(s.accept(event(epoch = "old"), "one", 0, 1).notify)
        assertTrue(s.accept(event(), "one", 0, 2).notify)
        assertFalse(s.accept(event(), "one", 0, 3).notify)
        assertEquals(Risk.DANGER, s.view(3).risk)
    }

    @Test fun muteAndCooldownNeverLowerRisk() {
        val s = MonitorState()
        assertTrue(s.accept(event(risk = Risk.CAUTION), "one", 0, 0).notify)
        assertFalse(s.accept(event(1, Risk.CAUTION), "one", 0, 1).notify)
        assertTrue(s.accept(event(2), "one", 0, 2).notify) // escalation bypasses caution cooldown
        s.muteFor(3, 1000)
        assertFalse(s.accept(event(3), "one", 0, 3).notify)
        assertEquals(Risk.DANGER, s.view(3).risk)
    }

    @Test fun disconnectKeepsLastRiskButCannotBeGreen() {
        val s = MonitorState()
        s.accept(event(), "one", 0, 0)
        val v = s.disconnected(1)
        assertEquals(Connection.OFFLINE, v.connection)
        assertEquals(Risk.DANGER, v.risk)
        assertEquals(Observation.UNOBSERVABLE, v.observation)
        assertFalse(v.green)
    }

    @Test fun newEpochClearsStateWithoutMovingRiskToAnotherPersonOrSession() {
        val s = MonitorState()
        s.accept(event(), "one", 0, 0)
        s.beginEpoch("two")
        assertNull(s.view(1).risk)
        assertFalse(s.accept(event(epoch = "one"), "two", 0, 2).notify)
    }

    @Test fun replayedEventIdWithNewSequenceDoesNotBeep() {
        val s = MonitorState()
        assertTrue(s.accept(event(), "one", 0, 0).notify)
        assertFalse(s.accept(event(1, id = "event-0"), "one", 0, 31000).notify)
    }

    @Test fun displayedSnapshotExpiresEvenWhileNetworkRequestIsBlocked() {
        val s = MonitorState()
        val snapshot = s.accept(event(risk = Risk.NORMAL, observation = Observation.OBSERVABLE), "one", 0, 0)
        assertTrue(snapshot.atTime(2499).green)
        assertFalse(snapshot.atTime(2500).green)
        assertEquals(Connection.STALE, snapshot.atTime(8000).connection)
    }
}
